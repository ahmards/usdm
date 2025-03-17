# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import torch
import librosa
import re
import os
from scipy.io.wavfile import write
from seamless_communication.models.unit_extractor import UnitExtractor
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor.models import _MODELS
from voicebox.util.model_util import reconstruct_speech, initialize_decoder
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import tempfile
import io

_MODELS['CustomMistralForCausalLM'] = _MODELS['MistralForCausalLM']

# Unified template generator
def default_template(user_unit, user_text=None, agent_text=None):
    template = (
        "Below is a conversation between the user and the agent. Each turn includes the user's speech and its corresponding transcript, "
        "along with the agent's response text and the corresponding speech.\n"
        "\n### User\n"
        f"{user_unit}<|correspond|>"
    )
    if user_text:
        template += f"{user_text}\n### Agent\n"
    if agent_text:
        template += f"{agent_text}<|correspond|>"
    return template

def strip_exact_multiple(text, patterns):
    for pattern in patterns:
        text = text.removeprefix(pattern).removesuffix(pattern)
    return text

# Custom logits processors
def bad_word_processor_unit2text(token_ids, logits):
    logits[32000:42003] = float("-inf")
    return logits

def bad_word_processor_text2text(token_ids, logits):
    logits[32002:42003] = float("-inf")
    return logits

def bad_word_processor_text2unit(token_ids, logits):
    logits[0:28705] = float("-inf")
    logits[28706:32002] = float("-inf")
    return logits

@torch.inference_mode()
def sample(user_wav, reference_path, model, unit_extractor, voicebox, vocoder,
           sampling_params_unit2text, sampling_params_text2text, 
           sampling_params_text2unit, pattern, device):
    user_unit = ''.join(
        [f'<|unit{i}|>' for i in unit_extractor.predict(torch.FloatTensor(user_wav).to(device), 35 - 1).cpu().tolist()]
    )

    model_input = default_template(user_unit=user_unit)
    outputs = model.generate([model_input], sampling_params_unit2text)
    user_text = strip_exact_multiple(outputs[0].outputs[0].text, ["\n", " "])

    model_input = default_template(user_unit=user_unit, user_text=user_text)
    outputs = model.generate([model_input], sampling_params_text2text)
    agent_text = strip_exact_multiple(outputs[0].outputs[0].text, ["\n", " ", "<|correspond|>"])

    model_input = default_template(user_unit=user_unit, user_text=user_text, agent_text=agent_text)
    outputs = model.generate([model_input], sampling_params_text2unit)
    agent_unit = strip_exact_multiple(outputs[0].outputs[0].text, ["\n", " "])

    matches = [int(x) for x in pattern.findall(agent_unit)]
    agent_unit = torch.LongTensor(matches).to(device)
    audio = reconstruct_speech(agent_unit, device, reference_path, unit_extractor, 
                              voicebox, vocoder, n_timesteps=50)
    return audio

app = FastAPI()

# Configuration
DEVICE = torch.device("cuda")
MODEL_CACHE_DIR = os.path.abspath("../model_cache")  # Corrected path
PATTERN = re.compile(r"<\|unit(\d+)\|>")

# Create model cache directory if not exists
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Initialize models once at startup
@torch.inference_mode()
def initialize_models():
    voicebox, vocoder = initialize_decoder(MODEL_CACHE_DIR, DEVICE)
    
    unit_extractor = UnitExtractor(
        "xlsr2_1b_v2",
        "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
        device=DEVICE
    )
    
    llm = LLM(
        model='naver-ai/USDM-DailyTalk',
        download_dir=MODEL_CACHE_DIR,
        gpu_memory_utilization=0.7
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        'naver-ai/USDM-DailyTalk', 
        cache_dir=MODEL_CACHE_DIR
    )
    
    return voicebox, vocoder, unit_extractor, llm, tokenizer

voicebox, vocoder, unit_extractor, model, tokenizer = initialize_models()

# Sampling parameters with logits processors
sampling_params_unit2text = SamplingParams(
    max_tokens=tokenizer.model_max_length,
    top_p=1.0,
    top_k=1,
    temperature=1.0,
    stop_token_ids=[tokenizer("\n").input_ids[-1]],
    logits_processors=[bad_word_processor_unit2text]
)

sampling_params_text2text = SamplingParams(
    max_tokens=tokenizer.model_max_length,
    top_p=1.0,
    top_k=1,
    temperature=1.0,
    stop_token_ids=[tokenizer("<|correspond|>").input_ids[-1]],
    logits_processors=[bad_word_processor_text2text]
)

sampling_params_text2unit = SamplingParams(
    max_tokens=tokenizer.model_max_length,
    top_p=1.0,
    top_k=1,
    temperature=1.0,
    stop_token_ids=[28705],
    logits_processors=[bad_word_processor_text2unit]
)

@app.post("/generate_speech/")
async def generate_speech(
    input_file: UploadFile = File(...),
    reference_file: UploadFile = File(None)
):
    try:
        # Process input audio in memory
        input_content = await input_file.read()
        user_wav, _ = librosa.load(io.BytesIO(input_content), sr=16000)
        
        # Process reference file
        reference_path = None
        if reference_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(await reference_file.read())
                reference_path = tmp.name
        
        # Generate speech
        audio = sample(
            user_wav=user_wav,
            reference_path=reference_path,
            model=model,
            unit_extractor=unit_extractor,
            voicebox=voicebox,
            vocoder=vocoder,
            sampling_params_unit2text=sampling_params_unit2text,
            sampling_params_text2text=sampling_params_text2text,
            sampling_params_text2unit=sampling_params_text2unit,
            pattern=PATTERN,
            device=DEVICE
        )
        
        # Create in-memory response
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            write(tmp.name, vocoder.h.sampling_rate, audio)
            return FileResponse(tmp.name, media_type="audio/wav", filename="response.wav")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if reference_path and os.path.exists(reference_path):
            os.remove(reference_path)
