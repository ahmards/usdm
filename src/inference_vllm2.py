# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import torch
import librosa
import re
from scipy.io.wavfile import write
from seamless_communication.models.unit_extractor import UnitExtractor
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor.models import _MODELS
from voicebox.util.model_util import reconstruct_speech, initialize_decoder
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import tempfile
import io

# Unified template generator for ASR, T2T, and TTS stages
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


# Function to strip multiple exact patterns from a string
def strip_exact_multiple(text, patterns):
    for pattern in patterns:
        if text.startswith(pattern):
            text = text[len(pattern):]
        if text.endswith(pattern):
            text = text[:-len(pattern)]
    return text


@torch.inference_mode()
def sample(user_wav, reference_path, model, unit_extractor, voicebox, vocoder,
           sampling_params_unit2text, sampling_params_text2text, sampling_params_text2unit):
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
    audio = reconstruct_speech(agent_unit, device, reference_path, unit_extractor, voicebox, vocoder,
                               n_timesteps=50)
    
    return audio


# FastAPI setup
app = FastAPI()

device = torch.device("cuda")

# Load voicebox, vocoder configuration and checkpoint
voicebox, vocoder = initialize_decoder("model_cache", device)

unit_extractor = UnitExtractor("xlsr2_1b_v2",
                               "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
                               device=device)

model = LLM(model='naver-ai/USDM-DailyTalk', download_dir="model_cache", gpu_memory_utilization=0.7)
tokenizer = AutoTokenizer.from_pretrained('naver-ai/USDM-DailyTalk', cache_dir="model_cache")

sampling_params_unit2text = SamplingParams(
    max_tokens=tokenizer.model_max_length, top_p=1.0, top_k=1, temperature=1.0,
    stop_token_ids=[tokenizer("\n").input_ids[-1]]
)

sampling_params_text2text = SamplingParams(
    max_tokens=tokenizer.model_max_length, top_p=1.0, top_k=1, temperature=1.0,
    stop_token_ids=[tokenizer("<|correspond|>").input_ids[-1]]
)

sampling_params_text2unit = SamplingParams(
    max_tokens=tokenizer.model_max_length, top_p=1.0, top_k=1, temperature=1.0,
    stop_token_ids=[28705]
)

pattern = re.compile(r"<\|unit(\d+)\|>")

@app.post("/generate_speech/")
async def generate_speech(input_file: UploadFile = File(...), reference_file: UploadFile = None):
    input_path = f"/tmp/{input_file.filename}"
    with open(input_path, "wb") as buffer:
        buffer.write(await input_file.read())
    
    reference_path = None
    if reference_file:
        reference_path = f"/tmp/{reference_file.filename}"
        with open(reference_path, "wb") as buffer:
            buffer.write(await reference_file.read())
    
    # Load user audio file
    user_wav, sr = librosa.load(input_path, sr=16000)
    
    # Perform inference
    try:
        audio = sample(user_wav, reference_path, model, unit_extractor, voicebox, vocoder,
                       sampling_params_unit2text, sampling_params_text2text, sampling_params_text2unit)
        
        # Save output to temporary file
        output_path = "/tmp/output.wav"
        write(output_path, vocoder.h.sampling_rate, audio)
        
        # Return the generated audio as a response
        return FileResponse(output_path, media_type="audio/wav")
    except Exception as e:
        return {"error": str(e)}
