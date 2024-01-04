import torch
import numpy as np
from dtw import dtw
from scipy.ndimage import median_filter
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from lib.utils import getTokens
from lib.plot import plotHead
from lib.dataset import getDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "distil-whisper/distil-small.en"

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).cuda()
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
audio_samples_per_token = processor.feature_extractor.hop_length * 2
audio_time_per_token = audio_samples_per_token / processor.feature_extractor.sampling_rate


def getCrossAttentions(processor: WhisperProcessor, model: WhisperForConditionalGeneration, waveform):
    with torch.no_grad():
        input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_features.cuda()
        predicted_ids = model.generate(input_values, return_dict_in_generate=True, output_attentions=True)
    cross_attentions = []
    for cross_attn in predicted_ids.cross_attentions:
        cross_attentions.append(torch.cat(cross_attn).squeeze().cpu())
    return cross_attentions


def normalizeWeights(duration, cross_attentions):
    weights = np.array(cross_attentions)
    weights = np.transpose(weights, (1, 2, 0, 3))
    weights = weights[:, :, :, : duration // audio_samples_per_token]
    weights = median_filter(weights, (1, 1, 1, 10))  # 10 filter width
    weights = torch.tensor(weights * 10)  # sf
    return weights / weights.norm(dim=-2, keepdim=True)


def calculateAligment(layerHead):
    metric = "cosine"
    np_head = -layerHead.double().numpy()
    alignment = dtw(np_head, dist_method=metric)
    return alignment


for audio, transcription in getDataset(10):
    duration = len(audio)
    words, tokens = getTokens(processor.tokenizer, transcription, duration, audio_samples_per_token)
    cross_attentions = getCrossAttentions(processor, model, audio)
    matrix = normalizeWeights(duration, cross_attentions)
    for li, layer in enumerate(matrix):
        for h, head in enumerate(layer):
            title = f"layer: {li}, head: {h}"
            alignment = calculateAligment(head)
            plotHead(title, head, alignment, words, tokens, audio_time_per_token)
