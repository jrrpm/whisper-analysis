import torch
import string
from transformers import WhisperTokenizer


def _splitTokensUnicode(tokens: torch.Tensor, tokenizer: WhisperTokenizer):
    words = []
    word_tokens = []
    current_tokens = []
    for token in tokens.tolist():
        current_tokens.append(token)
        decoded = tokenizer.decode([token], skip_special_tokens=False, decode_with_timestamps=True)
        if "\ufffd" not in decoded:
            words.append(decoded)
            word_tokens.append(current_tokens)
            current_tokens = []
    return words, word_tokens


def _splitTokens(tokens: torch.Tensor, tokenizer: WhisperTokenizer):
    subwords, subword_tokens_list = _splitTokensUnicode(tokens, tokenizer)
    words = []
    word_tokens = []
    for subword, subword_tokens in zip(subwords, subword_tokens_list):
        special = subword_tokens[0] >= tokenizer.added_tokens_encoder["<|endoftext|>"]
        with_space = subword.startswith(" ")
        punctuation = subword.strip() in string.punctuation
        if special or with_space or punctuation:
            words.append(subword)
            word_tokens.append(subword_tokens)
        else:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(subword_tokens)
    return words, word_tokens


def _tokenizeTranscription(tokenizer: WhisperTokenizer, transcription, duration, apt):
    return torch.tensor(
        [
            tokenizer.added_tokens_encoder["<|startoftranscript|>"],
            tokenizer.added_tokens_encoder["<|0.00|>"],
        ]
        + tokenizer(transcription, add_special_tokens=False).input_ids
        + [
            tokenizer.added_tokens_encoder["<|0.00|>"] + duration // apt,
            tokenizer.added_tokens_encoder["<|endoftext|>"],
        ]
    )


def getTokens(tokenizer: WhisperTokenizer, transcription, duration, apt):
    tokens = _tokenizeTranscription(tokenizer, transcription, duration, apt)
    return _splitTokens(tokens, tokenizer)
