#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from typing import Optional
import os.path
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from whisper.audio import CHUNK_LENGTH, N_FRAMES, log_mel_spectrogram, pad_or_trim, load_audio
from whisper.tokenizer import Tokenizer

from .dataloader import AudioDataset


class AudioVisualDataset(AudioDataset):
    def __init__(
        self,
        manifest_path: str,
        label_path: str,
        tokenizer: Tokenizer,
        fp16: bool = True,
        language: str = "en",
        no_timestamps_training: bool = False,
        max_prompt_length: int = 223,  # The maximum number of tokens to use for the prompt
        prompt_use_rate: float = 0.5,
        no_timestamps_rate: float = 0.5,
        context_len: int = 0,
        dev: bool = False,
        is_debug: bool = False
    ) -> None:
        super().__init__(
            manifest_path,
            label_path,
            tokenizer,
            fp16,
            language,
            no_timestamps_training,
            max_prompt_length,
            prompt_use_rate,
            no_timestamps_rate,
            context_len,
            dev
        )

        if is_debug:
            self.names = self.names[:5]
            self.labels = self.labels[:5]

    def _calculate_mel(
        self, audio_path: str, next_partial_segment_start: Optional[float], no_timestamps: bool
    ) -> torch.Tensor:
        mel = log_mel_spectrogram(audio_path)

        if self.context_len > 0:
            ## insert some zero frames to hold the places for prompts
            n_frames = self.context_len * 2  # 1 speaker embedding + 8 soft prompts, the down-sampling factor is 2 in encoder conv layers
            place_holder = torch.zeros((mel.size(0), n_frames))
            mel = torch.concat([place_holder, mel], dim=1)

        if no_timestamps and next_partial_segment_start is not None:
            mel = mel[:, : int(next_partial_segment_start * self.num_frames_per_second)]
        mel = pad_or_trim(mel, N_FRAMES)
        if self.fp16:
            mel = mel.half()

        return mel

    def load_video_seq(self, video_name):
        feats = np.load(os.path.join(self.dataset_root, video_name))["data"]
        video_feats = torch.tensor(feats, dtype=torch.float)
        return video_feats

    def __getitem__(self, index: int):
        video_fn, audio_fn = self.names[index]
        text = self.labels[index]
        no_timestamps = self.no_timestamps_training or torch.rand(1) < self.no_timestamps_rate
        prompt_tokens = self._get_prompt_tokens('@' * (self.context_len))  # hole the place where will be filled with speaker embedding.
        # prompt_tokens = []
        text_tokens, next_partial_segment_start = self._get_text_tokens(text.lower(), no_timestamps)
        is_text_empty = len(text_tokens) == 0
        special_tokens = self._get_special_tokens(is_text_empty, self.language, no_timestamps)

        decoder_input = prompt_tokens + special_tokens + text_tokens
        if len(decoder_input) > self.model_n_text_ctx:
            raise ValueError(f"Input is too long: {audio_fn} (length: {len(decoder_input)})")

        decoder_output = self._construct_decoder_output(prompt_tokens, special_tokens, text_tokens)
        mel = self._calculate_mel(os.path.join(self.dataset_root, audio_fn), next_partial_segment_start, no_timestamps)
        video_feats = self.load_video_seq(video_fn)
        if self.dev:
            return (
                mel,
                video_feats,
                torch.tensor(decoder_input, dtype=torch.long),
                torch.tensor(decoder_output, dtype=torch.long),
                text
            )
        else:
            return (
                mel,
                video_feats,
                torch.tensor(decoder_input, dtype=torch.long),
                torch.tensor(decoder_output, dtype=torch.long),
            )


def collate_fn(data):
    if len(data[0]) == 4:
        x, vfeat, y_in, y_out = zip(*data)
        x = pad_sequence(x, batch_first=True, padding_value=0)
        vfeat = pad_sequence(vfeat, batch_first=True, padding_value=0)
        y_in = pad_sequence(y_in, batch_first=True, padding_value=0)
        y_out = pad_sequence(y_out, batch_first=True, padding_value=-100)
        return x, vfeat, y_in, y_out
    else:
        x, vfeat, y_in, y_out, text = zip(*data)
        x = pad_sequence(x, batch_first=True, padding_value=0)
        vfeat = pad_sequence(vfeat, batch_first=True, padding_value=0)
        y_in = pad_sequence(y_in, batch_first=True, padding_value=0)
        y_out = pad_sequence(y_out, batch_first=True, padding_value=-100)
        return x, vfeat, y_in, y_out, text


def get_dataloader(
    manifest_path: str,
    label_path: str,
    tokenizer: Tokenizer,
    batch_size: int = 1,
    fp16: bool = True,
    language: str = "en",
    no_timestamps_training: bool = False,
    max_prompt_length: int = 223,
    prompt_use_rate: float = 0.5,
    no_timestamps_rate: float = 0.5,
    shuffle: bool = True,
    n_workers: int = 0,
    context_len: int = 8,
    dev: bool = False,
    is_debug: bool = False,
) -> DataLoader:
    dataset = AudioVisualDataset(
        manifest_path,
        label_path,
        tokenizer,
        fp16=fp16,
        language=language,
        no_timestamps_training=no_timestamps_training,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=prompt_use_rate,
        no_timestamps_rate=no_timestamps_rate,
        context_len=context_len,
        dev=dev,
        is_debug=is_debug,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
