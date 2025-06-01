from typing import Tuple, List
import os
import argparse
import numpy as np
from jiwer import wer

import torch
from torch import Tensor

import whisper
from whisper.audio import CHUNK_LENGTH
from whisper.tokenizer import get_tokenizer, Tokenizer
from whisper.normalizers import EnglishTextNormalizer
from whisper.model_avwhisper import ModelDimensions, Whisper
from whisper.decoding import (
    DecodingTask,
    DecodingOptions,
    Inference,
    MaximumLikelihoodRanker,
    BeamSearchDecoder,
    GreedyDecoder,
    SuppressBlank,
    SuppressTokens,
    ApplyTimestampRules,
    DecodingResult
)
from whisper.utils import compression_ratio

from data_utils.dataloader_av import get_dataloader
from training_avwhisper import System


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test script")
    # Dataloader-related arguments
    parser.add_argument(
        "--test_manifest_path",
        type=str,
        default=None,
        help="Where to load manifest file for test set."
    )
    parser.add_argument(
        "--test_label_path",
        type=str,
        default=None,
        help="Where to load label file for test set."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="support language"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use for training",
    )
    parser.add_argument(
        "--whisper_checkpoint_file",
        type=str,
        required=True,
        help="pre-trained whisper checkpoint file path"
    )
    parser.add_argument(
        "--prompt_length",
        type=int,
        required=True,
        help="soft prompt length",
    )

    parser.add_argument(
        "--use_mlp",
        action='store_true',
        help="whether to reparameterize the prompt",
    )
    parser.add_argument(
        "--deep",
        action='store_true',
        help="deep prompting",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=32,
        help="prompting depth",
    )
    parser.add_argument("--model_name", type=str, required=True, help="model_name")

    return parser


def test(loader, tokenizer, model, mytask, normal, device):
    real_wer_scores = []
    ideal_wer_scores = []
    # pbar = tqdm(loader)
    with torch.no_grad():
        for x, vfeat, y_in, y_out, text in loader:
            x, vfeat, y_in = x.to(device), vfeat.to(device), y_in.to(device)
            print(text)
            truth_results = mytask.run(x, vfeat)
            for result, t in zip(truth_results, text):
                wer_ = wer(normal(t), normal(result.text))
                real_wer_scores.append(wer_)
                print(result.text, wer_)


            logits = model(x, vfeat, y_in)
            tokens = logits.argmax(dim=-1)
            for token, t in zip(tokens, text):
                result = tokenizer.decode(token).replace("<|en|>", "").replace("<|transcribe|>", "").replace("<|notimestamps|>", "").replace("<|endoftext|>", "")
                wer_ = wer(normal(t), normal(result))
                ideal_wer_scores.append(wer_)
                print(tokenizer.decode(token), wer_)
                # wer_scores.append(wer_)
                # pbar.set_postfix({'wer': wer_, 'wer_mean': sum(wer_scores)/len(wer_scores), 'text': result})
                # print(result, wer_, t)

            print()
    return sum(real_wer_scores)/len(real_wer_scores), sum(ideal_wer_scores)/len(ideal_wer_scores)


class PyTorchInference(Inference):
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length

    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        return self.model.decoder(tokens, audio_features)


class AVWhisperDecodingTask(DecodingTask):
    def __init__(self, system, options):
        self.system = system

        language = options.language or "en"
        tokenizer = get_tokenizer(
            system.model.is_multilingual,
            num_languages=system.model.num_languages,
            language=language,
            task=options.task,
        )
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = system.model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or system.model.dims.n_text_ctx // 2

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = PyTorchInference(system.model, len(self.initial_tokens))

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(SuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(SuppressTokens(self._get_suppress_tokens()))
        if not options.without_timestamps:
            precision = CHUNK_LENGTH / system.model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision
                )
            self.logit_filters.append(
                ApplyTimestampRules(
                    tokenizer, self.sample_begin, max_initial_timestamp_index
                )
            )

    def _get_audio_features(self, mel, vfeat):
        prompts_encoder = self.system.prompt_layer(vfeat)
        audio_features = self.system.model.encoder(mel, prompts_encoder)

        if audio_features.dtype != (
            torch.float16 if self.options.fp16 else torch.float32
        ):
            return TypeError(
                f"audio_features has an incorrect dtype: {audio_features.dtype}"
            )

        return audio_features

    def _detect_language(self, audio_features, tokens):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.system.model.detect_language(
                audio_features, self.tokenizer
            )
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                tokens[:, self.sot_index + 1] = lang_tokens  # write language tokens

        return languages, lang_probs

    def _main_loop(self, audio_features: Tensor, tokens: Tensor):
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch

        for i in range(self.sample_len):
            logits = self.inference.logits(tokens, audio_features)

            if (
                i == 0 and self.tokenizer.no_speech is not None
            ):  # save no_speech_probs
                probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

            # now we need to consider the logits at the last token only
            logits = logits[:, -1]

            # apply the logit filters, e.g. for suppressing or applying penalty to
            for logit_filter in self.logit_filters:
                logit_filter.apply(logits, tokens)

            # expand the tokens tensor with the selected next tokens
            tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

            if completed or tokens.shape[-1] > self.n_ctx:
                break

        return tokens, sum_logprobs, no_speech_probs

    @torch.no_grad()
    def run(self, mel: Tensor, vfeat: Tensor) -> List[DecodingResult]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        audio_features = self._get_audio_features(mel, vfeat)  # encoder forward pass
        tokens: Tensor = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id":
            return [
                DecodingResult(
                    audio_features=features, language=language, language_probs=probs
                )
                for features, language, probs in zip(
                    audio_features, languages, language_probs
                )
            ]

        # repeat text tensors by the group size, for beam search or best-of-n sampling
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [
            [t[self.sample_begin: (t == tokenizer.eot).nonzero()[0, 0]] for t in s]
            for s in tokens
        ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        fields = (
            texts,
            languages,
            tokens,
            audio_features,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]


if __name__ == '__main__':
    args = get_parser().parse_args()
    print(args)

    tokenizer = get_tokenizer(multilingual=".en" not in os.path.basename(args.whisper_checkpoint_file), task="transcribe")
    with open(args.whisper_checkpoint_file, "rb") as fp:
        checkpoint = torch.load(fp, map_location=args.device)

    checkpoint["dims"]['depth'] = args.depth
    dims = ModelDimensions(**checkpoint["dims"])
    print(dims)
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    system = System(
        model,
        args.prompt_length,
        args.use_mlp,
        depth=args.depth if args.deep else 1
    )

    finetuned_weight = torch.load('./checkpoint/' + args.model_name)
    system.load_state_dict(finetuned_weight, strict=False)

    system.to(args.device)
    system.eval()

    #  -1 is for the special token `sot_prev` and the other half is for the transcribed tokens
    max_prompt_length = model.dims.n_text_ctx // 2 - 1
    dataloader = get_dataloader(
        manifest_path=args.test_manifest_path,
        label_path=args.test_label_path,
        tokenizer=tokenizer,
        batch_size=1,
        fp16=False,
        language=args.language,
        no_timestamps_training=True,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=1,
        no_timestamps_rate=0.0,
        shuffle=False,
        context_len=0,
        n_workers=4,
        dev=True
    )

    options = whisper.DecodingOptions(without_timestamps=True, fp16=False, language="en")
    print(options)
    task = AVWhisperDecodingTask(system, options)
    normalizer = EnglishTextNormalizer()

    score = test(loader=dataloader, tokenizer=tokenizer, model=system, mytask=task, normal=normalizer, device=args.device)
    print(score)














