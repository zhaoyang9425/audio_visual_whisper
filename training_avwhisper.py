#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import argparse
import random
import numpy as np
from tqdm import tqdm
from jiwer import wer

import torch
import torch.nn as nn
import torch.nn.functional as F

from whisper.tokenizer import get_tokenizer
from whisper.normalizers import EnglishTextNormalizer
from whisper.model_llama_adapter_v24 import ModelDimensions, Whisper

from data_utils.dataloader_av import get_dataloader


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a Whisper model for ASR")
    # Dataloader-related arguments
    parser.add_argument(
        "--train_manifest_path",
        type=str,
        default=None,
        help="Where to load manifest file for train set."
    )
    parser.add_argument(
        "--train_label_path",
        type=str,
        default=None,
        help="Where to load label file for train set."
    )
    parser.add_argument(
        "--test_manifest_path",
        type=str,
        default=None,
        help="Where to load manifest file for eval set."
    )
    parser.add_argument(
        "--test_label_path",
        type=str,
        default=None,
        help="Where to load label file for eval set."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="support language"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--dev_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--step_size", type=int, default=5, help="Update learning rate per step size.")
    parser.add_argument(
        "--no_timestamps_training",
        default=True,
        help="Always use the no-timestamps training mode",
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use for training",
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
    parser.add_argument("--whisper_checkpoint_file", type=str, required=True, help="freezed whisper checkpoint file path")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--exp_name", type=str, required=True, help="exp_name")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class ResMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn_dim = dim//2
        self.net = nn.Sequential(
            nn.Linear(dim, self.bn_dim),
            nn.ReLU(),
            nn.Linear(self.bn_dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        return x + self.net(x)


class Prompting(nn.Module):
    def __init__(self, dim, prompt_length, use_mlp=True, depth=32):
        super(Prompting, self).__init__()
        self.use_mlp = use_mlp
        self.depth = depth

        self.video_embedding = nn.Linear(1024, dim)

        self.soft_prompt_encoder = nn.Parameter(torch.Tensor(depth, prompt_length, dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.soft_prompt_encoder)

        if self.use_mlp:
            self.mlp_encoder = nn.ModuleList([ResMLP(dim) for _ in range(depth)])

    def forward(self, vfeat):
        bsz = vfeat.size(0)
        vfeat = self.video_embedding(vfeat)
        prompt_encoder = list(self.soft_prompt_encoder.chunk(self.depth, dim=0))

        for i in range(self.depth):
            # TODO:
            prompt_encoder[i] = torch.cat((prompt_encoder[i].repeat(bsz, 1, 1), vfeat), dim=1)
            if self.use_mlp:
                prompt_encoder[i] = self.mlp_encoder[i](prompt_encoder[i])

        return prompt_encoder


class System(nn.Module):
    def __init__(self, model, prompt_length, use_mlp, depth):
        super(System, self).__init__()
        self.model = model
        self.prompt_layer = Prompting(
            dim=model.dims.n_text_state,
            prompt_length=prompt_length,
            use_mlp=use_mlp,
            depth=depth
        )

    def forward(self, x, vfeat, y_in):
        prompts_encoder = self.prompt_layer(vfeat)
        logits = self.model.decoder(y_in, self.model.encoder(x, prompts_encoder))
        return logits


def train_test(
        model,
        device,
        train_loader,
        test_loader,
        epochs,
        optimizer,
        scheduler,
        exp_name,
        tokenizer,
        normalizer,
        train_loss=None,
        init_epoch=-1,
):
    model.to(device)
    train_loss = [] if train_loss is None else train_loss.tolist()
    for e in range(init_epoch + 1, epochs):
        model.train()
        pbar = tqdm(train_loader)
        for i, (x, vfeat, y_in, y_out) in enumerate(pbar):
            x, vfeat, y_in, y_out = x.to(device), vfeat.to(device), y_in.to(device), y_out.to(device)
            logits = model(x, vfeat, y_in)
            loss = F.cross_entropy(logits.transpose(1, 2), y_out)
            loss.backward()
            # if (i + 1) % 16 == 0:
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.detach().cpu().numpy())
            pbar.set_postfix({"loss": train_loss[-1], 'loss_mean': np.sum(train_loss) / len(train_loss)})
        scheduler.step()
        torch.save(save_state_dict(model), './checkpoint/avwhisper_' + exp_name + str(e))

        if test_loader is not None:
            wer_mean = epoch_validate(model, device, val_loader=test_loader, tokenizer=tokenizer, normalizer=normalizer)
            print(f"Epoch : {e}, validation WER: {wer_mean}")


def epoch_validate(
        model,
        device,
        val_loader,
        tokenizer,
        normalizer
):
    model.to(device)
    model.eval()
    wer_scores = []
    pbar = tqdm(val_loader)
    with torch.no_grad():
        for x, vfeat, y_in, y_out, text in pbar:
            x, vfeat, y_in = x.to(device), vfeat.to(device), y_in.to(device)
            logits = model(x, vfeat, y_in)
            tokens = logits.argmax(dim=-1)
            for token, t in zip(tokens, text):
                result = tokenizer.decode(token).replace("<|en|>", "").replace("<|transcribe|>", "").replace(
                    "<|notimestamps|>", "").replace("<|endoftext|>", "")
                wer_ = wer(normalizer(t), normalizer(result))
                wer_scores.append(wer_)
                pbar.set_postfix({'wer': wer_, 'wer_mean': sum(wer_scores) / len(wer_scores), 'text': result})
    return sum(wer_scores) / len(wer_scores)


def save_state_dict(model):
    to_return = {}
    my_state_dict = model.state_dict()
    for k in my_state_dict:
        if 'lora' in k or 'prompt' in k:
            to_return[k] = my_state_dict[k]
        elif ('query' in k or 'value' in k) and '.bias' in k:  # 对应 mark_only_lora_as_trainable 方法中参数 bias 为 lora_only
            to_return[k] = my_state_dict[k]
    return to_return


def main():
    args = get_parser().parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    print(args)

    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    tokenizer = get_tokenizer(multilingual=".en" not in os.path.basename(args.whisper_checkpoint_file), task="transcribe")
    with open(args.whisper_checkpoint_file, "rb") as fp:
        checkpoint = torch.load(fp, map_location=args.device)

    checkpoint["dims"]['depth'] = args.depth
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    system = System(
        model,
        args.prompt_length,
        args.use_mlp,
        depth=args.depth if args.deep else 1
    )
    print(system)

    #  -1 is for the special token `sot_prev` and the other half is for the transcribed tokens
    max_prompt_length = model.dims.n_text_ctx // 2 - 1
    fp16 = False
    train_loader = get_dataloader(
        manifest_path=args.train_manifest_path,
        label_path=args.train_label_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        fp16=fp16,
        language=args.language,
        no_timestamps_training=args.no_timestamps_training,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=1,
        no_timestamps_rate=0.0,
        shuffle=True,
        context_len=0,
        n_workers=4,
    )

    test_loader = None
    if args.test_manifest_path is not None:
        test_loader = get_dataloader(
            manifest_path=args.test_manifest_path,
            label_path=args.test_label_path,
            tokenizer=tokenizer,
            batch_size=16,
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

    # freeze the pure whisper model
    for name, p in system.named_parameters():
        if 'lora' in name or 'prompt' in name:
            p.requires_grad = True
        elif ('query' in name or 'value' in name) and '.bias' in name:  # 对应 mark_only_lora_as_trainable 方法中参数 bias 为 lora_only
            p.requires_grad = True
        else:
            p.requires_grad = False

    trainable_params = sum([param.nelement() for param in system.parameters() if param.requires_grad])
    total_params = sum([param.nelement() for param in system.parameters()])
    print('Number of trainable parameter: % .4fM, || trainable: % .3f' % (trainable_params / 1e6, 100*trainable_params/total_params))
    print('Number of total parameter: % .4fM' % (total_params / 1e6))
    print(f"trainable parameters: {[name for name, p in system.named_parameters() if p.requires_grad]}")
    optimizer = torch.optim.AdamW([param for param in system.parameters() if param.requires_grad], lr=args.lr,)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1, verbose=True)

    normalizer = EnglishTextNormalizer()
    train_test(
        model=system,
        device=args.device,
        train_loader=train_loader,
        test_loader=test_loader,
        tokenizer=tokenizer,
        normalizer=normalizer,
        epochs=args.num_train_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        exp_name=args.exp_name,
    )


if __name__ == "__main__":
    main()
