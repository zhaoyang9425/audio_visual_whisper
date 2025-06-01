import torch

import os
import argparse
import numpy as np
from tqdm import tqdm

from fairseq import tasks, utils, checkpoint_utils

utils.import_user_module(argparse.Namespace(user_dir="/path/to/avhubert/repository/"))


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="visual feature extraction")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Where to load the fine-tuned avhubert checkpoint."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/datasets/LRS3-TED/433h_data/",
        help="Data dir path for the dataset."
    )
    parser.add_argument(
        "--gen_subset",
        type=str,
        default="train",
        help="Data type."
    )
    parser.add_argument(
        "--saved_data_dir",
        type=str,
        default="/datasets/LRS3-TED/video_features/",
        help="Data dir path for the processed visual dataset."
    )


def main():
    args = get_parser().parse_args()

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([args.model_path])
    model = models[0].eval().cuda()

    # load dataset
    gen_subset = args.gen_subset
    max_sample_size = None # 3000
    ## image_aug: default is true

    saved_cfg.task.modalities = ["video"]
    saved_cfg.task.max_sample_size = max_sample_size
    saved_cfg.task.data = args.data_dir
    saved_cfg.task.label_dir = args.data_dir
    task = tasks.setup_task(saved_cfg.task)
    task.load_dataset(gen_subset, task_cfg=saved_cfg.task)
    itr = task.get_batch_iterator(dataset=task.dataset(gen_subset), max_sentences=1).next_epoch_itr(shuffle=False)
    print(f"total {len(itr)} samples")

    # save folder setting
    saved_data_dir = args.saved_data_dir
    os.makedirs(saved_data_dir, exist_ok=True)

    # inference
    for _ in tqdm(range(len(itr)), desc=gen_subset):
        try:
            sample = next(itr)
            sample = utils.move_to_cuda(sample)

            net_input = sample["net_input"]
            assert torch.sum(net_input["padding_mask"]).cpu().float().numpy() == 0.0
            with torch.no_grad():
                output = model.encoder.extract_wrapper(source=net_input["source"], padding_mask=net_input["padding_mask"])

            deep_feature = output["encoder_out"].squeeze(0)
            target_deep_path = saved_data_dir + sample["utt_id"][0] + ".npz"
            path, deep_filename = os.path.split(target_deep_path)
            os.makedirs(path, exist_ok=True)
            np.savez_compressed(target_deep_path, data=deep_feature.detach().cpu().numpy().astype(np.float32))

        except StopIteration:
            break


if __name__ == "__main__":
    main()










