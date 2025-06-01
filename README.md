# Injecting Visual Features into Whisper for Parameter-Efficient Noise-Robust Audio-Visual Speech Recognition
The proposed AVWhisper is a parameter-efficient AVSR model by injecting visual features from AV-HuBERT encoder into the pre-trained Whisper model through lightweight adapters based on Low-Rank Adaptation (LoRA) and prompt-based techniques.
## Setup
We used Python 3.8.19 and PyTorch 2.3.0 on NVIDIA 4090 GPU to train and test our model.
```bash
conda create -n avwhisper python=3.8
conda activate avwhisper

# install pytorch
conda install pytorch==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# install dependency
pip install -r requirements.txt

# install whisper
cd whisper
pip install -e .
```

## Data Preparing
### 1. LRS3 preparation
Follow the steps of [LRS3 Preprocessing](https://github.com/facebookresearch/av_hubert/tree/main/avhubert/preparation) in [AV-HuBERT](https://github.com/facebookresearch/av_hubert) for LRS3 preparation.

### 2. Extract and save visual features
First, download finetuned models for visual features extraction from [checkpoints](https://facebookresearch.github.io/av_hubert/):
```bash
# for example
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt
```

Then, extract and save visual features:
```bash
MODEL_PATH=
DATA_DIR=
SUBSET=
SAVED_DATA_DIR=

python data/generate_avhubert_features.py  \
--model_path ${MODEL_PATH}  \
--data_dir ${DATA_DIR}  \
--gen_subset ${SUBSET}  \
--saved_data_dir ${SAVED_DATA_DIR}
```

### 3. Create noisy speech (Optimal)
Adding noise with different SNRs to speech:
```bash
LRS3_MANIFEST_PATH=/datasets/LRS3-TED/30h_data/train.tsv
NEW_MANIFEST_PATH=/datasets/LRS3-TED/30h_data/train_noisy_random.tsv
NOISE_DIR=/datasets/MixASR/wham_noise
TYPE=train
SNR=-15 -10 -5 0 5
SAVED_DIRNAME=audio_30h_noisy_random
SAVED_INFO_FN=30h_train_noisy_random.csv

python data/create_noisy_lrs3.py  \
--lrs3_manifest_path ${LRS3_MANIFEST_PATH} \
--new_lrs3_manifest_path ${NEW_MANIFEST_PATH} \
--noise_dir ${NOISE_DIR}  \
--type ${TYPE} \
--noise_snr ${SNR}  \
--saved_dirname ${SAVED_DIRNAME}  \
--saved_info_fn ${SAVED_INFO_FN}
```

The processed test set can be download from [huggingface](https://huggingface.co/datasets/zhaoyang9425/Custom-LRS3/tree/main).

## Training
Download the [large-v2](https://huggingface.co/openai/whisper-large-v2) or [large-v3](https://huggingface.co/openai/whisper-large-v3-turbo) Whisper checkpoint. Then run:
```bash
DATA_ROOT=/path/to/data/30h_data
CHECKPOINT_PATH=/path/to/large-v2/whisper/checkpoint
EXP_NAME=

CUDA_VISIBLE_DEVICES=0 python training_avwhisper.py  \
--exp_name ${EXP_NAME} \
--train_manifest_path ${DATA_ROOT}/train.tsv  \
--train_label_path ${DATA_ROOT}/train.wrd  \
--whisper_checkpoint_file ${CHECKPOINT_PATH}  \
--video_type finetuned_deep  \
--prompt_length 16  \
--deep  \
--use_mlp \
--depth 16  \
--language en  \
--num_train_epochs 7  \
--lr 1e-3  \
--step_size 2  \
--batch_size 1 | tee logs/train.log
```

## Inference
```bash
DATA_ROOT=/path/to/data/30h_data
CHECKPOINT_PATH=/path/to/large-v2/whisper/checkpoint
MODEL_NAME=

CUDA_VISIBLE_DEVICES=0 python evaluate_avwhisper.py  \
--model_name ${MODEL_NAME} \
--test_manifest_path ${DATA_ROOT}/test.tsv  \
--test_label_path ${DATA_ROOT}/test.wrd  \
--whisper_checkpoint_file ${CHECKPOINT_PATH}  \
--video_type finetuned_deep  \
--prompt_length 16  \
--deep  \
--use_mlp \
--depth 16  \
--language en | tee logs/test.log
```



## Citation
If you find our paper useful, please kindly cite:
```
@inproceedings{yang2025injecting,
  title={Injecting Visual Features into Whisper for Parameter-Efficient Noise-Robust Audio-Visual Speech Recognition},
  author={Yang, Zhao and Yeo, Yue Heng and Jiang, Rui and Fu, Xiao and Chen, Weiguang and Xi, Wei and Zhao, Jizhong},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

## Acknowledgement
This repo benefits from [TS-Whisper](https://github.com/Aisaka0v0/TS-Whisper) and [Whisper](https://github.com/openai/whisper).
Thanks for their wonderful works.