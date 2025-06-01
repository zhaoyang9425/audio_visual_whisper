import argparse
import os
import glob
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lrs3_manifest_path', type=str, required=True,
                        help='Path to lrs3 manifest path')
    parser.add_argument('--new_lrs3_manifest_path', type=str, required=True,
                        help='Path to new lrs3 manifest path')
    parser.add_argument('--noise_dir', type=str, required=True,
                        help='Path to noise root directory')
    parser.add_argument('--type', type=str, required=True, choices=['train', 'valid', 'test'],
                        help='dataset type')
    parser.add_argument('--augment', default=False, action='store_true',
                        help='whether to apply augmentation')
    parser.add_argument('--noise_snr', nargs='+', type=int)
    parser.add_argument('--saved_dirname', type=str, required=True,
                        help='saved mixed speech data dirname')
    parser.add_argument('--saved_info_fn', type=str, required=True,
                        help='saved mixed speech information filename')
    return parser


def load_manifest(manifest_path):
    sample_ids, names, inds, sizes, lengths = [], [], [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            sz = int(items[-2])     #

            sample_ids.append(items[0])
            video_path = items[1]
            audio_path = items[2]
            names.append((video_path, audio_path))
            inds.append(ind)
            sizes.append(sz)
            lengths.append(int(items[-1]))

    tot = ind + 1
    return root, sample_ids, names, inds, tot, sizes, lengths


def main(args):
    print(args)

    # load lrs3 dataset
    root, sample_ids, names, _, total, sizes, lengths = load_manifest(args.lrs3_manifest_path)
    print(f"There are total {total} clean speech files")

    if args.type == "train":
        noise_dir = os.path.join(args.noise_dir, "tr")
    elif args.type == "valid":
        noise_dir = os.path.join(args.noise_dir, "cv")
    elif args.type == "test":
        noise_dir = os.path.join(args.noise_dir, "tt")
    else:
        raise ValueError
    # load noise dataset
    sound_paths = glob.glob(os.path.join(noise_dir, '*.wav'), recursive=True)
    if not args.augment:
        temp = []
        for sound_path in sound_paths:
            if "sp08" not in sound_path and "sp12" not in sound_path:
                temp.append(sound_path)
        sound_paths = temp
    print(f"There are total {len(sound_paths)} noise files")

    # Create the dataframe corresponding to this directory
    saved_data = open(args.new_lrs3_manifest_path, "w")
    saved_data.write(root + "\n")
    saved_info_md = pd.DataFrame(columns=['sample_id', 'clean_audio_path', 'noise_path', 'snr'])

    for idx, clean_av_name in enumerate(tqdm(names, total=total)):
        # load clean wav
        clean_av_path = os.path.join(root, clean_av_name[1])
        clean_wav, sample_rate = sf.read(clean_av_path)

        # select noise
        rand_index = np.random.randint(0, len(sound_paths))
        noise_wav, sample_rate = sf.read(sound_paths[rand_index])
        if len(noise_wav.shape) > 1:
            noise_wav = noise_wav.T
            noise_wav = noise_wav.sum(axis=0) / noise_wav.shape[0]
        assert sample_rate == 16000

        # generate mixed speech
        mix_wav, snr = add_noise(clean_wav, noise_wav, args.noise_snr)

        # save mixed speech
        mixed_save_path = clean_av_path.replace("/audio/", "/"+args.saved_dirname+"/")
        os.makedirs(os.path.dirname(mixed_save_path), exist_ok=True)
        sf.write(mixed_save_path, mix_wav, sample_rate)

        # Add information to the dataframe
        saved_data.write("\t".join([sample_ids[idx], clean_av_name[0], clean_av_name[1].replace("audio/", args.saved_dirname+"/"), str(sizes[idx]), str(lengths[idx])]) + "\n")
        saved_info_md.loc[len(saved_info_md)] = [sample_ids[idx], clean_av_name[1], sound_paths[rand_index].strip(args.noise_dir), str(snr)]

    saved_data.close()
    saved_info_md.to_csv(args.saved_info_fn, index=False)


def add_noise(clean_wav, noise_wav, noise_snr):
    if type(noise_snr) == int or type(noise_snr) == float:
        snr = noise_snr
    elif type(noise_snr) == list:
        snr = np.random.choice(noise_snr)
    clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
    if len(clean_wav) > len(noise_wav):
        ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
        noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
    if len(clean_wav) < len(noise_wav):
        start = 0
        noise_wav = noise_wav[start: start + len(clean_wav)]
    noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
    adjusted_noise_rms = clean_rms / (10**(snr/20))
    adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
    mixed = clean_wav + adjusted_noise_wav

    # Final check to see if there are any amplitudes exceeding +/- 1.
    # If so, normalize all the signals accordingly
    if is_clipped(mixed):
        noisyspeech_maxamplevel = max(abs(mixed)) / 0.99
        mixed = mixed / noisyspeech_maxamplevel
    return mixed, snr


def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)


# For lrs3 dataset
## 433 subset:  train-> 299646 files, val-> 1200, test-> 1321
## 30 subset: train-> 30782

# For wham noise dataset
## train: 60000 (include sp08, sp12)   val: 5000   test: 3000