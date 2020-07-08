import sys
import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
from librosa.core import resample, to_mono
from tqdm import tqdm
# import librosa
# (sig, rate) = librosa.load(path)
# print(rate)

TARGET = 'target'
LABELCOL = 'class'

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def downsample_mono(path, sr):
    # print()
    # print('XXX', path)
    rate, wav = wavfile.read(path)
    wav = wav.astype(np.float32, order='F')
    try:
        tmp = wav.shape[1]
        wav = to_mono(wav.T)
    except:
        pass
    wav = resample(wav, rate, sr)
    wav = wav.astype(np.int16)
    return sr, wav


def save_sample(sample, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    dst_path = os.path.join(target_dir.split('.')[0], fn+'_{}.wav'.format(str(ix)))
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, sample)


def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)


def split_wavs(args):
    df = pd.read_csv(args.csv)
    df[TARGET] = df[TARGET].str.replace('data/', '../../babbly-ml-pipeline/data2/')
    # src_root = args.src_root
    # dst_root = args.dst_root
    dst_root = args.dst_root
    dt = args.delta_time

    if args.exclude is not None:
        df_excl = pd.read_csv(args.exclude)
        exclusion_list = list(df_excl['path'])

    # wav_paths = glob('{}/**'.format(src_root), recursive=True)
    # wav_paths = df[TARGET]
    # wav_paths = [x for x in wav_paths if '.wav' in x]
    # dirs = os.listdir(src_root)
    # check_dir(dst_root)
    # classes = os.listdir(src_root)
    classes = list(df[LABELCOL].unique())
    print(f'{len(classes)} classes: {", ".join(classes)}')

    for _cls in classes:
        print(f'class: {_cls}')
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)
        # src_dir = os.path.join(src_root, _cls)

        paths = df.loc[df[LABELCOL] == _cls, TARGET]
        if args.head is not None:
            paths = paths.head(args.head)

        for src_fn in tqdm(paths):
            fn = os.path.basename(src_fn)
            if not os.path.exists(src_fn):
                continue
            if exclusion_list and src_fn in exclusion_list:
                # print('>>> excluding', src_fn)
                continue
            if os.path.exists(fn):
                continue

            rate, wav = downsample_mono(src_fn, args.sr)
            mask, y_mean = envelope(wav, rate, threshold=args.threshold)
            wav = wav[mask]
            delta_sample = int(dt*rate)

            # cleaned audio is less than a single sample
            # pad with zeros to delta_sample size
            if wav.shape[0] < delta_sample:
                sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
                sample[:wav.shape[0]] = wav
                save_sample(sample, rate, target_dir, fn, 0)
            # step through audio and save every delta_sample
            # discard the ending audio if it is too short
            else:
                trunc = wav.shape[0] % delta_sample
                for cnt, i in enumerate(np.arange(0, wav.shape[0]-trunc, delta_sample)):
                    start = int(i)
                    stop = int(i + delta_sample)
                    sample = wav[start:stop]
                    save_sample(sample, rate, target_dir, fn, cnt)


def test_threshold(args):
    src_root = args.src_root
    if args.file is None:
        wav_paths = glob('{}/**'.format(src_root), recursive=True)
        wav_path = [x for x in wav_paths if args.fn in x]
        if len(wav_path) != 1:
            print('audio file not found for sub-string: {}'.format(args.fn))
            return
        filename = wav_path[0]
    else:
        filename = args.file
    rate, wav = downsample_mono(filename, args.sr)
    mask, env = envelope(wav, rate, threshold=args.threshold)
    plt.style.use('ggplot')
    plt.title('Signal Envelope, Threshold = {}'.format(str(args.threshold)))
    plt.plot(wav[np.logical_not(mask)], color='r', label='remove')
    plt.plot(wav[mask], color='c', label='keep')
    plt.plot(env, color='m', label='envelope')
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--csv', type=str,
                        help='CSV with source paths')
    parser.add_argument('--head', type=int,
                        help='Process only first n rows of CSV.')
    parser.add_argument('--file', type=str,
                        help='Plot envelope for this file')
    parser.add_argument('--src_root', type=str, default='wavfiles',
                        help='directory of audio files in total duration')
    parser.add_argument('--dst_root', type=str, default='clean',
                        help='directory to put audio files split by delta_time')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='rate to downsample audio')

    parser.add_argument('--fn', type=str, default='3a3d0279',
                        help='file to plot over time to check magnitude')
    parser.add_argument('--threshold', type=float, default=20,
                        help='threshold magnitude for np.int16 dtype')

    parser.add_argument('--exclude', type=str,
                        help='exclude these files')
    args, _ = parser.parse_known_args()

    if args.file is not None:
        print(f'thresholding {args.file}')
        test_threshold(args)
    if args.csv is not None:
        print(f'cleaning')
        split_wavs(args)
