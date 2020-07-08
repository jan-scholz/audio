import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import os
import sys
from scipy.io import wavfile
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models import Conv1D, Conv2D, LSTM
from tqdm import tqdm
from glob import glob
import pathlib
import argparse

TARGET = 'target'
LABELCOL = 'class'

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        print('c', self.n_classes)
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # generate a batch of time data
        X = np.empty((self.batch_size, 1, int(self.sr*self.dt)), dtype=np.int16)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            X[i,] = wav.reshape(1, -1)
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def train(args):
    df = pd.read_csv(args.csv)
    # str.replace('data/', 'clean/')
    df_classes = list(df[LABELCOL].unique())

    src_root = args.src_root
    sr = args.sample_rate
    dt = args.delta_time
    batch_size = args.batch_size
    model_type = args.model_type
    
    if args.classes is not None:
        classes = list(set(args.classes).intersection(set(df_classes)))
    else:
        classes = df_classes
    print(f'{len(classes)} classes: {classes}')
    
    df = df[df[LABELCOL].isin(classes)]
   
    df0 = df.copy()
    df0['wav'] = df[[LABELCOL, TARGET]].apply(
        lambda x: os.path.join(args.src_root, x[0], pathlib.Path(x[1]).stem + '_0.wav'), axis=1)
    df1 = df.copy()
    df1['wav'] = df[[LABELCOL, TARGET]].apply(
        lambda x: os.path.join(args.src_root, x[0], pathlib.Path(x[1]).stem + '_1.wav'), axis=1)
    df2 = df.copy()
    df2['wav'] = df[[LABELCOL, TARGET]].apply(
        lambda x: os.path.join(args.src_root, x[0], pathlib.Path(x[1]).stem + '_2.wav'), axis=1)
    df3 = df.copy()
    df3['wav'] = df[[LABELCOL, TARGET]].apply(
        lambda x: os.path.join(args.src_root, x[0], pathlib.Path(x[1]).stem + '_3.wav'), axis=1)

    df = pd.concat([df0, df1, df2], axis=0)

    print('a', len(df))
    df = df[df['wav'].apply(lambda x: os.path.exists(x))]
    print('b', len(df))

    if len(df) < 1:
        raise Exception(f'No data for selected classes.')

    print(df[LABELCOL].value_counts())


    nsamples = {
        'baby_coo'         :  2000,
        'parentese_female' :  2000,
        'other'            :  2000,
        'baby_vocal_play'  :  2000,
        'baby_laugh'       :  2000,
        'baby_babble_var'  :  2000,
        'baby_cry'         :  2000,
        'parentese_male'   :  2000,
        'baby_babble_dup'  :  1000,
        'adult_female'     :   500,
        'adult_male'       :   500,
    }

    l = []
    for c, n in nsamples.items():
        if n < len(df[df[LABELCOL] == c]):
            replace = False
        else:
            replace = True
        tmp = df.loc[df[LABELCOL] == c].sample(n, replace=replace, random_state=123)
        l.append(tmp)
    df = pd.concat(l).sample(frac=1).reset_index(drop=True)
    del(l)

    print(df[LABELCOL].value_counts())

    params = {'N_CLASSES':len(classes),
              'SR':sr,
              'DT':dt}
    models = {'conv1d':Conv1D(**params),
              'conv2d':Conv2D(**params),
              'lstm':  LSTM(**params)}

    assert model_type in models.keys(), '{} not an available model'.format(model_type)
    csv_path = os.path.join('logs', '{}_history.csv'.format(model_type))

    # wav_paths = glob('{}/**'.format(src_root), recursive=True)
    # wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
    # classes = sorted(os.listdir(args.src_root))
    wav_paths = list(df['wav'])
    le = LabelEncoder()
    le.fit(classes)
    # labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    # labels = le.transform(labels)
    # labels = df[LABELCOL]
    labels = le.transform(df[LABELCOL])
    df[LABELCOL].to_csv(os.path.join('logs', f'{model_type}_labels.csv'), index=False)
    # print(min(labels), max(labels))


    wav_train, wav_val, label_train, label_val = train_test_split(wav_paths,
                                                                  labels,
                                                                  test_size=0.1,
                                                                  random_state=0)

    assert len(label_train) >= args.batch_size, 'number of train samples must be >= batch_size'

    tg = DataGenerator(wav_train, label_train, sr, dt,
                       len(le.classes_), batch_size=batch_size)
    vg = DataGenerator(wav_val, label_val, sr, dt,
                       len(le.classes_), batch_size=batch_size)

    model = models[model_type]
    cp = ModelCheckpoint('models/{}.h5'.format(model_type), monitor='val_loss',
                         save_best_only=True, save_weights_only=False,
                         mode='auto', save_freq='epoch', verbose=1)
    csv_logger = CSVLogger(csv_path, append=False)
    model.fit(tg, validation_data=vg,
              epochs=args.epochs,
              verbose=1,
              callbacks=[csv_logger, cp],
              workers=2,
              use_multiprocessing=True,
              )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--csv', type=str, required=True,
                        help='CSV with source paths')
    parser.add_argument('--model_type', type=str, default='conv1d',
                        help='model to run. i.e. conv1d, conv2d, lstm')
    parser.add_argument('--src_root', type=str, default='clean',
                        help='directory of audio files in total duration')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='epochs')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sample_rate', '-sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--classes', type=str, nargs='+',
                        help='classes to train on')
    args, _ = parser.parse_known_args()

    train(args)
