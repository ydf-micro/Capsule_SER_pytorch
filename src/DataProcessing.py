# *_*coding:utf-8 *_*

import re
import os
import time
import glob
import pickle
import numpy as np
import scipy.io.wavfile as wav
import python_speech_features as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

eps = 1e-5

def generate_label(emotion):
    label = -1
    if emotion == 'hap' or emotion == 'exc':
        label = 0
    elif emotion == 'sad':
        label = 1
    elif emotion == 'ang':
        label = 2
    elif emotion == 'neu':
        label = 3

    return label


def audio_segmentation(wav_dir, filter_num, speaker_emo, impro_or_script):
    '''
    If the audio is less than 3s, fill it with 0 to 3s
    If the audio is longer than 3s,
    take the first three seconds and the last three seconds of the audio,
    there may be some overlap
    :param wav_dir: directory of the audio
    :param filter_num: 40
    :param speaker_emo: emotion of every audio
    :param impro_or_script: spontaneous sessions or scripted sessions
    :return:
    '''

    # only consider four emotions
    # consider excited as happy
    emotion_dir = ['hap', 'ang', 'neu', 'sad', 'exc']

    # number of spontaneous session in emotion_dir is 4642
    # number of spontaneous session is 4784
    # number of scripted session in emotion_dir is 4199
    # number of scripted session is 5255
    wav_num = len(wav_dir)

    if impro_or_script == 0:
        data = np.empty((4642, 3, 300, filter_num), dtype=np.float32)
        label = np.empty((4642, 1), dtype=np.int8)
    else:
        data = np.empty((4199, 3, 300, filter_num), dtype=np.float32)
        label = np.empty((4199, 1), dtype=np.int8)

    data_num = 0

    for wav_dir in wav_dir:
        attr = wav_dir.split('/')
        session, dialogue, sub_dialogue = attr[-5], attr[-2], attr[-1]
        emotion = speaker_emo[session][dialogue][sub_dialogue]
        if emotion in emotion_dir:
            rate, signal = wav.read(wav_dir)
            # compute the mel_spec, deltas, delta-deltas
            mel_spec = ps.logfbank(signal, rate, nfilt=filter_num)
            deltas = ps.delta(mel_spec, 2)
            delta_deltas = ps.delta(deltas, 2)

            wav_time = mel_spec.shape[0]

            if wav_time <= 300:
                part = mel_spec
                delta1 = deltas
                delta2 = delta_deltas
                part = np.pad(part, ((0, 300-part.shape[0]), (0, 0)),
                              'constant', constant_values=0)
                delta1 = np.pad(delta1, ((0, 300 - deltas.shape[0]), (0, 0)),
                              'constant', constant_values=0)
                delta2 = np.pad(delta2, ((0, 300 - delta_deltas.shape[0]), (0, 0)),
                              'constant', constant_values=0)

                data[data_num, 0, :, :] = part
                data[data_num, 1, :, :] = delta1
                data[data_num, 2, :, :] = delta2

                emo_label = generate_label(emotion)
                label[data_num] = emo_label
                data_num += 1

            else:
                # if time > 300, intercepte the first 300 and the last 300
                part1 = mel_spec[0:300, :]
                part2 = mel_spec[wav_time-300:wav_time, :]

                delta11 = deltas[0:300, :]
                delta12 = deltas[wav_time-300:wav_time, :]

                delta21 = delta_deltas[0:300, :]
                delta22 = delta_deltas[wav_time-300:wav_time, :]

                data[data_num, 0, :, :] = part1
                data[data_num, 1, :, :] = delta11
                data[data_num, 2, :, :] = delta21

                emo_label = generate_label(emotion)
                label[data_num] = emo_label
                data_num += 1

                data[data_num, 0, :, :] = part2
                data[data_num, 1, :, :] = delta12
                data[data_num, 2, :, :] = delta22

                emo_label = generate_label(emotion)
                label[data_num] = emo_label
                data_num += 1

    return data_num, data, label


def datasets_segmentation(data, label, test_size1=0.4, test_size2=0.5, random_state=42):
    '''
    The data were divided into training set, cross validation set and test set
    :param data:
    :param label:
    :param test_size1:
    :param test_size2:
    :param random_state:
    :return:
    '''
    train_data, test_valid_data, \
    train_label, test_valid_label = \
        train_test_split(data, label, test_size=test_size1, random_state=random_state)

    valid_data, test_data, \
    valid_label, test_label = \
        train_test_split(test_valid_data, test_valid_label,
                         test_size=test_size2, random_state=random_state)

    return train_data, train_label, valid_data, valid_label, test_data, test_label


def input_normalization(train_data, valid_data, test_data, filter_num):
    '''
    normalize the input data
    :param train_data:
    :param valid_data:
    :param test_data:
    :param filter_num:
    :return:
    '''
    # train_data = train_data.reshape((-1, filter_num, 3))
    # valid_data = valid_data.reshape((-1, filter_num, 3))
    # test_data = test_data.reshape((-1, filter_num, 3))

    data_mean = {}
    data_std = {}

    data_mean['static'] = np.mean(train_data[:, 0, :, :].reshape(-1, filter_num),
                                  axis=0)
    data_mean['deltas'] = np.mean(train_data[:, 1, :, :].reshape(-1, filter_num),
                                  axis=0)
    data_mean['delta-deltas'] = np.mean(train_data[:, 2, :, :].reshape(-1, filter_num),
                                        axis=0)

    data_std['static'] = np.std(train_data[:, 0, :, :].reshape(-1, filter_num),
                                axis=0)
    data_std['deltas'] = np.std(train_data[:, 1, :, :].reshape(-1, filter_num),
                                axis=0)
    data_std['delta-deltas'] = np.std(train_data[:, 2, :, :].reshape(-1, filter_num),
                                      axis=0)

    for i, key in enumerate(data_mean):
        train_data[:, i, :, :] = (train_data[:, i, :, :] - data_mean[key]) / (data_std[key] + eps)
        valid_data[:, i, :, :] = (valid_data[:, i, :, :] - data_mean[key]) / (data_std[key] + eps)
        test_data[:, i, :, :] = (test_data[:, i, :, :] - data_mean[key]) / (data_std[key] + eps)

    # train_data = train_data.reshape((-1, 300, filter_num, 3))
    # valid_data = valid_data.reshape((-1, 300, filter_num, 3))
    # test_data = test_data.reshape((-1, 300, filter_num, 3))

    return train_data, valid_data, test_data


def save_data(output_path, data_label):
    (train_data, train_label, valid_data, valid_label, test_data, test_label) = data_label
    with open(output_path, 'wb') as f:
        pickle.dump((train_data, train_label,
                     valid_data, valid_label,
                     test_data, test_label), f)


def read_IEMOCAP():
    '''
    read IEMOCAP corpus datasets
    :return:
    '''
    rootdir = '/home/ydf_micro/datasets/IEMOCAP_full_release'

    speaker_impro_emo = {}
    speaker_script_emo = {}

    filter_num = 40  # the number of filters in the filterbank

    # get emotion
    for speaker in os.listdir(rootdir):
        if re.search('Session', speaker):  # Session1-5
            emoevl = os.path.join(rootdir, speaker, 'dialog/EmoEvaluation')
            sess_impro_map = {}
            sess_script_map = {}
            for sess in os.listdir(emoevl):
                if re.search('impro', sess):  # spontaneous sessions
                    emotdir = emoevl + '/' + sess
                    emot_map = {}
                    with open(emotdir, 'r') as emot_to_read:
                        while True:
                            line = emot_to_read.readline()
                            if not line:
                                break
                            if re.search(r'Ses[0-9]{2}[MF]_impro[0-9]{2}[ab]*_[MF][0-9]{3}', line):
                                t = line.split()
                                emot_map[t[3] + '.wav'] = t[4]

                    sess_impro_map[sess[:-4]] = emot_map

                elif re.search('script', sess):   # scripted sessions
                    emotdir = emoevl + '/' + sess
                    emot_map = {}
                    with open(emotdir, 'r') as emot_to_read:
                        while True:
                            line = emot_to_read.readline()
                            if not line:
                                break
                            if re.search(r'Ses[0-9]{2}[MF]_script[0-9]{2}_[0-9][ab]*_[MF][0-9]{3}', line):
                                t = line.split()
                                emot_map[t[3] + '.wav'] = t[4]

                    sess_script_map[sess[:-4]] = emot_map

            speaker_impro_emo[speaker] = sess_impro_map
            speaker_script_emo[speaker] = sess_script_map

    wav_impro_dir = []
    wav_script_dir = []

    # get directory
    for speaker in os.listdir(rootdir):
        if re.search('Session', speaker):  # Session1-5
            sub_dir = os.path.join(rootdir, speaker, 'sentences/wav')
            for sess in os.listdir(sub_dir):
                if re.search('impro', sess):   # spontaneous sessions
                    file_dir = os.path.join(sub_dir, sess, '*.wav')
                    files = glob.glob(file_dir)
                    wav_impro_dir.extend(files)

                elif re.search('script', sess):   # scripted sessions
                    file_dir = os.path.join(sub_dir, sess, '*.wav')
                    files = glob.glob(file_dir)
                    wav_script_dir.extend(files)

    impro_data_num, impro_data, impro_label = \
        audio_segmentation(wav_impro_dir, filter_num, speaker_impro_emo, 0)

    script_data_num, script_data, script_label = \
        audio_segmentation(wav_script_dir, filter_num, speaker_script_emo, 1)

    # # 4642 (4642, 300, 40, 3) (4642, 1)
    # print(impro_data_num, impro_data.shape, impro_label.shape)
    # # 4199 (4199, 300, 40, 3) (4199, 1)
    # print(script_data_num, script_data.shape, script_label.shape)

    #===================================spontaneous session=========================

    impro_train_data, impro_train_label, \
    impro_valid_data, impro_valid_label, \
    impro_test_data, impro_test_label = datasets_segmentation(impro_data, impro_label)

    # print(impro_train_data.shape[0]/impro_data_num)
    # print(impro_valid_data.shape[0] / impro_data_num)
    # print(impro_test_data.shape[0] / impro_data_num)

    # ===================================spontaneous session=========================

    # ===================================scripted session============================

    script_train_data, script_train_label, \
    script_valid_data, script_valid_label, \
    script_test_data, script_test_label = datasets_segmentation(script_data, script_label)

    # print(script_train_data.shape[0] / script_data_num)
    # print(script_valid_data.shape[0] / script_data_num)
    # print(script_test_data.shape[0] / script_data_num)

    # ===================================scripted session============================

    # =========================input normalization and save datasets=================

    impro_train_data, impro_valid_data, impro_test_data = \
        input_normalization(impro_train_data, impro_valid_data, impro_test_data, filter_num)

    print(impro_train_data.shape, impro_valid_data.shape, impro_test_data.shape)

    impro_data_label = (impro_train_data, impro_train_label,
                        impro_valid_data, impro_valid_label,
                        impro_test_data, impro_test_label)

    output_path = '../data/impro_IEMOCAP.pkl'
    save_data(output_path, impro_data_label)


    script_train_data, script_valid_data, script_test_data = \
        input_normalization(script_train_data, script_valid_data, script_test_data, filter_num)

    print(script_train_data.shape, script_valid_data.shape, script_test_data.shape)

    script_data_label = (script_train_data, script_train_label,
                        script_valid_data, script_valid_label,
                        script_test_data, script_test_label)

    output_path = '../data/script_IEMOCAP.pkl'
    save_data(output_path, script_data_label)


if __name__ == '__main__':
    start = time.time()
    read_IEMOCAP()
    end = time.time()
    print('所用时间:{}min {:.2f}s'.format((end - start) // 60, (end - start) % 60))