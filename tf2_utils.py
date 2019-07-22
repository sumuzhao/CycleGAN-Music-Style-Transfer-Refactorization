import datetime
import numpy as np
import copy
import write_midi
import tensorflow as tf


# new added functions for cyclegan
class ImagePool(object):

    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image


def load_npy_data(npy_data):
    npy_A = np.load(npy_data[0]) * 1.  # 64 * 84 * 1
    npy_B = np.load(npy_data[1]) * 1.  # 64 * 84 * 1
    npy_AB = np.concatenate((npy_A.reshape(npy_A.shape[0], npy_A.shape[1], 1),
                             npy_B.reshape(npy_B.shape[0], npy_B.shape[1], 1)),
                            axis=2)  # 64 * 84 * 2
    return npy_AB


def save_midis(bars, file_path, tempo=80.0):
    padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])),
                                  bars,
                                  np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))),
                                 axis=2)
    padded_bars = padded_bars.reshape(-1, 64, padded_bars.shape[2], padded_bars.shape[3])
    padded_bars_list = []
    for ch_idx in range(padded_bars.shape[3]):
        padded_bars_list.append(padded_bars[:, :, :, ch_idx].reshape(padded_bars.shape[0],
                                                                     padded_bars.shape[1],
                                                                     padded_bars.shape[2]))
    # this is for multi-track version
    # write_midi.write_piano_rolls_to_midi(padded_bars_list, program_nums=[33, 0, 25, 49, 0],
    #                                      is_drum=[False, True, False, False, False], filename=file_path, tempo=80.0)

    # this is for single-track version
    write_midi.write_piano_rolls_to_midi(piano_rolls=padded_bars_list,
                                         program_nums=[0],
                                         is_drum=[False],
                                         filename=file_path,
                                         tempo=tempo,
                                         beat_resolution=4)


def get_now_datetime():
    now = datetime.datetime.now().strftime('%Y-%m-%d')
    return str(now)


def to_binary(bars, threshold=0.0):
    """Turn velocity value into boolean"""
    track_is_max = tf.equal(bars, tf.reduce_max(bars, axis=-1, keepdims=True))
    track_pass_threshold = (bars > threshold)
    out_track = tf.logical_and(track_is_max, track_pass_threshold)
    return out_track
