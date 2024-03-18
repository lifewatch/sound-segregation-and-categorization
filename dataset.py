import datetime
import json
import os
import pathlib
import shutil
import sys

import fairseq
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import scipy
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from PIL import Image
# from maad import util
from tqdm import tqdm
import torch.optim as optim
from sklearn.model_selection import train_test_split
import copy
import suntime
import pytz
import noisereduce as nr

from transformers import ClapModel, ClapProcessor
from transformers import pipeline

import models
import utils as u

torchaudio.set_audio_backend(backend='soundfile')

# matplotlib.use('TkAgg')
# Get the color map by name:
cm = plt.get_cmap('jet')


class LifeWatchDataset:
    def __init__(self, config):
        # Spectrogram settings
        self.duration = config['duration']
        self.overlap = config['overlap']  # overlap of the chunks in %
        self.desired_fs = config['desired_fs']
        self.channel = config['channel']
        self.log = config['log']
        self.color = config['color']

        # Folders
        self.wavs_folder = pathlib.Path(config['wavs_folder'])
        self.dataset_folder = pathlib.Path(config['dataset_folder'])
        self.images_folder = self.dataset_folder.joinpath('images')
        self.labels_folder = self.dataset_folder.joinpath('labels')

        self.annotations_file = config['annotations_file']

        self.nfft = config['nfft']
        self.win_len = config['win_len']
        self.hop_length = int(self.win_len / config['hop_ratio'])
        self.win_overlap = self.win_len - self.hop_length

        self.normalization_style = config['normalization_style']

        if 'split_folders' in config.keys():
            self.split_folders = config['split_folders']
        else:
            self.split_folders = False

        if 'min_duration' in config.keys():
            self.MIN_DURATION = config['min_duration']
        else:
            self.MIN_DURATION = self.nfft / self.desired_fs

        if 'max_duration' in config.keys():
            self.MAX_DURATION = config['max_duration']
        else:
            self.MAX_DURATION = self.duration / 2
        self.MIN_SNR = 10

        self.blocksize = int(self.duration * self.desired_fs)

        self.config = config

    def __setitem__(self, key, value):
        if key in self.config.keys():
            self.config[key] = value
        self.__dict__[key] = value

    def save_config(self, config_path):
        with open(config_path, 'w') as f:
            json.dump(self.config, f)

    def create_spectrograms(self, overwrite=False, save_image=True, model=None, labels_path=None):
        # First, create all the images
        if self.split_folders:
            folders_list = []
            for f in self.wavs_folder.glob('*'):
                if f.is_dir():
                    folders_list.append(f)
                    if save_image:
                        if not self.images_folder.joinpath(f.name).exists():
                            os.mkdir(self.images_folder.joinpath(f.name))
                    if model is not None:
                        if not labels_path.joinpath(f.name).exists():
                            os.mkdir(labels_path.joinpath(f.name))
        else:
            folders_list = [self.wavs_folder]

        for folder_n, folder_path in tqdm(enumerate(folders_list), total=len(folders_list)):
            print('Spectrograms from folder %s/%s: %s' % (folder_n, len(folders_list), folder_path))
            for wav_path in tqdm(list(folder_path.glob('*.wav')), total=len(list(folder_path.glob('*.wav')))):
                waveform_info = torchaudio.info(wav_path)
                i = 0.0
                while (i * self.duration + self.duration) < (waveform_info.num_frames / waveform_info.sample_rate):
                    img_name = wav_path.name.replace('.wav', '_%s.png' % i)
                    if self.split_folders:
                        img_path = self.images_folder.joinpath(folder_path.name, img_name)
                    else:
                        img_path = self.images_folder.joinpath(img_name)

                    if overwrite or (not img_path.exists()):
                        start_chunk = int(i * self.blocksize)
                        start_chunk_s = start_chunk / self.desired_fs
                        if waveform_info.sample_rate > self.desired_fs:
                            start_chunk_old_fs = int(start_chunk_s * waveform_info.sample_rate)
                            blocksize_old_fs = int(self.duration * waveform_info.sample_rate)
                            chunk_old_fs, fs = torchaudio.load(wav_path,
                                                               normalize=True,
                                                               frame_offset=start_chunk_old_fs,
                                                               num_frames=blocksize_old_fs)
                            chunk = F.resample(waveform=chunk_old_fs[0, :], orig_freq=fs, new_freq=self.desired_fs)
                        else:
                            chunk, fs = torchaudio.load(wav_path, normalize=True, frame_offset=start_chunk,
                                                        num_frames=self.blocksize)
                            chunk = chunk[0, :]

                        if len(chunk) == self.blocksize:
                            if self.normalization_style == 'noisy':
                                img, f = self.create_chunk_spectrogram_noisy(chunk)
                            else:
                                img, f = self.create_chunk_spectrogram_low_freq(chunk)

                            if model is not None:
                                results = model(source=np.ascontiguousarray(np.flipud(img)[:, :, ::-1]),
                                                project=str(self.dataset_folder),
                                                name='predictions',
                                                save=False, show=False, save_conf=True, save_txt=False, conf=0.1,
                                                save_crop=False, agnostic_nms=False, stream=False, verbose=False,
                                                imgsz=640, exist_ok=True)
                                for r in results:
                                    label_name = img_name.replace('.png', '.txt')
                                    if self.split_folders:
                                        r.save_txt(labels_path.joinpath(folder_path.name, label_name), save_conf=True)
                                    else:
                                        r.save_txt(labels_path.joinpath(label_name), save_conf=True)

                            if save_image:
                                if self.log:
                                    fig, ax = plt.subplots()
                                    ax.pcolormesh(img[:, :, ::-1])
                                    ax.set_yscale('symlog')
                                    plt.axis('off')
                                    plt.ylim(bottom=3)
                                    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
                                else:
                                    Image.fromarray(np.flipud(img)).save(img_path)
                            plt.close()
                    i += self.overlap

    def create_chunk_spectrogram_noisy(self, chunk):
        f, t, sxx = scipy.signal.spectrogram(chunk, fs=self.desired_fs, window=('hamming'),
                                             nperseg=self.win_len,
                                             noverlap=self.win_overlap, nfft=self.nfft,
                                             detrend=False,
                                             return_onesided=True, scaling='density', axis=-1,
                                             mode='magnitude')
        sxx = sxx[f > 50, :]
        sxx = 10 * np.log10(sxx)
        per_min = np.percentile(sxx.flatten(), 1)
        per_max = np.percentile(sxx.flatten(), 99)
        sxx = (sxx - per_min) / (per_max - per_min)
        sxx[sxx < 0] = 0
        sxx[sxx > 1] = 1
        sxx = cm(sxx)  # convert to color

        img = np.array(sxx[:, :, :3] * 255, dtype=np.uint8)
        return img, f

    def create_chunk_spectrogram_low_freq(self, chunk):
        sos = scipy.signal.iirfilter(20, [5, 124], rp=None, rs=None, btype='band',
                                     analog=False, ftype='butter', output='sos',
                                     fs=self.desired_fs)
        chunk = scipy.signal.sosfilt(sos, chunk)
        f, t, sxx = scipy.signal.spectrogram(chunk, fs=self.desired_fs, window=('hamming'),
                                             nperseg=self.win_len,
                                             noverlap=self.win_overlap, nfft=self.nfft,
                                             detrend=False,
                                             return_onesided=True, scaling='density', axis=-1,
                                             mode='magnitude')
        sxx = 1 - sxx
        per = np.percentile(sxx.flatten(), 98)
        sxx = (sxx - sxx.min()) / (per - sxx.min())
        sxx[sxx > 1] = 1
        img = np.array(sxx * 255, dtype=np.uint8)
        return img

    def convert_raven_annotations_to_yolo(self, labels_to_exclude=None, values_to_replace=0):
        """

        :param annotations_file:
        :param labels_to_exclude: list
        :param values_to_replace: should be a dict with the name of the Tag as a key and an int as the value, for the
        yolo classes
        :return:
        """
        for _, selections in self.load_relevant_selection_table(labels_to_exclude=labels_to_exclude):
            selections['height'] = (selections['High Freq (Hz)'] - selections['Low Freq (Hz)']) / (self.desired_fs / 2)

            # The y is from the TOP!
            selections['y'] = 1 - (selections['High Freq (Hz)'] / (self.desired_fs / 2))

            # compute the width in pixels
            selections['width'] = ((selections['End Time (s)'] - selections['Begin Time (s)']) / self.duration)

            # Remove selections smaller than 2 pixels and longer than half the duration
            selections = selections.loc[(selections['End Time (s)'] - selections['Begin Time (s)']) < self.MAX_DURATION]
            selections = selections.loc[(selections['End Time (s)'] - selections['Begin Time (s)']) > self.MIN_DURATION]
            pbar = tqdm(total=len(selections['Begin File'].unique()))

            for wav_name, wav_selections in selections.groupby('Begin File'):
                if os.path.isdir(self.wavs_folder):
                    wav_file_path = self.wavs_folder.joinpath(wav_name)
                else:
                    wav_file_path = self.wavs_folder

                waveform_info = torchaudio.info(wav_file_path)
                fs = waveform_info.sample_rate
                waveform_duration = waveform_info.num_frames / fs

                # Re-compute the samples to match the new sampling rate
                wav_selections['End File Samp (samples)'] = wav_selections[
                                                                'End File Samp (samples)'] / fs * self.desired_fs
                wav_selections['Beg File Samp (samples)'] = wav_selections[
                                                                'Beg File Samp (samples)'] / fs * self.desired_fs

                i = 0.0
                while (i * self.duration + self.duration) < waveform_duration:
                    start_sample = int(i * self.blocksize)

                    chunk_selection = wav_selections.loc[(wav_selections['Beg File Samp (samples)'] >= start_sample) &
                                                         (wav_selections[
                                                              'Beg File Samp (samples)'] <= start_sample + self.blocksize)]

                    chunk_selection = chunk_selection.assign(
                        x=(chunk_selection['Beg File Samp (samples)'] - i * self.blocksize) / self.blocksize)

                    chunk_selection.loc[
                        chunk_selection['width'] + chunk_selection['x'] > 1, 'width'] = 1 - chunk_selection['x']

                    # Save the chunk detections so that they are with the yolo format
                    # <class > < x > < y > < width > < height >
                    chunk_selection['x'] = (chunk_selection['x'] + chunk_selection['width'] / 2)
                    chunk_selection['y'] = (chunk_selection['y'] + chunk_selection['height'] / 2)

                    if ((chunk_selection.x > 1).sum() > 0) or ((chunk_selection.y > 1).sum() > 0):
                        print('hey error')

                    if isinstance(values_to_replace, dict):
                        chunk_selection = chunk_selection.replace(values_to_replace)
                    else:
                        chunk_selection['Tags'] = 0
                    chunk_selection[[
                        'Tags',
                        'x',
                        'y',
                        'width',
                        'height']].to_csv(self.labels_folder.joinpath(wav_name.replace('.wav', '_%s.txt' % i)),
                                          header=None, index=None, sep=' ', mode='w')
                    # Add the station if the image adds it as well!
                    i += self.overlap
                    pbar.update(1)
            pbar.close()

    def convert_raven_annotations_to_df(self, labels_to_exclude=None, values_to_replace=0):
        total_selections = pd.DataFrame()
        for _, selections in self.load_relevant_selection_table(labels_to_exclude=labels_to_exclude):
            selections['height'] = (selections['High Freq (Hz)'] - selections['Low Freq (Hz)']) / (self.desired_fs / 2)

            # compute the width in pixels
            selections['width'] = ((selections['End Time (s)'] - selections['Begin Time (s)']) / self.duration)

            # The y is from the TOP!
            selections['y'] = 1 - (selections['High Freq (Hz)'] / (self.desired_fs / 2)) + selections['height'] / 2

            # Remove selections smaller than 2 pixels and longer than half the duration
            selections = selections.loc[(selections['End Time (s)'] - selections['Begin Time (s)']) < self.MAX_DURATION]
            selections = selections.loc[(selections['End Time (s)'] - selections['Begin Time (s)']) > self.MIN_DURATION]

            selections['wav'] = np.nan
            selections['wav_name'] = selections['Begin File']
            selections['duration'] = selections.width * self.duration
            selections['min_freq'] = 1 - (selections.y + selections.height / 2)
            selections['max_freq'] = 1 - (selections.y - selections.height / 2)

            if isinstance(values_to_replace, dict):
                selections = selections.replace(values_to_replace)
            else:
                selections['Tags'] = 0
            total_selections = pd.concat([total_selections, selections])

        return total_selections

    def all_predictions_to_dataframe(self, labels_folder, overwrite=True):
        detected_foregrounds = []
        if self.split_folders:
            wav_folder = self.wavs_folder.joinpath(labels_folder.name)
        else:
            wav_folder = self.wavs_folder
        for txt_label in tqdm(labels_folder.glob('*.txt'), total=len(list(labels_folder.glob('*.txt')))):
            name_parts = txt_label.name.split('_')
            wav_name = '_'.join(name_parts[:-1]) + '.wav'
            original_wav = wav_folder.joinpath(wav_name)
            offset_seconds = float(name_parts[-1].split('.txt')[0])
            detections = pd.read_table(txt_label, header=None, sep=' ', names=['class', 'x', 'y',
                                                                               'width', 'height', 'confidence'])
            detections['folder'] = str(original_wav.parent)
            detections['wav'] = str(original_wav)
            detections['wav_name'] = wav_name
            detections['start_seconds'] = (detections.x - detections.width / 2 + offset_seconds) * self.duration
            detections['image'] = txt_label.name.replace('.txt', '')

            # for _, row in detections.iterrows():
            detected_foregrounds.extend(detections.values)
            # detected_foregrounds = pd.concat([detected_foregrounds, detections], ignore_index=True)
        detected_foregrounds = np.stack(detected_foregrounds)
        detected_foregrounds = pd.DataFrame(detected_foregrounds, columns=detections.columns)
        detected_foregrounds['duration'] = detected_foregrounds.width * self.duration
        detected_foregrounds['min_freq'] = (1 - (detected_foregrounds.y + detected_foregrounds.height / 2)) * self.desired_fs / 2
        detected_foregrounds['max_freq'] = (1 - (detected_foregrounds.y - detected_foregrounds.height / 2)) * self.desired_fs / 2
        return detected_foregrounds

    def convert_detections_to_raven(self, predictions_folder, add_station_name=False, min_conf=None):
        # Convert to DataFrame
        labels_folder = predictions_folder.joinpath('labels')
        if self.split_folders:
            folders_list = [f for f in labels_folder.glob('*') if f.is_dir()]
        else:
            folders_list = [labels_folder]
        for f in folders_list:
            print('processing folder %s' % f.name)
            if self.split_folders:
                roi_path = predictions_folder.joinpath('roi_detections_clean_%s.txt' % f.name)
            else:
                roi_path = predictions_folder.joinpath('roi_detections_clean.txt')
            if not roi_path.exists():
                detected_foregrounds = self.all_predictions_to_dataframe(labels_folder=f)
                if min_conf is not None:
                    detected_foregrounds = detected_foregrounds.loc[detected_foregrounds.confidence >= min_conf]

                # Convert to RAVEN format
                columns = ['Selection', 'View', 'Channel', 'Begin File', 'End File', 'Begin Time (s)', 'End Time (s)',
                           'Beg File Samp (samples)', 'End File Samp (samples)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Tags']

                # convert the df to Raven format
                detected_foregrounds['Low Freq (Hz)'] = detected_foregrounds['min_freq']
                detected_foregrounds['High Freq (Hz)'] = detected_foregrounds['max_freq']
                detected_foregrounds['Tags'] = detected_foregrounds['class']

                detected_foregrounds.loc[detected_foregrounds['Low Freq (Hz)'] < 0, 'Low Freq (Hz)'] = 0
                detected_foregrounds = detected_foregrounds.loc[
                    detected_foregrounds['Low Freq (Hz)'] <= self.desired_fs / 2]
                detected_foregrounds.loc[detected_foregrounds['High Freq (Hz)'] > self.desired_fs / 2,
                                         'High Freq (Hz)'] = self.desired_fs / 2

                detected_foregrounds['View'] = 'Spectrogram 1'
                detected_foregrounds['Channel'] = 1
                detected_foregrounds['Begin File'] = detected_foregrounds['wav_name']
                detected_foregrounds['End File'] = detected_foregrounds['wav_name']
                detected_foregrounds['Begin Time (s)'] = detected_foregrounds['start_seconds']
                detected_foregrounds['End Time (s)'] = detected_foregrounds['start_seconds'] + detected_foregrounds[
                    'duration']

                detected_foregrounds['fs'] = np.nan
                detected_foregrounds['cummulative_sec'] = np.nan

                cummulative_seconds = 0
                if self.split_folders:
                    wavs_f_folder = self.wavs_folder.joinpath(f.name)
                else:
                    wavs_f_folder = self.wavs_folder
                wavs_to_check = list(wavs_f_folder.glob('*.wav'))
                if isinstance(wavs_to_check[0], pathlib.PosixPath):
                    wavs_to_check.sort()
                for wav_file_path in wavs_to_check:
                    if add_station_name:
                        wav_path_name = wav_file_path.parent.parent.parent.name.split('_')[0] + '_' + wav_file_path.name
                    else:
                        wav_path_name = wav_file_path.name

                    waveform_info = torchaudio.info(wav_file_path)
                    mask = detected_foregrounds['wav_name'] == wav_path_name
                    detected_foregrounds.loc[mask, 'cummulative_sec'] = cummulative_seconds
                    detected_foregrounds.loc[mask, 'fs'] = waveform_info.sample_rate
                    cummulative_seconds += waveform_info.num_frames / waveform_info.sample_rate

                detected_foregrounds['Beg File Samp (samples)'] = (detected_foregrounds['Begin Time (s)']
                                                                 * detected_foregrounds['fs']).astype(int)
                detected_foregrounds['End File Samp (samples)'] = (detected_foregrounds['End Time (s)']
                                                                 * detected_foregrounds['fs']).astype(int)
                detected_foregrounds['Begin Time (s)'] = detected_foregrounds['Begin Time (s)'] + detected_foregrounds[
                    'cummulative_sec']
                detected_foregrounds['End Time (s)'] = detected_foregrounds['End Time (s)'] + detected_foregrounds[
                    'cummulative_sec']

                detected_foregrounds = detected_foregrounds.sort_values('Begin Time (s)')
                detected_foregrounds = detected_foregrounds.reset_index(names='Selection')
                detected_foregrounds['Selection'] = detected_foregrounds['Selection'] + 1

                clean_detections = pd.DataFrame()
                for _, class_detections in detected_foregrounds.groupby('Tags'):
                    clean_detections_class = self.join_overlapping_detections_in_chunks(class_detections)
                    clean_detections = pd.concat([clean_detections, clean_detections_class])
                clean_detections['Selection'] = clean_detections['Selection'].astype(int)

                clean_detections.to_csv(roi_path, sep='\t', index=False)
            else:
                clean_detections = pd.read_table(roi_path)
        return clean_detections, roi_path


    @staticmethod
    def join_overlapping_detections(raven_detections_df, iou_threshold=0.5):
        # join all the detections overlapping an iou more than the threshold %
        selected_ids = []
        for wav_file_name, wav_selections in tqdm(raven_detections_df.groupby('Begin File'),
                                                  total=len(raven_detections_df['Begin File'].unique())):
            already_joined_ids = []
            if len(wav_selections) > 1:
                # wav_selections = wav_selections.sort_values('Begin Time (s)')
                for i, one_selection in wav_selections.iterrows():
                    selections_to_check = wav_selections.loc[~wav_selections.index.isin(already_joined_ids)].copy()
                    if i not in already_joined_ids:
                        min_end = np.minimum(one_selection['End Time (s)'], selections_to_check['End Time (s)'])
                        max_start = np.maximum(one_selection['Begin Time (s)'], selections_to_check['Begin Time (s)'])
                        max_bottom = np.maximum(one_selection['Low Freq (Hz)'], selections_to_check['Low Freq (Hz)'])
                        min_top = np.minimum(one_selection['High Freq (Hz)'], selections_to_check['High Freq (Hz)'])
                        # possible_overlaps = selections_to_check.loc[(min_end > max_start) & (min_top > max_bottom)]
                        inter = (min_end - max_start).clip(0) * (min_top - max_bottom).clip(0)
                        union = ((one_selection['End Time (s)'] - one_selection['Begin Time (s)']) *
                                 (one_selection['High Freq (Hz)'] - one_selection['Low Freq (Hz)'])) + \
                                ((selections_to_check['End Time (s)'] - selections_to_check['Begin Time (s)']) *
                                 (selections_to_check['High Freq (Hz)'] - selections_to_check['Low Freq (Hz)'])) - inter
                        iou = inter / union
                        overlapping_selections = selections_to_check.loc[iou > iou_threshold]
                        if len(overlapping_selections) > 1:
                            already_joined_ids = np.concatenate([already_joined_ids,
                                                                 overlapping_selections.index.values])

                            raven_detections_df.loc[i, 'Begin Time (s)'] = overlapping_selections[
                                'Begin Time (s)'].min()
                            raven_detections_df.loc[i, 'End Time (s)'] = overlapping_selections['End Time (s)'].max()
                            raven_detections_df.loc[i, 'Beg File Samp (samples)'] = overlapping_selections[
                                'Beg File Samp (samples)'].max()
                            raven_detections_df.loc[i, 'End File Samp (samples)'] = overlapping_selections[
                                'End File Samp (samples)'].max()
                            raven_detections_df.loc[i, 'Low Freq (Hz)'] = overlapping_selections['Low Freq (Hz)'].min()
                            raven_detections_df.loc[i, 'High Freq (Hz)'] = overlapping_selections[
                                'High Freq (Hz)'].max()
                            raven_detections_df.loc[i, 'confidence'] = overlapping_selections['confidence'].max()
                        else:
                            already_joined_ids = np.concatenate([already_joined_ids, [i]])
                        selected_ids.append(i)
            else:
                selected_ids += [wav_selections.index.values[0]]
        cleaned_detections = raven_detections_df.loc[selected_ids]
        return cleaned_detections

    def join_overlapping_detections_in_chunks(self, raven_detections_df, iou_threshold=0.5):
        # join all the detections overlapping an iou more than the threshold %
        selected_rows = []

        for chunk_n, chunk in tqdm(raven_detections_df.groupby('Begin File'),
                                   total=len(raven_detections_df.groupby('Begin File'))):
            already_joined_ids = []
            if len(chunk) > 1:
                # wav_selections = wav_selections.sort_values('Begin Time (s)')
                for i, one_selection in chunk.iterrows():
                    selections_to_check = chunk.loc[~chunk.index.isin(already_joined_ids)].copy()
                    if i not in already_joined_ids:
                        min_end = np.minimum(one_selection['End Time (s)'], selections_to_check['End Time (s)'])
                        max_start = np.maximum(one_selection['Begin Time (s)'], selections_to_check['Begin Time (s)'])
                        max_bottom = np.maximum(one_selection['Low Freq (Hz)'], selections_to_check['Low Freq (Hz)'])
                        min_top = np.minimum(one_selection['High Freq (Hz)'], selections_to_check['High Freq (Hz)'])
                        # possible_overlaps = selections_to_check.loc[(min_end > max_start) & (min_top > max_bottom)]
                        inter = (min_end - max_start).clip(0) * (min_top - max_bottom).clip(0)
                        union = ((one_selection['End Time (s)'] - one_selection['Begin Time (s)']) *
                                 (one_selection['High Freq (Hz)'] - one_selection['Low Freq (Hz)'])) + \
                                ((selections_to_check['End Time (s)'] - selections_to_check['Begin Time (s)']) *
                                 (selections_to_check['High Freq (Hz)'] - selections_to_check['Low Freq (Hz)'])) - inter
                        iou = inter / union
                        overlapping_selections = selections_to_check.loc[iou > iou_threshold]
                        if len(overlapping_selections) > 1:
                            already_joined_ids = np.concatenate([already_joined_ids,
                                                                 overlapping_selections.index.values])

                            one_selection['Begin Time (s)'] = overlapping_selections['Begin Time (s)'].min()
                            one_selection['End Time (s)'] = overlapping_selections['End Time (s)'].max()
                            one_selection['Beg File Samp (samples)'] = overlapping_selections[
                                'Beg File Samp (samples)'].max()
                            one_selection['End File Samp (samples)'] = overlapping_selections[
                                'End File Samp (samples)'].max()
                            one_selection['Low Freq (Hz)'] = overlapping_selections['Low Freq (Hz)'].min()
                            one_selection['High Freq (Hz)'] = overlapping_selections[
                                'High Freq (Hz)'].max()
                            one_selection['confidence'] = overlapping_selections['confidence'].max()
                        else:
                            already_joined_ids = np.concatenate([already_joined_ids, [i]])
                        selected_rows.append(one_selection.values)
            else:
                selected_rows.append(chunk.iloc[0].values)

        cleaned_detections = np.stack(selected_rows)
        cleaned_detections = pd.DataFrame(cleaned_detections, columns=chunk.columns)

        return cleaned_detections

    @staticmethod
    def compute_detection_overlap_with_dataset(detected_foregrounds, df_to_compare):
        detected_foregrounds['iou'] = np.nan
        for i, d in tqdm(detected_foregrounds.iterrows(), total=len(detected_foregrounds)):
            w = df_to_compare.width.copy()
            w.loc[w > d.width] = d.width

            top = df_to_compare.max_freq.copy()
            top.loc[top > d.max_freq] = d.max_freq

            bottom = df_to_compare.min_freq.copy()
            bottom.loc[bottom < d.min_freq] = d.min_freq

            iou = (top - bottom) * w / (d.height * d.width)
            iou[iou < 0] = 0
            detected_foregrounds.loc[i, 'iou'] = np.percentile(iou, 90)

        return detected_foregrounds

    # def plot_annotation(self, chunk_selection, sxx):
    #     px_freq = self.desired_fs / self.nfft
    #     sxx_width = int((self.desired_fs * self.duration) / self.hop_length) + 1
    #     sxx_height = self.desired_fs / px_freq / 2 + 1
    #
    #     if len(chunk_selection) > 0:
    #         fig, ax = plt.subplots()
    #         im = ax.pcolormesh(sxx, cmap='Greys_r')
    #
    #         for _, box in chunk_selection.iterrows():
    #             rect = patches.Rectangle(xy=(box.x * sxx_width, (1 - (box.y + box.height)) * sxx_height),
    #                                      width=box.width * sxx_width, height=box.height * sxx_height,
    #                                      linewidth=1, edgecolor='r', facecolor='none')
    #             ax.add_patch(rect)
    #             ax.text(box.x * sxx_width, (1 - box.y) * sxx_height, box.Tags)
    #         plt.title(len(chunk_selection))
    #         plt.colorbar(im)
    #         plt.savefig(output.joinpath('detections_bbox', img_path.name))
    #         plt.close()

    def all_snippets(self, detected_foregrounds, labels_to_exclude=None):

        file_list = os.listdir(self.wavs_folder)
        for i, row in tqdm(detected_foregrounds.iterrows(), total=len(detected_foregrounds)):
            wav_path = row['wav']
            waveform_info = torchaudio.info(wav_path)

            # If the selection is in between two files, open both and concatenate them
            if row['Beg File Samp (samples)'] > row['End File Samp (samples)']:
                waveform1, fs = torchaudio.load(wav_path,
                                                frame_offset=row['Beg File Samp (samples)'],
                                                num_frames=waveform_info.num_frames - row[
                                                    'Beg File Samp (samples)'])

                wav_path2 = self.wavs_folder.joinpath(file_list[file_list.index(row['Begin File']) + 1])
                waveform2, fs = torchaudio.load(wav_path2,
                                                frame_offset=0,
                                                num_frames=row['End File Samp (samples)'])
                waveform = torch.cat([waveform1, waveform2], -1)
            else:
                waveform, fs = torchaudio.load(wav_path,
                                               frame_offset=row['Beg File Samp (samples)'],
                                               num_frames=row['End File Samp (samples)'] - row[
                                                   'Beg File Samp (samples)'])
            if waveform_info.sample_rate > self.desired_fs:
                waveform = F.resample(waveform=waveform, orig_freq=fs, new_freq=self.desired_fs)[self.channel, :]
            else:
                waveform = waveform[self.channel, :]

            yield i, row, waveform

    def encode_aves(self, model_path, strategy='mean', labels_to_exclude=None, use_raven_filename=True):
        output_name = 'AVES_features_space_%s' % strategy
        features_path = self.dataset_folder.joinpath(output_name + '.pt')
        if not features_path.exists():
            model_list, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
            model = model_list[0]
            model.feature_extractor.requires_grad_(False)
            model.eval()
            features_list = []
            idxs = []
            for _, detected_foregrounds in self.load_relevant_selection_table(labels_to_exclude=labels_to_exclude):
                detected_foregrounds['height'] = detected_foregrounds[
                                                     'High Freq (Hz)'] - detected_foregrounds['Low Freq (Hz)']
                detected_foregrounds['width'] = detected_foregrounds[
                                                    'End Time (s)'] - detected_foregrounds['Begin Time (s)']

                for i, _, waveform in self.all_snippets(detected_foregrounds, labels_to_exclude=labels_to_exclude):
                    if strategy == 'mean':
                        features = model.extract_features(waveform.expand(1, -1))[0].mean(dim=1)
                    elif strategy == 'max':
                        features = model.extract_features(waveform.expand(1, -1))[0].max(dim=1).values
                    else:
                        raise Exception('Strategy %s is not defined. Only mean or max' % strategy)

                    features_list.append(features.squeeze(dim=0).detach().numpy())
                    idxs.append(i)

            features_space = torch.Tensor(np.stack(features_list).astype(float))
            torch.save(features_space, features_path)
            df = pd.DataFrame(features_space.numpy())
            df.index = idxs
            columns = ['Low Freq (Hz)', 'High Freq (Hz)', 'height', 'width', 'SNR NIST Quick (dB)', 'Tags']
            if 'SNR NIST Quick (dB)' not in detected_foregrounds.columns:
                columns = ['Low Freq (Hz)', 'High Freq (Hz)', 'height', 'width', 'Tags']
            total_df = pd.merge(df, detected_foregrounds[columns],
                                left_index=True, right_index=True)
            total_df = total_df.rename(
                columns={'Low Freq (Hz)': 'min_freq', 'High Freq (Hz)': 'max_freq', 'height': 'bandwidth',
                         'width': 'duration', 'SNR NIST Quick (dB)': 'snr',
                         'Tags': 'label'})

            total_df.to_pickle(self.dataset_folder.joinpath(output_name + '.pkl'))
        else:
            total_df = pd.read_pickle(self.dataset_folder.joinpath(output_name + '.pkl'))

        return

    def encode_clap(self, labels_to_exclude=None, max_duration=3):
        output_name = 'CLAP_features_space_filtered_%s' % max_duration
        features_path = self.dataset_folder.joinpath(output_name + '.pkl')
        if not features_path.exists():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = ClapModel.from_pretrained("davidrrobinson/BioLingual").to(device)
            processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual", sampling_rate=self.desired_fs)

            total_df = pd.DataFrame()
            folder_n = 0
            for _, selection_table in self.load_relevant_selection_table(labels_to_exclude=labels_to_exclude):
                if not self.dataset_folder.joinpath(output_name + '_%s.pkl' % folder_n).exists():
                    if 'wav' not in selection_table.columns:
                        if isinstance(self.wavs_folder, pathlib.PosixPath):
                            joining_str = '/'
                        else:
                            joining_str = '\\'
                        selection_table = selection_table.assign(wav=str(self.wavs_folder) + joining_str + selection_table['Begin File'])
                    selection_table['height'] = selection_table['High Freq (Hz)'] - selection_table['Low Freq (Hz)']
                    selection_table['width'] = selection_table['End Time (s)'] - selection_table['Begin Time (s)']
                    if 'min_freq' in selection_table.columns:
                        selection_table = selection_table.drop(columns=['min_freq', 'max_freq'])
                    selection_table = selection_table.rename(columns={'High Freq (Hz)': 'max_freq',
                                                                      'Low Freq (Hz)': 'min_freq'
                                                                      })

                    if 'begin_sample' not in selection_table.columns:
                        selection_table = selection_table.rename(columns={'Begin File': 'filename',
                                                                          'Beg File Samp (samples)': 'begin_sample',
                                                                          'End File Samp (samples)': 'end_sample'
                                                                          })

                    dataloader = torch.utils.data.DataLoader(
                        dataset=u.DatasetWaveform(df=selection_table, wavs_folder=self.wavs_folder, desired_fs=self.desired_fs,
                                                  max_duration=max_duration),
                        batch_size=16,
                        shuffle=False)
                    features_list, idxs = [], []
                    for x, i in tqdm(dataloader):
                        x = [s.cpu().numpy() for s in x]
                        inputs = processor(audios=x, return_tensors="pt", sampling_rate=self.desired_fs).to(
                            device)
                        audio_embed = model.get_audio_features(**inputs)
                        features_list.extend(audio_embed.cpu().detach().numpy())
                        idxs.extend(i.cpu().detach().numpy())

                    features_space = torch.Tensor(np.stack(features_list).astype(float))
                    torch.save(features_space, features_path)
                    features_df = pd.DataFrame(features_space.numpy())

                    features_df.index = idxs

                    columns = ['min_freq', 'max_freq', 'height', 'width', 'SNR NIST Quick (dB)', 'Tags']
                    if 'SNR NIST Quick (dB)' not in selection_table.columns:
                        columns = ['min_freq', 'max_freq', 'height', 'width', 'Tags']
                    df = pd.merge(features_df, selection_table[columns], left_index=True, right_index=True)
                    df = df.rename(
                        columns={'height': 'bandwidth',
                                 'width': 'duration', 'SNR NIST Quick (dB)': 'snr',
                                 'Tags': 'label'})

                    df.to_pickle(self.dataset_folder.joinpath(output_name + '_%s.pkl' % folder_n))
                else:
                    df = pd.read_pickle(self.dataset_folder.joinpath(output_name + '_%s.pkl' % folder_n))
                total_df = pd.concat([total_df, df])
                folder_n += 1
            total_df.to_pickle(self.dataset_folder.joinpath(output_name + '.pkl'))
        else:
            total_df = pd.read_pickle(self.dataset_folder.joinpath(output_name + '.pkl'))

        return total_df

    def zero_shot_learning(self, labels_to_exclude=None):
        output_name = 'zero_shot_space'
        features_path = self.dataset_folder.joinpath(output_name + '.pkl')
        if not features_path.exists():
            audio_classifier = pipeline(task="zero-shot-audio-classification", model="davidrrobinson/BioLingual")
            features_list = []
            idxs = []
            for _, detected_foregrounds in self.load_relevant_selection_table(labels_to_exclude=labels_to_exclude):

                detected_foregrounds['height'] = detected_foregrounds[
                                                     'High Freq (Hz)'] - detected_foregrounds['Low Freq (Hz)']
                detected_foregrounds['width'] = detected_foregrounds[
                                                    'End Time (s)'] - detected_foregrounds['Begin Time (s)']

                detected_foregrounds = detected_foregrounds.loc[~detected_foregrounds.Tags.isna()]
                detected_foregrounds = detected_foregrounds.replace({'jackhammer': 'repetitive sound of a fish',
                                                                     'grunt': 'grunting sound of a fish',
                                                                     'whistle': 'whistle sound of proably a dolphin',
                                                                     'loud_ship': 'sound of a ship passing by',
                                                                     'tap_dancing': 'pseudo-noise of invertebrates tapping the hydrophone',
                                                                     'metallic_bell': 'metallic sound like a bell with harmonics',
                                                                     'metallic_bell_nh': 'quiet metallic sound like a bell with no harmonics',
                                                                     'siren': 'sound like a siren',
                                                                     'hf_click': 'high frequency impulsive broaddband sound',
                                                                     'castanets': 'sound like repetitive castanets',
                                                                     'low_tap': 'low frequency impulsive sound',
                                                                     'rain': 'rain like sound',
                                                                     'chirp': 'seal chirp',
                                                                     'constant_freq': 'sound of constant frequency',
                                                                     'wave': 'sound of wave passing by',
                                                                     'boat_background': 'sound of a boat in the background',
                                                                     'woop': 'three consecutive woop sounds,maybe fish'})
                candidates_labels = detected_foregrounds.Tags.unique()
                for i, _, waveform in self.all_snippets(detected_foregrounds=detected_foregrounds,
                                                     labels_to_exclude=labels_to_exclude):
                    output = audio_classifier(waveform.numpy(), candidate_labels=candidates_labels)
                    max_score = 0
                    max_label = ''
                    for o in output:
                        if o['score'] > max_score:
                            max_label = o['label']
                            max_score = o['score']
                    features_list.append(max_label)
                    idxs.append(i)

                df = pd.DataFrame(features_list)
                df.index = idxs
                columns = ['Low Freq (Hz)', 'High Freq (Hz)', 'height', 'width', 'SNR NIST Quick (dB)', 'Tags']
                if 'SNR NIST Quick (dB)' not in detected_foregrounds.columns:
                    columns = ['Low Freq (Hz)', 'High Freq (Hz)', 'height', 'width', 'Tags']
                total_df = pd.merge(df, detected_foregrounds[columns],
                                    left_index=True, right_index=True)
                total_df = total_df.rename(
                    columns={'Low Freq (Hz)': 'min_freq', 'High Freq (Hz)': 'max_freq', 'height': 'bandwidth',
                             'width': 'duration', 'SNR NIST Quick (dB)': 'snr',
                             'Tags': 'label'})

                total_df.to_pickle(self.dataset_folder.joinpath(output_name + '.pkl'))
            else:
                total_df = pd.read_pickle(self.dataset_folder.joinpath(output_name + '.pkl'))

            return total_df

    def load_relevant_selection_table(self, labels_to_exclude=None):
        annotations_file = pathlib.Path(self.annotations_file)
        if annotations_file.is_dir():
            selections_list = list(annotations_file.glob('*.txt'))
        else:
            selections_list = [annotations_file]
        for selection_table_path in selections_list:
            print('Annotations table %s' % selection_table_path.name)
            selections = pd.read_table(selection_table_path)
            if 'Tags' in selections.columns:
                if labels_to_exclude is not None:
                    selections = selections.loc[~selections.Tags.isin(labels_to_exclude)]

            # Filter the selections
            selections = selections.loc[selections['Low Freq (Hz)'] < (self.desired_fs / 2)]
            selections = selections.loc[selections.View == 'Spectrogram 1']
            if 'SNR NIST Quick (dB)' in selections.columns:
                selections = selections.loc[selections['SNR NIST Quick (dB)'] > self.MIN_SNR]
            selections.loc[selections['High Freq (Hz)'] > (self.desired_fs / 2), 'High Freq (Hz)'] = self.desired_fs / 2
            selections = selections.loc[
                (selections['End Time (s)'] - selections['Begin Time (s)']) >= self.MIN_DURATION]
            selections = selections.loc[
                (selections['End Time (s)'] - selections['Begin Time (s)']) <= self.MAX_DURATION]

            yield selection_table_path, selections

    def convert_raven_to_ae_format(self, labels_to_exclude=None):
        total_encoder = pd.DataFrame()
        for _, selection_table in self.load_relevant_selection_table(labels_to_exclude=labels_to_exclude):
            for wav_file, wav_selections in selection_table.groupby('Begin File'):
                wav_path = list(self.wavs_folder.glob('**/' + wav_file))[0]
                wav = sf.SoundFile(wav_path)
                duration = wav_selections['End Time (s)'] - wav_selections['Begin Time (s)']
                two_files = wav_selections['Beg File Samp (samples)'] > wav_selections['End File Samp (samples)']
                pos = wav_selections['Beg File Samp (samples)'] / wav.samplerate + duration / 2
                encoder_df = pd.DataFrame({'begin_sample': wav_selections['Beg File Samp (samples)'],
                                           'end_sample': wav_selections['End File Samp (samples)'],
                                           'pos': pos,
                                           'duration': duration,
                                           'filename': wav_selections['Begin File'],
                                           'label': wav_selections['Tags'],
                                           'min_freq': wav_selections['Low Freq (Hz)'],
                                           'max_freq': wav_selections['High Freq (Hz)'],
                                           'two_files': two_files})
                if 'SNR NIST Quick (dB)' in wav_selections.columns:
                    encoder_df['snr'] = wav_selections['SNR NIST Quick (dB)']
                total_encoder = pd.concat([total_encoder, encoder_df])

        return total_encoder

    def encode_ae(self, model_path, nfft, sample_dur, n_mel, bottleneck, labels_to_exclude=None, input_type='fixed', ):
        self.MIN_DURATION = nfft / self.desired_fs
        features_path = self.dataset_folder.joinpath('CAE_%s_%s_%s_%s_%s_features_space.pkl' %
                                                     (input_type, nfft, sample_dur, n_mel, bottleneck))
        if not features_path.exists():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            if input_type == 'fixed':
                frontend = models.frontend(sr=self.desired_fs, nfft=nfft, sampleDur=sample_dur, n_mel=n_mel).to(device)
            elif input_type == 'cropsduration':
                frontend = models.frontend_crop_duration(sr=self.desired_fs, nfft=nfft,
                                                         sampleDur=sample_dur, n_mel=n_mel).to(device)
            elif input_type == 'crops':
                frontend = models.frontend_crop()

            encoder = models.sparrow_encoder(bottleneck // (n_mel // 32 * 4), (n_mel // 32, 4))
            decoder = models.sparrow_decoder(bottleneck, (n_mel // 32, 4))
            model = torch.nn.Sequential(encoder, decoder).to(device)

            model.load_state_dict(torch.load(model_path))
            model.eval()

            detections = self.convert_raven_to_ae_format(labels_to_exclude=labels_to_exclude)

            if input_type == 'fixed':
                annotations_ds = u.Dataset(df=detections, audiopath=str(self.wavs_folder), sr=self.desired_fs,
                                           sampleDur=sample_dur, channel=self.channel)
            elif input_type == 'cropsduration':
                annotations_ds = u.DatasetCropsDuration(detections, str(self.wavs_folder), self.desired_fs,
                                                        winsize=nfft, n_mel=n_mel, win_overlap=self.win_overlap,
                                                        sampleDur=sample_dur)
            elif input_type == 'crops':
                annotations_ds = u.DatasetCrops(detections, str(self.wavs_folder), self.desired_fs,
                                                winsize=nfft, n_mel=n_mel, win_overlap=self.win_overlap,
                                                sampleDur=sample_dur)

            loader = torch.utils.data.DataLoader(annotations_ds, batch_size=16, shuffle=False,
                                                 num_workers=0, collate_fn=u.collate_fn)
            encodings, idxs = [], []
            with torch.no_grad():
                for x, name in tqdm(loader, leave=False):
                    label = frontend(x.to(device))
                    encoding = model[:1](label)
                    idxs.extend(name.numpy())
                    encodings.extend(encoding.cpu().detach())
            encodings = np.stack(encodings)
            features_space = torch.Tensor(np.stack([encodings]).astype(float))
            features_space = pd.DataFrame(features_space.numpy()[0])
            idxs = np.stack(idxs)
            features_space.index = idxs
            columns = ['min_freq', 'max_freq', 'duration', 'label', 'snr']
            if 'snr' not in detections.columns:
                columns = ['min_freq', 'max_freq', 'duration', 'label']
            total_features = pd.merge(features_space, detections[columns],
                                      left_index=True, right_index=True)
            total_features['bandwidth'] = total_features['max_freq'] - total_features['min_freq']
            total_features.to_pickle(features_path)

        else:
            total_features = pd.read_pickle(features_path)

        return total_features

    def train_ae(self, sample_dur=5, bottleneck=48, n_mel=128, nfft=2048, input_type='fixed'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = models.sparrow_encoder(bottleneck // (n_mel // 32 * 4), (n_mel // 32, 4))
        decoder = models.sparrow_decoder(bottleneck, (n_mel // 32, 4))
        model = torch.nn.Sequential(encoder, decoder).to(device)

        if input_type == 'fixed':
            frontend = models.frontend(sr=self.desired_fs, nfft=nfft, sampleDur=sample_dur, n_mel=n_mel).to(device)
        elif input_type == 'cropsduration':
            frontend = models.frontend_crop_duration(sr=self.desired_fs, nfft=nfft,
                                                     sampleDur=sample_dur, n_mel=n_mel).to(device)
        elif input_type == 'crops':
            frontend = models.frontend_crop()

        # training / optimisation setup
        lr, wdL2, batch_size = 0.00001, 0.0, 64 if torch.cuda.is_available() else 16
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=wdL2, lr=lr, betas=(0.8, 0.999))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: .99 ** epoch)
        vgg16 = models.vgg16.eval().to(device)
        loss_fun = torch.nn.MSELoss()

        detections = self.convert_raven_to_ae_format(labels_to_exclude=None)

        if input_type == 'fixed':
            annotations_ds = u.Dataset(df=detections, audiopath=str(self.wavs_folder), sr=self.desired_fs,
                                       sampleDur=sample_dur, channel=self.channel)
        elif input_type == 'cropsduration':
            annotations_ds = u.DatasetCropsDuration(detections, str(self.wavs_folder), self.desired_fs,
                                                    winsize=nfft, n_mel=n_mel, win_overlap=self.win_overlap,
                                                    sampleDur=sample_dur)
        elif input_type == 'crops':
            annotations_ds = u.DatasetCrops(detections, str(self.wavs_folder), self.desired_fs,
                                            winsize=nfft, n_mel=n_mel, win_overlap=self.win_overlap,
                                            sampleDur=sample_dur)

        loader = torch.utils.data.DataLoader(annotations_ds, batch_size=16,
                                             shuffle=False, num_workers=0, collate_fn=u.collate_fn)

        modelname = f'CAE_{input_type}_{bottleneck}_mel{n_mel}'
        step, writer = 0, SummaryWriter(str(self.dataset_folder.joinpath(modelname)))
        print(f'Go for model {modelname} with {len(detections)} vocalizations')
        for epoch in range(100):
            for x, name in tqdm(loader, desc=str(epoch), leave=False):
                optimizer.zero_grad()
                label = frontend(x.to(device))
                x = encoder(label)
                pred = decoder(x)
                vgg_pred = vgg16(pred.expand(pred.shape[0], 3, *pred.shape[2:]))
                vgg_label = vgg16(label.expand(label.shape[0], 3, *label.shape[2:]))

                score = loss_fun(vgg_pred, vgg_label)
                score.backward()
                optimizer.step()
                writer.add_scalar('loss', score.item(), step)

                if step % 50 == 0:
                    images = [(e - e.min()) / (e.max() - e.min()) for e in label[:8]]
                    grid = make_grid(images)
                    writer.add_image('target', grid, step)
                    writer.add_embedding(x.detach(), global_step=step, label_img=label)
                    images = [(e - e.min()) / (e.max() - e.min()) for e in pred[:8]]
                    grid = make_grid(images)
                    writer.add_image('reconstruct', grid, step)

                step += 1
                if step % 500 == 0:
                    scheduler.step()
            torch.save(model.state_dict(), str(self.dataset_folder.joinpath(modelname + '.weights')))
        return str(self.dataset_folder.joinpath(modelname + '.weights'))

    def train_clap(self, log_folder, model_path='davidrrobinson/BioLingual', epochs=100, lr=5e-6,
                   batch_size=16, stop_shuffle=False, sample_dur=3):
        self.desired_fs = 48000
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_path = pathlib.Path(log_folder).joinpath('/logs.log')
        log_file = open(log_path, mode='w')

        detections = self.convert_raven_to_ae_format(labels_to_exclude=None)
        detections = detections.loc[~detections.label.isna()]
        num_classes = len(detections.label.unique())
        model = models.CLAPClassifier(model_path, num_classes, sr=self.desired_fs, device=device, multi_label=False)

        d_train, d_valid, _, _ = train_test_split(detections, detections['label'], test_size=0.2)

        dataloader_train = torch.utils.data.DataLoader(
            dataset=u.DatasetWaveform(df=d_train, wavs_folder=self.wavs_folder, desired_fs=self.desired_fs,
                                      max_duration=sample_dur),
            batch_size=batch_size,
            shuffle=not stop_shuffle)

        dataloader_val = torch.utils.data.DataLoader(
            dataset=u.DatasetWaveform(df=d_valid, wavs_folder=self.wavs_folder, desired_fs=self.desired_fs,
                                      max_duration=sample_dur),
            batch_size=batch_size,
            shuffle=not stop_shuffle)

        valid_metric_best = 0.
        best_model = None

        print(f"lr = {lr}", file=log_file)
        optimizer = optim.Adam(params=model.parameters(), lr=lr)

        for epoch in range(epochs):
            print(f'epoch = {epoch}', file=sys.stderr)

            model.train()

            train_loss = 0.
            train_steps = 0
            train_metric = u.Accuracy()

            for x, y in tqdm(dataloader_train, desc='train'):
                optimizer.zero_grad()

                x = x.to(device)
                y = y.to(device)

                loss, logits = model(x, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.to(device)
                train_steps += 1

                train_metric.update(logits, y)

            valid_loss, valid_metric = u.eval_pytorch_model(
                model=model,
                dataloader=dataloader_val,
                metric_factory=u.Accuracy,
                device=device,
                desc='valid')

            if valid_metric > valid_metric_best:
                valid_metric_best = valid_metric
                best_model = copy.deepcopy(model)
                model.clap.save_pretrained(pathlib.Path(log_folder).joinpath('best_model'))
            model.clap.save_pretrained(pathlib.Path(log_folder).joinpath('last_model'))

            print({
                'epoch': epoch,
                'train': {
                    'loss': (train_loss / train_steps).cpu().item(),
                    'metric': train_metric.get_metric(),
                },
                'valid': {
                    'loss': valid_loss,
                    'metric': valid_metric
                }
            }, file=log_file)
            log_file.flush()

        return best_model, valid_metric_best

    def add_clusters_to_raven(self, clusters):
        for _, selections in self.load_relevant_selection_table():
            selections.loc[:, 'clusters'] = clusters.values
            selections.loc[:, 'revise'] = 0
            selections.to_csv(str(self.annotations_file).replace('.txt', '_clusters.txt'), sep='\t', index=False)
            for cluster_number, cluster_selections in selections.groupby('clusters'):
                cluster_selections.loc[cluster_selections.sample(5, random_state=42).index, 'revise'] = 1
                cluster_selections.to_csv(
                    str(self.annotations_file).replace('.txt', '_cluster_%s.txt' % cluster_number),
                    sep='\t', index=False)
            yield selections

    def plot_clusters_polar_day(self, foregrounds_with_clusters):
        df = foregrounds_with_clusters
        df['datetime'] = df['Begin File'].str[-23:-4]
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d_%H-%M-%S')
        df['datetime'] = df['datetime'] + pd.to_timedelta(df['Beg File Samp (samples)'] / self.desired_fs,
                                                          unit='seconds')
        df['minute'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
        df['hour'] = df['datetime'].dt.hour
        start_time = df.datetime.round('h').min()
        end_time = df.datetime.round('h').max()
        hours_array = pd.date_range(start=start_time, end=end_time, freq='h')
        duration_array = np.arange(0, 24, 1)

        datetime_hour_df = pd.DataFrame(index=hours_array, columns=df['clusters'].unique())
        for (datetime_h, c), hour_detections in df.groupby([df.datetime.dt.round('h'), 'clusters']):
            datetime_hour_df.loc[datetime_h, c] = len(hour_detections.minute.unique()) / 60

        hour_df = datetime_hour_df.groupby(datetime_hour_df.index.hour).mean() * 100
        n = len(duration_array)
        theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        width = np.pi / n

        ax = plt.subplot(projection='polar')
        lower_limits = np.zeros_like(theta)
        colors = plt.get_cmap('tab20', len(hour_df.columns) + 1).colors
        labels = []
        lines = []
        for c_i, cluster_n in enumerate(hour_df.columns[:3]):
            radii = hour_df[cluster_n].values.astype(float)
            bars = ax.bar(x=theta, height=radii, width=width, bottom=lower_limits, color=colors[c_i])
            lower_limits += radii
            line_2d = plt.Line2D([], [], marker='o', linewidth=0, color=colors[c_i])
            lines.append(line_2d)
            labels.append(cluster_n)
        plt.legend(
            lines, labels,
            loc='lower right',
            title='Cluster',
        )

        ax.set_rorigin(-10)
        ax.set_rgrids([10, 25, 50])
        ax.set_thetagrids(theta * 360 / (np.pi * 2), labels=duration_array)
        plt.xlabel('')
        plt.tight_layout()
        plt.savefig(self.dataset_folder.joinpath('clusters_polar.png'))
        plt.show()

    def plot_clusters_evolution(self, foregrounds_with_clusters, latitude, longitude, tz):
        df = foregrounds_with_clusters
        df['datetime'] = df['Begin File'].str[-23:-4]
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d_%H-%M-%S')
        df['datetime'] = df['datetime'] + pd.to_timedelta(df['Beg File Samp (samples)'] / self.desired_fs,
                                                          unit='seconds')
        df['minute'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
        df['day'] = df.datetime.dt.date
        first_day = df.day.min()
        last_day = df.day.max()
        dates_range = pd.date_range(start=first_day, end=last_day, freq='1D')
        minutes_array = np.arange(0, 24 * 60)
        sun = suntime.Sun(latitude, longitude)
        sunrise_list = []
        sunset_list = []
        for d in dates_range:
            sunrise = sun.get_local_sunrise_time(d)
            sunset = sun.get_local_sunset_time(d)
            sunrise = sunrise.replace(tzinfo=pytz.timezone(tz)).astimezone(pytz.UTC)
            sunset = sunset.replace(tzinfo=pytz.timezone(tz)).astimezone(pytz.UTC)
            sunrise_list.append(sunrise.hour + sunrise.minute / 60)
            sunset_list.append(sunset.hour + sunset.minute / 60)

        for c, detections_c in df.groupby('clusters'):
            daily_patterns = pd.DataFrame(index=dates_range.date, columns=minutes_array)
            for (d, minute), minute_detections in detections_c.groupby(['day', 'minute']):
                daily_patterns.loc[d, minute] = len(minute_detections)

            daily_patterns = daily_patterns.fillna(0)
            fig, ax = plt.subplots()
            im = ax.pcolormesh(minutes_array/60, dates_range, daily_patterns.values, cmap='Greys')
            ax.plot(sunset_list, dates_range, color='blue', alpha=0.5)
            ax.plot(sunrise_list, dates_range, color='blue', alpha=0.5)
            plt.colorbar(im, label='number detections per minute', alpha=0.5)
            plt.xlabel('Time [h]')
            plt.ylabel('Day')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(self.dataset_folder.joinpath('daily_patterns_cluster_%s.png' % c))
            plt.show()

    def get_diversity_per_group(self, foregrounds_with_clusters, metadata_file, index, group, total_times,
                                clusters_to_consider='all'):
        df = foregrounds_with_clusters.merge(metadata_file, left_on='Begin File', right_on='wav_name')
        if type(clusters_to_consider) is list:
            df = df.loc[df['clusters'].isin(clusters_to_consider)]

        df['clusters'] = df['clusters'].astype('category')
        df_output = pd.DataFrame(index=df[group].unique(), columns=['adi', 'det_per_min', 'n_clusters'])
        for group_name, wav_selections in df.groupby(group):
            s = wav_selections['clusters'].value_counts()
            if index == "shannon":
                n = s.shape[0]
                adi = util.entropy(s, axis=0) * np.log(n)
            elif index == "simpson":
                s = s / sum(s)
                s = s ** 2
                adi = 1 - sum(s)
            elif index == "invsimpson":
                s = s / sum(s)
                s = s ** 2
                adi = 1 / sum(s)

            df_output.loc[group_name, 'adi'] = adi
            df_output.loc[group_name, 'det_per_min'] = len(wav_selections) / total_times[group][group_name]
            df_output.loc[group_name, 'n_clusters'] = len(wav_selections.clusters.unique())

        return df_output

    def plot_clusters_metadata_distr(self, foregrounds_with_clusters, metadata_file, total_times):
        df = foregrounds_with_clusters.merge(metadata_file, left_on='Begin File', right_on='wav_name')
        # df['datetime'] = df['Begin File'].str[-23:-4]
        # df['datetime'] = pd.to_datetime(df['datetime'],
        #                                                        format='%Y-%m-%d_%H-%M-%S')
        # df['datetime'] = df['datetime'] + pd.to_timedelta(df['Beg File Samp (samples)'] / self.desired_fs, unit='seconds')
        # df['minute'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute

        season_labels = {0: 'winter', 1: 'spring', 2: 'summer', 3: 'autumn'}
        moon_phase_labels = {0: 'new_moon', 1: 'growing', 2: 'full_moon', 3: 'decreasing'}
        day_labels = {0: 'Day', 1: 'Night', 2: 'Twilight'}
        # freq_array = np.arange(0, self.desired_fs / 2, 10)
        # duration_array = np.arange(0, 24 * 60, 1)
        # grid_duration, grid_freq = np.meshgrid(duration_array, freq_array)
        #
        # sum_iou_grid = np.zeros((len(freq_array), len(duration_array)))
        # metadata_file['datetime'] = metadata_file['wav_name'].str[-23:-4]
        # metadata_file['datetime'] = pd.to_datetime(metadata_file['datetime'], format='%Y-%m-%d_%H-%M-%S')
        # metadata_file['minute'] = metadata_file['datetime'].dt.hour * 60 + metadata_file['datetime'].dt.minute
        # for i, row in metadata_file.iterrows():
        #     row_end = row.minute + row.duration / 60
        #     mask = (grid_duration >= row.minute) & (grid_duration <= row_end)
        #     sum_iou_grid[mask] += 1

        for c, detections_c in df.groupby('clusters'):
            if c != -1:
                day_counts = detections_c['day_moment'].value_counts().sort_index()
                for k, v in total_times['day_moment'].items():
                    if k in day_counts.index:
                        day_counts.loc[k] = day_counts.loc[k] / v

                season_counts = detections_c['season'].value_counts().sort_index()
                for k, v in total_times['season'].items():
                    if k in season_counts.index:
                        season_counts.loc[k] = season_counts.loc[k] / v

                station_counts = detections_c['station'].value_counts().sort_index()
                for k, v in total_times['station'].items():
                    if k in station_counts.index:
                        station_counts.loc[k] = station_counts.loc[k] / v

                moon_phase_counts = detections_c['moon_phase_type'].value_counts().sort_index()
                for k, v in total_times['moon_phase_type'].items():
                    if k in moon_phase_counts.index:
                        moon_phase_counts.loc[k] = moon_phase_counts.loc[k] / v

                season_counts.index = season_counts.index.map(season_labels)
                moon_phase_counts.index = moon_phase_counts.index.map(moon_phase_labels)

                fig, ax = plt.subplots(2, 2)
                plt.suptitle('Cluster %s' % c)
                day_counts.plot.bar(x='Moment day', y='events/min', ax=ax[0, 0])
                season_counts.plot.bar(x='Season', y='events/min', ax=ax[0, 1])
                station_counts.plot.bar(x='Station', y='events/min', ax=ax[1, 0])
                moon_phase_counts.plot.bar(x='Moon phase', y='events/min', ax=ax[1, 1])
                plt.tight_layout()
                plt.savefig(self.dataset_folder.joinpath('cluster_%s_analysis.png' % c))
                plt.close()


def join_datasets(mother_config, child_config, path_to_new_folder, join_annotations=False):
    """
    Configs should be dictionaries

    :param mother_config:
    :param child_config:
    :param path_to_new_folder:
    :return:
    """
    # Copy the mother folder
    shutil.copytree(mother_config['dataset_folder'], path_to_new_folder)
    child_ds = LifeWatchDataset(child_config)
    shutil.copytree(str(child_ds.labels_folder), pathlib.Path(path_to_new_folder).joinpath('labels'),
                    dirs_exist_ok=True)
    shutil.copytree(str(child_ds.images_folder), pathlib.Path(path_to_new_folder).joinpath('images'),
                    dirs_exist_ok=True)
    if join_annotations:
        mother_annotations = pd.read_table(mother_config['annotations_file'])
        child_annotations = pd.read_table(child_config['annotations_file'])
        output_annotations = pd.concat([mother_annotations, child_annotations])
        output_annotations.to_csv(pathlib.Path(path_to_new_folder).joinpath('joined_annotations.txt'),
                                  sep='\t', index=False)
        return pathlib.Path(path_to_new_folder).joinpath('joined_annotations.txt')


def process_multiple_datasets():
    selection_table_metadata_path = pathlib.Path('//fs/shared/onderzoek/6. Marine Observation Center/Projects/sound_db/'
                                                 'sound_bpns/vliz_labelled_db/selection_tables/'
                                                 'selection_tables_metadata_RS.csv')
    selection_table_metadata = pd.read_csv(selection_table_metadata_path)
    selection_table_metadata = selection_table_metadata.dropna(
        subset=['sound_folder', 'station_name', 'selection_table'])
    # selection_table_metadata = selection_table_metadata.iloc[[10]]
    selection_table_metadata = selection_table_metadata.loc[~selection_table_metadata.sound_folder.str.contains('b&k')]

    for l_i, row in selection_table_metadata.iterrows():
        wavs_path = pathlib.Path(row['sound_folder'])
        selection_table_name = row['selection_table']
        station_name = row['station_name']
        print('working on station %s/%s: %s' % (l_i, len(selection_table_metadata), wavs_path))
        annotations_file = selection_table_metadata_path.parent.joinpath(selection_table_name)


def process_multiple_locations():
    locations = [
        'ElephantIsland2013',
        'ElephantIsland2014',
        'Greenwich2015',
        'kerguelen2005',
        'kerguelen2014',
        'kerguelen2015',
        'MaudRise2014',
        'RossSea2014',
        'casey2014',
        'casey2017',
        'BallenyIslands2015',
    ]
    classes_dict_foreground = {'20Plus': 0, '20Hz': 0, 'A': 0, 'B': 0, 'D': 0, 'Dswp': 0, 'Z': 0}
    classes_dict_single = {'20Plus': 0, '20Hz': 1, 'A': 2, 'B': 3, 'D': 4, 'Dswp': 5, 'Z': 6}
    classes_dict_joined = {'20Plus': 0, '20Hz': 0, 'A': 1, 'B': 1, 'D': 2, 'Dswp': 2, 'Z': 1}

    for l_i, l in enumerate(locations):
        print('working on station %s/%s: %s' % (l_i, len(locations), l))
        annotations_file = root_folder.joinpath('JoinedSelectionTables/new_frequency_limits_20hz', l + '.txt')

        wav_selections = pd.read_table(annotations_file)
        wav_path = root_folder.joinpath('LongWavs', l + '.wav')
        if not wav_path.exists():
            wav_path = root_folder.joinpath('LongWavs', 'untitled_folder', l + '.wav')
