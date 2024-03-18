import soundfile as sf
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, dataloader
import torchaudio
import torch
import numpy as np
from scipy.signal import resample
import torchvision.transforms.functional as F
import torch.nn.functional as F_general
import scipy
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
# from sliceguard.embeddings import generate_image_embeddings


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return dataloader.default_collate(batch)


class Dataset(Dataset):
    def __init__(self, df, audiopath, sr, sampleDur, channel=0):
        super(Dataset, self)
        self.audiopath, self.df, self.sr, self.sampleDur, self.channel = audiopath, df, sr, sampleDur, channel
        self.file_list = os.listdir(audiopath)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig = self.read_snippet(row)
        if len(sig) < self.sampleDur * self.sr:
            sig = np.concatenate([sig, np.zeros(int(self.sampleDur * self.fs) - len(sig))])

        return Tensor(norm(sig)).float(), row.name

    def _get_duration(self, row):
        return self.sampleDur

    def read_snippet(self, row):
        info = sf.info(self.audiopath + '/' + row.filename)
        dur, fs = info.duration, info.samplerate
        sample_dur = self._get_duration(row)
        start = int(np.clip(row.pos - sample_dur / 2, 0, max(0, dur - sample_dur)) * fs)
        if row.two_files:
            stop = info.frames
            extra_dur = sample_dur - (info.frames - start) / fs
        else:
            stop = start + int(sample_dur * fs)
        try:
            sig, fs = sf.read(self.audiopath + '/' + row.filename, start=start, stop=stop, always_2d=True)
            if row.two_files:
                second_file = self.file_list[self.file_list.index(row.filename) + 1]
                stop2 = int(extra_dur * fs)
                sig2, fs2 = sf.read(self.audiopath + '/' + second_file, start=0, stop=stop2, always_2d=True)
                sig = np.concatenate([sig, sig2])
            sig = sig[:, self.channel]
        except Exception as e:
            print(f'Failed to load sound from row {row.name} with filename {row.filename}', e)

        if fs != self.sr:
            sig = resample(sig, int(len(sig)/fs*self.sr))
        return sig


class DatasetCropsDuration(Dataset):
    def __init__(self, df, audiopath, sr, sampleDur, winsize, win_overlap, n_mel, channel=0):
        super(Dataset, self)
        self.audiopath, self.df, self.sr, self.channel = audiopath, df, sr, channel
        self.winsize = winsize
        self.win_overlap = win_overlap
        self.n_mel = n_mel
        # self.norm = nn.InstanceNorm2d(1)
        self.file_list = os.listdir(audiopath)
        self.sampleDur = sampleDur

    def _get_duration(self, row):
        return row.duration + 0.2

    def get_spectrogram(self, sig):
        hopsize = int((len(sig) - self.winsize) / 128)
        f, t, sxx = scipy.signal.spectrogram(sig, fs=self.sr, window=('hamming'),
                                             nperseg=self.winsize,
                                             noverlap=self.winsize - hopsize, nfft=self.winsize,
                                             detrend=False,
                                             return_onesided=True, scaling='density', axis=-1,
                                             mode='magnitude')
        return f, t, sxx

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig = self.read_snippet(row)
        f, t, sxx = self.get_spectrogram(sig)
        sxx = sxx[:, :self.n_mel]
        sxx = Tensor(sxx).float()
        return sxx.unsqueeze(0), row.name


class DatasetCrops(DatasetCropsDuration):
    def __init__(self, df, audiopath, sr, sampleDur, winsize, win_overlap, n_mel, channel=0):
        super(Dataset, self)
        self.audiopath, self.df, self.sr, self.channel = audiopath, df, sr, channel
        self.winsize = winsize
        self.win_overlap = win_overlap
        self.n_mel = n_mel
        # self.norm = nn.InstanceNorm2d(1)
        self.file_list = os.listdir(audiopath)
        self.sampleDur = sampleDur

    def get_spectrogram(self, sig, row):
        winsize = min(int(len(sig)/2), int(128 * row.max_freq / (row.max_freq - row.min_freq)) * 2)
        hopsize = min(int((len(sig) - self.winsize) / 128), int(winsize/2))
        f, t, sxx = scipy.signal.spectrogram(sig, fs=self.sr, window=('hamming'),
                                             nperseg=winsize,
                                             noverlap=winsize - hopsize, nfft=winsize,
                                             detrend=False,
                                             return_onesided=True, scaling='density', axis=-1,
                                             mode='magnitude')

        return f, t, sxx

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig = self.read_snippet(row)
        f, t, sxx = self.get_spectrogram(sig, row)

        sxx = Tensor(sxx).float()
        max_freq = min(int(row.max_freq / (self.sr / 2) * sxx.shape[0]) + 1, sxx.shape[0] - 1)
        min_freq = max(0, int(row.min_freq / (self.sr / 2) * sxx.shape[0]) - 1)

        # min_dur = max(int(((self.sampleDur / 2) - (row.duration / 2 + 0.2)) / self.sampleDur * sxx.shape[1]) - 1, 0)
        # max_dur = min(int(((self.sampleDur / 2) + (row.duration / 2) - 0.2) / self.sampleDur * sxx.shape[1]) + 1, sxx.shape[1] - 1)
        sxx_cropped = sxx[min_freq: max_freq, :]  # min_dur:max_dur

        # sxx_mel = sxx_cropped - torch.quantile(sxx_cropped, 0.2, dim=-1, keepdim=True)[0]

        # plt.imshow(sxx_mel, origin='lower')
        # plt.axis('off')
        # plt.savefig(
        #     '/mnt/fscompute_shared/roi/datasets/bpns/stratified_test_set/crops_ae/%s.png' % row.name)
        # plt.close()

        # plt.pcolormesh(t[min_dur:max_dur], f[min_freq:max_freq], sxx_out.numpy()[0], cmap='jet', shading='nearest')
        # plt.savefig('/mnt/fscompute_shared/roi/datasets/bpns/stratified_test_set/predictions/crops_ae/%s.png' % row.name)

        sxx_out = F.resize(sxx_cropped.unsqueeze(0), (128, 128))

        return sxx_out, row.name


class DatasetWaveform(DataLoader):
    def __init__(self, df, wavs_folder, desired_fs, max_duration, channel=0):
        self.file_list = os.listdir(wavs_folder)
        self.df = df.copy()
        self.wavs_folder = wavs_folder
        self.desired_fs = desired_fs
        self.channel = channel
        self.max_duration = max_duration
        #self.label_to_id = {lbl: i for i, lbl in enumerate(df['label'].unique())}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row['wav']
        waveform_info = torchaudio.info(wav_path)

        # If the selection is in between two files, open both and concatenate them
        if row['begin_sample'] > row['end_sample']:
            waveform1, fs = torchaudio.load(wav_path,
                                            frame_offset=row['begin_sample'],
                                            num_frames=waveform_info.num_frames - row[
                                                'begin_sample'])

            wav_path2 = self.wavs_folder.joinpath(self.file_list[self.file_list.index(row['filename']) + 1])
            waveform2, fs = torchaudio.load(wav_path2,
                                            frame_offset=0,
                                            num_frames=row['end_sample'])
            waveform = torch.cat([waveform1, waveform2], -1)
        else:
            waveform, fs = torchaudio.load(wav_path,
                                           frame_offset=row['begin_sample'],
                                           num_frames=row['end_sample'] - row[
                                               'begin_sample'])
        if waveform_info.sample_rate != self.desired_fs:
            transform = torchaudio.transforms.Resample(fs, self.desired_fs)
            waveform = transform(waveform)
        else:
            waveform = waveform

        max_samples = self.max_duration * self.desired_fs
        waveform = waveform[self.channel, :max_samples]
        if waveform.shape[0] < max_samples:
            waveform = F_general.pad(waveform, (0, max_samples - waveform.shape[0]))

        if (row['min_freq'] > 10) and (row['max_freq'] < (self.desired_fs / 2)):
            sos = scipy.signal.iirfilter(4, [row['min_freq'],
                                             row['max_freq']],
                                         rp=None, rs=None, btype='band',
                                         analog=False, ftype='butter', output='sos',
                                         fs=self.desired_fs)
            waveform = scipy.signal.sosfilt(sos, waveform)

        elif (row['min_freq'] == 0) and (row['max_freq'] < (self.desired_fs / 2)):
            sos = scipy.signal.iirfilter(4, row['max_freq'],
                                         rp=None, rs=None, btype='lowpass',
                                         analog=False, ftype='butter', output='sos',
                                         fs=self.desired_fs)
            waveform = scipy.signal.sosfilt(sos, waveform)
        elif (row['max_freq'] == (self.desired_fs / 2)) and (row['min_freq'] > 0):
            sos = scipy.signal.iirfilter(4, row['min_freq'],
                                         rp=None, rs=None, btype='highpass',
                                         analog=False, ftype='butter', output='sos',
                                         fs=self.desired_fs)
            waveform = scipy.signal.sosfilt(sos, waveform)

        else:
            waveform = waveform

        return torch.Tensor(waveform), torch.tensor(row.name)


def norm(arr):
    return (arr - np.mean(arr)) / np.std(arr)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


class Croper2D(nn.Module):
    def __init__(self, *shape):
        super(Croper2D, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x[:,:,:self.shape[0],(x.shape[-1] - self.shape[1])//2:-(x.shape[-1] - self.shape[1])//2]


class Accuracy:
    def __init__(self):
        self.num_total = 0
        self.num_correct = 0

    def update(self, logits, y):
        self.num_total += logits.shape[0]
        self.num_correct += torch.sum(logits.argmax(axis=1) == y).cpu().item()

    def get_metric(self):
        return {'acc': 0. if self.num_total == 0 else self.num_correct / self.num_total}

    def get_primary_metric(self):
        return self.get_metric()['acc']


def eval_pytorch_model(model, dataloader, metric_factory, device, desc):
    model.eval()
    total_loss = 0.
    steps = 0
    metric = metric_factory()
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=desc):
            x = x.to(device)
            y = y.to(device)

            loss, logits = model(x, y)
            total_loss += loss.cpu().item()
            steps += 1

            metric.update(logits.to("cpu"), y.to("cpu"))

    total_loss /= steps

    return total_loss, metric.get_primary_metric()

