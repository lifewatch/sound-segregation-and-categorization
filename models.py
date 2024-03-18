import torchvision.models as torchmodels
from torch import nn
import utils as u
from filterbank import STFT, MelFilter, Log1p, MedFilt
import torch
from transformers import AutoProcessor, ClapModel, ClapAudioModelWithProjection, ClapProcessor

torch.hub.set_dir('/data/cleap/torch/')

vgg16 = torchmodels.vgg16(weights=torchmodels.VGG16_Weights.DEFAULT)
vgg16 = vgg16.features[:13]
for nm, mod in vgg16.named_modules():
    if isinstance(mod, nn.MaxPool2d):
        setattr(vgg16, nm,  nn.AvgPool2d(2, 2))


frontend = lambda sr, nfft, sampleDur, n_mel : nn.Sequential(
    STFT(nfft, int((sampleDur*sr - nfft)/128)),
    MelFilter(sr, nfft, n_mel, 0, sr//2),
    Log1p(7, trainable=False),
    nn.InstanceNorm2d(1),
    u.Croper2D(n_mel, 128)
  )


frontend_medfilt = lambda sr, nfft, sampleDur, n_mel: nn.Sequential(
  STFT(nfft, int((sampleDur*sr - nfft)/128)),
  MelFilter(sr, nfft, n_mel, sr//nfft, sr//2),
  Log1p(7, trainable=False),
  nn.InstanceNorm2d(1),
  MedFilt(),
  u.Croper2D(n_mel, 128)
)


frontend_crop = lambda: nn.Sequential(
  Log1p(7, trainable=False),
  nn.InstanceNorm2d(1)
)

frontend_crop_duration = lambda sr, nfft, sampleDur, n_mel : nn.Sequential(
    MelFilter(sr, nfft, n_mel, 0, sr//2),
    Log1p(7, trainable=False),
    nn.InstanceNorm2d(1)
)


sparrow_encoder = lambda nfeat, shape : nn.Sequential(
  nn.Conv2d(1, 32, 3, stride=2, bias=False, padding=(1)),
  nn.BatchNorm2d(32),
  nn.ReLU(True),
  nn.Conv2d(32, 64, 3, stride=2, bias=False, padding=1),
  nn.BatchNorm2d(64),
  nn.ReLU(True),
  nn.Conv2d(64, 128, 3, stride=2, bias=False, padding=1),
  nn.BatchNorm2d(128),
  nn.ReLU(True),
  nn.Conv2d(128, 256, 3, stride=2, bias=False, padding=1),
  nn.BatchNorm2d(256),
  nn.ReLU(True),
  nn.Conv2d(256, nfeat, 3, stride=2, padding=1),
  u.Reshape(nfeat * shape[0] * shape[1])
)

sparrow_decoder = lambda nfeat, shape : nn.Sequential(
  u.Reshape(nfeat//(shape[0]*shape[1]), *shape),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(nfeat//(shape[0]*shape[1]), 256, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(256),
  nn.ReLU(True),
  nn.Conv2d(256, 256, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(256),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(256, 128, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(128),
  nn.ReLU(True),
  nn.Conv2d(128, 128, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(128),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(128, 64, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(64),
  nn.ReLU(True),
  nn.Conv2d(64, 64, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(64),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(64, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),
  nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(32, 1, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(1),
  nn.ReLU(True),
  nn.Conv2d(1, 1, (3, 3), bias=False, padding=1),
)


class CLAPClassifier(nn.Module):
    def __init__(self, model_path, num_classes, sr, device, multi_label=False) -> None:
        super().__init__()
        self.clap = ClapAudioModelWithProjection.from_pretrained(model_path)
        self.clap.eval()
        print("fixed clap.")

        self.linear = nn.Linear(in_features=512, out_features=16)
        self.linear2 = nn.Linear(in_features=16, out_features=num_classes)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.multi_label = multi_label

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
        self.device = device
        self.sr = sr

    def forward(self, x, y=None):
        x = [s.cpu().numpy() for s in x]
        inputs = self.processor(audios=x, return_tensors="pt", sampling_rate=self.sr, padding=True).to(self.device)
        out = self.clap(**inputs).audio_embeds
        out = self.linear(out)
        out = self.linear2(out)
        loss = self.loss_func(out, y)
        return loss, out


class CLAPZeroShotClassifier(nn.Module):
    def __init__(self, model_path, labels, sr, device, multi_label=False) -> None:
        super().__init__()
        print("model!", model_path)
        self.clap = ClapModel.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.loss_func = nn.CrossEntropyLoss()
        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        self.labels = labels
        print("labels", self.labels)
        self.multi_label = multi_label
        self.device = device
        self.sr = sr

    def forward(self, x, y=None):
        x = [s.cpu().numpy() for s in x]
        inputs = self.processor(audios=x, text=self.labels, return_tensors="pt", sampling_rate=self.sr, padding=True).to(
          self.device)
        out = self.clap(**inputs).logits_per_audio
        loss = self.loss_func(out, y)
        return loss, out
