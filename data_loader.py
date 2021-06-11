import torch
import torchvision.transforms as VT
from torchvision.transforms.functional import pad
import torchaudio
import torchaudio.transforms as AT
from PIL import Image
from tqdm import tqdm
import csv
import glob
import random


# transforms applied to a test image
transform_test = VT.Compose([
  VT.Resize((224, 224)),
  VT.ToTensor(),
  VT.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# transforms applied to a train image
transform_train = VT.Compose([
  VT.Resize((224, 224)),
  VT.RandomHorizontalFlip(),
  VT.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
  VT.ToTensor(),
  VT.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class Dataset_Loader(torch.utils.data.Dataset):
  def __init__(self, dataset: str, metadata: dict, device: torch.device, data_path: str):
    if dataset == "train":
      self.transform = transform_train
      initials = "FGHIJKLMNOPQRSTUVWXYZ"
      samples = 1200000
    elif dataset == "valid":
      self.transform = transform_test
      initials = "AB"
      samples = 10000
    elif dataset == "test":
      self.transform = transform_test
      initials = "CDE"
      samples = 10000
    else:
      raise RuntimeError("Invalid dataset")

    self.data_path = data_path
    # get samples
    names = list(metadata)
    valid_names = [name for name in names if name[0] in initials]
    chosen = dict()
    print(f"Generating {dataset} dataset of size: {samples}...")
    for _ in tqdm(range(samples)):
      # get random -unique- sample
      attempts = 0
      label = random.randint(0,1)
      while True:
        iS = random.randint(0, len(valid_names) - 1)
        name = valid_names[iS]
        key, iWS = self.__random_sample(metadata, names, name)
        if not key in chosen:
          break
        attempts += 1
        if attempts >= 100:
          raise RuntimeError("Exceeded max attempts to obtain a sample")
      chosen[key] = (label, iS, iWS)
    # store info and finish
    self.data = list()
    for (voice, rightF, wrongF), (label, iS, iWS) in chosen.items():
      if label == 0:
        self.data.append((voice, rightF, wrongF, label, iS, iWS))
      else:
        self.data.append((voice, wrongF, rightF, label, iS, iWS))
    self.device = device

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    v, f1, f2, label, iS, iWS = self.data[idx]
    v = self.__get_audio(v)
    f1 = self.__get_img(f1)
    f2 = self.__get_img(f2)
    label = torch.tensor(label, dtype=torch.long)
    # NOTE: iS and iWS are two different integers uniquely assigned to the 
    # correct and incorrect speakers, respectively.
    # Both are expected to fit in 16-bits.
    iS = torch.tensor(iS, dtype=torch.int16)
    iWS = torch.tensor(iWS, dtype=torch.int16)
    return v, f1, f2, label, iS, iWS

  def __random_sample(self, metadata: dict, names: list, name: str):
    v_files, f_files = self.__get_files(metadata, name)
    voice = random.choice(v_files)
    rightF = random.choice(f_files)
    wrongF, iWS = self.__random_face(names, name, metadata)
    return (voice, rightF, wrongF), iWS

  def __random_face(self, names: list, name: str, metadata: dict):
    assert len(names) > 1
    f_name = name
    while f_name == name:
      idx = random.randint(0, len(names)-1)
      f_name = names[idx]
    _, f_files = self.__get_files(metadata, f_name)
    return random.choice(f_files), idx

  def __get_files(self, metadata: dict, name: str):
    val = metadata[name]
    if isinstance(val, str): # files not fetched yet
      voice_files = glob.glob(f"{self.data_path}/vox1/{val}/*/*.wav")
      face_files = glob.glob(f"{self.data_path}/faces/{name}/*/*/*.jpg")
      assert len(voice_files) > 0 and len(face_files) > 0
      val = (voice_files, face_files)
      metadata[name] = val
    return val

  def __get_audio(self, path: str):
    wave, rate = torchaudio.load(path, normalize=True) 
    channels, _ = wave.shape
    assert channels == 1 and rate == 16000
    # randomize rate to range: [0.95*rate, 1.05*rate]
    # (tensor is moved to device at this point to speed up preprocessing)
    rate = int(rate * random.uniform(0.95, 1.05))
    wave = AT.Resample(orig_freq=16000, new_freq=rate)(wave.to(self.device))
    channels, samples = wave.shape
    # extract random 3-second segment and convert to spectrogram
    extr_len = 3 * rate
    assert channels == 1 and samples >= extr_len
    transform = VT.Compose([
      AT.Spectrogram(
        n_fft=1022,
        win_length=round(25*rate/1000), # 25ms window
        hop_length=round(rate/100), # 10ms hop
        wkwargs={"device": self.device},
        center=True,
        pad_mode="reflect",
      ),
      VT.Resize((512, 300)),
    ])
    offset = random.randint(0, samples - extr_len)
    return transform(wave[:, offset : offset + extr_len])

  def __get_img(self, path: str):
    img = Image.open(path)
    # -dynamically- pad image before appliying other transforms (to preserve proportions)
    w, h = img.size
    max_dim = w if w > h else h
    w_pad = int((max_dim - w) / 2)
    h_pad = int((max_dim - h) / 2)
    img = pad(img, (w_pad, h_pad))
    return self.transform(img)


# generates batches for the Dataset_Loader class
def batch_fn(samples: list):
  voices = torch.stack([sample[0] for sample in samples])
  faces1 = torch.stack([sample[1] for sample in samples])
  faces2 = torch.stack([sample[2] for sample in samples])
  labels = torch.stack([sample[3] for sample in samples])
  rSpeakers = torch.stack([sample[4] for sample in samples])
  wSpeakers = torch.stack([sample[5] for sample in samples])
  return voices, faces1, faces2, labels, rSpeakers, wSpeakers


def get_metadata(data_path: str):
  metadata = dict()
  with open(data_path + '/metadata.csv') as f:
    reader = list(csv.reader(f))
    for id, name, _, _ in reader[1:]:
      metadata[name] = id
  return metadata


def load_data(train: bool, batch_sz: int, device: torch.device, data_path: str):
  assert batch_sz > 0
  metadata = get_metadata(data_path)
  if train:
    trainset = Dataset_Loader("train", metadata, device, data_path)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_sz, 
      shuffle=True, num_workers=0, collate_fn=batch_fn)
    validset = Dataset_Loader("valid", metadata, device, data_path)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_sz, 
      shuffle=False, num_workers=0, collate_fn=batch_fn)
    return train_loader, valid_loader
  else:
    testset = Dataset_Loader("test", metadata, device, data_path)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_sz, 
      shuffle=False, num_workers=0, collate_fn=batch_fn)
    return test_loader