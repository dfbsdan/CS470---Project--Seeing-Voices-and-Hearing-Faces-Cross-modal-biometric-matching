import enum
import torch
from torchvision import transforms
from torchvision.transforms.functional import pad
import torchaudio
from PIL import Image
import csv
import glob
import random


##########################################################################################################NOT FINISHED
transform_test = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.RandomHorizontalFlip(),
  transforms.ColorJitter(),
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
######################################################################################################################

class Dataset_Loader(torch.utils.data.Dataset):
  def __init__(self, dataset: str, metadata: dict):
    if dataset == "train":
      self.transform = transform_train
      initials = "FGHIJKLMNOPQRSTUVWXYZ"
    elif dataset == "valid":
      self.transform = transform_test
      initials = "AB"
    elif dataset == "test":
      self.transform = transform_test
      initials = "CDE"
    else:
      raise RuntimeError("Invalid dataset")

    self.data = list()
    names = list(metadata)
    for iS, name in enumerate(names):
      if name[0] in initials:
        voice_files, face_files = self.__get_files(metadata, name)
        for voice in voice_files:
          for rightF in face_files:
            # randomly select a wrong face
            wrongF, iWS = self.__random_face(names, name, metadata)
            # generate label and store info
            label = random.randint(0,1)
            if label == 0:
              self.data.append((voice, rightF, wrongF, label, iS, iWS))
            else:
              self.data.append((voice, wrongF, rightF, label, iS, iWS))
	
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    v, f1, f2, label, iS, iWS = self.data[idx]    
    v = self.__get_audio(v)
    f1 = self.__get_img(f1)
    f2 = self.__get_img(f2)
    label = torch.tensor(label).long()
    iS = torch.tensor(iS, dtype=torch.int16)
    iWS = torch.tensor(iWS, dtype=torch.int16)
    # NOTE: iS and iWS are two different integers uniquely assigned to the 
    # correct and incorrect speakers, respectively.
    # Both are expected to fit in 16-bits.
    return v, f1, f2, label, iS, iWS

##########################################################################################################NOT FINISHED  
  '''
  16-bit streams at a 16kHz sampling rate for consistency. 
  Spectrograms are then generated in a similar manner to that in [33], giving spectrograms of size
512 ⇥ 300 for three seconds of speech. 

  For the audio segments, we change the speed of each segment by choosing a
random speed ratio between 0.95 to 1.05. We then extract a
random 3s segment from the audio clip at train time. Training uses 1.2M triplets that are selected at random (and the
choice is then fixed). Networks are trained for 10 epochs, or
until validation error stops decreasing, whichever is sooner.'''
  def __get_audio(self, path: str):
    waveform, sample_rate = torchaudio.load(path, normalization=True)
    chs, frames = waveform.shape
    assert chs == 1 and sample_rate == 16000
    return
######################################################################################################################

  def __get_img(self, path: str):
    img = Image.open(path)
    # -dynamically- pad image before appliying other transforms (to preserve proportions)
    w, h = img.size
    max_dim = w if w > h else h
    w_pad = int((max_dim - w) / 2)
    h_pad = int((max_dim - h) / 2)
    img = pad(img, (w_pad, h_pad))
    return self.transform(img)

  def __random_face(self, names: list, name: str, metadata: dict):
    assert len(names) > 1
    f_name = name
    while f_name == name:
      idx = random.randint(0, len(names)-1)
      f_name = names[idx]
    _, f_files = self.__get_files(metadata, f_name)
    return random.choice(f_files), idx

  def __get_files(self, metadata: dict, name: str):
    files = metadata[name]
    if isinstance(files, str): # files not fetched yet
      voice_files = glob.glob("./data/vox1/" + files + "/*/*.wav")
      face_files = glob.glob("./data/faces/" + name + "/*/*/*.jpg")
      assert len(voice_files) > 0 and len(face_files) > 0
      files = (voice_files, face_files)
      metadata[name] = files
    return files


# generates batches for the Dataset_Loader class
def batch_fn(samples: list):
  voices = torch.stack([sample[0] for sample in samples])
  faces1 = torch.stack([sample[1] for sample in samples])
  faces2 = torch.stack([sample[2] for sample in samples])
  labels = torch.stack([sample[3] for sample in samples])
  rSpeakers = torch.stack([sample[4] for sample in samples])
  wSpeakers = torch.stack([sample[5] for sample in samples])
  return voices, faces1, faces2, labels, rSpeakers, wSpeakers


def get_metadata():
  metadata = dict()
  with open('./data/metadata.csv') as f:
    reader = list(csv.reader(f))
    for id, name, _, _ in reader[1:]:
      metadata[name] = id
  return metadata


def load_data(train: bool, batch_sz: int):
  assert batch_sz > 0
  metadata = get_metadata()
  if train:
    trainset = Dataset_Loader("train", metadata)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_sz, 
      shuffle=True, num_workers=2, collate_fn=batch_fn)
    validset = Dataset_Loader("valid", metadata)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_sz, 
      shuffle=False, num_workers=1, collate_fn=batch_fn)
    return train_loader, valid_loader
  else:
    testset = Dataset_Loader("test", metadata)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_sz, 
      shuffle=False, num_workers=1, collate_fn=batch_fn)
    return test_loader