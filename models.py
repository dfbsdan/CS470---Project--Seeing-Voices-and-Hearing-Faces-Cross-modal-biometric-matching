import torch
import torch.nn as nn


# Layer constructor helpers

def new_shape(old_shape: int, k_size: int, stride: int, padding: int):
  assert old_shape > 0 and k_size > 0 and stride > 0 and padding >= 0
  shape = int(((old_shape + 2*padding - k_size) / stride) + 1)
  assert shape > 0
  return shape

def conv(shape: tuple, in_chs: int, out_chs: int, k_size: tuple, stride: tuple, 
          padding: int):
  assert (len(shape) == 2 and in_chs > 0 and out_chs > 0 and len(k_size) == 2
    and len(stride) == 2 and padding >= 0)
  w = new_shape(shape[0], k_size[0], stride[0], padding)
  h = new_shape(shape[1], k_size[1], stride[1], padding)
  layer = nn.Conv2d(in_chs, out_chs, k_size, stride, padding)
  return (layer, out_chs, (w, h))

def maxpool2d(shape: tuple, k_size: tuple, stride: tuple, padding: int):
  assert len(shape) == 2 and len(k_size) == 2 and len(stride) == 2 and padding >= 0
  w = new_shape(shape[0], k_size[0], stride[0], padding)
  h = new_shape(shape[1], k_size[1], stride[1], padding)
  layer = nn.MaxPool2d(k_size, stride, padding)
  return (layer, (w, h))

def linear(in_features: int, out_features: int):
  assert in_features > 0 and out_features > 0
  return (nn.Linear(in_features, out_features), out_features)


# MODELS

# baseline model
class SVHF_Net(nn.Module):
  def __init__(self, load_path: str = None):
    super(SVHF_Net, self).__init__()
    
    self.voice = SVHF_Voice()
    self.face = SVHF_Face()
    l1, features = linear(3*1024, 1024)
    l2, features = linear(features, 512)
    l3, _ = linear(features, 2)
    self.last = nn.Sequential(l1, nn.ReLU(), l2, nn.ReLU(), l3, nn.LogSoftmax(dim=1))

    if isinstance(load_path, str):
      status = torch.load(load_path)
      self.load_state_dict(status)
  
  def forward(self, v, f1, f2):
    v = self.voice(v)
    f1 = self.face(f1)
    f2 = self.face(f2)
    out = torch.cat((v, f1, f2), 1)
    return self.last(out)


# voice sublayer of the SVHF NN
class SVHF_Voice(nn.Module):
  def __init__(self):
    super(SVHF_Voice, self).__init__()
    shape = (512, 300)
    channels = 1
    l1, channels, shape = conv(shape, channels, 96, (7,7), (2,2), 1)
    a1 = nn.ReLU()
    b1 = nn.BatchNorm2d(channels)
    p1, shape = maxpool2d(shape, (3,3), (2,2), 0)
    l2, channels, shape = conv(shape, channels, 256, (5,5), (2,2), 1)
    a2 = nn.ReLU()
    b2 = nn.BatchNorm2d(channels)
    p2, shape = maxpool2d(shape, (3,3), (2,2), 0)
    l3, channels, shape = conv(shape, channels, 256, (3,3), (1,1), 1)
    a3 = nn.ReLU()
    l4, channels, shape = conv(shape, channels, 256, (3,3), (1,1), 1)
    a4 = nn.ReLU()
    l5, channels, shape = conv(shape, channels, 256, (3,3), (1,1), 1)
    a5 = nn.ReLU()
    p5, shape = maxpool2d(shape, (5,3), (3,2), 0)
    self.seq1 = nn.Sequential(l1, a1, b1, p1, l2, a2, b2, p2, l3, a3, l4, a4, l5, a5, p5)

    features = 9*256
    l1, features = linear(features, 4096)
    a1 = nn.ReLU()
    self.seq2 = nn.Sequential(l1, a1)

    l1 = nn.AvgPool1d(8, 8, 0)
    fl = nn.Flatten()
    l2, _ = linear(features, 1024)
    a2 = nn.ReLU()
    self.seq3 = nn.Sequential(l1, fl, l2, a2)

  def forward(self, x):
    x = self.seq1(x).permute(0, 3, 1, 2).reshape((x.shape[0], 8, 9*256))
    x = self.seq2(x).permute(0, 2, 1)
    return self.seq3(x)


# face sublayer of the SVHF NN
class SVHF_Face(nn.Module):
  def __init__(self):
    super(SVHF_Face, self).__init__()
    shape = (224, 224)
    channels = 3
    l1, channels, shape = conv(shape, channels, 96, (7,7), (2,2), 1)
    a1 = nn.ReLU()
    b1 = nn.BatchNorm2d(channels)
    p1, shape = maxpool2d(shape, (3,3), (2,2), 0)
    l2, channels, shape = conv(shape, channels, 256, (5,5), (2,2), 1)
    a2 = nn.ReLU()
    b2 = nn.BatchNorm2d(channels)
    p2, shape = maxpool2d(shape, (3,3), (2,2), 0)
    l3, channels, shape = conv(shape, channels, 256, (3,3), (1,1), 1)
    a3 = nn.ReLU()
    l4, channels, shape = conv(shape, channels, 256, (3,3), (1,1), 1)
    a4 = nn.ReLU()
    l5, channels, shape = conv(shape, channels, 256, (3,3), (1,1), 1)
    a5 = nn.ReLU()
    p5, shape = maxpool2d(shape, (3,3), (2,2), 0)
    features = channels * shape[0] * shape[1]
    fl = nn.Flatten()
    l6, features = linear(features, 4096)
    a6 = nn.ReLU()
    l7, features = linear(features, 1024)
    a7 = nn.ReLU()
    self.seq = nn.Sequential(l1, a1, b1, p1, l2, a2, b2, p2, l3, a3, l4, a4, l5, 
      a5, p5, fl, l6, a6, l7, a7)

  def forward(self, x):
    return self.seq(x)


# Random NN used for testing
class RandomNet(nn.Module):
  def __init__(self, _):
    super(RandomNet, self).__init__()
    self.device = torch.device("cuda")
    self.linear, _ = linear(1, 1)
    self.lsm = nn.LogSoftmax(dim=1)
  
  def forward(self, v, f1, f2):
    x = torch.rand((len(v), 2), device=self.device, requires_grad=True)
    return self.lsm(x)