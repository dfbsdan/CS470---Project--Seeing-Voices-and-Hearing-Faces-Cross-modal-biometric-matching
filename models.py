import torch
import torch.nn as nn


# Layer constructor helpers

def new_shape(old_shape: int, k_size: int, stride: int, padding: int):
  assert old_shape > 0 and k_size > 0 and stride > 0 and padding >= 0
  shape = int(((old_shape + 2*padding - k_size) / stride) + 1)
  assert shape > 0
  return shape

def conv(shape: tuple, in_chs: int, out_chs: int, k_size: tuple, stride: int, 
          padding: int):
  assert (len(shape) == 2 and in_chs > 0 and out_chs > 0 and len(k_size) == 2
    and stride > 0 and padding >= 0)
  w = new_shape(shape[0], k_size[0], stride, padding)
  h = new_shape(shape[1], k_size[1], stride, padding)
  layer = nn.Conv2d(in_chs, out_chs, k_size, stride, padding)
  return (layer, out_chs, (w, h))

def maxpool(shape: tuple, k_size: tuple, stride: int, padding: int):
  assert len(shape) == 2 and len(k_size) == 2 and stride > 0 and padding >= 0
  w = new_shape(shape[0], k_size[0], stride, padding)
  h = new_shape(shape[1], k_size[1], stride, padding)
  layer = nn.MaxPool2d(k_size, stride, padding)
  return (layer, (w, h))

def avgpool(shape: tuple, k_size: tuple, stride: int, padding: int):
  assert len(shape) == 2 and len(k_size) == 2 and stride > 0 and padding >= 0
  w = new_shape(shape[0], k_size[0], stride, padding)
  h = new_shape(shape[1], k_size[1], stride, padding)
  layer = nn.AvgPool2d(k_size, stride, padding)
  return (layer, (w, h))

def linear(in_features: int, out_features: int):
  assert in_features > 0 and out_features > 0
  return (nn.Linear(in_features, out_features), out_features)



# MODELS

# baseline model
class SVHF_Net(nn.Module):
  def __init__(self, load_file: str = None):
    super(SVHF_Net, self).__init__()
    
    ############################################################################Create sub-models as different classes?
    self.voice = self.__createLayer(True)
    self.face1 = self.__createLayer(False)
    self.face2 = self.__createLayer(False)######################################Same as face1?
    features = 3*1024
    l1, features = linear(features, 1024)
    l2, features = linear(features, 512)
    l3, _ = linear(features, 2)
    self.last = nn.Sequential(l1, nn.ReLU(), l2, nn.ReLU(), l3, nn.Softmax())

    if isinstance(load_file, str):
      status = torch.load("saved_models/" + load_file)
      self.load_state_dict(status)
  
  def forward(self, v, f1, f2):
    v = self.voice(v)
    f1 = self.face1(f1)
    f2 = self.face2(f2)
    out = torch.cat((v, f1, f2))
    return self.last(out)

  def __createLayer(self, voiceLayer: bool):
    if voiceLayer:
      shape = (512, 300)
      channels = 1
    else: # face layer
      shape = (224, 224)
      channels = 3
    l1, channels, shape = conv(shape, channels, 96, (7,7), 2, 0)
    a1 = nn.ReLU()
    p1, shape = maxpool(shape, (2,2), 1, 0)
    l2, channels, shape = conv(shape, channels, 256, (5,5), 2, 0)
    a2 = nn.ReLU()
    p2, shape = maxpool(shape, (2,2), 1, 0)
    l3, channels, shape = conv(shape, channels, 256, (3,3), 1, 0)
    a3 = nn.ReLU()
    l4, channels, shape = conv(shape, channels, 256, (3,3), 1, 0)
    a4 = nn.ReLU()
    l5, channels, shape = conv(shape, channels, 256, (3,3), 1, 0)
    a5 = nn.ReLU()
    p5, shape = maxpool(shape, (2,2), 0, 0)
    layers = [l1, a1, p1, l2, a2, p2, l3, a3, l4, a4, l5, a5, p5]
    features = channels * shape[0] * shape[1]
    if voiceLayer:
##########################################################################################################NOT FINISHED
      l6, features = linear(features, )
      a6 = nn.ReLU()
      p6, shape = avgpool()
######################################################################################################################
      layers += [l6, a6, p6]
    else:
      l6, features = linear(features, 4096)
      a6 = nn.ReLU()
      layers += [l6, a6]
    l7, features = linear(features, 1024)
    a7 = nn.ReLU()
    layers += [l7, a7]
    return nn.Sequential(layers)