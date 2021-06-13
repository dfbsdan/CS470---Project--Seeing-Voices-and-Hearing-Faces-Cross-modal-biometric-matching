import torch
import torch.nn as nn
from collections import OrderedDict


# MODELS

# baseline model
class SVHF_Net(nn.Module):
  def __init__(self, load_path: str = None):
    super(SVHF_Net, self).__init__()
    
    self.voice = SVHF_Voice()
    self.face = SVHF_Face()
    self.last=nn.Sequential(OrderedDict([
      ('fc1', nn.Linear(3*1024, 1024)),
      ('relu1', nn.ReLU()),
      ('fc2', nn.Linear(1024, 512)),
      ('relu2', nn.ReLU()),
      ('fc3', nn.Linear(512, 2)),
      ('lsm3', nn.LogSoftmax(dim=1)),
    ]))

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
    self.features = nn.Sequential(OrderedDict([
      ('conv1', nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(7,7), stride=(2,2), padding=1)),
      ('bn1', nn.BatchNorm2d(96, momentum=0.9)),
      ('relu1', nn.ReLU()),
      ('mpool1', nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))),
      ('conv2', nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), stride=(2,2), padding=1)),
      ('bn2', nn.BatchNorm2d(256, momentum=0.9)),
      ('relu2', nn.ReLU()),
      ('mpool2', nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))),
      ('conv3', nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), stride=(1,1), padding=1)),
      ('bn3', nn.BatchNorm2d(384, momentum=0.9)),
      ('relu3', nn.ReLU()),
      ('conv4', nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)),
      ('bn4', nn.BatchNorm2d(256, momentum=0.9)),
      ('relu4', nn.ReLU()),
      ('conv5', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)),
      ('bn5', nn.BatchNorm2d(256, momentum=0.9)),
      ('relu5', nn.ReLU()),
      ('mpool5', nn.MaxPool2d(kernel_size=(5,3), stride=(3,2))),
      ('fc6', nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(9,1), stride=(1,1))),
      ('bn6', nn.BatchNorm2d(4096, momentum=0.9)),
      ('relu6', nn.ReLU()),
      ('apool6', nn.AdaptiveAvgPool2d((1,1))),
      ('flatten', nn.Flatten())
    ]))
            
    self.classifier=nn.Sequential(OrderedDict([
      ('fc7', nn.Linear(4096, 1024)),
      ('relu7', nn.ReLU())
    ]))
    
  def forward(self, x):
    x = self.features(x)
    return self.classifier(x)


# face sublayer of the SVHF NN
class SVHF_Face(nn.Module):
  def __init__(self):
    super(SVHF_Face, self).__init__()
    self.features = nn.Sequential(OrderedDict([
      ("conv1", nn.Conv2d(3, 96, kernel_size=[7, 7], stride=(2, 2))), 
      ("bn1", nn.BatchNorm2d(96, momentum=0.9)), 
      ("relu1", nn.ReLU()), 
      ("mpool1", nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, ceil_mode=False)), 
      ("conv2", nn.Conv2d(96, 256, kernel_size=[5, 5], stride=(2, 2), padding=(1, 1))), 
      ("bn2", nn.BatchNorm2d(256, momentum=0.9)),
      ("relu2", nn.ReLU()), 
      ("mpool2", nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=(0, 0), ceil_mode=True)), 
      ("conv3", nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))), 
      ("bn3", nn.BatchNorm2d(512, momentum=0.9)), 
      ("relu3", nn.ReLU()), 
      ("conv4", nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))),
      ("bn4", nn.BatchNorm2d(512, momentum=0.9)), 
      ("relu4", nn.ReLU()), 
      ("conv5", nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))), 
      ("bn5", nn.BatchNorm2d(512, momentum=0.9)), 
      ("relu5", nn.ReLU()), 
      ("mpool5", nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, ceil_mode=False)),
      ("fc6", nn.Conv2d(512, 4096, kernel_size=[6, 6], stride=(1, 1))), 
      ("bn6", nn.BatchNorm2d(4096, momentum=0.9)), 
      ("relu6", nn.ReLU()), 
      ("fc7", nn.Conv2d(4096, 4096, kernel_size=[1, 1], stride=(1, 1))), 
      ("bn7", nn.BatchNorm2d(4096, momentum=0.9)), 
      ("relu7", nn.ReLU())
    ]))

    self.classifier=nn.Sequential(OrderedDict([
      ('fc7', nn.Linear(4096, 1024)),
      ('relu7', nn.ReLU())
    ]))

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    return self.classifier(x)
