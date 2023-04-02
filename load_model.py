import torch
from torchvision import models



model=torch.load("squeezenet1_0-a815701f.pth") #load pre_train model
print(model)