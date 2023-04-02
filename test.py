import torch
import torchvision
import torchvision.transforms as transforms
from model.squeezenet import SqueezeNet
from torchvision import models
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform1 = transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])

test_dataset = torchvision.datasets.ImageFolder(root='./data/test/',
                                            transform=transform1,
                                            )

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=32,
                                           shuffle=False,
                                           )
###Call the model and the trained ckpt file
model_path = "checkpoints/model_14.ckpt"
model=models.squeezenet1_0(pretrained=True)
model.load_state_dict(torch.load("squeezenet1_0-a815701f.pth"))
model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=100,
                                kernel_size=1) # #Change the output of the last layer of the network to 20 categories
model.num_classes = 100 #Change the number of classification categories of the network

model.load_state_dict(torch.load(model_path)) # read ckpt file 

model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))