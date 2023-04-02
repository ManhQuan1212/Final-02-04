import torch
import torchvision.transforms as transforms
from model.squeezenet import SqueezeNet
import numpy as np
from PIL import Image
from torchvision import models
from PIL import Image
import torch.nn as nn

# Đọc ảnh đầu vào và áp dụng các phép biến đổi tương tự với phần đọc dữ liệu
img = Image.open("nhan2.bmp")
# from PIL import Image

def check_and_convert_to_rgb(img):
    # Check if image is in RGB format, if not convert it
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img
img = check_and_convert_to_rgb(img)


transform1 = transforms.Compose([                         
                                 transforms.ToTensor(),                               
                                 transforms.Resize((32, 32)),
                                 transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
                                 
])
img = transform1(img)
img = img.unsqueeze(0) # Thêm chiều batch (batch size = 1)
#调用模型和训练好的ckpt文件
model_path = "checkpoints/model_8.ckpt"
model=models.squeezenet1_0(pretrained=True)
model.load_state_dict(torch.load("squeezenet1_0-a815701f.pth"))
model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=31,
                                kernel_size=1)#改变网络的最后一层输出为90分类
model.num_classes = 31 #改变网络的分类类别数目

model.load_state_dict(torch.load(model_path)) 

model.num_classes = 31


with torch.no_grad():
    model.eval()  # Chuyển sang chế độ inference
    output = model(img) # Dự đoán kết quả
    _, predicted = torch.max(output.data, 1)
    label_list= ['010', '011', '012', '013', '014', '015', '016', '017', 
     '018', '019', '020', 'DU', 'DUY', 'DUYKHANG', 'HAN', 'HIEU', 'KHOA', 'NHAN', 
     'NHI', 'QUAN', 'QUANG', 'TAI', 'THAI', 'THONG', 'TIN', 'TRINH', 'T_QUAN', 
     'VIET', 'VINH', 'VY']
    
    # class_dict = predicted.item() # Lấy chỉ số của lớp dự đoán được
    label = label_list[predicted]

# define mapping of class indices to label names
# class_dict = {0: "010", 1: "011", 2: "012",
#               3:"013",4:"014",5:"015",
#                6:"016",7:"017",8:"018",
#                 9:"019",10:"020",11:"DU",
#                 12:"DUY",13:"KHANG",14:"HAN",
#                 15:"HIEU",16:"KHOA",17:"NHAN",
#                 18:"NHI",19:"QUAN",20:"QUANG",
#                 21:"T_QUAN",22:"TAI",23:"TAN",
#                 24:"THAI",25:"THONG",26:"TIN",
#                 27:"TRINH",28:"VIET",29:"VINH",
#                 30:"VY"              
              
#               }

# print predicted class name
print("Predicted class:",label)

# print predicted class
print("Predicted class:", label)
# print(type(str(class_dict[predicted.item()])))
