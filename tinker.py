import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import torch
import torchvision.transforms as transforms
from model.squeezenet import SqueezeNet
import numpy as np
from PIL import Image
from torchvision import models
from PIL import Image
import torch.nn as nn
def check_and_convert_to_rgb(img):
    # Check if image is in RGB format, if not convert it
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img
transform1 = transforms.Compose([                         
                                 transforms.ToTensor(),                               
                                 transforms.Resize((32, 32)),
                                 transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
                                 
])
model_path = "checkpoints/model_12.ckpt"
model=models.squeezenet1_0(pretrained=True)
model.load_state_dict(torch.load("squeezenet1_0-a815701f.pth"))
model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=100,
                                kernel_size=1) # #Change the output of the last layer of the network to 20 categories
model.num_classes = 100 #Change the number of classification categories of the network

model.load_state_dict(torch.load(model_path)) # read ckpt file 
class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Image Recognition'
        self.left = 100
        self.top = 100
        self.width = 400
        self.height = 300
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setGeometry(50, 50, 300, 200)

        self.btn1 = QPushButton('Select Image', self)
        self.btn1.move(50, 260)
        self.btn1.clicked.connect(self.openFileNameDialog)

        self.btn2 = QPushButton('Predict', self)
        self.btn2.move(250, 260)
        self.btn2.clicked.connect(self.predict)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Image Files (*.bmp *.png *.jpg *.jpeg);;All Files (*)", options=options)
        if fileName:
            self.image_path = fileName
            pixmap = QPixmap(fileName)
            self.label.setPixmap(pixmap.scaled(300, 200, Qt.KeepAspectRatio))

    def predict(self):
        img = Image.open(self.image_path)
        img = check_and_convert_to_rgb(img)
        img = transform1(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            model.eval()
            output = model(img)
            probs = nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
            class_dict = predicted.item()
        class_dict = ['010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', 
                      '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', 
                      '038', '039', '040', 'DU', 'DUY', 'DUYKHANG', 'HAN', 'HIEU', 'KHOA', 'NHAN',
                       'NHI', 'QUAN', 'QUANG', 'TAI', 'THAI', 'THONG', 'TIN', 'TRINH', 'T_QUAN', 'VIET', 'VINH', 'VY']
        label = class_dict[predicted.item()]
        confidence = probs[0][predicted].item()
        print("Predicted class:", label)
        print("Confidence:", confidence)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
