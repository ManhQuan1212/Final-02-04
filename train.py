import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model.squeezenet import SqueezeNet
from torchvision import models
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters 超参数设置
num_epochs = 20
num_classes = 100
batch_size = 32
learning_rate = 0.0002
def check_and_convert_to_rgb(img):
    # Check if image is in RGB format, if not convert it
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img
transform1 = transforms.Compose([ # #chuẩn hóa dữ liệu, giá trị trung bình và phương sai của 3 kênh là 0.5
                                 transforms.ToTensor(),                             
                                 transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])
#đọc dữ liệu theo từng mục mỗi mục là một nhãn 
train_dataset = torchvision.datasets.ImageFolder(root='./data/train/',
                                            transform=transform1,
                                            )
label_list=[]
label_list = list(train_dataset.class_to_idx.keys())


test_dataset = torchvision.datasets.ImageFolder(root='./data/test/',
                                            transform=transform1,
                                            )

# Data loader 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True, #shuffle_ Trainning data
                                           drop_last=True, #Nếu drop_test là true, dữ liệu ít hơn 1 bacth sẽ bị bỏ
                                           )

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
#load model
model=models.squeezenet1_0(pretrained=True)
#读取参数
model.load_state_dict(torch.load("squeezenet1_0-a815701f.pth")) #load pre_train model

model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=num_classes,
                                kernel_size=1)
model.num_classes = num_classes
#model.cuda()
# model = SqueezeNet(1.0,90).to(device) #call network model 

criterion = nn.CrossEntropyLoss() #cross entropy


# Train the model 

#Collect loss and correct rate for later drawing
train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

total_step = len(train_loader)  # every epoch has how many step

for epoch in range(num_epochs):  # how many epoch
    model.train()
    if epoch ==12:
        learning_rate=learning_rate*0.1
    if epoch ==30:
        learning_rate=learning_rate*0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)# optimizẻr
    for i, (images, labels) in enumerate(train_loader): # load data
        images = images.to(device)  # image
        labels = labels.to(device)   # label

        correct = 0
        # Forward pass
        outputs = model(images)   # outputs 输出
        
        loss = criterion(outputs, labels)  #计算损失函数
        _, predicted = torch.max(outputs.data, 1)  ##The predicted value (the index of the element that returns the largest value in each row)

        # Backward and optimize ( Backward and optimize (Backpropagation Gradient Descent Optimization))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (predicted == labels).sum().item()
        train_acc = correct / labels.size(0)
        if (i + 1) % 1 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, train_accuracy: {}'
                     .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), train_acc))
    train_loss.append(loss.item())
    train_accuracy.append(train_acc)

    #validtion set 
    model.eval()
    with torch.no_grad(): ##Do not calculate the gradient, save memory
        correct = 0
        total = 0
        step_test = len(test_loader)
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            te_loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_acc = correct / total
        test_loss.append(te_loss.item())
        test_accuracy.append(test_acc)
    # # every epoch save the model Save the model (note that only the network parameters are saved using state_dict here)   
    torch.save(model.state_dict(), './checkpoints/'+'model_'+str(epoch)+'.ckpt')
    

# matpolite -- Train acc - test acc 
# Loss  acc / train acc 
x1 = range(0, len(train_loss))

plt.subplot(2, 1, 1)
plt.plot(x1, train_accuracy, 'b-')
plt.plot(x1, test_accuracy, 'r-')
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.legend(('train_acc', 'test_acc'), loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(x1, train_loss, 'b-')
plt.plot(x1, test_loss, 'r-')
plt.title('Loss')
plt.ylabel('loss')
plt.legend(('train_loss', 'test_loss'), loc='upper right')

plt.savefig('./result.png')
plt.show()




print(label_list)




