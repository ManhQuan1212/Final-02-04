from ctypes import sizeof
from tkinter import W
import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import os
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
model_path = "checkpoints/model_8.ckpt"
model=models.squeezenet1_0(pretrained=True)
model.load_state_dict(torch.load("squeezenet1_0-a815701f.pth"))
model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=31,
                                kernel_size=1)#改变网络的最后一层输出为90分类
model.num_classes = 31 #改变网络的分类类别数目

model.load_state_dict(torch.load(model_path)) 

model.num_classes = 31

def IncreaseContrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    #result = np.hstack((img, enhanced_img))
    return enhanced_img
cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while True:
        OK,frame=cap.read()
        if not OK:
            print("Ignoring empty camera frame.")
            continue
        imgaeResize = IncreaseContrast(frame)
        imgaeRGB = imgaeResize
        imgaeResize.flags.writeable = False
        imgaeRGB.flags.writeable = False
        imgaeRGB = imgaeResize
        results = hands.process(imgaeResize)
        # cv2.imshow("RESIZE ", imgaeResize)
        cropped_image = cv2.cvtColor(imgaeResize, cv2.COLOR_BGR2GRAY)
        h = cropped_image.shape[0]
        w = cropped_image.shape[1]
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                pixelCoordinatesLandmarkPoint5 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[5].x, hand_landmark.landmark[5].y, w, h)
                pixelCoordinatesLandmarkPoint9 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[9].x, hand_landmark.landmark[9].y, w, h)
                pixelCoordinatesLandmarkPoint13 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[13].x, hand_landmark.landmark[13].y, w, h)
                pixelCoordinatesLandmarkPoint17 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[17].x, hand_landmark.landmark[17].y, w, h)
                pixelCoordinatesLandmarkPoint0 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[0].x, hand_landmark.landmark[0].y, w, h)
                
                print(pixelCoordinatesLandmarkPoint5)
                print(pixelCoordinatesLandmarkPoint17)
                center5 = np.array(
                    [np.mean(hand_landmark.landmark[5].x)*w, np.mean(hand_landmark.landmark[5].y)*h]).astype('int32')
                center9 = np.array(
                    [np.mean(hand_landmark.landmark[9].x)*w, np.mean(hand_landmark.landmark[9].y)*h]).astype('int32')
                center13 = np.array(
                    [np.mean(hand_landmark.landmark[13].x)*w, np.mean(hand_landmark.landmark[13].y)*h]).astype('int32')
                center17 = np.array(
                    [np.mean(hand_landmark.landmark[17].x)*w, np.mean(hand_landmark.landmark[17].y)*h]).astype('int32')
                

                # cv2.circle(imgaeResize, tuple(center5), 10, (255, 0, 0), 1)
                # cv2.circle(imgaeResize, tuple(center9), 10, (255, 0, 0), 1)
                # cv2.circle(imgaeResize, tuple(center13), 10, (255, 0, 0), 1)
                # cv2.circle(imgaeResize, tuple(center17), 10, (255, 0, 0), 1)



                cropped_image = cropped_image[0:pixelCoordinatesLandmarkPoint0[1] + 50, 0:pixelCoordinatesLandmarkPoint5[0] + 100]
                x1 = (pixelCoordinatesLandmarkPoint17[0] +  pixelCoordinatesLandmarkPoint13[0]) / 2 + (pixelCoordinatesLandmarkPoint17[0] - pixelCoordinatesLandmarkPoint13[0])
                y1 = (pixelCoordinatesLandmarkPoint17[1] + pixelCoordinatesLandmarkPoint13[1]) / 2 - 50
                x2 = (pixelCoordinatesLandmarkPoint5[0] + pixelCoordinatesLandmarkPoint9[0]) / 2 - 50
                y2 = (pixelCoordinatesLandmarkPoint5[1] + pixelCoordinatesLandmarkPoint9[1]) / 2 - 50

                theta = np.arctan2((y2 - y1), (x2 - x1))*180/np.pi 

                if (theta >= -15 and theta < 0):
                    print("theta", theta)
                    R = cv2.getRotationMatrix2D(
                        (int(x2), int(y2)), theta, 1)
                    align_img = cv2.warpAffine(cropped_image, R, (w, h)) 
                    # imgaeRGB = cv2.warpAffine(imgaeRGB, R, (w, h)) 
                    point_1 = [x1, y1]
                    point_2 = [x2, y2]

                    point_1 = (R[:, :2] @ point_1 + R[:, -1]).astype(np.int)
                    point_2 = (R[:, :2] @ point_2 + R[:, -1]).astype(np.int)

                    landmarks_selected_align = {
                                        "x": [point_1[0], point_2[0]], "y": [point_1[1], point_2[1]]}

                    point_1 = np.array([landmarks_selected_align["x"]
                                                [0], landmarks_selected_align["y"][0]])
                    point_2 = np.array([landmarks_selected_align["x"]
                                                        [1], landmarks_selected_align["y"][1]])                   

                    ux = point_1[0]
                    uy = point_1[1] + (point_2-point_1)[0]//3
                    lx = point_2[0]
                    ly = point_2[1] + 4*(point_2-point_1)[0]//3
                    roi_zone_img = cv2.cvtColor(align_img, cv2.COLOR_GRAY2BGR)

                    
                    roi_img = align_img[uy:ly + 85, ux:lx + 85]
                    roi_img = cv2.resize(roi_img, (32,32))
                    # roi_img = check_and_convert_to_rgb(roi_img)
                    roi_img=cv2.cvtColor(roi_img,cv2.COLOR_RGB2BGR)
                    roi_img = transform1(roi_img)
                    roi_img = roi_img.unsqueeze(0) # Thêm chiều batch (batch size = 1)
                    with torch.no_grad():
                        model.eval()  # Chuyển sang chế độ inference
                        output = model(roi_img) # Dự đoán kết quả
                        _, predicted = torch.max(output.data, 1)
                        class_dict = predicted.item() # Lấy chỉ số của lớp dự đoán được
                    cv2.rectangle(imgaeResize, (ux, uy),
                        (lx+85, ly + 85), (10, 255, 15), 2)
                    # cv2.putText(roi_img,"nhan",(w,h),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,25,255),2)
                    # cv2.putText(imgaeResize, str(class_dict[predicted.item()]), (uxROI, uyROI), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 25, 255), 2)
                    # cv2.putText(imgaeResize, str(class_dict[(predicted.item())]), (uxROI, uyROI), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 25, 255), 2)
                    # text=class_dict[predicted.item()]
                    # 
                    class_dict = {  0: "010", 1: "011", 2: "012",
                                    3:"013",4:"014",5:"015",
                                    6:"016",7:"017",8:"018",
                                    9:"019",10:"020",11:"DU",
                                    12:"DUY",13:"KHANG",14:"HAN",
                                    15:"HIEU",16:"KHOA",17:"NHAN",
                                    18:"NHI",19:"QUAN",20:"QUANG",
                                    21:"T_QUAN",22:"TAI",23:"TAN",
                                    24:"THAI",25:"THONG",26:"TIN",
                                    27:"TRINH",28:"VIET",29:"VINH",
                                    30:"VY"              
                
                                }
                    cv2.putText(imgaeResize, class_dict[(predicted.item())], (ux, uy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 25, 255), 2)
           
                      
                
                cv2.imshow('Frame',imgaeResize)
                if cv2.waitKey(1)&0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()