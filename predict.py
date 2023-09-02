import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import transforms as T
import cv2
import time 
from tqdm.notebook import tqdm
import glob

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

im_dir = glob.glob('/home/sathya/DL/dataset/ForHairTest/HairDamageTest/Images/*')
save_dir = '/home/sathya/DL/dataset/ForHairTest/HairDamageTest/res/'
os.makedirs(save_dir , exist_ok=True)

def predict_final_mask(model, image , mean=[0.485, 0.456, 0.406],
                       std = [0.229 , 0.224 ,0.225]):
    model.eval()
    t= T.Compose([T.ToTensor() ,T.Normalize(mean, std)])
    image = t(image)
    # model.to(device) ; 
    image = image.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)

        output = model(image)
        masked = torch.argmax(output , dim =1)
        masked = masked.cpu().squeeze(0)
    return masked

# Load pytorch model 
model = torch.load('output/model.pth')
model=model.to(device)
kernel_sharp = np.array(([-2, -2, -2], [-2, 17, -2], [-2, -2, -2]), dtype='int')

for i,im_path in enumerate(im_dir):
    # if i%25 == 0:
    im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    h,w = im.shape[:2]
    im = cv2.filter2D(im, -1, kernel_sharp)
    im = cv2.resize(im, (512 , 512) , interpolation = cv2.INTER_NEAREST)
    pred_mask = predict_final_mask(model , im)
    # print(np.unique(pred_mask))
    # pred mask = 0 for background (0),  2 for foreground (255) (Binary Segmentaion)
    pred_mask = pred_mask.cpu().numpy()
    pred_mask[pred_mask == 0] = 0
    pred_mask[pred_mask == 1] = 255

    pred_mask = cv2.resize(pred_mask, (w , h) , interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(save_dir + '/' + im_path.split('/')[-1] , pred_mask)
    print(i+1, save_dir + '/' + im_path.split('/')[-1] )