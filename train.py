import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import random
from PIL import Image
import cv2
import albumentations as A
from sklearn.metrics import f1_score
from rmi import RMILoss
import json
import time 
from tqdm.auto import tqdm

import segmentation_models_pytorch as smp
import warnings
import wandb
import argparse
from utils.lovasz_losses import lovasz_softmax
import ssl
import yaml
ssl._create_default_https_context = ssl._create_unverified_context

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

encoder = config["encoder_name"]


work_dir = os.path.join(config["work_dir"], config["model_name"], config["encoder_name"])
os.makedirs(work_dir, exist_ok=True)

# Wandb configuration
WANDB_API_KEY = config["WANDB_API_KEY"]
WANDB_PROJECT_NAME = config["WANDB_PROJECT_NAME"]
WANDB_ENTITY = config["WANDB_ENTITY"]
WANDB_NAME = work_dir.split('/')[-3] + '_' + work_dir.split('/')[-2]

# Initialize wandb
wandb.login(key=WANDB_API_KEY)
wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, name = WANDB_NAME)

warnings.filterwarnings("ignore")

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

n_classes = config["num_classes"]
activation = config["activation"]
if activation == 'None':
    activation = None
    activation_applied = False
else:
    activation = activation
    activation_applied = True
encoder_weights = config["encoder_weights"]
if encoder_weights == 'None':
    encoder_weights = None
else:
    encoder_weights = encoder_weights

def find_image_by_id(image_id, path):
    supported_extensions = ['png', 'jpg', 'jpeg', 'bmp','tif']
    for ext in supported_extensions:
        file = glob.glob(f"{path}/{image_id}.{ext}")
        if file:
            return file[0]
    return None

train_images = config["input"]["train_images"]+'/'
train_targets = config["input"]["train_mask"]+'/'
val_images = config["input"]["test_images"]+'/'
val_targets = config["input"]["test_mask"]+'/'

def _df_(images_path, targets_path):
    name = []
    for dirname, _, filenames in os.walk(images_path):
        for filename in filenames:
#            print("file name->",filename.split('.')[1])
            if filename.split('.')[1] == "png" or filename.split('.')[1] == "jpeg" or filename.split('.')[1] == "jpg" or filename.split('.')[1] == "bmp" or filename.split('.')[1] == "tif":
                name.append(filename.split('.')[0])
            else:
                continue

    # return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))
    return pd.DataFrame({'id': name, 'extension': '.'+filename.split('/')[-1].split('.')[1]}, index=np.arange(0, len(name)))

df_train = _df_(train_images, train_targets)
df_val = _df_(val_images, val_targets)

X_train = df_train['id'].values
X_val = df_val['id'].values
print('Train : ', len(X_train))
print('Val : ', len(X_val))


img = Image.open(train_images + df_train['id'][100] + df_train['extension'][100])
targ = Image.open(train_targets + df_train['id'][100] + df_train['extension'][100])

print('Image Size', np.asarray(img).shape)
print('Mask Size' , np.asarray(targ).shape)

plt.figure(figsize=(10,10))
plt.imshow(img)
plt.imshow(targ , alpha =0.4)
plt.savefig(work_dir + '/sample.png')

img_width = config["image_shape"]["width"]
img_height = config["image_shape"]["height"]


def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ], p=0.1),
        A.RandomBrightnessContrast(p=0.1),
        A.OneOf(
            [
                A.HueSaturationValue(p=0.5),
                A.CLAHE(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2,p=0.5),
            ],
            p=0.1,
        ),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5)
        ], p=0.1),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.GaussianBlur(p=0.5),
            A.ISONoise(p=0.5),
            A.MotionBlur(p=0.5),
            A.MedianBlur(p=0.5),
        ], p=0.05),            
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8, p=0.05),
        
    ]
    return A.Compose(train_transform, p=0.9)

def get_validation_augmentation():   
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        A.Resize(height=img_height, width=img_width, always_apply=True, p=1.0),
    ]    
    return A.Compose(test_transform, p=1)

def dense_target(tar: np.ndarray):
    classes =np.unique(tar)
    dummy= np.zeros_like(tar)
    for idx , value in enumerate(classes):
        mask = np.where(tar == value)
        dummy[mask] = idx
    return dummy

class SegData(Dataset):

    def __init__(self ,  train_image_path, train_target_path, val_image_path ,val_target_path, X , mean , std , transform =None , val=False, test=False):
        self.train_image_path = train_image_path
        self.train_target_path = train_target_path
        self.val_image_path = val_image_path
        self.val_target_path = val_target_path
        self.X = X
        self.transform =transform
        self.mean = mean
        self.std = std
        self.test =test
        self.val = val

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image_id = self.X[idx]

        if self.val:
            image_file = find_image_by_id(image_id, self.val_image_path)
            target_file = find_image_by_id(image_id, self.val_target_path)
        else:
            image_file = find_image_by_id(image_id, self.train_image_path)
            target_file = find_image_by_id(image_id, self.train_target_path)

        if image_file is None or target_file is None :
            raise ValueError(f"Image or target file not found for id {image_id}")

        img = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        target = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)

        # After Changes 
        kernel_sharp = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), dtype='int')
        img = cv2.filter2D(img, -1, kernel_sharp)
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        target = cv2.resize(target , (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        # target = np.where( target > 0,255,0)


        if self.transform is not None:
            aug = self.transform(image = img , target = target)
            img = Image.fromarray(aug['image'])
            target = aug['target']
        
        if self.transform is None:
            img = Image.fromarray(img) 

        t = T.Compose([T.ToTensor() , T.Normalize(self.mean , self.std)])

        target = dense_target(target)
        target = torch.from_numpy(target).long().unsqueeze(0)  # Adding channel dimension

        if self.test is False:
            img = t(img)
            
        return img, target

mean = [0.485 ,0.456 ,0.406]
std = [0.229 , 0.224 , 0.225]

train_set = SegData(train_images, train_targets,val_images , val_targets, X_train, mean, std, transform=get_training_augmentation(), val=False, test=False)
val_set = SegData(train_images, train_targets,val_images,val_targets, X_val, mean, std, transform=None, val=True, test=False)

batch_size = config["batch_size"]
num_workers = config["num_workers"]
train_loader= DataLoader(train_set , batch_size= batch_size , shuffle = True,drop_last=True, num_workers=num_workers)
val_loader = DataLoader(val_set , batch_size = 1 , shuffle = False , num_workers=num_workers)

x , y =next(iter(train_loader))

print(f' x = shape : {x.shape} ; type :{x.dtype}')
print(f' x = min : {x.min()} ; max : {x.max()}')
print(f' y = shape: {y.shape}; class : {y.unique()}; type: {y.dtype}')
print("Model : ", config["model_name"])
print("Encoder : ", encoder)
print("Encoder Weights : ", encoder_weights)
print("Activation : ", activation)
print("Number of Classes : ", n_classes)

if config["model_name"] == 'UnetPlusPlus':
    model = smp.UnetPlusPlus(encoder_name=encoder,encoder_weights=encoder_weights, classes = n_classes, activation=activation)
elif config["model_name"] == 'DeepLavbV3':
    model = smp.DeepLabV3(encoder_name=encoder,encoder_weights=encoder_weights, classes = n_classes, activation=activation)
elif config["model_name"] == 'FPN':
    model = smp.FPN(encoder_name=encoder,encoder_weights=encoder_weights, classes = n_classes, activation=activation)
elif config["model_name"] == 'PSPNet':
    model = smp.PSPNet(encoder_name=encoder,encoder_weights=encoder_weights, classes = n_classes, activation=activation)
elif config["model_name"] == 'Linknet':
    model = smp.Linknet(encoder_name=encoder,encoder_weights=encoder_weights, classes = n_classes, activation=activation)
elif config["model_name"] == 'Unet':
    model = smp.Unet(encoder_name=encoder,encoder_weights=encoder_weights, classes = n_classes, activation=activation)
elif config["model_name"] == 'MAnet':
    model = smp.MAnet(encoder_name=encoder,encoder_weights=encoder_weights, classes = n_classes, activation=activation)
elif config["model_name"] == 'PAN':
    model = smp.PAN(encoder_name=encoder,encoder_weights=encoder_weights, classes = n_classes, activation=activation)
elif config["model_name"] == 'DeepLabV3Plus':
    model = smp.DeepLabV3Plus(encoder_name=encoder,encoder_weights=encoder_weights, classes = n_classes, activation=activation)

model=model.to(device)
wandb.watch(model)  # Add this line to watch the model

def pixel_wise_accuracy(output, mask, num_classes=1, activation_applied=False):
    with torch.no_grad():
        if num_classes == 1:
            if not activation_applied:
                output = torch.sigmoid(output)
            output = (output > 0.5).long()
        else:
            if not activation_applied:
                output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())  # total number
    return accuracy


def IoU(pred, true_pred, smooth=1e-10, num_classes=2, activation_applied=False):
    with torch.no_grad():
        if num_classes > 1:
            if not activation_applied:
                pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
        else:
            if not activation_applied:
                pred = torch.sigmoid(pred)
            pred = (pred > 0.5).long()

        pred = pred.contiguous().view(-1)
        true_pred = true_pred.contiguous().view(-1)

        iou_class = []
        for value in range(0, num_classes):
            true_class = pred == value
            true_label = true_pred == value

            if true_label.long().sum().item() == 0:
                iou_class.append(np.nan)
            else:
                inter = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (inter + smooth) / (union + smooth)
                iou_class.append(iou)

        return np.nanmean(iou_class)


def DiceBceLoss(true, logits, eps=1e-7, activation_applied=False):
    num_classes = logits.shape[1]
    
    # Handle single and multi-class cases differently
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1, device=logits.device)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        
        if activation_applied:
            pos_prob = logits.squeeze(1)  # Remove the channel dimension
        else:
            pos_prob = torch.sigmoid(logits).squeeze(1)  # Apply sigmoid and remove the channel dimension

        neg_prob = 1 - pos_prob
        probas = torch.stack([pos_prob, neg_prob], dim=1)  # Stack along the channel dimension

    else:
        true_1_hot = torch.eye(num_classes, device=logits.device)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()

        if activation_applied:
            probas = logits
        else:
            probas = F.softmax(logits, dim=1)
    
    true_1_hot = true_1_hot.type(probas.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    
    dice_loss = 1 - ((2. * intersection + eps) / (cardinality + eps)).mean()
    
    if num_classes == 1:
        # Add the missing channel dimension
        pos_prob = pos_prob.unsqueeze(1)
        bce = F.binary_cross_entropy(pos_prob, true.float(), reduction="mean")
    else:
        bce = F.cross_entropy(logits, true, reduction="mean")
    
    dice_bce = bce + dice_loss
    return dice_bce

def RMIloss(pred, target, num_classes=1, activation_applied=False):
    """
    Generalized RMILoss function that can handle both logits and probabilities.
    Args:
    - pred (torch.Tensor): The predicted output from the neural network.
    - target (torch.Tensor): The ground truth.
    - num_classes (int): The number of classes.
    - activation_applied (bool): Whether the activation function (sigmoid or softmax) has been applied to pred.
    
    Returns:
    - torch.Tensor: The RMILoss
    """
    # Initialize RMILoss
    loss = RMILoss(with_logits=not activation_applied)
    
    # Apply activation if not already applied
    if num_classes == 1 and not activation_applied:
        pred = torch.sigmoid(pred)
    elif num_classes > 1 and not activation_applied:
        pred = torch.softmax(pred, dim=1)
        
    # Explicitly convert target to float
    target = target.float()
        
    # Compute RMILoss
    output = loss(pred, target)
    return output


def tversky_loss(true, logits, alpha=0.5, beta=0.5, eps=1e-7):
    num_classes = logits.shape[1]
    true_1_hot = torch.eye(num_classes, device=logits.device)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()

    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())

    dims = (0,) + tuple(range(2, true.ndimension()))
    tp = torch.sum(probas * true_1_hot, dims)
    fp = torch.sum(probas * (1 - true_1_hot), dims)
    fn = torch.sum((1 - probas) * true_1_hot, dims)

    tversky_coeff = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    tversky_loss = 1 - tversky_coeff.mean()
    return tversky_loss

def LovaszTverskyLoss(true, logits, alpha=0.5, beta=0.5, eps=1e-7, lovasz_weight=0.5):
    lovasz_loss = lovasz_softmax(logits, true, ignore=None)
    tversky_loss_value = tversky_loss(true, logits, alpha=alpha, beta=beta, eps=eps)
    combined_loss = lovasz_weight * lovasz_loss + (1 - lovasz_weight) * tversky_loss_value
    return combined_loss

def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.

    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes, device=device)[true.squeeze(1)]
        # true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

def calculate_f1_score(true, pred, n_classes):
    true = true.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    
    if n_classes == 1:
        # Binary segmentation
        true = true.squeeze(1).flatten()
        pred = pred.squeeze(1).flatten()
        pred = np.round(pred).astype(int)
    else:
        # Multi-class segmentation
        true = true.flatten()
        pred = np.argmax(pred, axis=1).flatten()
        
    score = f1_score(true, pred, average="weighted")
    return score


    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Save a checkpoint
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

# Load a checkpoint
def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch

model_path = work_dir + '/ckpts/'
os.makedirs(model_path, exist_ok=True)
os.makedirs(model_path + '/ckpt_state_dict/', exist_ok=True)

def fit(epochs, model, train_loader, val_loader, optimizer, scheduler, patch=False, resume=False, checkpoint_path=None):
    # Initialize the best models, their IoU scores, and file paths
    best_iou = 0.0
    best_model_path = model_path + "best_model.pth"

    train_losses = []
    val_losses = []
    epoch_train_f1_scores = [] 
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    # If resume flag is set, load the checkpoint
    start_epoch = 0
    if resume and checkpoint_path:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    model.to(device)
    fit_time = time.time()
    for e in range(start_epoch, epochs):
        since = time.time()
        running_loss = 0.0
        iou_score = 0.0
        accuracy = 0.0
        sum_f1_score = 0.0  # Initialize sum_f1_score
        epoch_val_f1_scores = []
        val_f1_scores = []
        model.train()
        
        train_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)

        for i, data in train_loop:
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()
                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            output = model(image)
            loss = DiceBceLoss(mask, output, activation_applied=activation_applied) + RMIloss(output, mask, activation_applied=activation_applied)
            iou_score += IoU(output, mask, num_classes=n_classes, activation_applied=activation_applied)
            accuracy += pixel_wise_accuracy(output, mask, num_classes=n_classes, activation_applied=activation_applied)

            #backward
            loss.backward()
            optimizer.step() #update weight          
            optimizer.zero_grad() #reset gradient
            
            #step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step() 
            
            running_loss += loss.item()

            # Calculate and append batch-level F1 score
            train_f1_score = calculate_f1_score(mask, output, n_classes)
            sum_f1_score += train_f1_score  # Update sum_f1_score
            mean_f1_score = sum_f1_score / (i + 1)  # Compute mean F1 score up to this batch
            train_loop.set_description(f"Epoch {e+1}/{epochs}")
            train_loop.set_postfix(IoU=iou_score/(i+1), LR=lrs[-1], Accuracy=accuracy/(i+1), Dice_BCE_RMI_Loss=running_loss/(i+1), F1_Score=mean_f1_score )
            # Calculate average metrics for the batch
            avg_train_loss = running_loss / (i+1)
            avg_iou_score = iou_score / (i+1)
            avg_accuracy = accuracy / (i+1)     
            # Log batch-level metrics to wandb
            wandb.log({"train_f1_score_batch": train_f1_score,
                    "train_loss_batch": avg_train_loss,
                    "train_iou_batch": avg_iou_score,
                    "train_acc_batch": avg_accuracy,
                    "lr_batch": get_lr(optimizer)})
        else:
            model.eval()
            val_f1_scores = []
            test_loss = 0.0
            val_iou_score = 0.0
            test_accuracy = 0.0
            #validation loop
            with torch.no_grad():
                val_loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=True)
                for i, data in val_loop:
                    image_tiles, mask_tiles = data
                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()
                        image_tiles = image_tiles.view(-1,c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)
                    
                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    output = model(image)
                    val_f1_score = calculate_f1_score(mask, output, n_classes)
                    val_f1_scores.append(val_f1_score)

                    val_loss = DiceBceLoss(mask, output, activation_applied=activation_applied) + RMIloss(output, mask, activation_applied=activation_applied)
                    test_loss += val_loss.item()

                    val_iou_score += IoU(output, mask, num_classes=n_classes, activation_applied=activation_applied)
                    test_accuracy += pixel_wise_accuracy(output, mask, num_classes=n_classes, activation_applied=activation_applied)
                    val_loop.set_description(f"Epoch {e+1}/{epochs}")
                    val_loop.set_postfix(IoU=iou_score/(i+1), Accuracy=accuracy/(i+1), Dice_BCE_RMI_Loss=running_loss/(i+1), F1_Score=val_f1_score)
                    avg_val_loss = test_loss / (i+1)
                    avg_val_iou = val_iou_score / (i+1)
                    avg_val_accuracy = test_accuracy / (i+1)

                    wandb.log({"val_f1_score_batch": val_f1_score,
                            "val_loss_batch": avg_val_loss,
                            "val_iou_batch": avg_val_iou,
                            "val_acc_batch": avg_val_accuracy})

                current_iou = val_iou_score / len(val_loader)

                # Save the best model based on IoU
                if current_iou > best_iou:
                    best_iou = current_iou
                    torch.save({
                        'epoch': e + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'iou': best_iou
                    }, best_model_path)



        # Your existing code for updating metrics lists
        train_losses.append(running_loss / len(train_loader))
        val_losses.append(test_loss / len(val_loader))
        train_iou.append(iou_score / len(train_loader))
        val_iou.append(current_iou)
        avg_train_f1_score = sum_f1_score / len(train_loader)
        epoch_train_f1_scores.append(avg_train_f1_score)
        avg_val_f1_score = np.mean(val_f1_scores)
        epoch_val_f1_scores.append(avg_val_f1_score)
        # Inside your training loop
        train_acc.append(accuracy / len(train_loader))
        val_acc.append(test_accuracy / len(val_loader))
        # Log epoch-level metrics to wandb
        wandb.log({
            "epoch": e + 1,
            "train_f1_score_epoch": avg_train_f1_score,
            "val_f1_score_epoch": avg_val_f1_score,
            "lr_epoch": scheduler.get_last_lr()[0],
            "train_loss_epoch": avg_train_loss,
            "val_loss_epoch": avg_val_loss,
            "train_iou_epoch": avg_iou_score,
            "val_iou_epoch": avg_val_iou,
            "train_acc_epoch": avg_accuracy,
            "val_acc_epoch": avg_val_accuracy,
            "Time": (time.time() - since) / 60
        })

    
        # Save a checkpoint at the end of every 50 epochs
        if (e + 1) % 50 == 0:
            save_checkpoint(model, optimizer, e + 1, model_path + f'/ckpt_state_dict/checkpoint_epoch_{e + 1}.pt')
        # Save the last model
        torch.save(model, model_path + 'last_model_epoch_{}.pt'.format(e + 1))
        # CHeck if the last model exists and delete it
        if os.path.exists(model_path + 'last_model_epoch_{}.pt'.format(e)):
            os.remove(model_path + 'last_model_epoch_{}.pt'.format(e))

        # Create DataFrame dynamically based on current epoch
        current_epoch = e + 1
        # metrics_df = pd.DataFrame({
        #     'epoch': list(range(1, current_epoch + 1)),  # Adjust the range
        #     'train_loss': train_losses[:current_epoch],
        #     'val_loss': val_losses[:current_epoch],
        #     'train_f1_score': epoch_train_f1_scores[:current_epoch],
        #     'val_f1_score': epoch_val_f1_scores[:current_epoch],
        #     'train_iou': train_iou[:current_epoch],
        #     'val_iou': val_iou[:current_epoch],
        #     'train_acc': train_acc[:current_epoch],
        #     'val_acc': val_acc[:current_epoch],
        # })

        # metrics_df.to_csv(f"{work_dir}/metrics.csv", index=False)

    history = {'train_loss': train_losses, 'val_loss': val_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_acc': train_acc, 'val_acc': val_acc,
               'lrs': lrs}
    print('Total time: {:.2f} m' .format((time.time()- fit_time)/ 60))
    return history



max_lr = 1e-3
epoch = config["epoch"]

# Use a smaller learning rate for the weight decay to prevent overfitting
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

# Calculate pct_start
# pct_start = 120 / epoch
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                            steps_per_epoch=len(train_loader))
history = fit(epoch, model, train_loader, val_loader, optimizer, sched, resume=False)
wandb.finish()

# Save history using json
with open(f"{work_dir}/history.json", 'w') as fp:
    json.dump(history, fp)
    

