from torch.utils.data import Dataset
import PIL.Image as Image
import os
import cv2
import glob
import numpy as np
import torch

def make_dataset(root1,root2,j,n):
    imgs = []
    name1=os.listdir(root1)
    
    #name2=os.listdir(root2)
    #filename=glob.glob(root1+'*.bmp')
    #n = len(os.listdir(root1))
    #name1 = sorted(name1, key=lambda name: int(name.split("_")[0]))
    #name=name1[j:n]#列表从第零个开始
    name = name1
    for filename in name:
        
        img=os.path.join(root1,filename)
        mask=os.path.join(root2,filename.replace(".jpg",".png"))
        #mask=os.path.join(root2,filename)
        imgs.append((img, mask))
        
    return imgs
def make_dataset_unlabel(root1,root2,j,n):
    imgs = []
    name1=os.listdir(root1)
    
    name1 = sorted(name1, key=lambda name: int(name.split("_")[0]))
    name = name1[j:n]#列表从第零个开始
   
    for filename in name:
        
        img=os.path.join(root1,filename)
        mask=os.path.join(root2,filename)
        imgs.append((img,mask))
      
        
    return imgs

class LiverDataset(Dataset):
    def __init__(self, root1,root2,i=0,n=0 ,transform=None, target_transform=None):
        imgs = make_dataset(root1,root2,j=i,n=n)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        #size=(512,512)
        img_x=img_x.resize((256,256))
        img_y=img_y.resize((256,256),Image.NEAREST)
        
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
            #img_y = img_y.type(torch.LongTensor)

        return img_x, img_y
    def __len__(self):
        return len(self.imgs)

class LiverDataset1(Dataset):
    def __init__(self, root1,root2,i,n ,transform=None, target_transform=None):
        imgs = make_dataset(root1,root2,i,n)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        # if img_y==0:
        #     pass
        #size=(512,512)
        img_x=img_x.resize((256,256))
        img_y=img_y.resize((256,256))
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
            img_y = img_y.type(torch.LongTensor)

        return img_x, img_y
    def __len__(self):
        return len(self.imgs)
class LiverDataset_smi(Dataset):
    def __init__(self, root1,root2,i,n, transform=None,target_transform=None):
        imgs = make_dataset_unlabel(root1,root2,i,n)
        self.imgs = imgs
        
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        x_path,y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        #img_x=img_x.resize((248,248))
        #img_y=img_y.resize((248,248),Image.NEAREST)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
            img_y = img_y.type(torch.LongTensor)
        return img_x,img_y
    def __len__(self):
        return len(self.imgs)
        
def make_dataset_txt(root1,root2,file):
    imgs = []
    f = open(file, 'r') 
    filname = f.readlines()
    for line in filname:
        line = line.strip('\n')
        img=os.path.join(root1,line)
        mask=os.path.join(root2,line.replace(".jpg",".png"))
        imgs.append((img, mask))
   
   
 
    return imgs
class LiverDataset_txt(Dataset):
    def __init__(self, root1,root2,file, transform=None,target_transform=None):
        imgs = make_dataset_txt(root1,root2,file)
        self.imgs = imgs
        
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        x_path,y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        img_x = img_x.resize((256,192))
        img_y = img_y.resize((256,192),Image.NEAREST)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
            
        return img_x,img_y
    def __len__(self):
        return len(self.imgs)

