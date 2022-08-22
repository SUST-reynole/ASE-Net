import numpy as np
from scipy import ndimage
#VOE代表IOU VOE = 1-IOU

def IoU(prediction, target):   #准确
    prediction = prediction.squeeze(1).cpu().detach().numpy()
    target = target.squeeze(1).cpu().detach().numpy()
    batch,_,_ = prediction.shape
    count = 0
    for i in range(batch):
        pred = prediction[i]
        tar = target[i]
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0
        pred = np.array(pred, dtype='uint8')
        pred = ndimage.binary_fill_holes(pred).astype(int)
        tar[tar>0.5] = 1
        tar[tar<=0.5] = 0
        delta = 1e-10
        IoU = ((pred * tar).sum() + delta) / (pred.sum() + tar.sum() - (pred * tar).sum() + delta)
        count = count + IoU
    return count/batch

def Dice(prediction, target):   #准确
    prediction = prediction.squeeze(1).cpu().detach().numpy()
    target = target.squeeze(1).cpu().detach().numpy()
    batch,_,_ = prediction.shape
    count = 0
    for i in range(batch):
        pred = prediction[i]
        tar = target[i]
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0
        pred = np.array(pred, dtype='uint8')
        pred = ndimage.binary_fill_holes(pred).astype(int)
        tar[tar>0.5] = 1
        tar[tar<=0.5] = 0
        delta = 1e-10
        dice = (((pred * tar).sum())*2 + delta) / (pred.sum() + tar.sum() + delta)
        count = count + dice
    #ious.append(IoU)
    #return np.nanmean(IoU)
    return count/batch
def acc(prediction, target):
    prediction = prediction.squeeze(1).cpu().detach().numpy()
    target = target.squeeze(1).cpu().detach().numpy()
    batch,row,col= prediction.shape
    count = 0
    TP=0
    TN=0
    for i in range(batch):
        pred = prediction[i]
        tar = target[i]
        pred[pred>=0.5] = 1
        pred[pred<0.5] = 0
        pred = np.array(pred, dtype='uint8')
        pred = ndimage.binary_fill_holes(pred).astype(int)
        tar[tar>=0.5] = 1
        tar[tar<0.5] = 0

        TP=(pred * tar).sum()
        #print('TP的值',TP)
        pred1=np.copy(pred)
        tar1=np.copy(tar)

        pred1[pred1==0]=2
        pred1[pred1==1]=0
        pred1[pred1==2]=1
        tar1[tar1==0]=2
        tar1[tar1==1]=0
        tar1[tar1==2]=1
        TN=(pred1 * tar1).sum()
        
        acc = (TP +TN) / (row * col)
        count = count + acc
    #ious.append(IoU)
    #return np.nanmean(IoU)
    return count/batch
def SE(prediction, target):   #准确
    prediction = prediction.squeeze(1).cpu().detach().numpy()
    target = target.squeeze(1).cpu().detach().numpy()
    batch,_,_ = prediction.shape
    count = 0
    for i in range(batch):
        pred = prediction[i]
        tar = target[i]
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0
        pred = np.array(pred, dtype='uint8')
        pred = ndimage.binary_fill_holes(pred).astype(int)
        # pred = np.array(pred, dtype='uint8')
        # pred = ndimage.binary_fill_holes(pred).astype(int)
        # tar[tar>0.5] = 1
        # tar[tar<=0.5] = 0
        delta = 1e-10
        se = (((pred * tar).sum()) + delta) / (tar.sum() + delta)
        count = count + se
    #ious.append(IoU)
    #return np.nanmean(IoU)
    return count/batch
def SP(prediction, target):   #准确
    prediction = prediction.squeeze(1).cpu().detach().numpy()
    target = target.squeeze(1).cpu().detach().numpy()
    batch,row,col= prediction.shape
    count = 0
    TP=0
    TN=0
    for i in range(batch):
        pred = prediction[i]
        tar = target[i]
        pred[pred>=0.5] = 1
        pred[pred<0.5] = 0
        pred = np.array(pred, dtype='uint8')
        pred = ndimage.binary_fill_holes(pred).astype(int)
        tar[tar>=0.5] = 1
        tar[tar<0.5] = 0

        TP = (pred * tar).sum()
        FP =  pred.sum()-TP
        #print('TP的值',TP)
        pred1=np.copy(pred)
        tar1=np.copy(tar)

        pred1[pred1==0]=2
        pred1[pred1==1]=0
        pred1[pred1==2]=1
        tar1[tar1==0]=2
        tar1[tar1==1]=0
        tar1[tar1==2]=1
        TN=(pred1 * tar1).sum()
        
        delta = 1e-10
        SP = (TN+delta) / (TN + FP+ delta)
        count = count + SP
   
    return count/batch
if __name__ =='__main__':

 
    a = np.array([[1,2,0],[4,0,1],[0,5,0]])
    
    print(a)
