import cv2
import mediapipe as mp
import numpy as np

class segmentation_filter(object):
    def __init__(self) -> None:
        self.filter = mp.solutions.selfie_segmentation
        return
    def get_mask(self):
        with self.filter.SelfieSegmentation(model_selection=1) as filter:
            result = filter.process(self.img)
        self.mask=result.segmentation_mask
    
    def get_filter_img(self,target_img,bg_img,filter_model):
        self.img=target_img
        if isinstance(bg_img, np.ndarray):
            self.bg=bg_img
        else:
            self.bg=cv2.GaussianBlur(self.img,(0,0),15)
        self.get_mask()

        if filter_model==0:
            self.img[np.where(self.mask<0.6)]=self.bg[np.where(self.mask<0.6)]
        else:
            self.img[np.where(self.mask>0.6)]=self.bg[np.where(self.mask>0.6)]

        return self.img

class filter_with_mask(object):
    def __init__(self,threshold) -> None:
        self.threshold=threshold
        pass
    def filter_img(self,target_img,bg_img,mask,filter_model):
        self.img=target_img
        self.mask=mask
        if isinstance(bg_img, np.ndarray):
            self.bg=bg_img
        else:
            self.bg=cv2.GaussianBlur(self.img,(0,0),15)

        if filter_model==0:
            self.img[np.where(self.mask<self.threshold)]=self.bg[np.where(self.mask<self.threshold)]
        else:
            self.img[np.where(self.mask>self.threshold)]=self.bg[np.where(self.mask>self.threshold)]

        return self.img
        
