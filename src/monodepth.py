import cv2
import torch
import numpy as np
from src.segmentation import *


def run(background_path:str,filter_target:int):
    model_type =  "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform


    cap = cv2.VideoCapture(0)
    bg = cv2.imread(background_path)
    video_filter=filter_with_mask(0.05)
    while True:
        ret,frame=cap.read()
        cv2.imshow("origin",frame)
        input_batch = transform(frame).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output_msk = 1/prediction.cpu().numpy()
        video_filter.filter_img(frame,bg,output_msk,filter_target)
        cv2.imshow("filter",frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    run()
