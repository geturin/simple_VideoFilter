from src.segmentation import *
from src.monodepth import *
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('-filter_model')
parser.add_argument('-background')
parser.add_argument('-filter_target')




args = parser.parse_args() 

if args.filter_model=="segmentation":
    img_filter=segmentation_filter()
    cap = cv2.VideoCapture(0)
    bg = cv2.imread(args.background)

    while True:
        ret,frame=cap.read()
        cv2.imshow("origin",frame)
        img_filter.get_filter_img(frame,bg,int(args.filter_target))
        cv2.imshow("1",frame)
        cv2.waitKey(1)
elif args.filter_model=="depth":
    run(args.background,int(args.filter_target))

else:
    pass