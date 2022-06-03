from src.segmentation import *

img_filter=segmentation_filter()

cap = cv2.VideoCapture(0)
bg = cv2.imread("images/universe.png")
while True:
    ret,frame=cap.read()
    cv2.imshow("origin",frame)
    img_filter.get_filter_img(frame,bg,1)
    cv2.imshow("1",frame)
    cv2.waitKey(1)
