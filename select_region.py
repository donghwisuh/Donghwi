# import the necessary packages
import argparse
from subprocess import REALTIME_PRIORITY_CLASS
import cv2
import config
import time

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
select_ready = False


def click_and_crop(event, x, y, flags, image):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)
        cv2.putText(image, "length = {} pixels".format(refPt[1][0] - refPt[0][0]), refPt[0], cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

def select_region(cap):

    if(config.real_time == 0): 
        select_ready = True
        ret, image = cap.read()
        #video_name = config.video_name

    elif(config.real_time == 1):
        video_name = 0 #use webcam
        
        #cap = cv2.VideoCapture(video_name)
        while True:
            ret, image = cap.read()
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, "press esc when ready", (10,450), font, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow("webcam image", image)
            k = cv2.waitKey(33)
            if(k == 27):#is press esc
                select_ready = True
                break

    if(select_ready):
        #cap = cv2.VideoCapture(video_name)
        #ret, image = cap.read()
        #image = cv2.imread('computerimage.jpg')
        clone = image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_and_crop, image)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", image)
            key = cv2.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                image = clone.copy()
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break
        # if there are two reference points, then crop the region of interest
        # from the image and display it
        if len(refPt) == 2:
            roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            cv2.imshow("ROI", roi)
            cv2.waitKey(0)
            config.box_start = refPt[0]
            config.box_end = refPt[1]
        # close all open windows
        
        
        cv2.destroyAllWindows()


testing = 0
if(testing):
    cap1 = cv2.VideoCapture(0)
    select_region(cap1)
