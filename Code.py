#code for raspi pi 10/04

import numpy as np
import cv2
import math
from adafruit_servokit import ServoKit
from time import sleep
import time


kit = ServoKit(channels=16)  # initialise the servo driver
servo = 3  # the number of channels being used

rt = 1.73
r =  96 #guessed radius of plate in pixels

#factor value for P and D 
yp = 500 # previous y values .initialized value = 500
xp = 500 # previous y values .initialized value = 500
v = 1 #initialized value 
k = 0.45
st = 1  #time spent stationary at a moment
t =  1 #no.of frames for speed calc
f = 1 #st factor


def nothing(x):
    pass

def disp(f, m, r):
    cv2.imshow('stream', f)
    cv2.imshow('mask', m)
    cv2.imshow('result', r)
    
# defining a kernel to perform opening on the mask
kernel = np.ones((5, 5), np.uint8)

capture = cv2.VideoCapture(0)

'''
# create a window with trackbars to set upper & lower bounds for color detection
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

# picking the color to track
while True:

    status, frame = capture.read()

    frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # convert frame from a BGR image to an HSV image

    # read values from the trackbars
    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")
    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    # lower and upper bounds for color detection
    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    # create an image mask
    mask = cv2.inRange(hsv, l_b, u_b)

    result = cv2.bitwise_and(frame, frame, mask=mask)
    result = cv2.circle(result, (int(result.shape[1] / 2), int(result.shape[0] / 2)), int(result.shape[0]/ 2) - 1, (252, 3, 248), 1)
    result = cv2.line(result, (int(result.shape[1]/2), int(result.shape[0]/2)), (0, int(result.shape[0]/2)), (252, 3, 248), 1)
    
    frame = cv2.circle(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), int(frame.shape[0] / 2) - 1, (252, 3, 248), 1)
    frame = cv2.line(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), (0, int(frame.shape[0]/2)), (252, 3, 248), 1)

    disp(frame, mask, result)
    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()'''

# print(l_b)
# print(u_b)
# 
l_b = np.array([0,0,0])
u_b = np.array([20,255,255])

d1 = 0.001
d2 = 0.001
d3 = 0.001

# color tracking
while True:
    
    status, frame = capture.read()

    frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # convert frame from a BGR image to an HSV image

    # create an image mask and open (erosion then dilation) it
    mask = cv2.morphologyEx(cv2.inRange(hsv, l_b, u_b), cv2.MORPH_OPEN, kernel, iterations=1)

    # result = cv2.bitwise_and(frame, frame, mask=mask)

    # calculate the moments of the mask
    M = cv2.moments(mask)
    
    # calculate x,y coordinate of the centroid of the image mask
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # put text and highlight the center
    # cv2.circle(mask, (cX, cY), 5, (0, 0, 0), -1)
    # cv2.putText(mask, "(" + str(cX-(mask.shape[1]/2)) + "," + str(-(cY-(mask.shape[0]/2))) + ")", (cX-50, cY-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # change the coordinates inorder to make the center of the image the center of the coordinate plane
    cX -= mask.shape[1] / 2
    cY = - (cY - (mask.shape[0] / 2))
    
    if math.sqrt(cX**2+cY**2) > r:
        d1 = 0.0001
        d2 = 0.0001
        d3 = 0.0001
        st = 0
    elif cY > rt*cX and cY > -rt*cX:
        d1 =   cY/rt - cX
        d2 = 0.001
        d3 = -cY/rt -  cX
    elif  0 < cY < rt*cX: 
        d1 = (cY*2)/rt
        d2 = - cX + cY/rt
        d3 = 0.001
    elif -rt*cX < cY < 0:
        d1 = 0.001
        d2 = - cX - cY/rt
        d3 = -(cY*2)/rt
    elif cY < rt*cX and cY < -rt*cX : 
        d1 =  cY/rt +cX
        d2 = 0.001
        d3 = -cY/rt + cX
    elif rt*cX < cY < 0:
        d1 = (cY*2)/rt
        d2 = -cX + cY/rt
        d3= 0.001
    elif 0 < cY < -rt*cX:
        d1 = 0.001
        d2 = -cX - cY/rt
        d3 = -(cY*2)/rt
        
    if xp != 500 and yp != 500:
        v = math.sqrt((xp-cX)**2+(yp-cY)**2)/t #-- calculating v
        if v == 0.0:
            v=1 # avoiding zero error
        if xp == cX and yp == cY and cX != -128 and cY != 96:
            st = st + (1/f) # accumaltion of time spent stationary
        else:
            st = 1
    
    yp = cY
    xp = cX


    t01 = ((d1*k*st)/v)
    t02 = ((d2*k*st)/v)
    t03 = ((d3*k*st)/v)

    t1 = int(((t01+r)/r)*31.5)+38
    t2 = int(((t02+r)/r)*32)+39-1
    t3 = int(((t03+r)/r)*31)+34
#     t1 = int(((t01+r)/r)*15.75)+85.25
#     t2 = int(((t02+r)/r)*16)+87
#     t3 = int(((t03+r)/r)*15.5)+80.5

    if t1 > 101:
        t1 = 101
    elif t1 < 38:
        t1 = 38
    if t2 > 103: 
        t2=71
    elif t2 < 39:
        t2 = 39
    if t3 > 96:
        t3=65
    elif t3 < 34:
        t3 = 34

# 
#     if t1 > 101:
#         t1 = 101
#     elif t1 < 85.25:
#         t1 = 85.25
#     if t2 > 103: 
#         t2=71
#     elif t2 < 87:
#         t2 = 87
#     if t3 > 96:
#         t3=65
#     elif t3 < 80.5:
#         t3 = 80.5

        
#     print("d1 ",int(d1), "\t\t" , "t01" , int(t01) , "\t\t" , "t1" , int(t1))
#     print("d2 ",int(d2), "\t\t" , "t02" , int(t02) , "\t\t" , "t2" , int(t2))
#     print("d3 ",int(d3), "\t\t" , "t03" , int(t03) , "\t\t" , "t3" , int(t3))
# 
#     print("x,y: ", cX, cY)
#     print("st",st)  
#     print("speed",v)
#     print('factor',((d1*k*st)/v))

    kit.servo[4].angle=int(int(t1)*18/12)
    kit.servo[5].angle=int(int(t2)*18/12)
    kit.servo[6].angle=int(int(t3)*18/12)

    # frame = cv2.circle(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), int(frame.shape[0] / 2) - 1, (0, 0, 0), 3)
    # mask = cv2.circle(mask, (int(mask.shape[1] / 2), int(mask.shape[0] / 2)), int(mask.shape[0] / 2) - 1, (255, 255, 255), 3)
    # result = cv2.circle(result, (int(result.shape[1] / 2), int(result.shape[0] / 2)), int(result.shape[0] / 2) - 1, (0, 0, 0), 3)

    # disp(frame, mask, result)
    
    cv2.imshow('mask', mask)
    
    #time.sleep(1)
    
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
capture.release()


