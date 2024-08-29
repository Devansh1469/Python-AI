import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
imgs=cv.imread('path-of-photo')
img=cv.resize(imgs,(600,600),interpolation=cv.INTER_CUBIC)
cv.imshow('lady',img)
#haarcascade dont read colors it just uses edges to determine if there is a person or not
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray person',gray)

haar_cascades=cv.CascadeClassifier('haar_face.xml')
face_rect=haar_cascades.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
print(f'number of faces detected are {len(face_rect)}')
for(x,y,w,h) in face_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow('detected faces',img)
cv.waitKey(0)