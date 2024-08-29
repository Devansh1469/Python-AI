import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
imgs=cv.imread('photos/boston.jpg')
img=cv.resize(imgs,(600,600),interpolation=cv.INTER_CUBIC)
cv.imshow('cat',img)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)
blank=np.zeros(img.shape[:2],dtype='uint8')
mask=cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),100,255,-1)
masked=cv.bitwise_and(gray,gray,mask=mask)
cv.imshow('masked',masked)
#grayscale historgram
gray_hist=cv.calcHist([gray],[0],mask,[256],[0,256])

plt.figure()
plt.title('grayscale histogram')
plt.xlabel('bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()

#color_histogram
plt.figure()
plt.figure()
plt.title('grayscale histogram')
plt.xlabel('bins')
plt.ylabel('# of pixels')
colors=('b','g','r')
for i,col in enumerate(colors):
    hist=cv.calcHist([imgs],[i],None,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])
plt.show()


cv.waitKey(0)
