import numpy as np
import cv2 as cv
import os
haar_cascades=cv.CascadeClassifier('haar_face.xml')
features=np.load('features.npy',allow_pickle=True)
labels=np.load('labels.npy',allow_pickle=True)
people=[]
for i in os.listdir(r'path-of-storage-folder-of-pics'):
    people.append(i)
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_recognizer.yml')
img=cv.imread(r'photo-you-want-to-recognise')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('person',gray)

#detect the face
faces_rect=haar_cascades.detectMultiScale(gray,1.1,4)

for(x,y,w,h) in faces_rect:
    faces_roi=gray[y:y+h,x:x+w]
    label,confidence=face_recognizer.predict(faces_roi)
    print(f'Label={people[label]}with confidence of {confidence}')
    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow('Detected face',img)
cv.waitKey(0)