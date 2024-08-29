import numpy as np
import cv2 as cv
import os
haar_cascades=cv.CascadeClassifier('haar_face.xml')
features=np.load('features.npy',allow_pickle=True)
labels=np.load('labels.npy',allow_pickle=True)
people=[]
for i in os.listdir(r'folder'):
    people.append(i)
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_recognizer.yml')
webcam=cv.VideoCapture(0)
unlock=False
while True:
    _,img=webcam.read()
    img=cv.flip(img,1)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces_rect=haar_cascades.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces_rect:
        faces_roi=gray[y:y+h,x:x+w]
        label,confidence=face_recognizer.predict(faces_roi)
        if people[label] == 'your name':
            cv.putText(img,f'{people[label]}-Unlocked',(x,y-10),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),thickness=2)
            print("Unlocked")
            unlock=True
        else:
            cv.putText(img,str("Unknown person-Access Denied"),(20,20),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),thickness=2)
            print("Access Denied")
            unlock=False
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv.imshow("Face Lock", img)
        cv.waitKey(2000)
        if unlock:
            break

    if unlock:
        break
    if cv.waitKey(10)==27:
        break
webcam.release()
cv.destroyAllWindows()