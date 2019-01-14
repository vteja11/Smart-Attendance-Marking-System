import cv2
import numpy as np
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cap=cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read('recognizer/trainingdata.yml');
fontface=cv2.FONT_HERSHEY_PLAIN;
fontscale=5
fontcolor=(255,255,255)

while(True):
    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        Id, conf = rec.predict(gray[y:y+h,x:x+w])
        cv2.putText(img,str(Id),(x,y+h),fontface,fontscale,fontcolor)
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
