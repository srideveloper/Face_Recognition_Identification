import numpy as np
import cv2
import pickle


facecascade = cv2.CascadeClassifier("Data/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
# original_label = {}
# labels = {"personName": 1}
with open("label.pickle",'rb') as f:
    original_label = pickle.load(f)
    labels = {v:k for v,k in original_label.items()}
cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()    

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(frame, 
                                 scaleFactor=1.3, 
                                 minNeighbors= 4, 
                                 minSize=(30, 30))

    for x,y,w,h in faces:
        # print(x,y,w,h)
        # frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)        
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf>=30 and conf <=80:
            print(id_)
            print(labels[id_])
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    

cap.release()
cv2.destroyAllWindows()