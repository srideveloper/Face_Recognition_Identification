import os
from PIL import Image
import numpy as np
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
image_dir = os.path.join(BASE_DIR,"Trainy_Images")

x_trains = []
y_label = []
current_ids = 0
label_ids_dict = {}


facecascade = cv2.CascadeClassifier("Data/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

for root,dir1,files in os.walk(image_dir):
    for file in files:
        if file.endswith("jfif") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.abspath(root)).replace(" ","-").lower()
            # print(label,path)
            
#             Creating label ide with dictionary
            if not label in label_ids_dict:
                label_ids_dict[label] = current_ids
                current_ids+=1
            id_ = label_ids_dict[label]
            # print(label_ids_dict)
                        
                            
            pil_image = Image.open(path).convert("L")
            Image_array = np.array(pil_image,"uint8")
#             print(Image_array)
            
            faces = facecascade.detectMultiScale(Image_array, 
                                 scaleFactor=1.3, 
                                 minNeighbors= 4, 
                                 minSize=(30, 30))
        
            for x,y,w,h in faces:
#                 print(x,y,w,h)
                ROI = Image_array[y:y+h,x:x+w]  
                x_trains.append(ROI)
                y_label.append(id_)
with open("label.pickle","wb") as f:
    pickle.dump(label_ids_dict,f)
recognizer.train(x_trains,np.array(y_label))
recognizer.save("trainer.yml")