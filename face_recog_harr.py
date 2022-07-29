import numpy as np
import cv2
import pickle
import os
import psutil
import time
from datetime import timedelta

start_time = time.time()

#Harr Casscade
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("./pickles/face-labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}

video_capture = cv2.VideoCapture("./video/BlackpinkTiktok.mp4")

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
    
        if conf >= 50 :
            print(id_)
            print(labels[id_])
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

       

        color = (255, 0, 0) #BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    cv2.imshow('LBPHFaceRecognizer',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#Execution time
elapsed_time_secs = time.time() - start_time
msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
print(msg)
print('The CPU usage is: ', psutil.cpu_percent(4))
print('RAM memory % used:', psutil.virtual_memory()[2])

video_capture.release()
cv2.destroyAllWindows()