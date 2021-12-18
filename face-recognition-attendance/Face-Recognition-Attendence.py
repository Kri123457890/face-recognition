import cv2 
import numpy as np   
import face_recognition 
import os 
from datetime import datetime 
import pyodbc  

path='trainimg' 
images=[]   
imgLabel=[]   
mylst=os.listdir(path) 
#print(mylst)

for cl in mylst:
    curimg=cv2.imread(f'{path}\\{cl}')   
    images.append(curimg)  
    imgLabel.append(os.path.splitext(cl)[0])  

def findEncodings(images): 
    encodLst=[]    
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)    
        encode=face_recognition.face_encodings(img)[0]   
        encodLst.append(encode)    
    return encodLst   

encodlstKnowFaces=findEncodings(images)   


def markAttendance(name):     
    with open('Attendence.csv','r+') as f:    
        myDataList = f.readlines()   
        nameList = []   
        for line in myDataList:
            entry = line.split(',') 
            nameList.append(entry[0]) 
        if name not in nameList:
            now = datetime.now()   
            dtString = now.strftime('%H:%M:%S')  
            f.writelines(f'\n{name},{dtString}')  

webcam=cv2.VideoCapture(0) 


while True:
    success, img=webcam.read()  
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)   
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)  

    faceCurFrm= face_recognition.face_locations(imgS)   
    encodeCurFrm=face_recognition.face_encodings(imgS,faceCurFrm) 

    for encodFace, faceLocation in zip(encodeCurFrm,faceCurFrm):
        maches=face_recognition.compare_faces(encodlstKnowFaces,encodFace) 
        faceDis=face_recognition.face_distance(encodlstKnowFaces,encodFace) 
        
        machesIndex=np.argmin(faceDis)   

        if maches[machesIndex]:
            name = imgLabel[machesIndex].upper()
            # print(name)
            y1,x2,y2,x1=faceLocation  
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4   
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3) 
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED) 
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX ,1,(255,255,255),2) 
            markAttendance(name)
    
    cv2.imshow('Frame',img)  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

webcam.release() 
cv2.destroyAllWindows() 