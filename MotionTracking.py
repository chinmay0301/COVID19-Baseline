import cv2
import numpy as np
import imutils
import os
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# load video of person
cap = cv2.VideoCapture('p9_normal.MP4')
print("number of frames in video: ",int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print("Capture FPS rate: ", cap.get(cv2.CAP_PROP_FPS))
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.005,
                       minDistance = 5,
                       blockSize = 50)

lk_params = dict( winSize = (15,15),
                  maxLevel = 1,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0,255,(100,3))

# save dimensions of the first frame 
ret,old_frame = cap.read()
old_frame = imutils.rotate_bound(old_frame, 0)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(old_gray, 1.3, 5)
mask = np.zeros((old_gray.shape[0], old_gray.shape[1]),np.uint8)
for (x,y,w,h) in faces:
    # width - 50 % in middle, height 90% from top
    img = cv2.rectangle(old_frame, (int(x+w/4),y), (int(x+3*w/4),int(y+0.9*h)), (255,0,0,),2)
    mask[y + int(0.50*h):y + h, int(x+w/4):int(x + 3*w/4)] = 1
    mask[y:y + int(0.2*h), int(x+w/4):int(x + 3*w/4)] = 1
    print(mask.shape)
    print(old_frame.shape)
    old_crop_img = cv2.bitwise_and(old_frame, old_frame, mask=mask)
    # old_crop_img = old_frame[y + int(0.50*h):y + h, int(x+w/4):int(x + 3*w/4)]

old_gray_crop = cv2.cvtColor(old_crop_img, cv2.COLOR_BGR2GRAY)    

# select feature points on face to track
p0 = cv2.goodFeaturesToTrack(old_gray_crop, mask=None, **feature_params)

traj = []
traj.append(p0)
counter = 0
while(cap.isOpened()):
    ret, frame =  cap.read()
    if ret == True:
        counter = counter + 1
        frame = imutils.rotate_bound(frame, 0)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for (x,y,w,h) in faces:
            crop_img = cv2.bitwise_and(frame, frame, mask=mask)
            # crop_img = frame[y + int(0.50*h):y + h, int(x+w/4):int(x + 3*w/4)]
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_crop_img,crop_img,p0,None, **lk_params)
        traj.append(p1)
        gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)    
        
        for i in np.int0(p1):
            x,y = i.ravel()
            cv2.circle(crop_img,(x,y),3,255,-1)
        cv2.imshow('Frame', crop_img)


        if cv2.waitKey(25) & 0xFF  == ord('q'):
            break
            
    # elif counter == 2700:
    #     break
    else:
        break

cap.release()
np.save('extracted_files/traj_p9.npy', traj)
traj = np.squeeze(np.array(traj))
cv2.destroyAllWindows()
for i in range(traj.shape[1]):
    plt.plot(traj[:,i,1])
plt.show()
