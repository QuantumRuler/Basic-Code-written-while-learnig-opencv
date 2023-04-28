
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2 as cv


# In[2]:


#then we get the video feed and display the video 


# In[3]:


cap = cv.VideoCapture(0)
frame_width = int( cap.get(cv.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc('X','V','I','D')

out = cv.VideoWriter("output.avi", fourcc, 5.0, (1280,720))


# In[4]:


# now i would get one frame from the feed
ret, frame = cap.read()
imgGrey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
_, thrash = cv.threshold(imgGrey, 20, 255, cv.THRESH_BINARY)
contours, _ = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)


# In[5]:


# we will put a while loop to only run the further parts if video feed is open
while cap.isOpened():
    dump = 1
    # now i would get one frame from the feed
    ret, frame = cap.read()
    imgGrey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, thrash = cv.threshold(imgGrey, 125, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.01* cv.arcLength(contour, True), True)
        cv.drawContours(frame, [approx], 0, (0, 0, 0), 5)
        # now get the x and y cordinate 
        x = approx.ravel()[0]
        y = approx.ravel()[1] 
        if cv.contourArea(contour) < 4000:
            continue
        if len(approx) == 3:
            cv.putText(frame, "Triangle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif len(approx) == 4:
            # for 4 sides figure we need to distinguish square or quadrilatral
            x1 ,y1, w, h = cv.boundingRect(approx)
            aspectRatio = float(w)/h
           # print(aspectRatio)
            if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                cv.putText(frame, "square", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            else:
                cv.putText(frame, "rectangle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif len(approx) == 5:
            cv.putText(frame, "Pentagon", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        else:
            cv.putText(frame, "Circle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        
        image = cv.resize(frame, (1280,720))
        out.write(image)
        cv.imshow("feed", frame)
        if cv.waitKey(20) & 0xFF == ord('d'):
            dump = 0
            break
    if dump == 0:
        break
        
        #then break through the video



# In[6]:


cv.destroyAllWindows()
cap.release()
out.release()
