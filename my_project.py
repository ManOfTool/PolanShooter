import cv2
import sys
import numpy as np
from collections import deque
import random

pts = deque(maxlen=100)

monsters = []
bullets = []


class Monster:
    def __init__(self, m_x, m_y, m_speed, m_logo):
        self.m_x = m_x
        self.m_y = m_y
        self.m_speed = m_speed
        self.m_logo = m_logo
        self.m_rows = m_logo.shape[0]
        self.m_cols = m_logo.shape[1]
        self.live = True
    def move(self):
        self.m_x = self.m_x + self.m_speed

class Bullet:
    def __init__(self, b_x, b_y, b_speed):
        self.b_x = b_x
        self.b_y = b_y
        self.b_speed = b_speed
        self.valid = True
    def move(self):
        self.b_x = self.b_x - self.b_speed

def InsertLogo(img1, img2, ix, iy):
    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1[ix:ix+rows, iy:iy+cols ]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, imask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(imask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = imask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[ix:ix+rows, iy:iy+cols ] = dst
    return img1

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

lower_blue = np.array([100, 50, 50], dtype=np.uint8)
upper_blue = np.array([140,255,255], dtype=np.uint8)

video_capture = cv2.VideoCapture(0)
get_face = False
Alive = 50
killed = 0

player_x = 450
player_y = 80

loose_logo = cv2.imread('loose.jpg')
win_logo = cv2.imread('win.jpg')

E1 = cv2.imread('E1.jpg')
E1 = cv2.resize(E1, (100,100))
E1_rows, E1_cols, E1_ch = E1.shape
for i in range(15):
    y = random.randint(0, 640)
    rand_speed = random.randint(1, 15)
    monsters.append(Monster(0, y, rand_speed, E1))


while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    scene = np.zeros((frame.shape[0] + 350,frame.shape[1] + 100,3), np.uint8)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 5)

    # Draw a rectangle around the faces
    if(get_face == False):
        if (len(faces) > 0):
            (x,y,h,w) = faces[0]
            your_face = frame[y:y+h, x:x+w]
            your_face = cv2.resize(your_face, (100, 100))

    else: # Face get!!! Game start!!!
        if Alive > 0 and killed != len(monsters):

            # Collision of enemys and player
            for i in range(len(monsters)):
                if monsters[i].live == True:
                    if player_x <= (monsters[i].m_x + monsters[i].m_rows) and player_x >= monsters[i].m_x:
                        if player_y <= (monsters[i].m_y + monsters[i].m_cols) and player_y >= monsters[i].m_y:
                            Alive -= 1
                            print(Alive)
                            break

                    if player_x + 100 <= (monsters[i].m_x + monsters[i].m_rows) and player_x + 100 >= monsters[i].m_x:
                        if player_y <= (monsters[i].m_y + monsters[i].m_cols) and player_y >= monsters[i].m_y:
                            Alive -= 1
                            print(Alive)
                            break

                    if player_x <= (monsters[i].m_x + monsters[i].m_rows) and player_x >= monsters[i].m_x:
                        if player_y + 100 <= (monsters[i].m_y + monsters[i].m_cols) and player_y + 100>= monsters[i].m_y:
                            Alive -= 1
                            print(Alive)
                            break

                    if player_x + 100 <= (monsters[i].m_x + monsters[i].m_rows) and player_x + 100 >= monsters[i].m_x:
                        if player_y + 100 <= (monsters[i].m_y + monsters[i].m_cols) and player_y + 100 >= monsters[i].m_y:
                            Alive -= 1
                            print(Alive)
                            break

            cv2.imshow('your_face', your_face)
            InsertLogo(scene, your_face, player_x, player_y)
            
            # Bullets create & movement
            if k == ord(' '):
                bullets.append(Bullet(player_x,player_y+50,20))

            for j in range(0, len(bullets)):
                if len(bullets) > 0:
                    if bullets[j].valid == True:
                        cv2.circle(scene,(bullets[j].b_y,bullets[j].b_x),6,(0,255,0),-1)
                        bullets[j].move()
                        if bullets[j].b_x < 0:
                            bullets[j].valid = False

                        # # Bullets hit enemys
                        for i in range(len(monsters)):
                            if len(monsters) > 0:
                                if bullets[j].b_x + 3 <= (monsters[i].m_x + monsters[i].m_rows) and bullets[j].b_x + 3 >= monsters[i].m_x:
                                    if bullets[j].b_y + 3 <= (monsters[i].m_y + monsters[i].m_cols) and bullets[j].b_y + 3 >= monsters[i].m_y:
                                        if monsters[i].live == True:
                                            monsters[i].live = False
                                            bullets[j].valid = False
                                            killed += 1

            if len(bullets) > 0:
                if bullets[0].valid == False:
                    del bullets[0]

            #print(len(bullets))

            # Enemy movement
            for i in range(len(monsters)):
                if monsters[i].live == True:
                    InsertLogo(scene, monsters[i].m_logo, monsters[i].m_x, monsters[i].m_y)
                    monsters[i].move()
                    if(monsters[i].m_x + monsters[i].m_logo.shape[0] >= scene.shape[0]):
                        monsters[i].m_x = 0

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask = cv2.erode(mask, kernel, iterations = 2)
            mask = cv2.dilate(mask, kernel, iterations = 2)

            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None
            # only proceed if at least one contour was found
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((ny, nx), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
         
                player_x = (int)(nx+250)
                player_y = (int)(ny)

            pts.appendleft(center) 

            # loop over the set of tracked points
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue

            res = cv2.bitwise_and(frame,frame, mask= mask)

        elif killed == len(monsters):
            InsertLogo(scene, win_logo, 200, 100)
        elif Alive <= 0:
            InsertLogo(scene, loose_logo, 150, 50)

    cv2.imshow('Video', frame)
    cv2.imshow('Scene', scene)
    cv2.imshow('E1', E1)

    # Press 'm' to capture face
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        get_face = not get_face
    elif k == 27:
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()