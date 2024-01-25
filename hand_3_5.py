import cv2
import math
import numpy
import random
import time
from hand_1_6 import HandDetector
from hand_1_7 import put_text_rectangle
webcam = cv2.VideoCapture(0)
WEBCAM_WIDTH = 1_280
WEBCAM_HEIGHT = 720
webcam.set(3, WEBCAM_WIDTH)
webcam.set(4, WEBCAM_HEIGHT)
detector = HandDetector(min_detection_confidence = 0.8,max_num_hands = 1)
raw_distance = [450, 400, 360, 300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
cm_interpolation = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coefficients = numpy.polyfit(x = raw_distance,y = cm_interpolation,deg = 2)
game_x = 250
game_y = 250
counter = 0
score = 0
colour = PURPLE = (255, 0, 255)
GREEN = (0, 255, 0)
time_start = time.time()
game_time = 30
while True:
    response, frame = webcam.read()
    frame = cv2.flip(frame,flipCode = 1)  
    if time.time() - time_start < game_time:
        list_hands, _ = detector.find_hands(frame = frame,draw = False)
        if list_hands != None and list_hands != []:
            list_landmarks = list_hands[0]['list_landmarks']
            x_1, y_1, _ = list_landmarks[5]
            x_2, y_2, _ = list_landmarks[17]
            distance_raw = int(math.sqrt((y_2 - y_1) ** 2 + (x_2 - x_1) ** 2))            
            w_1, w_2, bias = coefficients
            distance_estimate = (w_1 * distance_raw ** 2) \
                + (w_2 * distance_raw) \
                + bias
            x, y, width, height = list_hands[0]['box_bounding']
            if distance_estimate < 40:
                if x < game_x < x + width and y < game_y < y + height:
                    counter = 1
            cv2.rectangle(frame,(x, y),(x + width, y + height),PURPLE,3)            
            put_text_rectangle(frame = frame,text = f'{int(distance_estimate)} cm',position = (x + 5, y - 10))
        if counter != 0:
            counter += 1
            colour = GREEN
            if counter == 3:
                game_x = random.randint(100, WEBCAM_WIDTH - 100)
                game_y = random.randint(100, WEBCAM_HEIGHT - 100)
                colour = PURPLE
                score += 1
                counter = 0
        cv2.circle(frame,(game_x, game_y),30,colour,cv2.FILLED)
        cv2.circle(frame,(game_x, game_y),30,(50, 50, 50),2)
        cv2.circle(frame,(game_x, game_y),20,(255, 255, 255),2)
        cv2.circle(frame,(game_x, game_y),10,(255, 255, 255),cv2.FILLED)
        put_text_rectangle(frame = frame,text = f'Time: {int(game_time - (time.time() - time_start)):02d}',position = (1_000, 75),scale = 3,offset = 20)
        put_text_rectangle(frame = frame,text = f'Score: {score:02d}',position = (60, 75),scale = 3,offset = 20)
    else:
        put_text_rectangle(frame = frame,text = 'Game Over',position = (400, 400),scale = 5,offset = 30,thickness = 7)
        put_text_rectangle(frame = frame,text = f'Your Score: {score:02d}',position = (450, 500),scale = 3,offset = 20,)
        put_text_rectangle(frame = frame,text = f'Press "R" to Restart',position = (460, 575),scale = 2,offset = 10,)
    cv2.imshow('Webcam', frame)
    key = cv2.waitKey(1)
    if key == ord('r'):
        time_start = time.time()
        score = 0