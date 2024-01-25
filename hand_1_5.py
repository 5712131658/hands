import cv2
import mediapipe
import time
import math
class HandDetector():
    def __init__(self,static_image_mode = False,max_num_hands = 2,min_detection_confidence = 0.5,min_tracking_confidence = 0.5,):
        self.mp_hands = mediapipe.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode = static_image_mode,max_num_hands = max_num_hands,min_detection_confidence = min_detection_confidence,min_tracking_confidence = min_tracking_confidence)
        self.mp_draw = mediapipe.solutions.drawing_utils
    def find_hands(self,frame,draw = True):
        frame_RGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_RGB)
        if self.results.multi_hand_landmarks != None:
              for each_hand in self.results.multi_hand_landmarks:
                    if draw == True:
                        self.mp_draw.draw_landmarks(frame,each_hand,self.mp_hands.HAND_CONNECTIONS)
        return frame
    def find_position (self,frame,hand_number = 0,draw = False,):
        list_landmarks = []
        if self.results.multi_hand_landmarks != None:
            hand = self.results.multi_hand_landmarks[hand_number]
            for each_id, each_landmark in enumerate(hand.landmark):
                height, width, channels = frame.shape
                centre_x = int(each_landmark.x * width)
                centre_y = int(each_landmark.y * height)
                list_landmarks.append([each_id, centre_x, centre_y])
                if draw == True:
                    cv2.circle(frame,(centre_x, centre_y),15,(255, 0, 255),cv2.FILLED)
        return list_landmarks
    def find_distance(self,point_1,point_2,frame = None,colour = (255, 0, 255),scale = 15):
        x_1, y_1 = point_1 
        x_2, y_2 = point_2 
        centre_x = (x_1 + x_2) // 2 
        centre_y = (y_1 + y_2) // 2 
        length = math.hypot(x_2 - x_1, y_2 - y_1) 
        if frame is not None: 
            cv2.circle(frame, (x_1, y_1), scale, colour, cv2.FILLED) 
            cv2.circle(frame, (x_2, y_2), scale, colour, cv2.FILLED) 
            cv2.circle(frame, (centre_x, centre_y), scale, colour, cv2.FILLED) 
            cv2.line(frame, (x_1, y_1), (x_2, y_2), colour, max(1, scale // 3)) 
        return length, frame
if __name__ == '__main__':
    webcam = cv2.VideoCapture(0)
    time_previous = 0
    time_current = 0
    detector = HandDetector()
    while True:
        webcam = cv2.VideoCapture(0)
        response, frame = webcam.read()
        frame = detector.find_hands(frame = frame)
        list_landmarks = detector.find_position(frame = frame)
        time_current = time.time()
        fps = 1 / (time_current - time_previous)
        time_previous = time_current
        cv2.putText(frame,str(int(fps)),(10, 70),cv2.FONT_HERSHEY_PLAIN,3,(255, 0, 255),3)
        cv2.imshow('Webcam', frame)
        cv2.waitKey(1)