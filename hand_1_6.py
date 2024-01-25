import cv2
import mediapipe
import time
import math
class HandDetector():
    def __init__(self,static_image_mode = False,max_num_hands = 2,min_detection_confidence = 0.5,min_tracking_confidence = 0.5,):
        self.mp_hands = mediapipe.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode = static_image_mode,max_num_hands = max_num_hands,min_detection_confidence = min_detection_confidence,min_tracking_confidence = min_tracking_confidence)
        self.mp_draw = mediapipe.solutions.drawing_utils
    def find_hands(self,frame,draw = True,flip = True,):
        frame_RGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_RGB)
        list_hands = []
        height, width, channels = frame.shape
        if self.results.multi_hand_landmarks != None:
            for each_hand_type, each_hand_landmarks in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                dict_each_hand = {}
                list_landmarks = []
                list_x = []
                list_y = []
                for each_index, each_landmark in enumerate (each_hand_landmarks.landmark):
                    point_x = int(each_landmark.x * width)
                    point_y = int(each_landmark.y * height)
                    point_z = int(each_landmark.z * width)
                    list_landmarks.append([point_x, point_y, point_z])
                    list_x.append(point_x)
                    list_y.append(point_y)
                min_x = min(list_x)
                max_x = max(list_x)
                min_y = min(list_y)
                max_y = max(list_y)
                box_width = max_x - min_x
                box_height = max_y - min_y
                box_bounding = min_x, min_y, box_width, box_height
                centre_x = box_bounding[0] + (box_bounding[2] // 2)
                centre_y = box_bounding[1] + (box_bounding[3] // 2)
                dict_each_hand['list_landmarks'] = list_landmarks
                dict_each_hand['box_bounding'] = box_bounding
                dict_each_hand['centre'] = (centre_x, centre_y)
                if flip == True:
                    if each_hand_type.classification[0].label.lower() == 'right':
                        dict_each_hand['type'] = 'left'
                    else:
                        dict_each_hand['type'] = 'right'
                else:
                    dict_each_hand['type'] = each_hand_type.classification[0].label.lower()
                list_hands.append(dict_each_hand)
                if draw == True:
                    self.mp_draw.draw_landmarks(frame,each_hand_landmarks,self.mp_hands.HAND_CONNECTIONS)
                    cv2.rectangle(frame,(int(box_bounding[0] - 20),int(box_bounding[1] - 20)),(int(box_bounding[0] + box_bounding[2] + 20),int(box_bounding[1] + box_bounding[3] + 20)),(255, 0, 255),2,)
                    cv2.putText(frame,str(dict_each_hand['type']),(int(box_bounding[0] - 30), int(box_bounding[1] - 30)),cv2.FONT_HERSHEY_PLAIN,2,(255, 0, 255),2,)
        return list_hands, frame
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
    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        """
        Find the distance between two landmarks input should be (x1,y1) (x2,y2)
        :param p1: Point1 (x1,y1)
        :param p2: Point2 (x2,y2)
        :param img: Image to draw output on. If no image input output img is None
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)
        return length, info, img
if __name__ == '__main__':
    webcam = cv2.VideoCapture(1)
    time_previous = 0
    time_current = 0
    detector = HandDetector()
    while True:
        response, frame = webcam.read()
        frame = detector.find_hands(frame = frame)
        list_landmarks = detector.find_position(frame = frame)
        time_current = time.time()
        fps = 1 / (time_current - time_previous)
        time_previous = time_current
        cv2.putText(frame,str(int(fps)),(10, 70),cv2.FONT_HERSHEY_PLAIN,3,(255, 0, 255),3)
        cv2.imshow('Webcam', frame)
        cv2.waitKey(1)