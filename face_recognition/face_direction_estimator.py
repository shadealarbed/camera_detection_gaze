import cv2
import dlib
import numpy as np
from .direction_detection import DirectionDetector
from .light_detection import LightDetector
from .face_counter import FaceCounter
from .depth_detection import DepthDetector
from .roi_face import FaceROIDetector
import time
import psutil

class FaceDirectionEstimator:
    def __init__(self, shape_predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        self.threshold = 10
        self.margin = 100
        self.threshold_value = 250
        self.threshold_value_dark = 10
        self.depth = False
        self.face_roi_detector = FaceROIDetector()
        self.direction_detector = DirectionDetector()
        self.light_detector = LightDetector()
        self.face_counter = FaceCounter()
        self.depth_detector = DepthDetector()  # Create an instance of DepthDetector
        self.green_color = (0, 255, 0)
        self.red_color = (0, 0, 255)
        
    def screen_text(self, frame, text, position,color = (0,0,255)):
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    
    def get_landmarks(self,gray, face):
        landmarks1 = self.predictor(gray, face)
        return landmarks1, np.array([[point.x, point.y] for point in landmarks1.parts()])
    
    def measure_depth(self, frame, landmarks):
        color = self.red_color
        
        if self.depth_detector.calculate_depth(landmarks) <= 60:
            message = "too far"
        elif self.depth_detector.calculate_depth(landmarks) >= 75:
            message = "too close"
        else:
            message = "good"
            self.depth = True
            color = self.green_color
            
        text = f"Distance to camera is {message}: {self.depth_detector.calculate_depth(landmarks):.2f}"
        self.screen_text(frame, text, (10,60), color)
    
    def measure_brightness(self, frame, landmarks):
        _, thresholded_face, thresholded_face_low = self.face_roi_detector.detect_roi_face_light(frame, landmarks)
        bright_pixel_count = np.sum(thresholded_face == 255)
        # bright_pixel_count_low = np.sum(thresholded_face_low == 255) 
        color = self.red_color
        self.light = False
        print(bright_pixel_count)
        
        if bright_pixel_count >= 4000:
            message =  "too high brightness"
        # elif bright_pixel_count_low >= 2000:
        #     message = "too low brightness"
        else:
            message =  "good brightness"
            self.light = True
            color = self.green_color
            
        self.screen_text(frame, message,(500, 30), color)
    
    def detect_direction(self, frame, landmarks, landmarks1):
        
        center_x = np.mean(landmarks[58:64, 0])
        direction = self.direction_detector.estimate_face_direction(
            left_landmarks=landmarks[landmarks[:, 0] < center_x],
            right_landmarks=landmarks[landmarks[:, 0] >= center_x],
            landmarks1=landmarks1
        )
        self.screen_text(
            frame,
            direction,
            (10, 30),
            self.green_color if direction == "Center" else self.red_color
        )
        return direction

    def draw_landmarks(self,frame,landmarks):
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 2, self.red_color, -1)
 
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.detector(gray)
    
################ for run function ####################
    def get_roi_capture(self,frame,frame_copy,landmarks):
        return self.face_roi_detector.detect_roi_image_capture(frame_copy, landmarks) ,self.face_roi_detector.detect_roi_image_capture(frame, landmarks)

    def condition_capture(self,direction,frame,frame_copy,landmarks):
        roi_image_capture , roi_image_capture1 = self.get_roi_capture(frame,frame_copy,landmarks)
        if self.light == True and direction == "Center" and self.depth == True:
            cv2.imwrite("/Users/shadialarbed/Desktop/image_recognition/optimized_face_detection/images/without_landmark_1.jpg",
                        roi_image_capture)
            cv2.imwrite("/Users/shadialarbed/Desktop/image_recognition/optimized_face_detection/images/with_landmark_1.jpg",
                        roi_image_capture1)
            print("Captured an image named: landmark_1.jpg") 
        else:
            cv2.putText(frame, "not ready to take a pic", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, self.red_color, 2)
            
    def capture_picture(self , frame, frame_copy,direction,landmarks,key):
        if key & 0xFF == ord('c'):
            self.condition_capture(direction,frame,frame_copy,landmarks)

    def detect_condetions(self,frame,frame_copy,faces,key):
        for face in faces:
            landmarks1, landmarks = self.get_landmarks(frame, face)
            self.measure_depth(frame, landmarks)
            self.draw_landmarks(frame,landmarks)
            self.measure_brightness(frame, landmarks)
            direction = self.detect_direction(frame, landmarks, landmarks1)
            self.capture_picture(frame, frame_copy,direction,landmarks,key)
                    
################ for run function ####################
    def run(self):
        
        while True:
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            ret, frame = self.cap.read()
            if not ret:
                break
            faces = self.detect_faces(frame)
            frame_copy = frame.copy()

            if len(faces) == 1:
                self.detect_condetions(frame,frame_copy,faces,key)
            
            else:
                print(len(faces))
                self.screen_text(
                    frame,
                    text="Only one face allowed in the frame",
                    position=(500, 60)
                )

            cv2.imshow("Face Direction Estimation", frame)

        self.cap.release()
        cv2.destroyAllWindows()
