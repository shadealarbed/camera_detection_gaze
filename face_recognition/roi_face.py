import cv2
import numpy as np

class FaceROIDetector:
    def __init__(self, margin=100, threshold_value=250, threshold_value_dark=10):
        self.margin = margin
        self.threshold_value = threshold_value
        self.threshold_value_dark = threshold_value_dark

    def mapXY(self,landmarks):
        return np.min(landmarks[:, 0]), np.max(landmarks[:, 0]), np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
    
    def margen_cordents(self,landmarks):
        min_x, max_x, min_y, max_y = self.mapXY(landmarks)
        min_x -= self.margin
        min_y -= self.margin + 50
        max_x += self.margin
        max_y += self.margin - 50
        return min_x ,min_y,max_x,max_y
    
    
    def detect_margen_cordents(self,landmarks,frame):
        min_x ,min_y,max_x,max_y = self.margen_cordents(landmarks)
        return max(min_x, 0),max(min_y, 0),min(max_x, frame.shape[1]),min(max_y, frame.shape[0])
    
    
    def detect_roi_image_capture(self, frame, landmarks):
        min_x ,min_y,max_x,max_y = self.detect_margen_cordents(landmarks,frame)
        return frame[int(min_y):int(max_y), int(min_x):int(max_x)]


########### roi face capture #############

    def roi_marks(self,landmarks,frame):
        min_x1, max_x1, min_y1, max_y1 = self.mapXY(landmarks)
        return frame[int(min_y1):int(max_y1), int(min_x1):int(max_x1)]
    
    def color_threshold(self,threshold_value,roi_face_light):
        if len(roi_face_light) !=0:
            gray1 = cv2.cvtColor(roi_face_light, cv2.COLOR_BGR2GRAY)
        
            return cv2.threshold(gray1, threshold_value, 255, cv2.THRESH_BINARY)
        else:
            print("no faces to detected")

    def detect_roi_face_light(self, frame, landmarks):
        roi_face_light = self.roi_marks(landmarks,frame)
        if len(roi_face_light) != 0:
            _, thresholded_face = self.color_threshold(self.threshold_value,roi_face_light)
            _, thresholded_face_low = self.color_threshold(self.threshold_value_dark,roi_face_light)
        else:
            print("no faces")
        return roi_face_light, thresholded_face, thresholded_face_low