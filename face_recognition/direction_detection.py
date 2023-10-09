import numpy as np

class DirectionDetector:
    def __init__(self, threshold=10):
        self.threshold = threshold

    def get_side_landmarks(self, left_landmarks, right_landmarks):
        return (
            np.std(left_landmarks[:, 0]) if len(left_landmarks) > 0 else 0,
            np.std(right_landmarks[:, 0]) if len(right_landmarks) > 0 else 0
        )

    def get_vertical_distance(self, landmarks1):
        x1, y1 = landmarks1.part(1).x, landmarks1.part(1).y
        x15, y15 = landmarks1.part(15).x, landmarks1.part(15).y
        x29, y29 = landmarks1.part(29).x, landmarks1.part(29).y
        return np.array([x1, y1]), np.array([x15, y15]), np.array([x29, y29])

    def calculate_distance_between_landmarks(self, landmarks1):
        p1, p2, p3 = self.get_vertical_distance(landmarks1)
        distance_1_29 = np.linalg.norm(p1 - p3)
        distance_15_29 = np.linalg.norm(p2 - p3)
        return distance_1_29, distance_15_29

    def calculate_theta_between_landmarks(self, p1, p2, p3, distance_1_29, distance_15_29):
        VD = np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)
        return np.arcsin(VD / distance_1_29), np.arcsin(VD / distance_15_29)

    def combine_dis_points(self, landmarks1):
        p1, p2, p3 = self.get_vertical_distance(landmarks1)
        distance_1_29, distance_15_29 = self.calculate_distance_between_landmarks(landmarks1)
        return self.calculate_theta_between_landmarks(p1, p2, p3, distance_1_29, distance_15_29)

    def pitched_angle(self, landmarks1):
        theta_1, theta_2 = self.combine_dis_points(landmarks1)
        return (theta_1 + theta_2) / 2

    def left_right_side(self, left_landmarks, right_landmarks):
        left_std, right_std = self.get_side_landmarks(left_landmarks, right_landmarks)
        if left_std - right_std > 10:
            return "Left"
        elif right_std - left_std > 15:
            return "Right"

    def up_down_directions(self, landmarks1):
        if self.pitched_angle(landmarks1) > 0.15:
            return "Down"
        elif self.pitched_angle(landmarks1) < -0.15:
            return "Up"

    def estimate_face_direction(self, left_landmarks, right_landmarks, landmarks1):
        if self.left_right_side(left_landmarks, right_landmarks):
            return self.left_right_side(left_landmarks, right_landmarks)
        elif self.up_down_directions(landmarks1):
            return self.up_down_directions(landmarks1)
        else:
            return "Center"
