# main.py
from face_recognition.face_direction_estimator import FaceDirectionEstimator
import psutil

if __name__ == "__main__":
    process = psutil.Process()
    shape_predictor_path = "/Users/shadialarbed/img_proc/shape_predictor_68_face_landmarks.dat"
    face_estimator = FaceDirectionEstimator(shape_predictor_path)
    face_estimator.run()
    print(f"memory used: {process.memory_info().rss}")