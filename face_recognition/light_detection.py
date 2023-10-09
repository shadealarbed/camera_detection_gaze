import cv2
import numpy as np

class LightDetector:
    def __init__(self, brightness_threshold=10):
        """
        Initialize the LightDetector.

        Args:
            brightness_threshold (int): Threshold for determining if the frame is bright.
        """
        self.brightness_threshold = brightness_threshold

    def is_bright(self, frame):
        """
        Analyze the frame to determine if it's bright.

        Args:
            frame (numpy.ndarray): Input frame in BGR format.

        Returns:
            Tuple[bool, bool]: A tuple containing two boolean values.
                - The first element indicates if the frame is considered bright (True) or not (False).
                - The second element indicates if the frame is too dark (True) based on a lower brightness threshold.
        """
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to detect high pixel values
        _, thresholded_frame = cv2.threshold(gray_frame, self.brightness_threshold, 255, cv2.THRESH_BINARY)

        # Count the number of bright pixels
        bright_pixel_count = np.sum(thresholded_frame == 255)

        # Compare the bright pixel count with the threshold
        is_bright_frame = bright_pixel_count > self.brightness_threshold
        is_too_dark = bright_pixel_count <= 10  # You can adjust the lower threshold as needed

        return is_bright_frame, is_too_dark