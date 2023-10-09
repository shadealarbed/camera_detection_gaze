class FaceCounter:
    def __init__(self):
        """
        Initialize the FaceCounter.
        """
        pass  # You can add any initialization code here if needed.

    def count_faces(self, faces):
        """
        Count the number of faces detected in the frame.

        Args:
            faces (list): List of dlib face objects detected in the frame.

        Returns:
            int: The number of faces detected in the frame.
        """
        return len(faces)