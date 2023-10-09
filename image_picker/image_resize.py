import cv2

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
    
    def load_image(self):
        if self.image is None:
            print("Error: Could not load the image.")
        else:
            # Get image properties
            height, width, _ = self.image.shape
            print(f"Image resolution: {width}x{height}")
    
    def resize_image(self, desired_width, desired_height):
        # Calculate the new size while maintaining the aspect ratio
        width, height, _ = self.image.shape
        if width >= height:
            new_width = min(desired_width, width)
            new_height = int((new_width / width) * height)
        else:
            new_height = min(desired_height, height)
            new_width = int((new_height / height) * width)
    
        print(f"Resized resolution: {new_width}x{new_height}")
    
        # Resize the image to the desired size
        self.resized_image = cv2.resize(self.image, (new_width, new_height))
    
    def process_image(self):
        # Perform your image processing on self.resized_image here
        pass
    
    def save_resized_image(self, output_path):
        cv2.imwrite(output_path, self.resized_image)
    
    def show_image(self):
        # Display the resized image
        cv2.imshow("Resized Image", self.resized_image)
        # Wait for a key press and then close the OpenCV window
        cv2.waitKey(0)
        cv2.destroyAllWindows()