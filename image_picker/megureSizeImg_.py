from PIL import Image
image_path = Image.open("/Users/shadialarbed/Desktop/image_recognition/image_recognition/adjusted_image3.jpg")
width_in_inches, height_in_inches = image_path.size
print(f"width: {width_in_inches} , hight: {height_in_inches}")

import cv2

image_path1 = "/Users/shadialarbed/Desktop/image_recognition/image_recognition/adjusted_image3.jpg" 
image = cv2.imread(image_path1)

# Get the pixel dimensions of the image
height_px, width_px, _ = image.shape

# Specify the physical dimensions of the image in inches
width_in = 6.0  # Replace with the actual width in inches
height_in = 4.0  # Replace with the actual height in inches

# Calculate DPI
dpi_x = int(width_px / width_in_inches)
dpi_y = int(height_px / height_in_inches)

print(f"DPI (X-axis): {dpi_x}")
print(f"DPI (Y-axis): {dpi_y}")