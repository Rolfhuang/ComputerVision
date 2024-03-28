import cv2
import numpy as np
from matplotlib import pyplot as plt

# def traffic_light_detection(img_in, radii_range):
#     image = cv2.imread(img_in)

#     if image is None:
#         print("Error: Could not read the image.")
#         return

#     # Step 2: Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Iterate through the radii range
#     for radius in radii_range:
#         # Step 3: Apply Gaussian blur
#         blurred = cv2.GaussianBlur(gray, (9, 9), 2)

#         # Step 4: Detect circles using the Hough Circle Transform
#         circles = cv2.HoughCircles(
#             blurred,
#             cv2.HOUGH_GRADIENT,
#             dp=1,
#             minDist=20,
#             param1=50,
#             param2=30,
#             minRadius=radius,
#             maxRadius=radius
#         )

#         # Step 5: Draw yellow circles with a red boundary on the original image
#         if circles is not None:
#             circles = np.uint16(np.around(circles))
#             for circle in circles[0, :]:
#                 center = (circle[0], circle[1])
#                 radius = circle[2]
#                 # Check if the circle is yellow (assuming BGR color space)
#                 b, g, r = image[circle[1], circle[0]]
#                 if b > 150 and g > 150 and r < 100:
#                     cv2.circle(image, center, radius, (0, 255, 0), 2)  # Draw a green circle
#                     cv2.circle(image, center, radius, (0, 0, 255), 2)  # Draw a red boundary

#     # Step 6: Display or save the image with the detected circle(s)
#     cv2.imshow('Traffic Light Detection', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # To save the image with circles
#     cv2.imwrite('output_image.jpg', image)

if __name__ == "__main__":
#     input_images = r'C:\Users\rolfh\OneDrive\Desktop\GeoTech\CSC6476_CV\Assignment_2\CSC6476_CV_PS2\input_images\tl_template - Copy.png'
#     radii_range = range(10, 40, 1)
#     traffic_light_detection(input_images, radii_range)
    print (np.__version__)
# input_images = r'C:\Users\rolfh\OneDrive\Desktop\GeoTech\CSC6476_CV\Assignment_2\CSC6476_CV_PS2\input_images\scene_constr_1 - Copy.png'
# ip=cv2.imread(input_images)

# gray = cv2.cvtColor(ip, cv2.COLOR_BGR2GRAY)

# # Apply edge detection (e.g., Canny)
# edges = cv2.Canny(gray, 1, 30)

# # Apply Hough Line Transform to detect lines
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, None, 0, 10)

# detected_quadrilateral = None

# if lines is not None:
#     # Filter and combine lines to form quadrilaterals
#     for line1 in lines:
#         for line2 in lines:
#             if line1 is not line2:
#                 x1, y1, x2, y2 = line1[0]
#                 x3, y3, x4, y4 = line2[0]
                
#                 # Calculate the angle between lines
#                 angle = np.arctan2(y2 - y1, x2 - x1) - np.arctan2(y4 - y3, x4 - x3)
#                 angle = np.abs(angle * 180 / np.pi)  # Convert to degrees
                
#                 # Check if the lines are approximately perpendicular
#                 if 85 < angle < 95:
#                     # Check if the lines are of similar length
#                     length1 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#                     length2 = np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2)
#                     length_ratio = length1 / length2
                    
#                     # You can adjust this ratio as needed
#                     if 0.9 < length_ratio < 1.1:
#                         # Calculate the center of the quadrilateral
#                         center_x = (x1 + x2 + x3 + x4) // 4
#                         center_y = (y1 + y2 + y3 + y4) // 4
#                         detected_quadrilateral = (center_x, center_y)
#                         break  # Exit the loop once a valid quadrilateral is found
#         if detected_quadrilateral is not None:
#             break  # Exit the loop once a valid quadrilateral is found

#     # Draw detected quadrilateral (optional)
#     if detected_quadrilateral:
#         center_x, center_y = detected_quadrilateral
#         cv2.circle(ip, (center_x, center_y), 5, (255, 0, 0), -1)
    
#     # Display the result (optional)
#     cv2.imshow('Detected Signs', ip)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()