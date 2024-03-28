import cv2
import numpy as np
from matplotlib import pyplot as plt



def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.
    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.
    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.
    It is recommended you use Hough tools to find these circles in
    the image.
    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.
    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.
    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    # raise NotImplementedError
    temp_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    temp_img = cv2.GaussianBlur(temp_in, (9, 9), 2)
    active_light = None
    light_center = None
    traffic_light=[]
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    # lower_green = np.array([20, 100, 100])
    # upper_green = np.array([40, 255, 255])
    # lower_yellow = np.array([60, 100, 100])
    # upper_yellow = np.array([80, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    lower_green = np.array([60, 100, 100])
    upper_green = np.array([80, 255, 255])
    for radius in radii_range:
        circles = cv2.HoughCircles(
            temp_img,
            cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30,
            minRadius=radius, maxRadius=radius)
        if circles is not None:
            circles = np.uint16(circles)
            circles_sorted = sorted(circles[0, :], key=lambda x: x[0])
            x, y, r = circles_sorted[1]
            area = img_in[y - r:y + r, x - r:x + r]
            hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
            mask_r = cv2.inRange(hsv, lower_red, upper_red)
            mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_g = cv2.inRange(hsv, lower_green, upper_green)
            red_pixels = cv2.countNonZero(mask_r)
            yellow_pixels = cv2.countNonZero(mask_y)
            green_pixels = cv2.countNonZero(mask_g)
            if red_pixels > yellow_pixels and red_pixels > green_pixels:
                active_light = "red"
            elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
                active_light = "yellow"
            else:
                active_light = "green"
            light_center = (x, y)
            if active_light:
                break
    return light_center,active_light

def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.
    Args:
        img_in (numpy.array): image containing a traffic light.
    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    # raise NotImplementedError

    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,1, 20)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 45, None, 0, 20)
    detected_quadrilateral = None
    if lines is not None:
        for line1 in lines:
            for line2 in lines:
                if line1 is not line2:
                    x1, y1, x2, y2 = line1[0]
                    x3, y3, x4, y4 = line2[0]
                    # angle = np.arctan2(y2 - y1, x2 - x1) - np.arctan2(y4 - y3, x4 - x3)
                    angle = np.abs((np.arctan2(y2 - y1, x2 - x1) - np.arctan2(y4 - y3, x4 - x3)) * 180 / np.pi)
                    if angle == 90:
                        length1 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        length2 = np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2)
                        length_ratio = length1 / length2
                        if 0.9 < length_ratio < 1.1:
                            mylistx=[x1,x2,x3,x4]
                            mylisty=[y1,y2,y3,y4]
                            sorted(mylistx)
                            sorted(mylisty)
                            center_x=(mylistx[1]+mylistx[2]) // 2
                            center_y=(mylisty[1]+mylisty[2]) // 2
                            return (center_x, center_y)
    


def template_match(img_orig, img_template, method):
    """Returns the location corresponding to match between original image and provided template.
    Args:
        img_orig (np.array) : numpy array representing 2-D image on which we need to find the template
        img_template: numpy array representing template image which needs to be matched within the original image
        method: corresponds to one of the four metrics used to measure similarity between template and image window
    Returns:
        Co-ordinates of the topmost and leftmost pixel in the result matrix with maximum match
    """
    """Each method is calls for a different metric to determine
       the degree to which the template matches the original image
       We are required to implement each technique using the
       sliding window approach.
       Suggestion : For loops in python are notoriously slow
       Can we find a vectorized solution to make it faster?
    """
    result = np.zeros(
        (
            (img_orig.shape[0] - img_template.shape[0] + 1),
            (img_orig.shape[1] - img_template.shape[1] + 1),
        ),
        float,
    )
    top_left = []
    new_img_64=img_orig.astype(np.float64)
    new_tam_64=img_template.astype(np.float64)
    """Once you have populated the result matrix with the similarity metric corresponding to each overlap, return the topmost and leftmost pixel of
    the matched window from the result matrix. You may look at Open CV and numpy post processing functions to extract location of maximum match"""
    # Sum of squared differences
    if method == "tm_ssd":
        # """Your code goes here"""
        # raise NotImplementedError
        best_match = float('inf')
        best_match_p = None
        for y in range(result.shape[0]):
            for x in range(result.shape[1]):
                roi = new_img_64[y:y + new_tam_64.shape[0], x:x + new_tam_64.shape[1]]
                ssd = np.sum((roi - new_tam_64)**2)
                if ssd < best_match:
                    best_match = ssd
                    best_match_p = (x, y)
        top_left.append(best_match_p)

    # Normalized sum of squared differences
    elif method == "tm_nssd":
        # """Your code goes here"""
        # raise NotImplementedError
        best_match = np.inf
        best_match_p = (0, 0)
        for y in range(result.shape[0]):
            for x in range(result.shape[1]):
                roi = new_img_64[y:y+new_tam_64.shape[0], x:x+new_tam_64.shape[1]]
                nssd = np.sum((roi - new_tam_64) ** 2) / (new_tam_64.shape[0] * new_tam_64.shape[1])
                if nssd < best_match:
                    best_match = nssd
                    best_match_p = (x, y)
        top_left.append(best_match_p)

    # # Cross Correlation
    elif method == "tm_ccor":
    #     """Your code goes here"""
    #     raise NotImplementedError
        best_match = -np.inf
        best_match_p = (0, 0)
        for y in range(result.shape[0]):
            for x in range(result.shape[1]):
                roi = new_img_64[y:y+new_tam_64.shape[0], x:x+new_tam_64.shape[1]]
                ccor = np.sum(roi * new_tam_64)
                if ccor > best_match:
                    best_match = ccor
                    best_match_p = (x, y)
        top_left.append(best_match_p)

    # # Normalized Cross Correlation
    elif method == "tm_nccor":
        """Your code goes here"""
        # raise NotImplementedError
        best_match = -np.inf
        best_match_p = (0, 0)
        tem_mean=np.mean(new_tam_64)
        for y in range(result.shape[0]):
            for x in range(result.shape[1]):
                relate_img = new_img_64[y:y+new_tam_64.shape[0], x:x+new_tam_64.shape[1]]
                relate_img_mean=np.mean(relate_img)
                nccor = np.sum((relate_img - relate_img_mean) * (new_tam_64 - tem_mean))
                if nccor > best_match:
                    best_match = nccor
                    best_match_p = (x, y)
        top_left.append(best_match_p)

    else:
    #     """Your code goes here"""
    #     # Invalid technique
    # raise NotImplementedError
        return "Invalid Method"
    return top_left[0]


'''Below is the helper code to print images for the report'''
    # cv2.rectangle(img_orig,top_left, bottom_right, 255, 2)
    # plt.subplot(121),plt.imshow(result,cmap = 'gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img_orig,cmap = 'gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # plt.suptitle(method)
    # plt.show()


def dft(x):
    """Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing Fourier Transformed Signal

    """
    x = np.asarray(x, dtype=np.complex_)
    return np.fft.fft(x)
    # raise NotImplementedError


def idft(x):
    """Inverse Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing Fourier-Transformed signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing signal

    """
    x = np.asarray(x, dtype=np.complex_)
    return np.fft.ifft(x)
    # raise NotImplementedError


def dft2(img):
    """Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,m) representing image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,m) representing Fourier-Transformed image

    """
    return np.fft.fft2(img)
    # raise NotImplementedError


def idft2(img):
    """Inverse Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,m) representing Fourier-Transformed image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,m) representing image

    """
    return np.fft.ifft2(img)
    # raise NotImplementedError


def compress_image_fft(img_bgr, threshold_percentage):
    """Return compressed image by converting to fourier domain, thresholding based on threshold percentage, and converting back to fourier domain
    Args:
        img_bgr (np.array): numpy array of shape (n,m,3) representing bgr image
        threshold_percentage (float): between 0 and 1 representing what percentage of Fourier image to keep
    Returns:
        img_compressed (np.array): numpy array of shape (n,m,3) representing compressed image. (Make sure the data type of the np array is float64)
        compressed_frequency_img (np.array): numpy array of shape (n,m,3) representing the compressed image in the frequency domain

    """
    # raise NotImplementedError
    pre_channels = np.zeros_like(img_bgr, dtype=float)
    frequency_domain = np.zeros_like(img_bgr, dtype=np.complex_)

    for index in range(3):
        channle_domain = dft2(img_bgr[:,:,index])
        # channle_domain_shifted = np.fft.fftshift(channle_domain)
        magnitude = np.abs(channle_domain).flatten()
        sorted_magnitude = np.sort(magnitude)[::-1]
        threshold_value = sorted_magnitude[int(np.floor(threshold_percentage * magnitude.size))]
        mask=np.where(np.abs(channle_domain) > threshold_value,1,0)
        filtered_domain = channle_domain * mask
        # filtered_domain_shifted = np.fft.ifftshift(filtered_domain)
        filtered_channel = idft2(filtered_domain).real
        pre_channels[:,:,index]=filtered_channel.astype(float)
        # frequency_domain[:,:,index]=mask.astype(float)
        frequency_domain[:,:,index]=np.fft.fftshift(filtered_domain)
        # frequency_domain[:,:,index]=filtered_domain
    final_frequency_domain=np.log(np.abs(frequency_domain)+1)*20
    return pre_channels, final_frequency_domain



def low_pass_filter(img_bgr, r):
    """Return low pass filtered image by keeping a circle of radius r centered on the frequency domain image
    Args:
        img_bgr (np.array): numpy array of shape (n,m,3) representing bgr image
        r (float): radius of low pass circle
    Returns:
        img_low_pass (np.array): numpy array of shape (n,m,3) representing low pass filtered image. (Make sure the data type of the np array is float64)
        low_pass_frequency_img (np.array): numpy array of shape (n,m,3) representing the low pass filtered image in the frequency domain

    """
    # raise NotImplementedError
    # img_bgr = img_bgr.astype(np.float64)
    pre_channels = np.zeros_like(img_bgr)
    frequency_domain = np.zeros_like(img_bgr)
    height, weight,_ = img_bgr.shape
    y, x = np.ogrid[-height//2:height//2, -weight//2:weight//2]
    mask = x**2 + y**2 <= r**2
    for index in range(3):
        channel_domain = dft2(img_bgr[:,:,index])
        channel_domain_shifted = np.fft.fftshift(channel_domain)
        channel_domain_shifted_low_pass = channel_domain_shifted * mask
        channel_domain_low_pass = np.fft.ifftshift(channel_domain_shifted_low_pass)
        filtered_channel = np.abs(idft2(channel_domain_low_pass))
        pre_channels[:,:,index]=filtered_channel
        frequency_domain[:,:,index]=np.abs(channel_domain_shifted_low_pass)
    final_frequency_domain=np.log(np.abs(frequency_domain)+1)*20
    return pre_channels, final_frequency_domain

