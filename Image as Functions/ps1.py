import math
import numpy as np
import cv2

# # Implement the functions below.


def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    temp_image = np.copy(image)
    b, g, r = cv2.split(temp_image)
    return np.asarray(r)
    # raise NotImplementedError


def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """
    temp_image = np.copy(image)
    b, g, r = cv2.split(temp_image)
    return np.asarray(g)
    # raise NotImplementedError


def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """
    temp_image = np.copy(image)
    b, g, r = cv2.split(temp_image)
    return np.asarray(b)
    # raise NotImplementedError


def swap_green_blue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """
    temp_image=np.copy(image)
    b, g, r = cv2.split(temp_image)
    return cv2.merge((g,b,r))
    # raise NotImplementedError


def copy_paste_middle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    temp_image_src = np.copy(src).astype(np.float32)
    temp_image_dst = np.copy(dst).astype(np.float32)
    middle_row_start = temp_image_src.shape[0] // 2 - shape[0] // 2
    middle_row_end = middle_row_start + shape[0]
    middle_col_start = temp_image_src.shape[1] // 2 - shape[1] // 2
    middle_col_end = middle_col_start + shape[1]
    middle_region = temp_image_src[middle_row_start:middle_row_end, middle_col_start:middle_col_end]
    if middle_region.size == 0 or middle_region.shape[0] != shape[0] or middle_region.shape[1] != shape[1]:
        return None
    middle_region_resized = cv2.resize(middle_region, (shape[1], shape[0]))

    paste_row_start = temp_image_dst.shape[0] // 2 - shape[0] // 2
    paste_row_end = paste_row_start + shape[0]
    paste_col_start = temp_image_dst.shape[1] // 2 - shape[1] // 2
    paste_col_end = paste_col_start + shape[1]
    temp_image_dst[paste_row_start:paste_row_end, paste_col_start:paste_col_end] = middle_region_resized
    return temp_image_dst
    # raise NotImplementedError

def center_circle(length):
    if length%2==1:
        return length//2
    else:
        return length//2-1

def copy_paste_middle_circle(src, dst, radius):
    """ Copies the middle circle region of radius "radius" from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

    Args:
        src (numpy.array): 2D array where the circular shape will be copied from.
        dst (numpy.array): 2D array where the circular shape will be copied to.
        radius (scalar): scalar value of the radius.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    temp_image_src = np.copy(src).astype(np.uint8)
    temp_image_dst = np.copy(dst).astype(np.uint8)
    radius = int(radius)
    
    center_row = center_circle(temp_image_src.shape[0])
    center_col = center_circle(temp_image_src.shape[1])
    
    circle_mask = np.zeros_like(temp_image_src, dtype=np.uint8)
    r_grid, c_grid = np.ogrid[:circle_mask.shape[0], :circle_mask.shape[1]]
    circle_mask[((r_grid - center_row) ** 2 + (c_grid - center_col) ** 2) <= radius ** 2] = 255
    
    circle_region = temp_image_src * (circle_mask / 255).astype(temp_image_src.dtype)
    
    add=1 if temp_image_dst.shape[1]<temp_image_src.shape[1] else 0
    start_x = center_col - radius
    end_x = center_col + radius +add
    start_y = center_row - radius
    end_y = center_row + radius +add
    
    circle_width = end_x - start_x
    circle_height = end_y - start_y
    
    paste_x = center_circle(temp_image_dst.shape[1] - circle_width)
    paste_y = center_circle(temp_image_dst.shape[0] - circle_height)
    
    circle_mask2 = np.zeros_like(temp_image_dst, dtype=np.uint8)
    r_grid_d, c_grid_d = np.ogrid[:circle_mask2.shape[0], :circle_mask2.shape[1]]
    circle_mask2[((r_grid_d - center_circle(temp_image_dst.shape[0])) ** 2 + (c_grid_d - center_circle(temp_image_dst.shape[1])) ** 2) <= radius ** 2] = 255
    
    invertimage2 = cv2.bitwise_not(circle_mask2)
    circle_region2 = cv2.bitwise_and(temp_image_dst, invertimage2)
    
    invertimage2 = 255 - circle_mask2
    circle_region2 = temp_image_dst * (invertimage2 / 255).astype(temp_image_dst.dtype)
    
    result = circle_region2[paste_y:paste_y + circle_height, paste_x:paste_x + circle_width] + circle_region[start_y:end_y, start_x:end_x]
    result = np.clip(result, 0, 255)
    temp_image_dst[paste_y:paste_y + circle_height, paste_x:paste_x + circle_width] = result
    return temp_image_dst
    # raise NotImplementedError


def image_stats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """
    temp_image = np.copy(image)
    return (np.min(temp_image).astype(np.float64),np.max(temp_image).astype(np.float64),np.mean(temp_image),np.std(temp_image))
    # raise NotImplementedError


def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """
    temp_image = np.copy(image).astype(np.float64)
    result=((temp_image-np.mean(temp_image))/np.std(temp_image))*scale
    return result+np.mean(temp_image)
    # raise NotImplementedError


def shift_image_left(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """
    temp_image = np.copy(image).astype(np.float32)
    x=temp_image.shape[1]
    y=temp_image.shape[0]

    newshape = np.array([[1, 0, -shift], [0, 1, 0]], dtype=np.float32)

    newimage = cv2.warpAffine(temp_image, newshape, (x, y))

    newimage=newimage[:,:-shift]
    border=cv2.copyMakeBorder(temp_image, 0, 0, 0, shift, cv2.BORDER_REPLICATE)
    border=border[:,-shift:]
    image_combined = np.concatenate((newimage, border), axis=1)

    return image_combined
    # raise NotImplementedError


def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """
    temp_image1 = np.copy(img1)
    temp_image2 = np.copy(img2)

    diff = temp_image1- temp_image2
    min_image, max_image, _ , _ =image_stats(diff)

    if max_image == min_image:
        diff = np.zeros_like(diff)
    else:
        diff = (diff - min_image) * (255 / (max_image - min_image))
    return diff
    # raise NotImplementedError


def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """
    temp_image = np.float64(np.copy(image))

    rows, cols, _ = temp_image.shape
    gaussian_noise = np.random.normal(0, sigma, (rows, cols))

    temp_image[:, :, channel] = np.clip(temp_image[:, :, channel] + gaussian_noise, 0, 255)
    return temp_image
    # raise NotImplementedError


def build_hybrid_image(image1, image2, cutoff_frequency):
    """ 
    Takes two images and creates a hybrid image given a cutoff frequency.
    Args:
        image1: numpy nd-array of dim (m, n, c)
        image2: numpy nd-array of dim (m, n, c)
        cutoff_frequency: scalar
    
    Returns:
        hybrid_image: numpy nd-array of dim (m, n, c)

    Credits:
        Assignment developed based on a similar project by James Hays. 
    """

    filter = cv2.getGaussianKernel(ksize=cutoff_frequency*4+1,
                                   sigma=cutoff_frequency)
    filter = np.dot(filter, filter.T)
    
    low_frequencies = cv2.filter2D(image1,-1,filter)

    high_frequencies = image2 - cv2.filter2D(image2,-1,filter)

    return low_frequencies + high_frequencies

    # raise NotImplementedError


def vis_hybrid_image(hybrid_image):
    """ 
    Tools to visualize the hybrid image at different scale.

    Credits:
        Assignment developed based on a similar project by James Hays. 
    """


    scales = 5
    scale_factor = 0.5
    padding = 5
    original_height = hybrid_image.shape[0]
    num_colors = 1 if hybrid_image.ndim == 2 else 3

    output = np.copy(hybrid_image)
    cur_image = np.copy(hybrid_image)
    for scale in range(2, scales+1):
      # add padding
      output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                          dtype=np.float32)))

      # downsample image
      cur_image = cv2.resize(cur_image, (0, 0), fx=scale_factor, fy=scale_factor)

      # pad the top to append to the output
      pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                     num_colors), dtype=np.float32)
      tmp = np.vstack((pad, cur_image))
      output = np.hstack((output, tmp))

    return output
