"""Problem Set 4: Motion Detection"""

import cv2
import numpy as np

# Utility function
def read_video(video_file, show=False):
    """Reads a video file and outputs a list of consecuative frames
  Args:
      image (string): Video file path
      show (bool):    Visualize the input video. WARNING doesn't work in
                      notebooks
  Returns:
      list(numpy.ndarray): list of frames
  """
    frames = []
    cap = cv2.VideoCapture(video_file)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        # Opens a new window and displays the input
        if show:
            cv2.imshow("input", frame)
            # Frames are read by intervals of 1 millisecond. The
            # programs breaks out of the while loop when the
            # user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # The following frees up resources and
    # closes all windows
    cap.release()
    if show:
        cv2.destroyAllWindows()
    return frames
    
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    return cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3, scale=1/8.0, borderType=cv2.BORDER_DEFAULT)
    # raise NotImplementedError


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """
    return cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3, scale=1/8.0, borderType=cv2.BORDER_DEFAULT)
    # raise NotImplementedError


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """

    # raise NotImplementedError
    # Ix_a=cv2.Sobel(img_a,cv2.CV_64F,1,0,ksize=3, scale=1/8.0, borderType=cv2.BORDER_DEFAULT)
    # Iy_a=cv2.Sobel(img_a,cv2.CV_64F,0,1,ksize=3, scale=1/8.0, borderType=cv2.BORDER_DEFAULT)
    Ix_a=gradient_x(img_a)
    Iy_a=gradient_y(img_a)
    if k_type == 'uniform':
        kernel = np.ones((k_size, k_size)) / (k_size ** 2)
    elif k_type == 'gaussian':
        kernel = cv2.getGaussianKernel(k_size, sigma)
    It = img_b - img_a
    Ix = cv2.filter2D(Ix_a * Ix_a, -1, kernel)
    Iy = cv2.filter2D(Iy_a * Iy_a, -1, kernel)
    Ixy = cv2.filter2D(Ix_a * Iy_a, -1, kernel)
    denominator = Ix * Iy - Ixy**2
    deno_inv = np.zeros_like(denominator)
    mask = (denominator != 0)
    deno_inv[mask] = 1 / denominator[mask]
    U = -deno_inv * (Iy * cv2.filter2D(Ix_a * It, -1, kernel) - Ixy * cv2.filter2D(Iy_a * It, -1, kernel))
    V = -deno_inv * (Ix * cv2.filter2D(Iy_a * It, -1, kernel) - Ixy * cv2.filter2D(Ix_a * It, -1, kernel))
    return U, V

def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """

    # raise NotImplementedError
    # kernel = np.array([1, 4, 6, 4, 1]) / 16.0
    # image_conv_h = cv2.filter2D(image, -1, kernel.reshape(1, -1))
    # image_conv_v = cv2.filter2D(image_conv_h, -1, kernel.reshape(-1, 1))
    # reduced_image = image_conv_v[::2, ::2]
    # return reduced_image

    kernel = np.array([1, 4, 6, 4, 1]) / 16.0
    image_conv_h = cv2.filter2D(image, -1, kernel.reshape(1, -1))
    image_conv_v = cv2.filter2D(image_conv_h, -1, kernel.reshape(-1, 1))
    output_height = (image_conv_v.shape[0] + 1) // 2
    output_width = (image_conv_v.shape[1] + 1) // 2
    # output_height = max(output_height, 1)
    # output_width = max(output_width, 1)
    reduced_image = image_conv_v[:output_height*2-1:2, :output_width*2-1:2]
    return reduced_image


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """

    # raise NotImplementedError
    pyramid=[image]
    for _ in range(levels - 1):
        reduced_image= reduce_image(pyramid[-1])
        # print(reduced_image.shape)
        pyramid.append(reduced_image)
    return pyramid


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """

    # raise NotImplementedError
    normalized_images = [normalize_and_scale(img) for img in img_list]
    image_dimensions = [img.shape for img in normalized_images]
    max_width = sum([dim[1] for dim in image_dimensions])
    height = max([dim[0] for dim in image_dimensions])
    img_out = np.zeros((height, max_width), dtype=np.float64)
    start_x = 0
    for img in normalized_images:
        end_x = start_x + img.shape[1]
        img_out[:img.shape[0],start_x:end_x] = img
        start_x = end_x
    return img_out

def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    kernel = np.array([1, 4, 6, 4, 1]) / 8.0
    upsampled_image = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
    upsampled_image[::2, ::2] = image
    expanded_image_v = cv2.filter2D(upsampled_image, -1, kernel.reshape(-1, 1))
    expanded_image_h = cv2.filter2D(expanded_image_v, -1, kernel.reshape(1, -1))
    return expanded_image_h
    # raise NotImplementedError


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    l_pyramid = []
    num_levels = len(g_pyr)
    for i in range(num_levels - 1):
        expanded_next_level = expand_image(g_pyr[i + 1])
        h, w = g_pyr[i].shape
        expanded_next_level = expanded_next_level[:h, :w]
        laplacian_image = g_pyr[i] - expanded_next_level
        l_pyramid.append(laplacian_image)
    l_pyramid.append(g_pyr[-1])
    return l_pyramid
    # raise NotImplementedError


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """

    # raise NotImplementedError
    # height, width = image.shape
    # x_coords, y_coords = np.arange(0, width), np.arange(0, height)
    # remap_x, remap_y = np.meshgrid(x_coords, y_coords)
    # remap_x, remap_y = np.clip(remap_x + U, 0, width - 1), np.clip(remap_y + V, 0, height - 1)
    # warped_image = cv2.remap(image, remap_x.astype(np.float32), remap_y.astype(np.float32),
    #                          interpolation=interpolation, borderMode=border_mode)
    # return warped_image
    height, width = image.shape
    y_coords, x_coords = np.indices(image.shape)
    U_resized = cv2.resize(U, (width, height), interpolation=cv2.INTER_LINEAR)
    V_resized = cv2.resize(V, (width, height), interpolation=cv2.INTER_LINEAR)
    remap_x, remap_y = np.clip(x_coords + U_resized, 0, width - 1), np.clip(y_coords + V_resized, 0, height - 1)
    warped_image = cv2.remap(image, remap_x.astype(np.float32), remap_y.astype(np.float32),
                             interpolation=interpolation, borderMode=border_mode)
    return warped_image


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """

    # raise NotImplementedError
    # U = np.zeros_like(img_a, dtype=np.float64)
    # V = np.zeros_like(img_a, dtype=np.float64)
    # for _ in range(levels, -1, -1):
    #     img_a_reduced = reduce_image(img_a)
    #     img_b_reduced = reduce_image(img_b)
    #     img_a_warped = warp(img_a_reduced, U, V, interpolation, border_mode)
    #     U_process, V_process = optic_flow_lk(img_a_warped, img_b_reduced, k_size, k_type, sigma)
    #     U_expanded, V_expanded = expand_image(U_process), expand_image(V_process)
    #     U, V = U_expanded + U, V_expanded + V
    #     U *= 2
    #     V *= 2
    #     U, V = np.clip(U, -img_a.shape[1], img_a.shape[1]), np.clip(V, -img_a.shape[0], img_a.shape[0])
    # return U, V
    A_process, B_process = [img_a], [img_b]
    for i in range(levels):
        A_process.append(reduce_image(A_process[-1]))
        B_process.append(reduce_image(B_process[-1]))
    U = None
    V = None
    for i in range(len(A_process) - 1, -1, -1):
        A_levelImg = A_process[i]
        B_levelImg = B_process[i]
        if U is None or V is None:
            U = np.zeros(A_levelImg.shape, dtype=np.float64)
            V = np.zeros(B_levelImg.shape, dtype=np.float64)
        else:
            U = 2*expand_image(U)
            V = 2*expand_image(V)
        warp_img = warp(B_levelImg, U, V, interpolation, border_mode)
        U_lk, V_lk = optic_flow_lk(A_levelImg, warp_img, k_size, k_type, sigma)
        U_resized = cv2.resize(U_lk, (U.shape[1], U.shape[0]), interpolation=cv2.INTER_CUBIC)
        V_resized = cv2.resize(V_lk, (V.shape[1], V.shape[0]), interpolation=cv2.INTER_CUBIC)
        U = U + U_resized
        V = V + V_resized
    return U, V

def classify_video(images):
    """Classifies a set of frames as either
        - int(1) == "Running"
        - int(2) == "Walking"
        - int(3) == "Clapping"
    Args:
        images list(numpy.array): greyscale floating-point frames of a video
    Returns:
        int:  Class of video
    """
    def extract_features(optical_flow_frames):
        # Extract average velocity magnitude as features
        features = np.array([np.mean(frame) for frame in optical_flow_frames])
        return features.reshape(-1, 1)
    X = extract_features(images)
    # from sklearn.tree import DecisionTreeClassifier #uncomment this for test
    classifier = DecisionTreeClassifier(random_state=42)
    y = np.array([1] * len(images))
    y[len(images)//3:2*len(images)//3] = 2
    y[2*len(images)//3:] = 3
    classifier.fit(X, y)
    accuracy=classifier.score(X,y)
    predicted_classes = classifier.predict(X)
    action_counts = np.bincount(predicted_classes)
    predicted_class = np.argmax(action_counts)
    return (predicted_class,accuracy)
    # raise NotImplementedError
