"""
CS6476 Assignment 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np

import cv2
import numpy as np
from typing import Tuple
import math
from scipy.signal import convolve2d
import scipy.ndimage,scipy.signal

class Mouse_Click_Correspondence(object):

    def __init__(self,path1='',path2='',img1='',img2=''):
        self.sx1 = []
        self.sy1 = []
        self.sx2 = []
        self.sy2 = []
        self.img = img1
        self.img2 = img2
        self.path1 = path1
        self.path2 = path2


    def click_event(self,event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x y', x, ' ', y)

            sx1=self.sx1
            sy1=self.sy1

            sx1.append(x)
            sy1.append(y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image 1', self.img)

            # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img[y, x, 0]
            g = self.img[y, x, 1]
            r = self.img[y, x, 2]
            cv2.putText(self.img, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x2, y2), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 1', self.img)

        # driver function

    def click_event2(self,event2, x2, y2, flags, params):
        # checking for left mouse clicks
        if event2 == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x2 y2', x2, ' ', y2)

            sx2= self.sx2
            sy2 = self.sy2

            sx2.append(x2)
            sy2.append(y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img2, str(x2) + ',' +
                        str(y2), (x2, y2), font,
                        1, (0, 255, 255), 2)
            cv2.imshow('image 2', self.img2)

            # checking for right mouse clicks
        if event2 == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x2, ' ', y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img2[y2, x2, 0]
            g = self.img2[y2, x2, 1]
            r = self.img2[y2, x2, 2]
            cv2.putText(self.img2, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x2, y2), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 2', self.img2)

    # driver function
    def driver(self,path1,path2):
        # reading the image
        # path = r'D:\GaTech\TA - CV\ps05\ps05\ps5-1-b-1.png'
        #path1 = r'1a_notredame.jpg'
        #path2 = r'1b_notredame.jpg'


        #path1 = self.path1
        #path2 = self.path2

        # path1 = r'crop1.jpg'
        # path2 = r'crop2.jpg'

        self.img = cv2.imread(path1, 1)
        self.img2 = cv2.imread(path2, 2)

        # displaying the image
        cv2.namedWindow("image 1", cv2.WINDOW_NORMAL)
        cv2.imshow('image 1', self.img)
        cv2.namedWindow("image 2", cv2.WINDOW_NORMAL)
        cv2.imshow('image 2', self.img2)

        # setting mouse hadler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image 1', self.click_event)
        cv2.setMouseCallback('image 2', self.click_event2)

        # wait for a key to be pressed to exit

        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()

        print('sx1 sy1', self.sx1, self.sy1)
        print('sx2 sy2', self.sx2, self.sy2)

        points1, points2 = [], []
        for x, y in zip(self.sx1, self.sy1):
            points1.append((x, y))

        points_1 = np.array(points1)

        for x, y in zip(self.sx2, self.sy2):
            points2.append((x, y))

        points_2 = np.array(points2)

        np.save('p1.npy', points_1)
        np.save('p2.npy', points_2)



def euclidean_distance(p0, p1):
    """Get the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1
        p1 (tuple): Point 2
    Return:
        float: The distance between points
    """

    # raise NotImplementedError
    return math.dist(p0,p1)


def get_corners_list(image):
    """List of image corner coordinates used in warping.

    Args:
        image (numpy.array of float64): image array.
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """
    height, width = image.shape[:2]
    corners = [(0,0),(0,height-1),(width-1,0),(width-1,height-1)]

    # raise NotImplementedError
    return corners


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding and convolution to find the
    four markers in the image.

    Args:
        image (numpy.array of uint8): image array.
        template (numpy.array of unint8): template of the markers
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """

    # raise NotImplementedError
    # out_image=[]
    # # img_removeNoise=cv2.fastNlMeansDenoisingColored(image,None,10,10,5,20)
    # gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # template_gray=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    # # imagedeno = cv2.GaussianBlur(gray_image, (1, 1), 0)
    # # img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # # template_gray=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    # img_removeNoise=cv2.bilateralFilter(gray_image,15,40,40)
    # template_removeNoise=cv2.bilateralFilter(template_gray,15,40,40)
    # # img_removeNoise = cv2.GaussianBlur(img_gray, (1, 1), 0)
    # # template_gray = cv2.GaussianBlur(template_gray, (1, 1), 0)
    # process_template=[]
    # result=cv2.matchTemplate(img_removeNoise,template_removeNoise,cv2.TM_CCOEFF_NORMED)
    # threshold=0.8
    # matchs=np.where(result>=threshold)
    # for item in zip(*matchs[::-1]):
    #     process_template.append((item[0],item[1]))
    # # locations=[(int(item[0]+template.shape[:2][0]/2),int(item[1]+template.shape[:2][0]/2)) for item in process_template]
    # # first_sort=sorted(locations, key=lambda x: x[1])
    # # top_left,top_right=sorted(first_sort[:2], key=lambda x: x[0])
    # # bottom_left,bottom_right=sorted(first_sort[2:], key=lambda x: x[0])
    # # out_image=[top_left,bottom_left,top_right,bottom_right]
    # if len(process_template)<4:
    #     pre_image=[]
    #     circle_centers_with_2_lines=[]
    #     circles = cv2.HoughCircles(img_removeNoise, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=100, param2=10, minRadius=10, maxRadius=50)
    #     if circles is not None:
    #         circles = np.uint16(np.around(circles))
    #         circle_centers_with_2_lines = []
    #         for circle in circles[0, :]:
    #             center = (circle[0], circle[1])
    #             radius = circle[2]
    #             lines = cv2.HoughLinesP(img_removeNoise,1,np.pi / 180,threshold=50,minLineLength=10,maxLineGap=100)
    #             if lines is not None and len(lines) >= 2:
    #                 vertical_lines = []
    #                 for line in lines:
    #                     x1, y1, x2, y2 = line[0]
    #                     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #                 # for line in lines:
    #                 #     x1, y1, x2, y2 = line[0]
    #                 #     angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    #                 #     if 85 <= abs(angle) <= 95:
    #                 #         vertical_lines.append(line)
    #                 # if len(vertical_lines) >= 2:
    #                 circle_centers_with_2_lines.append(center)
    #     locations=[(int(item[0]),int(item[1])) for item in circle_centers_with_2_lines]
    #     # first_sort=sorted(locations, key=lambda x: x[1])
    #     # top_left,top_right=sorted(first_sort[:2], key=lambda x: x[0])
    #     # bottom_left,bottom_right=sorted(first_sort[2:], key=lambda x: x[0])
    #     # out_image=[top_left,bottom_left,top_right,bottom_right]
    # else:
    #     locations=[(int(item[0]+template.shape[:2][0]/2),int(item[1]+template.shape[:2][0]/2)) for item in process_template]
    # first_sort=sorted(locations, key=lambda x: x[1])
    # top_left,top_right=sorted(first_sort[:2], key=lambda x: x[0])
    # bottom_left,bottom_right=sorted(first_sort[2:], key=lambda x: x[0])
    # out_image=[top_left,bottom_left,top_right,bottom_right]
    # return out_image
    out_image=[]
    temp_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    temp_template=cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    denoise_img = cv2.GaussianBlur(temp_img, (1, 1), 0)
    process_img = cv2.cornerHarris(denoise_img, 30, 31, 0.04)
    # process_img = Automatic_Corner_Detection().harris_corner(denoise_img,100)
    cv2.normalize(process_img, process_img, 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow('Detected Signs', process_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    process_img_location = np.argwhere(np.uint8(process_img) > 100)
    _, _, centers = cv2.kmeans(np.float32(process_img_location), 4, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100000, 0.02), 10, cv2.KMEANS_RANDOM_CENTERS)
    kernel = np.ones((10, 10), np.uint8)
    dilated_img = cv2.dilate(denoise_img, kernel, iterations=1)
    # dilated_img = cv2.erode(denoise_img, kernel, iterations=1)
    process_template = []
    for center in centers.astype(int):
        if dilated_img[center[0], center[1]] > 0:
            process_template.append((center[1], center[0]))

    # if template is not None:
    #     template_match = cv2.matchTemplate(temp_img, temp_template, cv2.TM_CCOEFF_NORMED)
    #     _, _, _, max_loc = cv2.minMaxLoc(template_match)
    #     process_template.append(max_loc)

    # if len(process_template) != 4:
    #     return None
    # locations = sorted(process_template, key=lambda x: (-x[1], x[0]))
    # top_left, bottom_left, top_right, bottom_right = locations
    first_sort=sorted(process_template, key=lambda x: x[1])
    top_left,top_right=sorted(first_sort[:2], key=lambda x: x[0])
    bottom_left,bottom_right=sorted(first_sort[2:], key=lambda x: x[0])
    out_image=[top_left,bottom_left,top_right,bottom_right]
    return out_image
    # return 0

def draw_box(image, markers, thickness=1):
    """Draw 1-pixel width lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line and leave the default "thickness" and "lineType".

    Args:
        image (numpy.array of uint8): image array
        markers(list of tuple): the points where the markers were located
        thickness(int): thickness of line used to draw the boxes edges
    Returns:
        numpy.array: image with lines drawn.
    """
    count=0
    for i in range(len(markers)):
        if count==len(markers)-1:
            count=0
        else:
            count+=1
        new_image=cv2.line(image,markers[count-1],markers[count],(0, 0, 255),thickness)
    # raise NotImplementedError
    return new_image


def project_imageA_onto_imageB(imageA, imageB, homography,default=None):
    """Using the four markers in imageB, project imageA into the marked area.

    You should have used your find_markers method to find the corners and then
    compute the homography matrix prior to using this function.

    Args:
        image (numpy.array of uint8): image array
        image (numpy.array of uint8): image array
        homography (numpy.array): Perspective transformation matrix, 3 x 3
    Returns:
        numpy.array: combined image
    """
    if default!=None:
        heightB,widthB=imageB.shape[:2]
        heightA,widthA=imageA.shape[:2]
        total_width=widthA+widthB
        out_image = np.zeros((heightB, total_width , imageA.shape[2]), dtype=imageA.dtype)
    else:
        out_image = imageB.copy()
        heightB,total_width,_=out_image.shape
    # dst_image = np.zeros((heightB, widthB, imageA.shape[2]), dtype=np.float32)
    for y_out in range(heightB):
        for x_out in range(total_width):
            mapped_point = np.dot(np.linalg.inv(homography), np.array([x_out, y_out, 1]))
            # mapped_point /= mapped_point[2]
            x_in, y_in,_= mapped_point / mapped_point[2]
            if 0 <= x_in < imageA.shape[1] and 0 <= y_in < imageA.shape[0]:
                # out_image[y_out, x_out] = imageA[y_in, x_in]
                x1, y1 = int(x_in), int(y_in)
                x2, y2 = min(imageA.shape[1] - 1,x1 + 1), min(imageA.shape[0] - 1,y1 + 1)
                dx = x_in - x1
                dy = y_in - y1
                interpolated_value = (
                    imageA[y1, x1] * (1 - dx) * (1 - dy) +
                    imageA[y1, x2] * dx * (1 - dy) +
                    imageA[y2, x1] * (1 - dx) * dy +
                    imageA[y2, x2] * dx * dy
                )
                out_image[y_out, x_out]= interpolated_value
    return out_image

    # return 0


def find_four_point_transform(srcPoints, dstPoints):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform
    Hint: You will probably need to use least squares to solve this.
    Args:
        srcPoints (list): List of four (x,y) source points
        dstPoints (list): List of four (x,y) destination points
    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values
    """

    # raise NotImplementedError
    pre_matrix=[]
    for i in range(4):
        x, y = srcPoints[i]
        u, v = dstPoints[i]
        pre_matrix.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        pre_matrix.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    _, _, least_sq = np.linalg.svd(np.array(pre_matrix))
    h = least_sq[-1, :]
    homography = h.reshape((3, 3))
    return homography


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename
    """

    # Open file with VideoCapture and set result to 'video'. (add 1 line)
    video = cv2.VideoCapture(filename)

    # TODO
    # raise NotImplementedError
    while True:
        reach,frame=video.read()
        if not reach:
            video.release()
            yield None
            return
    # Close video (release) and yield a 'None' value. (add 2 lines)
    # video.release()
    #     yield None
        yield frame



class Automatic_Corner_Detection(object):

    def __init__(self):

        self.SOBEL_X = np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]).astype(np.float32)
        self.SOBEL_Y = np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]).astype(np.float32)



    def gradients(self, image_bw):
        '''Use convolution with Sobel filters to calculate the image gradient at each
            pixel location
            Input -
            :param image_bw: A numpy array of shape (M,N) containing the grayscale image
            Output -
            :return Ix: Array of shape (M,N) representing partial derivatives of image
                    in x-direction
            :return Iy: Array of shape (M,N) representing partial derivative of image
                    in y-direction
        '''

        # raise NotImplementedError
        padded_image = np.pad(image_bw, pad_width=1, mode='constant', constant_values=0)
        Ix = convolve2d(padded_image, self.SOBEL_X, mode='valid')
        Iy = convolve2d(padded_image, self.SOBEL_Y, mode='valid')
        reverse_Ix=np.where(Ix == 0, 0, -Ix)
        reverse_Iy=np.where(Iy == 0, 0, -Iy)
        return reverse_Ix,reverse_Iy


    def second_moments(self, image_bw, ksize=7, sigma=10):
        """ Compute second moments from image.
            Compute image gradients, Ix and Iy at each pixel, the mixed derivatives and then the
            second moments (sx2, sxsy, sy2) at each pixel,using convolution with a Gaussian filter. You may call the
            previously written function for obtaining the gradients here.
            Input -
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param ksize: size of 2d Gaussian filter
            :param sigma: standard deviation of Gaussian filter
            Output -
            :return sx2: np array of shape (M,N) containing the second moment in x direction
            :return sy2: np array of shape (M,N) containing the second moment in y direction
            :return sxsy: np array of shape (M,N) containing the second moment in the x then the
                    y direction
        """
        sx2, sy2, sxsy = None, None, None
        # raise NotImplementedError
        Ix, Iy = self.gradients(image_bw)
        sx2 = scipy.ndimage.gaussian_filter(Ix ** 2, sigma, mode='constant', truncate=(ksize-1)/(2*sigma))
        sy2 = scipy.ndimage.gaussian_filter(Iy ** 2, sigma, mode='constant', truncate=(ksize-1)/(2*sigma))
        sxsy = scipy.ndimage.gaussian_filter(Ix * Iy, sigma, mode='constant', truncate=(ksize-1)/(2*sigma))
        return sx2, sy2, sxsy


    def harris_response_map(self, image_bw, ksize=7, sigma=5, alpha=0.05):
        """Compute the Harris cornerness score at each pixel (See Szeliski 7.1.1)
            R = det(M) - alpha * (trace(M))^2
            where M = [S_xx S_xy;
                       S_xy  S_yy],
                  S_xx = Gk * I_xx
                  S_yy = Gk * I_yy
                  S_xy  = Gk * I_xy,
            and * is a convolutional operation over a Gaussian kernel of size (k, k).
            (You can verify that this is equivalent to taking a (Gaussian) weighted sum
            over the window of size (k, k), see how convolutional operation works here:
                http://cs231n.github.io/convolutional-networks/)
            Ix, Iy are simply image derivatives in x and y directions, respectively.
            Input-
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param ksize: size of 2d Gaussian filter
            :param sigma: standard deviation of gaussian filter
            :param alpha: scalar term in Harris response score
            Output-
            :return R: np array of shape (M,N), indicating the corner score of each pixel.
            """


        # raise NotImplementedError
        sx2, sy2, sxsy = self.second_moments(image_bw, ksize, sigma)
        R = sx2 * sy2 - sxsy**2 - alpha * ((sx2 + sy2)**2)
        R_normalized = (R - np.min(R)) / (np.max(R) - np.min(R))
        return R_normalized

    def nms_maxpool(self, R, k, ksize):
        """ Get top k interest points that are local maxima over (ksize,ksize)
        neighborhood.
        One simple way to do non-maximum suppression is to simply pick a
        local maximum over some window size (u, v). Note that this would give us all local maxima even when they
        have a really low score compare to other local maxima. It might be useful
        to threshold out low value score before doing the pooling.
        Threshold globally everything below the median to zero, and then
        MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
        with the maximum nearby value. Binarize the image according to
        locations that are equal to their maximum. Multiply this binary
        image, multiplied with the cornerness response values.
        Args:
            R: np array of shape (M,N) with score response map
            k: number of interest points (take top k by confidence)
            ksize: kernel size of max-pooling operator
        Returns:
            x: np array of shape (k,) containing x-coordinates of interest points
            y: np array of shape (k,) containing y-coordinates of interest points
        """

        R_threshold = np.where(R >= np.median(R), R, 0)
        max_pooled = scipy.ndimage.maximum_filter(R_threshold, size=(ksize,ksize))
        binary_image = (R_threshold == max_pooled).astype(np.float32)
        interested_points = binary_image * R
        top_K = np.argsort(interested_points.flatten())[::-1]
        top_K = np.unravel_index(top_K[:k], interested_points.shape)
        return top_K[1], top_K[0]
        


    def harris_corner(self, image_bw, k=100):
        """
            Implement the Harris Corner detector. You can call harris_response_map(), nms_maxpool() functions here.
            Input-
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param k: maximum number of interest points to retrieve
            Output-
            :return x: np array of shape (p,) containing x-coordinates of interest points
            :return y: np array of shape (p,) containing y-coordinates of interest points
            """
        R=self.harris_response_map(np.uint16(image_bw))
        x,y=self.nms_maxpool(R,k,7)
        return x,y
        # return 0
        # raise NotImplementedError
        # return [3,3]
        # return x1, y1





class Image_Mosaic(object):

    def __int__(self):
        pass

    def image_warp_inv(self, im_src, im_dst, H):
        '''
        Input -
        :param im_src: Image 1
        :param im_dst: Image 2
        :param H: numpy ndarray - 3x3 homography matrix
        Output -
        :return: Inverse Warped Resulting Image
        '''


        # raise NotImplementedError
        H_inverse=np.linalg.inv(H)
        # height, width=im_src.shape[:2]
        # warped_img=cv2.warpPerspective(im_dst,H,(width+im_dst.shape[:2][1],height))
        warped_img=project_imageA_onto_imageB(im_dst,im_src,H,100)
        return warped_img



    def output_mosaic(self, img_src, img_warped):
        '''
        Input -
        :param img_src: Image 1
        :param img_warped: Warped Image
        Output -
        :return: Output Image Mosiac
        '''


        # raise NotImplementedError
        out_img=img_warped.copy()
        out_img[0:img_src.shape[0],0:img_src.shape[1]]=img_src
        return out_img

def ransac(srcPoints, dstPoints):
    max_inliers = 0
    best_homography = None
    for _ in range(100):
        random_indices = np.random.choice(len(srcPoints), 4, replace=False)
        src_samples = np.array([srcPoints[i] for i in random_indices])
        dst_samples = np.array([dstPoints[i] for i in random_indices])
        homography =find_four_point_transform(src_samples, dst_samples)
        transformed_src = np.zeros_like(dstPoints)
        for i in range(len(srcPoints)):
            x, y = srcPoints[i]
            u, v, w = np.dot(homography, [x, y, 1])
            transformed_src[i] = (u / w, v / w)
        num_inliers = np.sum((np.linalg.norm(np.array(transformed_src) - np.array(dstPoints), axis=1)) < 0.1)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_homography = homography
    return best_homography

def detectAndDescribe(imageA,imageB):
    descriptor = cv2.xfeatures2d.SIFT_create()
    (detectA, featuresA) = descriptor.detectAndCompute(imageA, None)
    (detectB, featuresB) = descriptor.detectAndCompute(imageB, None)

    matcher=cv2.BFMatcher()
    AB_match=matcher.knnMatch(featuresA, featuresB, 2)
    match=[]
    for item in AB_match:
        if len(item) == 2 and item[0].distance<item[1].distance * 0.80:
            match.append((item[0].trainIdx, item[0].queryIdx))
    detectA=np.float32([item.pt for item in detectA])
    detectB=np.float32([item.pt for item in detectB])
    if len(match)>4:
        A_point = np.float32([detectA[i] for (_, i) in match])
        B_point = np.float32([detectB[i] for (i, _) in match])
        return A_point,B_point