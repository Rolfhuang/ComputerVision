"""Problem Set 4: Motion Detection"""

import os

import cv2
import numpy as np
import ps4
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
import pandas as pd
# I/O directories
input_dir = "input_images"
output_dir = "./"

# Utility code
def quiver(u, v, scale, stride, color=(0, 255, 0)):

    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in range(0, v.shape[0], stride):

        for x in range(0, u.shape[1], stride):

            cv2.line(img_out, (x, y),
                     (x + int(u[y, x] * scale), y + int(v[y, x] * scale)),
                     color, 1)
            cv2.circle(img_out,
                       (x + int(u[y, x] * scale), y + int(v[y, x] * scale)), 1,
                       color, 1)
    return img_out


# Functions you need to complete:


def scale_u_and_v(u, v, level, pyr):
    """Scales up U and V arrays to match the image dimensions assigned 
    to the first pyramid level: pyr[0].

    You will use this method in part 3. In this section you are asked 
    to select a level in the gaussian pyramid which contains images 
    that are smaller than the one located in pyr[0]. This function 
    should take the U and V arrays computed from this lower level and 
    expand them to match a the size of pyr[0].

    This function consists of a sequence of ps4.expand_image operations 
    based on the pyramid level used to obtain both U and V. Multiply 
    the result of expand_image by 2 to scale the vector values. After 
    each expand_image operation you should adjust the resulting arrays 
    to match the current level shape 
    i.e. U.shape == pyr[current_level].shape and 
    V.shape == pyr[current_level].shape. In case they don't, adjust
    the U and V arrays by removing the extra rows and columns.

    Hint: create a for loop from level-1 to 0 inclusive.

    Both resulting arrays' shapes should match pyr[0].shape.

    Args:
        u: U array obtained from ps4.optic_flow_lk
        v: V array obtained from ps4.optic_flow_lk
        level: level value used in the gaussian pyramid to obtain U 
               and V (see part_3)
        pyr: gaussian pyramid used to verify the shapes of U and V at 
             each iteration until the level 0 has been met.

    Returns:
        tuple: two-element tuple containing:
            u (numpy.array): scaled U array of shape equal to 
                             pyr[0].shape
            v (numpy.array): scaled V array of shape equal to 
                             pyr[0].shape
    """

    # TODO: Your code here
    # raise NotImplementedError
    for curr_level in range(level - 1, -1, -1):
        # Expand U and V using the expand_image function and scale by 2
        u = ps4.expand_image(u) * 2
        v = ps4.expand_image(v) * 2

        # Adjust the shapes to match the current level
        target_shape = pyr[curr_level].shape
        u = u[:target_shape[0], :target_shape[1]]
        v = v[:target_shape[0], :target_shape[1]]

    return u, v


def part_1a():

    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'),
                         0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR2.png'),
                          0) / 255.
    shift_r5_u5 = cv2.imread(
        os.path.join(input_dir, 'TestSeq', 'ShiftR5U5.png'), 0) / 255.

    # Optional: smooth the images if LK doesn't work well on raw images
    k_size = 15  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 1  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-1.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    # input images first.

    k_size = 31  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 1  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r5_u5, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)


def part_1b():
    """Performs the same operations applied in part_1a using the images
    ShiftR10, ShiftR20 and ShiftR40.

    You will compare the base image Shift0.png with the remaining
    images located in the directory TestSeq:
    - ShiftR10.png
    - ShiftR20.png
    - ShiftR40.png

    Make sure you explore different parameters and/or pre-process the
    input images to improve your results.

    In this part you should save the following images:
    - ps4-1-b-1.png
    - ps4-1-b-2.png
    - ps4-1-b-3.png

    Returns:
        None
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'),
                         0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'),
                           0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'),
                           0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'),
                           0) / 255.

    # raise NotImplementedError
    k_size = 40  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 1  # TODO: Select a sigma value if you are using a gaussian kernel
    u10, v10 = ps4.optic_flow_lk(shift_0, shift_r10, k_size, k_type, sigma)
    u20, v20 = ps4.optic_flow_lk(shift_0, shift_r20, k_size, k_type, sigma)
    u40, v40 = ps4.optic_flow_lk(shift_0, shift_r40, k_size, k_type, sigma)

    # Flow image
    u_v10 = quiver(u10, v10, scale=3, stride=10)
    u_v20 = quiver(u20, v20, scale=3, stride=10)
    u_v40 = quiver(u40, v40, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-1.png"), u_v10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-2.png"), u_v20)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-3.png"), u_v40)


def part_2():

    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.

    # 2a
    levels = 4
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_01_g_pyr_img = ps4.create_combined_img(yos_img_01_g_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-a-1.png"),
                yos_img_01_g_pyr_img)

    # 2b
    yos_img_01_l_pyr = ps4.laplacian_pyramid(yos_img_01_g_pyr)

    yos_img_01_l_pyr_img = ps4.create_combined_img(yos_img_01_l_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-b-1.png"),
                yos_img_01_l_pyr_img)


def part_3a_1():
    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.

    levels = 1  # Define the number of pyramid levels
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)

    level_id = 0  # TODO: Select the level number (or id) you wish to use
    k_size = 15  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 1  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
                             yos_img_02_g_pyr[level_id], k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-1.png"),
                ps4.normalize_and_scale(diff_yos_img_01_02))


def part_3a_2():
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    yos_img_03 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.

    levels = 2  # Define the number of pyramid levels
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
    yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03, levels)

    level_id = 1  # TODO: Select the level number (or id) you wish to use
    k_size = 15  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 1  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id],
                             yos_img_03_g_pyr[level_id], k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)

    diff_yos_img = yos_img_02 - yos_img_03_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-2.png"),
                ps4.normalize_and_scale(diff_yos_img))


def part_4a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'),
                         0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'),
                           0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'),
                           0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'),
                           0) / 255.

    levels = 3  # TODO: Define the number of levels
    k_size = 15  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 1  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u10, v10, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-1.png"), u_v)

    # You may want to try different parameters for the remaining function
    # calls.
    u20, v20 = ps4.hierarchical_lk(shift_0, shift_r20, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u20, v20, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-2.png"), u_v)

    u40, v40 = ps4.hierarchical_lk(shift_0, shift_r40, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    u_v = quiver(u40, v40, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-3.png"), u_v)


def part_4b():
    urban_img_01 = cv2.imread(os.path.join(input_dir, 'Urban2', 'urban01.png'),
                              0) / 255.
    urban_img_02 = cv2.imread(os.path.join(input_dir, 'Urban2', 'urban02.png'),
                              0) / 255.

    levels = 1  # TODO: Define the number of levels
    k_size = 115  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-1.png"), u_v)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
                                   border_mode)

    diff_img = urban_img_01 - urban_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
                ps4.normalize_and_scale(diff_img))


def part_5a():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    def frame_interpolation(img1, img2, u, v, interpolation_factors):
        interpolated_frames = []
        for factor in interpolation_factors:
            interpolated_u = u * -factor
            interpolated_v = v * -factor
            warped_img1 = ps4.warp(img1, interpolated_u, interpolated_v, cv2.INTER_LINEAR, cv2.BORDER_REFLECT101)
            # warped_img2 = ps4.warp(img2, interpolated_u, interpolated_v, cv2.INTER_LINEAR, cv2.BORDER_REFLECT101)
            # interpolated_frame = ((1 - factor) * ps4.normalize_and_scale(warped_img1)) + (factor * ps4.normalize_and_scale(warped_img2))
            # interpolated_frame = np.clip(interpolated_frame, 0, 1)
            interpolated_frames.append(warped_img1)

        return interpolated_frames
    # raise NotImplementedError
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'),
                         0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'),
                           0) / 255.

    levels = 5  # TODO: Define the number of levels
    k_size = 31  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 1  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    interpolation_factors = [0.0,0.2, 0.4, 0.6, 0.8,1.0]
    interpolated_frames = frame_interpolation(shift_0, shift_r10, u10, v10, interpolation_factors)
    first_combine=ps4.create_combined_img(interpolated_frames[:3])
    second_combine=ps4.create_combined_img(interpolated_frames[3:])
    combine_out= np.zeros((first_combine.shape[0]+second_combine.shape[0], first_combine.shape[1]), dtype=np.float64)
    combine_out[:first_combine.shape[0],:]=first_combine
    combine_out[first_combine.shape[0]:,:]=second_combine
    
    # u_v = quiver(u10, v10, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-5-a-1.png"), combine_out)
    # cv2.imwrite(os.path.join(output_dir, "ps4-5-a-2.png"), second_combine)


def part_5b():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    def frame_interpolation(img1, img2, u, v, interpolation_factors):
        interpolated_frames = []
        for factor in interpolation_factors:
            interpolated_u = u * factor
            interpolated_v = v * factor
            warped_img1 = ps4.warp(img1, interpolated_u, interpolated_v, cv2.INTER_CUBIC, cv2.BORDER_REFLECT101)
            warped_img2 = ps4.warp(img2, interpolated_u, interpolated_v, cv2.INTER_CUBIC, cv2.BORDER_REFLECT101)
            interpolated_frame = ((1 - factor) * warped_img1) + (factor * 0.5 * warped_img2)
            interpolated_frame = np.clip(interpolated_frame, 0, 1)
            interpolated_frames.append(interpolated_frame)

        return interpolated_frames
    
    shift_mc1 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc01.png'),
                         0) / 255.
    shift_mc2 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc02.png'),
                           0) / 255.
    shift_mc3 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc03.png'),
                           0) / 255.

    levels = 4  # TODO: Define the number of levels
    k_size = 51  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 1  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u12, v12 = ps4.hierarchical_lk(shift_mc1, shift_mc2, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    u23, v23 = ps4.hierarchical_lk(shift_mc2, shift_mc3, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    # raise NotImplementedError
    # u_v = quiver(u12, v12, scale=3, stride=10)
    # cv2.imwrite(os.path.join(output_dir, "ps4-testuv.png"), u_v)
    interpolation_factors = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # interpolated_frames_12 = create_missing_frames(shift_mc1, u12, v12, interpolation_factors, interpolation, border_mode)
    # first_combine_12=ps4.create_combined_img(interpolated_frames_12[:3])
    # second_combine_12=ps4.create_combined_img(interpolated_frames_12[3:])
    # combine_out_12= np.zeros((first_combine_12.shape[0]+second_combine_12.shape[0], first_combine_12.shape[1]), dtype=np.float64)
    # combine_out_12[:first_combine_12.shape[0],:]=first_combine_12
    # combine_out_12[first_combine_12.shape[0]:,:]=second_combine_12
    interpolated_frames_12 = frame_interpolation(shift_mc1, shift_mc2, u12, v12, interpolation_factors)
    first_combine_12=ps4.create_combined_img(interpolated_frames_12[:3])
    second_combine_12=ps4.create_combined_img(interpolated_frames_12[3:])
    combine_out_12= np.zeros((first_combine_12.shape[0]+second_combine_12.shape[0], first_combine_12.shape[1]), dtype=np.float64)
    combine_out_12[:first_combine_12.shape[0],:]=first_combine_12
    combine_out_12[first_combine_12.shape[0]:,:]=second_combine_12
    
    interpolated_frames_23 = frame_interpolation(shift_mc2, shift_mc3, u23, v23, interpolation_factors)
    first_combine_23=ps4.create_combined_img(interpolated_frames_23[:3])
    second_combine_23=ps4.create_combined_img(interpolated_frames_23[3:])
    combine_out_23= np.zeros((first_combine_23.shape[0]+second_combine_23.shape[0], first_combine_23.shape[1]), dtype=np.float64)
    combine_out_23[:first_combine_23.shape[0],:]=first_combine_23
    combine_out_23[first_combine_23.shape[0]:,:]=second_combine_23
    # u_v = quiver(u10, v10, scale=3, stride=10)

    cv2.imwrite(os.path.join(output_dir, "ps4-5-b-1.png"), combine_out_12)
    cv2.imwrite(os.path.join(output_dir, "ps4-5-b-2.png"), combine_out_23)


def part_6():
    """Challenge Problem

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    def read_video(video_file, show=False):
        frames = []
        cap = cv2.VideoCapture(video_file)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    video_path=os.path.join(input_dir, 'videos')
    video_list=os.listdir(video_path)
    classifed=[]
    levels = 4  # TODO: Define the number of levels
    k_size = 50  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 1  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_LINEAR  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    video_features = []
    video_action_labels = []
    video_labels = {'Running':1, 'Walking':2, 'Clapping':3}
    data_set=[]
    X=[]
    y=[]
    for item in video_list:
        sep_video_path=os.path.join(video_path,item)
        frames=ps4.read_video(sep_video_path)
        flow_magnitude=None
        max_magnitude=None
        optical_flow_data=[]
        label=item.split("_")[1]
        optical_flow_frames=[]
        for i in range(1,len(frames)):
            frame1=frames[i-1]
            frame2=frames[i]
            frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            U, V= ps4.hierarchical_lk(frame1,frame2,levels,k_size,k_type,sigma,interpolation,border_mode)
            velocity = np.sqrt((U ** 2) + (V ** 2))

            # Append the velocity as optical flow for this frame
            optical_flow_frames.append(velocity)
        pclass=ps4.classify_video(optical_flow_frames)
        print("Classify", item, "is", pclass[0], "and the confindence is", pclass[1])
        # if max_magnitude > 10.0:
        #     print("Classify", item, "is", 1)
        # elif max_magnitude > 5.0:
        #     print("Classify", item, "is", 2)
        # else:
        #     print("Classify", item, "is", 3)
        # new_x = np.array(X)
        # new_y = np.array(y)
        # X_train, X_test, y_train, y_test = train_test_split(new_x, new_y, test_size=0.2, random_state=42)
        # clf = DecisionTreeClassifier()
        # clf.fit(X_train, y_train)
        # important_features = np.argsort(clf.feature_importances_)[-4:]
        # feature_names = [f'Feature_{i}' for i in important_features]

        # # Export the decision tree to text using the selected feature names
        # tree_text = export_text(clf, feature_names=feature_names)
        # print("Decision Tree:\n", tree_text)
    # raise NotImplementedError


if __name__ == '__main__':
    # part_1a()
    # part_1b()
    # part_2()
    # part_3a_1()
    # part_3a_2()
    # part_4a()
    # part_4b()
    # part_5a()
    part_5b()
    # part_6()
