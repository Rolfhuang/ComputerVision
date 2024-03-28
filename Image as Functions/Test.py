import numpy as np
import cv2

# def center_circle(length):
#     if length%2!=0:
#         return length//2
#     else:
#         return length//2-1

# def copy_paste_middle_circle(src, dst, radius):
#     temp_image_src = np.copy(src).astype(np.uint8)
#     temp_image_dst = np.copy(dst).astype(np.uint8)
#     radius=int(radius)
#     center_row = center_circle(temp_image_src.shape[0])
#     center_col = center_circle(temp_image_src.shape[1])
#     circle_mask = np.zeros_like(temp_image_src, dtype=np.uint8)
#     r_grid, c_grid = np.ogrid[:circle_mask.shape[0], :circle_mask.shape[1]]
#     circle_mask[((r_grid - center_row) ** 2 + (c_grid - center_col) ** 2) <= radius ** 2] = 255
    
#     circle_mask = cv2.resize(circle_mask, (temp_image_src.shape[1], temp_image_src.shape[0]))

#     circle_region = temp_image_src * (circle_mask / 255).astype(temp_image_src.dtype)

#     start_x = center_col - radius
#     end_x = center_col + radius
#     start_y = center_row - radius
#     end_y = center_row + radius

#     circle_width = end_x - start_x
#     circle_height = end_y - start_y

#     paste_x = center_circle(temp_image_dst.shape[1]- circle_width)
#     paste_y = center_circle(temp_image_dst.shape[0]- circle_height)

#     circle_mask2= np.zeros_like(temp_image_dst, dtype=np.uint8)
#     r_grid_d, c_grid_d = np.ogrid[:circle_mask2.shape[0], :circle_mask2.shape[1]]
#     circle_mask2[((r_grid_d - center_circle(temp_image_dst.shape[0])) ** 2 + (c_grid_d - center_circle(temp_image_dst.shape[1])) ** 2) <= radius ** 2] = 255

#     invertimage2=cv2.bitwise_not(circle_mask2)
#     circle_region2 = cv2.bitwise_and(temp_image_dst , invertimage2)

#     invertimage2 = 255 - circle_mask2
#     circle_region2 = temp_image_dst * (invertimage2 / 255).astype(temp_image_dst.dtype)

#     result = circle_region2[paste_y:paste_y + circle_height, paste_x:paste_x + circle_width] + circle_region[start_y:end_y, start_x:end_x]
#     result = np.clip(result, 0, 255)
#     temp_image_dst[paste_y:paste_y + circle_height, paste_x:paste_x + circle_width] = result
#     return temp_image_dst

# def main():
#     mono1=[[101,102,103,104,105],[106,107,108,109,110],[111,112,113,114,115],[116,117,118,119,120],[121,122,123,124,125]]
#     mono2=[[ 1, 2,  3,  4],[ 6,  7,  8,  9],[11, 12, 13, 14],[16, 17, 18, 19]]
#     replaced_img_circle = copy_paste_middle_circle(mono1, mono2, 1)
#     print(replaced_img_circle)
    
# if __name__ == "__main__":
#     main()

# def center_circle(length):
#     if length % 2 != 0:
#         return length // 2
#     else:
#         return length // 2 - 1

# def copy_paste_middle_circle(src, dst, radius):
#     temp_image_src = np.copy(src).astype(np.uint8)
#     temp_image_dst = np.copy(dst).astype(np.uint8)
#     radius = int(radius)
    
#     center_row = center_circle(temp_image_src.shape[0])
#     center_col = center_circle(temp_image_src.shape[1])
    
#     circle_mask = np.zeros_like(temp_image_src, dtype=np.uint8)
#     r_grid, c_grid = np.ogrid[:circle_mask.shape[0], :circle_mask.shape[1]]
#     circle_mask[((r_grid - center_row) ** 2 + (c_grid - center_col) ** 2) <= radius ** 2] = 255
    
#     circle_region = temp_image_src * (circle_mask / 255).astype(temp_image_src.dtype)
    
#     start_x = center_col - radius 
#     end_x = center_col + radius +1
#     start_y = center_row - radius 
#     end_y = center_row + radius +1
    
#     circle_width = end_x - start_x
#     circle_height = end_y - start_y
    
#     paste_x = center_circle(temp_image_dst.shape[1] - circle_width)
#     paste_y = center_circle(temp_image_dst.shape[0] - circle_height)
    
#     circle_mask2 = np.zeros_like(temp_image_dst, dtype=np.uint8)
#     r_grid_d, c_grid_d = np.ogrid[:circle_mask2.shape[0], :circle_mask2.shape[1]]
#     circle_mask2[((r_grid_d - center_circle(temp_image_dst.shape[0])) ** 2 + (c_grid_d - center_circle(temp_image_dst.shape[1])) ** 2) <= radius ** 2] = 255
    
#     invertimage2 = cv2.bitwise_not(circle_mask2)
#     circle_region2 = cv2.bitwise_and(temp_image_dst, invertimage2)
    
#     invertimage2 = 255 - circle_mask2
#     circle_region2 = temp_image_dst * (invertimage2 / 255).astype(temp_image_dst.dtype)
    
#     result = circle_region2[paste_y:paste_y + circle_height, paste_x:paste_x + circle_width] + circle_region[start_y:end_y, start_x:end_x]
#     result = np.clip(result, 0, 255)
#     temp_image_dst[paste_y:paste_y + circle_height, paste_x:paste_x + circle_width] = result
#     return temp_image_dst

def center_circle(length):
    if length % 2 != 0:
        return length // 2
    else:
        return length // 2 - 1

def copy_paste_middle_circle(src, dst, radius):
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
    end_x = center_col + radius + add# Fixed end_x index
    start_y = center_row - radius
    end_y = center_row + radius + add# Fixed end_y index
    
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

def main():
    # mono1 = np.array([[101, 102, 103, 104, 105],
    #                   [106, 107, 108, 109, 110],
    #                   [111, 112, 113, 114, 115],
    #                   [116, 117, 118, 119, 120],
    #                   [121, 122, 123, 124, 125]])
    
    # mono2 = np.array([[ 1, 2,  3,  4],
    #                   [ 6,  7,  8,  9],
    #                   [11, 12, 13, 14],
    #                   [16, 17, 18, 19]])
    # # mono1=np.array([[1, 2, 3, 4, 5],
    # #             [6, 7, 8, 9, 10],
    # #             [11, 12, 13, 14, 15],
    # #             [16, 17, 18, 19, 20],
    # #             [21, 22, 23, 24, 25]])
    # # mono2=np.array([[101, 102, 103, 104, 105],
    # #              [106, 107, 108, 109, 110],
    # #              [111, 112, 113, 114, 115],
    # #              [116, 117, 118, 119, 120],
    # #              [121, 122, 123, 124, 125]])
    # replaced_img_circle = copy_paste_middle_circle(mono1, mono2, 1)
    # print(replaced_img_circle)
    image1=r"./southafricaflagface.png"
    image1_arr=cv2.imread(image1)
    gray_image1=cv2.cvtColor(image1_arr,cv2.COLOR_BGR2GRAY)
    cv2.imwrite('ps1-southafricaflagface.png', gray_image1)
    
if __name__ == "__main__":
    main()