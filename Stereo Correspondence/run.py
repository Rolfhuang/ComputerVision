import numpy as np
import cv2
import networkx as nx
import maxflow
from scipy.signal import convolve2d


def calculate_ssd(img1, img2, pixels):
    ssd = np.zeros((img1.shape[0], img1.shape[1], pixels))
    for i in range(pixels):
        diff = img1 - np.roll(img2, i, axis=1)
        ssd[:, :, i] = convolve2d(diff**2, np.ones((1, 1)), mode='valid')
    return ssd

def compute_disparity_ssd(img1, img2, direction, kernel_size, pixels):
    '''
    ssd disparity map
    '''
    ssd = calculate_ssd(img1, img2, pixels)
    disparity_map = np.zeros_like(img1)
    for i in range(0, img1.shape[1] - kernel_size + 1):
        for j in range(0, img1.shape[0] - kernel_size + 1):
            if direction == 0:
                ssd_tx = np.sum(ssd[j:(j+kernel_size), i:(i+kernel_size), :], axis=(0, 1))
            elif direction == 1:
                ssd_tx = np.sum(ssd[(j - kernel_size + 1):(j + 1), i:(i + kernel_size), :], axis=(0, 1))
            disparity_map[j, i] = np.argmin(ssd_tx)
    return disparity_map

def compute_disparity_advanced(left_image, right_image, cost_weight, smoothness_weight=20):
    '''
    graph cut
    '''
    disparity_map = np.abs(left_image - right_image)
    data_cost = disparity_map.astype(np.float32)
    G = nx.Graph()
    height, width = disparity_map.shape
    for i in range(height):
        for j in range(width):
            node_id = i * width + j
            G.add_node(node_id, data_cost=data_cost[i, j])

    for i in range(height - 1):
        for j in range(width - 1):
            node_id1 = i * width + j
            node_id2 = (i + 1) * width + (j + 1)
            smoothness_cost = smoothness_weight * np.abs(disparity_map[i, j] - disparity_map[i + 1, j + 1])
            G.add_edge(node_id1, node_id2, weight=cost_weight * data_cost[i, j] + smoothness_cost)
    _, partition = nx.minimum_cut(G, 0, (height - 1) * width + (width - 1))
    disparity_map_result = np.zeros_like(disparity_map)
    for node_id, _ in enumerate(partition[0]):
        i = node_id // width
        j = node_id % width
        disparity_map_result[i, j] = disparity_map[i, j]
    return disparity_map_result


if __name__ == '__main__':
    left_img = cv2.imread("left3.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    right_img = cv2.imread("right3.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)

    l = compute_disparity_ssd(left_img, right_img, 0, 5, 50)
    cv2.normalize(l,l,0,255,cv2.NORM_MINMAX)
    # cl = cv2.applyColorMap(l.astype(np.uint8),cv2.COLORMAP_RAINBOW)
    cv2.imwrite("Stereo_ltr.png",l)

    r = compute_disparity_ssd(left_img, right_img, 1, 5, 50)
    cv2.normalize(r,r,0,255,cv2.NORM_MINMAX)
    # cr = cv2.applyColorMap(r.astype(np.uint8),cv2.COLORMAP_RAINBOW)
    cv2.imwrite("Stereo_rtl.png",r)

    disparity_map_graph_cut = compute_disparity_advanced(left_img, right_img, 0.0001)
    # disparity_map_graph_cut = stereo_matching(left_img, right_img)
    cv2.normalize(disparity_map_graph_cut,disparity_map_graph_cut,0,255,cv2.NORM_MINMAX)
    # cgt = cv2.applyColorMap(disparity_map_graph_cut.astype(np.uint8),cv2.COLORMAP_JET)
    cv2.imwrite("Stereo_graph_cut.png",disparity_map_graph_cut)


