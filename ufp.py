import math
import random
import numpy as np

from copy import deepcopy
from collections import namedtuple
import sys

Rectangle = namedtuple('Rectangle', ['x', 'y', 'w', 'h'])


def phspprg(width, rectangles, sorting="height"):
    """
    The PH heuristic for the Strip Packing Problem. This is the RG variant, which means that rotations by
    90 degrees are allowed and that there is a guillotine constraint.

    Parameters
    ----------
    width
        The width of the strip.

    rectangles
        List of list containing width and height of every rectangle, [[w_1, h_1], ..., [w_n,h_h]].
        It is assumed that all rectangles can fit into the strip.

    sorting : string, {'width', 'height'}, default='width'
        The heuristic uses sorting to determine which rectangles to place first.
        By default sorting happens on the width but can be changed to height.

    Returns
    -------
    height
        The height of the strip needed to pack all the items.
    rectangles : list of namedtuple('Rectangle', ['x', 'y', 'w', 'h'])
        A list of rectangles, in the same order as the input list. This contains bottom left x and y coordinate and
        the width and height (which can be flipped compared to input).

    """
    if sorting not in ["width", "height" ]:
        raise ValueError("The algorithm only supports sorting by width or height but {} was given.".format(sorting))
    if sorting == "width":
        wh = 0
    else:
        wh = 1
    # logger.debug('The original array: {}'.format(rectangles))
    result = [None] * len(rectangles)
    remaining = deepcopy(rectangles)
    for idx, r in enumerate(remaining):
        if r[0] > r[1]:
            remaining[idx][0], remaining[idx][1] = remaining[idx][1], remaining[idx][0]
    # logger.debug('Swapped some widths and height with the following result: {}'.format(remaining))
    sorted_indices = sorted(range(len(remaining)), key=lambda x: -remaining[x][wh])
    # logger.debug('The sorted indices: {}'.format(sorted_indices))
    sorted_rect = [remaining[idx] for idx in sorted_indices]
    # logger.debug('The sorted array is: {}'.format(sorted_rect))
    x, y, w, h, H = 0, 0, 0, 0, 0
    while sorted_indices:
        idx = sorted_indices.pop(0)
        r = remaining[idx]
        if r[1] > width:
            result[idx] = Rectangle(x, y, r[0], r[1])
            x, y, w, h, H = r[0], H, width - r[0], r[1], H + r[1]
        else:
            result[idx] = Rectangle(x, y, r[1], r[0])
            x, y, w, h, H = r[1], H, width - r[1], r[0], H + r[0]
        recursive_packing(x, y, w, h, 1, remaining, sorted_indices, result)
        x, y = 0, H
    # logger.debug('The resulting rectangles are: {}'.format(result))

    return H, result


def phsppog(width, rectangles, sorting="width"):
    """
    The PH heuristic for the Strip Packing Problem. This is the OG variant, which means that rotations are
    NOT allowed and that there is a guillotine contraint.

    Parameters
    ----------
    width
        The width of the strip.

    rectangles
        List of list containing width and height of every rectangle, [[w_1, h_1], ..., [w_n,h_h]].
        It is assumed that all rectangles can fit into the strip.

    sorting : string, {'width', 'height'}, default='width'
        The heuristic uses sorting to determine which rectangles to place first.
        By default sorting happens on the width but can be changed to height.

    Returns
    -------
    height
        The height of the strip needed to pack all the items.
    rectangles : list of namedtuple('Rectangle', ['x', 'y', 'w', 'h'])
        A list of rectangles, in the same order as the input list. This contains bottom left x and y coordinate and
        the width and height (which can be flipped compared to input).

    """
    if sorting not in ["width", "height" ]:
        raise ValueError("The algorithm only supports sorting by width or height but {} was given.".format(sorting))
    if sorting == "width":
        wh = 0
    else:
        wh = 1
    # logger.debug('The original array: {}'.format(rectangles))
    result = [None] * len(rectangles)
    remaining = deepcopy(rectangles)
    # logger.debug('Swapped some widths and height with the following result: {}'.format(remaining))
    sorted_indices = sorted(range(len(remaining)), key=lambda x: -remaining[x][wh])
    # logger.debug('The sorted indices: {}'.format(sorted_indices))
    sorted_rect = [remaining[idx] for idx in sorted_indices]
    # logger.debug('The sorted array is: {}'.format(sorted_rect))
    x, y, w, h, H = 0, 0, 0, 0, 0
    while sorted_indices:
        idx = sorted_indices.pop(0)
        r = remaining[idx]
        result[idx] = Rectangle(x, y, r[0], r[1])
        x, y, w, h, H = r[0], H, width - r[0], r[1], H + r[1]
        recursive_packing(x, y, w, h, 0, remaining, sorted_indices, result)
        x, y = 0, H

    return H, result


def recursive_packing(x, y, w, h, D, remaining, indices, result):
    """Helper function to recursively fit a certain area."""
    priority = 6
    for idx in indices:
        for j in range(0, D + 1):
            if priority > 1 and remaining[idx][(0 + j) % 2] == w and remaining[idx][(1 + j) % 2] == h:
                priority, orientation, best = 1, j, idx
                break
            elif priority > 2 and remaining[idx][(0 + j) % 2] == w and remaining[idx][(1 + j) % 2] < h:
                priority, orientation, best = 2, j, idx
            elif priority > 3 and remaining[idx][(0 + j) % 2] < w and remaining[idx][(1 + j) % 2] == h:
                priority, orientation, best = 3, j, idx
            elif priority > 4 and remaining[idx][(0 + j) % 2] < w and remaining[idx][(1 + j) % 2] < h:
                priority, orientation, best = 4, j, idx
            elif priority > 5:
                priority, orientation, best = 5, j, idx
    if priority < 5:
        if orientation == 0:
            omega, d = remaining[best][0], remaining[best][1]
        else:
            omega, d = remaining[best][1], remaining[best][0]
        result[best] = Rectangle(x, y, omega, d)
        indices.remove(best)
        if priority == 2:
            recursive_packing(x, y + d, w, h - d, D, remaining, indices, result)
        elif priority == 3:
            recursive_packing(x + omega, y, w - omega, h, D, remaining, indices, result)
        elif priority == 4:
            min_w = sys.maxsize
            min_h = sys.maxsize
            for idx in indices:
                min_w = min(min_w, remaining[idx][0])
                min_h = min(min_h, remaining[idx][1])
            # Because we can rotate:
            min_w = min(min_h, min_w)
            min_h = min_w
            if w - omega < min_w:
                recursive_packing(x, y + d, w, h - d, D, remaining, indices, result)
            elif h - d < min_h:
                recursive_packing(x + omega, y, w - omega, h, D, remaining, indices, result)
            elif omega < min_w:
                recursive_packing(x + omega, y, w - omega, d, D, remaining, indices, result)
                recursive_packing(x, y + d, w, h - d, D, remaining, indices, result)
            else:
                recursive_packing(x, y + d, omega, h - d, D, remaining, indices, result)
                recursive_packing(x + omega, y, w - omega, h, D, remaining, indices, result)


def scale_boxes(bboxes, scale, image_shape=[1333, 1333]):
    """Expand an array of boxes by a given scale.

    Args:
        bboxes (Tensor): Shape (m, 4)
        scale (float): The scale factor of bboxes

    Returns:
        (Tensor): Shape (m, 4). Scaled bboxes
    """
    assert bboxes.shape[1] == 4
    w_half = (bboxes[:, 2] - bboxes[:, 0]) * 0.5
    h_half = (bboxes[:, 3] - bboxes[:, 1]) * 0.5
    x_c = (bboxes[:, 2] + bboxes[:, 0]) * 0.5
    y_c = (bboxes[:, 3] + bboxes[:, 1]) * 0.5

    w_half *= scale
    h_half *= scale
    w, h = image_shape

    boxes_scaled = np.zeros_like(bboxes)
    boxes_scaled[:, 0] = np.clip(x_c - w_half, 0, w - 1)
    boxes_scaled[:, 2] = np.clip(x_c + w_half, 0, w - 1)
    boxes_scaled[:, 1] = np.clip(y_c - h_half, 0, h - 1)
    boxes_scaled[:, 3] = np.clip(y_c + h_half, 0, h - 1)
    return boxes_scaled


def get_merge_bbox_aera(bbox1, bbox2):
    x11, y11, x12, y12 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x21, y21, x22, y22 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    bbox1_area = (x12 - x11) * (y12 - y11)
    bbox2_area = (x22 - x21) * (y22 - y21)
    x_min = min(x11, x21)
    y_min = min(y11, y21)
    x_max = max(x12, x22)
    y_max = max(y12, y22)
    merge_area = (x_max - x_min) * (y_max - y_min)
    return merge_area, bbox1_area + bbox2_area, [x_min, y_min, x_max, y_max]


def get_bbox_area(bbox1):
    x11, y11, x12, y12 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    bbox1_area = (x12 - x11) * (y12 - y11)
    return bbox1_area



def ForegroundRegionGeneration(bbox_list, scaled_bbox_list):
    num_bbox = bbox_list.shape[0]
    x1 = bbox_list[:, 0]
    y1 = bbox_list[:, 1]
    x2 = bbox_list[:, 2]
    y2 = bbox_list[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    avg_areas = areas
    cnt = np.array([1] * num_bbox)
    is_used = [True] * num_bbox
    for idx in range(num_bbox):
        if not is_used[idx]:
            continue
        A = scaled_bbox_list[idx]
        for jdx in range(num_bbox):
            if not is_used[jdx] or idx == jdx:
                continue
            merge_area, origin_area, merge_bbox = get_merge_bbox_aera(A, scaled_bbox_list[jdx])
            if merge_area < origin_area:
                A = merge_bbox
                is_used[jdx] = False
                avg_areas[idx] += avg_areas[jdx]
                cnt[idx] += cnt[jdx]
        scaled_bbox_list[idx] = A
    avg_areas = avg_areas/cnt
    scale_factor = np.array([1] * num_bbox)
    for idx in range(num_bbox):
        if avg_areas[idx] < 32 * 32:
            scale_factor[idx] = 4
        elif avg_areas[idx] < 96 * 96:
            scale_factor[idx] = 2
        else:
            scale_factor[idx] = 1
    return scaled_bbox_list[is_used], scale_factor[is_used]
   
def Packing(foreground_region, scale_factor, output_shape=[1333,800]):
    boxes = []
    for idx, _flag in enumerate(scale_factor):
        w = foreground_region[idx][2] - foreground_region[idx][0]
        h = foreground_region[idx][3] - foreground_region[idx][1]
        factor = scale_factor[idx]

        boxes.append([w * factor, h * factor])
    width_min = 300
    width_max = 2666
    while (width_min <= width_max):
        width_mid = (width_min + width_max) / 2
        height, rectangles = phsppog(width_mid, boxes, sorting='height')
        if height > width_mid:
            width_min = width_mid + 1
        else:
            width_max = width_mid - 1

    flag = [True] * foreground_region.shape[0]
    result = []
    new_width = 0
    new_height = 0
    for post_rec in rectangles:
        x = post_rec.x
        y = post_rec.y
        w = post_rec.w
        h = post_rec.h
        new_width = max(new_width, x + w)
        new_height = max(new_height, y + h)
        for idx in range(foreground_region.shape[0]):
            if not flag[idx]:
                continue
            factor = scale_factor[idx]
            _w = foreground_region[idx, 2] - foreground_region[idx, 0]
            _h = foreground_region[idx, 3] - foreground_region[idx, 1]
            if _w * factor == w and _h * factor == h:
                flag[idx] = False
                result.append([foreground_region[idx, 0], foreground_region[idx, 1], _w, _h, x, y, factor])

    return result, new_width, new_height
    


def UnifiedForegroundPacking(bbox_list, scale, input_shape, output_shape=[1333,800]):
    
    # scale bbox
    scaled_bbox_list = scale_boxes(bbox_list, scale, input_shape)

    
    # Foreground Region Generation Algorithm
    foreground_region, scale_factor = ForegroundRegionGeneration(bbox_list, scaled_bbox_list)


    # Packing
    result, new_width, new_height = Packing(foreground_region, scale_factor, output_shape)

    return result, new_width, new_height
