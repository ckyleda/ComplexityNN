import cv2
from skimage.util import view_as_blocks, view_as_windows
import numpy as np


def patch_based_local_symmetry(input_image, window_size=256, step_size=32):
    image = input_image
    image = image / 255.0
    block_shape = (window_size, window_size, 3)
    view = view_as_windows(image, block_shape, step=step_size)
    view = np.squeeze(view)
    rows = view.shape[0]
    columns = view.shape[1]
    horizontal_symmetry = np.zeros((rows, columns))
    vertical_symmetry = np.zeros((rows, columns))

    for r in range(rows):
        for c in range(columns):
            patch = view[r][c]
            patch_h, patch_w, patch_c = patch.shape
            # This is horizontal splitting
            subdivide_1 = patch[0:patch_h//2, :, :]
            subdivide_2 = patch[patch_h//2:, :, :]
            # Flip first subdivide horizontally as if it "mirrors" the second.
            subdivide_1 = cv2.flip(subdivide_1, 0)
            horizontal_symmetry[r, c] = np.linalg.norm(subdivide_1 - subdivide_2)

            # This is vertical splitting
            vert_left = patch[:, 0:patch_w//2, :]
            vert_right = patch[:, patch_w//2:, :]
            # Flip first subdivide vertically as if it "mirrors" the second.
            vert_left = cv2.flip(vert_left, 1)
            vertical_symmetry[r, c] = np.linalg.norm(vert_left - vert_right)

    return horizontal_symmetry, vertical_symmetry


def plot_patch_symmetry_scores(image_path, window_size, step_size, pad_image=False, blend=False, store_single=False):
    image = cv2.imread(image_path)
    channels = 1
    pad_amount = window_size
    if pad_image:
        image = cv2.copyMakeBorder(image, top=pad_amount, bottom=pad_amount, left=pad_amount, right=pad_amount,
                                   borderType=cv2.BORDER_REPLICATE, value=0)

    image_w, image_h, image_c = image.shape
    output_image = np.zeros((image_w, image_h, channels))
    horiz_scores, vert_scores = patch_based_local_symmetry(image, window_size=window_size, step_size=step_size)

    maximum_value = np.linalg.norm(np.zeros((window_size, window_size, channels)) - np.ones((window_size, window_size, channels)))
    norm_horiz_scores = 1.0 - (horiz_scores / maximum_value)
    norm_vert_scores = 1.0 - (vert_scores / maximum_value)
    view = view_as_windows(output_image, (window_size, window_size, channels), step=step_size)
    view = np.squeeze(view)
    rows = view.shape[0]
    columns = view.shape[1]

    for r in range(rows):
        for c in range(columns):
            if channels == 3:
                view[r, c, :, :, 0] += (norm_horiz_scores[r, c] + norm_vert_scores[r, c])
            if channels == 1:
                view[r, c, :, :] += (norm_horiz_scores[r, c] + norm_vert_scores[r, c])

    output_image = output_image / output_image.max()
    if store_single:
        return np.mean(output_image)

    output_image = output_image[pad_amount:image_w-pad_amount, pad_amount:image_h-pad_amount, :]

    if blend:
        original_image = cv2.imread(image_path)
        output_image = output_image.astype(np.uint8)
        alpha = 0.4
        blended_image = cv2.addWeighted(original_image, alpha, output_image, 1 - alpha, 0)
        cv2.imwrite('blended.png', blended_image)
    return output_image, np.mean(output_image)
