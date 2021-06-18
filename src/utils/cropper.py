import math
import numpy as np


def calculate_lastx_y_z(resolution, x_steps, z_steps, crop_size_xy, crop_size_z):
    """
    Calculates where last crop in x, last crop in y and last crop in z begins.

    Parameters
    ----------
    resolution:  	    tuple
                    	Resolution of original tof mra, e.g. (560,560,140)
    x_steps:  	        int
                    	steps in the x,y direction (e.g. if 560x560 with crop_size_xyz 64 -> 9x9)
    z_steps:  	        int
                    	steps in the z direction (e.g. if 140 slices with crop_size_xyz 64 -> 3)
    crop_size_xy:  	    int
                    	crop size, x=y, e.g.: 256
    crop_size_z:  	    int
                    	crop size, z, e.g.: 64

    Returns
    -------
    last_x:		        int
            			the pixel in the image, where last x crop begins, such that it fits -> the crop is possible.
    last_y:		        int
            			the pixel in the image, where last y crop begins, such that it fits -> the crop is possible.
    last_z:		        int
            			the pixel in the image, where last z crop begins, such that it fits -> the crop is possible.
    """

    rest_x = resolution[0] - (crop_size_xy * x_steps)
    overlap_rest_x = crop_size_xy - rest_x
    last_x = (crop_size_xy * x_steps) - overlap_rest_x
    last_y = last_x

    rest_z = resolution[2] - (crop_size_z * z_steps)
    overlap_rest_z = crop_size_z - rest_z
    last_z = (crop_size_z * z_steps) - overlap_rest_z

    return last_x, last_y, last_z


def calculate_cropped_array(data, crop_size_xy, crop_size_z, overlap):
    """
    Calculates the cropped array, containing all cropps of the current MRA-TOF image (data) with desired crop size and overlap
    (e.g. image 560x560x140 will be cropped into many 64x64x64 images)

    Parameters
    ----------
    data:		        list
            			List of npy arrays containing data to be cropped, e.g. one 560x560x140 image.
    crop_size_xy        int
                        Crop in the x and y dim. Due to the quadratic form of the images, crop size of x and y dims are equal
    crop_size_z         int
                        Crop in the z dim (slices dim). Amount of slices is not dependent on x and y dim.
    overlap             int
                        Overlap for the cropping. We don't want to cut the aneurysms through cropping.

    Returns
    -------
    cropped_array:	    list
                        List of npy arrays including cropped data.
    """

    x_steps = math.ceil((data.shape[0] - crop_size_xy) / (crop_size_xy - overlap)) + 1
    y_steps = math.ceil((data.shape[1] - crop_size_xy) / (crop_size_xy - overlap)) + 1
    z_steps = math.ceil((data.shape[2] - crop_size_z) / (crop_size_z - overlap)) + 1

    startx, starty, startz = 0, 0, 0
    last_x, last_y, last_z = calculate_lastx_y_z(data.shape, x_steps, z_steps, crop_size_xy, crop_size_z)

    cropped_array = []
    for z_step in range(z_steps):
        for y_step in range(y_steps):
            for x_step in range(x_steps):
                if x_step == x_steps - 1 and y_step == y_steps - 1 and z_step == z_steps - 1:
                    cropped_array.append(data[last_x:, last_y:, last_z:])
                elif x_step == x_steps - 1 and y_step == y_steps - 1:
                    cropped_array.append(data[last_x:, last_y:, startz: startz + crop_size_z])
                elif x_step == x_steps - 1 and z_step == z_steps - 1:
                    cropped_array.append(data[last_x:, starty: starty + crop_size_xy, last_z:])
                elif y_step == y_steps - 1 and z_step == z_steps - 1:
                    cropped_array.append(data[startx: startx + crop_size_xy, last_y:, last_z:])

                elif x_step == x_steps - 1:
                    cropped_array.append(data[last_x:, starty: starty + crop_size_xy, startz: startz + crop_size_z])

                elif y_step == y_steps - 1:
                    cropped_array.append(data[startx: startx + crop_size_xy, last_y:, startz: startz + crop_size_z])

                elif z_step == z_steps - 1:
                    cropped_array.append(data[startx: startx + crop_size_xy, starty: starty + crop_size_xy, last_z:])
                else:
                    cropped_array.append(data[startx: startx + crop_size_xy, starty: starty + crop_size_xy,
                                         startz: startz + crop_size_z])
                startx = startx + crop_size_xy - overlap

            starty = starty + crop_size_xy - overlap
            startx = 0
        startz = startz + crop_size_z - overlap
        startx = 0
        starty = 0

    return cropped_array


### reconstruction

def reconstruct_orig_img(crp_arr, orig_resolution, crop_size_xy, crop_size_z, overlap):
    """
    Calculates the reconstruction of the cropped_array from method "calculate_cropped_array"

    Parameters
    ----------
    crp_arr:		    list
            			Array with cropped data which should be reconstructed. (e.g. list with 64x64x64 cubes)
    orig_resolution     tuple
                        Desired resolution of the data before cropping, e.g. (560,560,140)
    crop_size_xy        int
                        Crop in the x and y dim. Due to the quadratic form of the images, crop size of x and y dims are equal
    crop_size_z         int
                        Crop in the z dim (slices dim). Amount of slices is not dependent on x and y dim.
    overlap             int
                        Overlap for the cropping. We don't want to cut the aneurysms through cropping.

    Returns
    -------
    orig_img:	        list
                        Original reconstructed image. Here overlapped data is summed up at the corresponding pixels.
    orig_img_overlap_count:	    list
                        Array of the same form as the orig_img. It counts how many times each pixel was overlapped.
                        Can be used in order to calculate the previous value through division orig_img / orig_img_overlap_count.
    """

    x_steps = math.ceil((orig_resolution[0] - crop_size_xy) / (crop_size_xy - overlap)) + 1
    y_steps = math.ceil((orig_resolution[1] - crop_size_xy) / (crop_size_xy - overlap)) + 1
    z_steps = math.ceil((orig_resolution[2] - crop_size_z) / (crop_size_z - overlap)) + 1

    reshaped_crp_arr = np.array(crp_arr).reshape((z_steps, y_steps, x_steps, crop_size_xy, crop_size_xy, crop_size_z))
    startx = 0
    starty = 0
    startz = 0

    x_shape = reshaped_crp_arr.shape[1]
    y_shape = reshaped_crp_arr.shape[2]
    z_shape = reshaped_crp_arr.shape[0]
    last_overlapxy = orig_resolution[0] - crop_size_xy
    last_overlapz = orig_resolution[2] - crop_size_z

    orig_img = np.zeros(orig_resolution)
    orig_img_overlap_count = np.zeros(orig_resolution)
    for z in range(z_shape):
        for y in range(y_shape):
            for x in range(x_shape):

                if x == x_shape - 1 and y == y_shape - 1 and z == z_shape - 1:
                    orig_img[last_overlapxy:, last_overlapxy:, last_overlapz:] += reshaped_crp_arr[z][y][x]
                    orig_img_overlap_count[last_overlapxy:, last_overlapxy:, last_overlapz:] += 1

                elif x == x_shape - 1 and y == y_shape - 1:
                    orig_img[last_overlapxy:, last_overlapxy:, startz: startz + crop_size_z] += reshaped_crp_arr[z][y][
                        x]
                    orig_img_overlap_count[last_overlapxy:, last_overlapxy:, startz: startz + crop_size_z] += 1

                elif x == x_shape - 1 and z == z_shape - 1:
                    orig_img[last_overlapxy:, starty: starty + crop_size_xy, last_overlapz:] += reshaped_crp_arr[z][y][
                        x]
                    orig_img_overlap_count[last_overlapxy:, starty: starty + crop_size_xy, last_overlapz:] += 1

                elif y == y_shape - 1 and z == z_shape - 1:
                    orig_img[startx: startx + crop_size_xy, last_overlapxy:, last_overlapz:] += reshaped_crp_arr[z][y][
                        x]
                    orig_img_overlap_count[startx: startx + crop_size_xy, last_overlapxy:, last_overlapz:] += 1

                elif x == x_shape - 1:
                    orig_img[last_overlapxy:, starty: starty + crop_size_xy, startz: startz + crop_size_z] += \
                    reshaped_crp_arr[z][y][x]
                    orig_img_overlap_count[last_overlapxy:, starty: starty + crop_size_xy,
                    startz: startz + crop_size_z] += 1

                elif y == y_shape - 1:
                    orig_img[startx: startx + crop_size_xy, last_overlapxy:, startz: startz + crop_size_z] += \
                    reshaped_crp_arr[z][y][x]
                    orig_img_overlap_count[startx: startx + crop_size_xy, last_overlapxy:,
                    startz: startz + crop_size_z] += 1

                elif z == z_shape - 1:
                    orig_img[startx: startx + crop_size_xy, starty: starty + crop_size_xy, last_overlapz:] += \
                    reshaped_crp_arr[z][y][x]
                    orig_img_overlap_count[startx: startx + crop_size_xy, starty: starty + crop_size_xy,
                    last_overlapz:] += 1

                else:
                    orig_img[startx: startx + crop_size_xy, starty: starty + crop_size_xy,
                    startz: startz + crop_size_z] += reshaped_crp_arr[z][y][x]
                    orig_img_overlap_count[startx: startx + crop_size_xy, starty: starty + crop_size_xy,
                    startz: startz + crop_size_z] += 1

                startx = startx + crop_size_xy - overlap
            starty = starty + crop_size_xy - overlap
            startx = 0
        startz = startz + crop_size_z - overlap
        startx = 0
        starty = 0

    return orig_img, orig_img_overlap_count
