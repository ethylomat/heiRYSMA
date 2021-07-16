import nibabel as nib
import os
from skimage.transform import resize

from src.utils import cropper


def load_niigz_as_npy_challenge(data_path):
    """
    Loads the nii.gz data as npy array.

    Parameters
    ----------
    data_path:  	    String
                    	Absolute path to the data folder with single TOF MRA nii.gz data.

    Returns
    -------
    data_orig:		    array
            			npy array containing TOF MRA data.
    """

    sub_dirs = [x[0] for x in  os.walk(data_path)]

    if(os.path.join(data_path, 'orig') not in sub_dirs):
        raise ValueError("No orig directory in input directory.")

    data_orig = nib.load(os.path.join(data_path, 'orig', 'TOF.nii.gz')).get_fdata()

    print('DONE: niigz data loaded as npy')
    return data_orig


def resize_width_height_skimage_challenge(data_orig, resolution_xy):
    """
    Resizes the numpy array to the 256x256xAmountSlices.

    Parameters
    ----------
    data_orig:		    array
            			npy array, TOF-MRA data.
    resolution_xy:      int
                        needed for down/upscaling, e.g. 256 for 256x256

    Returns
    -------
    resized_data_orig:	array
            			npy array, resized to 256x256xX.

    """

    target_resolution_curr = (resolution_xy, resolution_xy, data_orig.shape[2])
    resized_data_orig = resize(data_orig, target_resolution_curr)

    print('DONE: resizing images')
    return resized_data_orig


def crop_data_challenge(data_orig):
    """
    Crops the TOF MRA 3D image into smaller cubes with wanted size and overlap (e.g. image 560x560x140 will be cropped into many 256x256x8 images)


    Parameters
    ----------
    data_orig:		    list
            			List of npy arrays containing TOF MRA data.

    Returns
    -------
    cropped_data_orig:	list
                        List of npy arrays including cropped data and augmented data. Also balancing is involved. So many non aneurysm datasets are there as with aneurysms.
    """

    cropped_data_orig = cropper.calculate_cropped_array(data_orig, 256, 8, 1)

    print('DONE: cropping images')

    return cropped_data_orig