import numpy as np
import nibabel as nib
import os
from skimage.transform import resize


def load_niigz_as_npy(data_path):
    """
    Loads the nii.gz data as list of npy arrays.

    Parameters
    ----------
    data_path:  	    String
                    	Path to the data folder.

    Returns
    -------
    data_orig:		    list
            			List of (different shapes) npy arrays containing TOF MRA data.
    data_mask:		    list
            			List of (different shapes) npy arrays containing binary masks corresponding to data_orig.
    """

    sub_dirs = [x[0] for x in  os.walk(data_path)]
    if((os.path.join(data_path, '10001') not in sub_dirs) and (os.path.join(data_path, 'preprocessed') not in sub_dirs)):
        raise ValueError("There is no data in the directory to work with! Check the data directory / data_path to data directory and try again...")
    data_orig = []
    data_mask = []

    print('Preprocessing from niigz files in progress...')
    for dir in sub_dirs:
        if(dir.endswith('orig')):
            data_orig.append(nib.load(os.path.join(dir, 'TOF.nii.gz')).get_fdata())
            data_mask.append(nib.load(os.path.join(dir.partition('orig')[0], 'aneurysms.nii.gz')).get_fdata())

    print('DONE: niigz data loaded as npy')
    return data_orig, data_mask


def save_data_as_npy(data_path, data_orig, data_mask, data_orig_name, data_mask_name):
    """
    Stores list of npy arrays from method "load_niigz_as_npy(data_path)" to the folder data/preprocessed.

    Parameters
    ----------
    data_path:  	    String
                    	Path to the data folder.
    data_orig:		    list
            			List of (different shapes) npy arrays containing TOF MRA data.
    data_mask:		    list
            			List of (different shapes) npy arrays containing binary masks corresponding to data_orig.
    data_orig_name:		String
            			Name of the target data_orig file name without .npy ending.
    data_mask_name:		String
            			Name of the target data_mask file name without .npy ending.
    """

    if not os.path.exists(os.path.join(data_path, 'preprocessed')):
        os.makedirs(os.path.join(data_path, 'preprocessed'))
    np.save(os.path.join(os.path.join(data_path, 'preprocessed'), data_orig_name) + '.npy', data_orig)
    np.save(os.path.join(os.path.join(data_path, 'preprocessed'), data_mask_name) + '.npy', data_mask)
    print('DONE: npy arrays stored in data_path/preprocessed')


def load_data_from_npy(data_path, data_orig_name, data_mask_name):
    """
    Loads the npy arrays to the list of npy arrays from the folder data/preprocessed.

    Parameters
    ----------
    data_path:  	    String
                    	Path to the data folder.
    data_orig_name:		String
            			Name of the data_orig file name without .npy ending.
    data_mask_name:		String
            			Name of the data_mask file name without .npy ending.

    Returns
    -------
    data_orig:		    list
            			List of (different shapes) npy arrays containing TOF MRA data.
    data_mask:		    list
            			List of (different shapes) npy arrays containing binary masks corresponding to data_orig.
    """

    data_orig = np.load(os.path.join(os.path.join(data_path, 'preprocessed'), data_orig_name) + '.npy')
    data_mask = np.load(os.path.join(os.path.join(data_path, 'preprocessed'), data_mask_name) + '.npy')
    print('DONE: npy arrays loaded from data_path/preprocessed')
    return  data_orig, data_mask


def resize_width_height_skimage(data, target_resolution):
    """
    Resizes the list of numpy arrays (e.g. 113x(.,.,.)) to the target_resolution (e.g. (560,560,140)).
    see https://scikit-image.org/docs/stable/auto_examples/transform/plot_rescale.html

    Parameters
    ----------
    data:		        list
            			List of npy arrays (of different shapes).
    target_resolution:	tuple
            			Tuple consisting of 3 elements: width, height and amount of slices, e.g.: (560,560,140).

    Returns
    -------
    resized_data_list:	list
            			List of npy arrays (of same, resized shape, e.g.: 113x560x560x140).
    """

    resized_data_list = []
    for d in data:
        resized_data = resize(d, target_resolution)
        resized_data_list.append(resized_data)

    print('DONE: resizing images')
    return resized_data_list






