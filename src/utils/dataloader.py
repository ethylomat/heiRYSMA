import torch
import os
import numpy as np
from utils import preprocessing
import nibabel as nib
import warnings


class AneurysmDataset(torch.utils.data.Dataset):
    """Aneurysm dataset."""

    def __init__(self, data_path, target_resolution, overlap, include_augmented_data, include_resizing, train_eval_test):
        """
        Args:
            data_path (string): Path to dir with all the (not processed) data.
            target_resolution (tuple): 3 positional tuple containing the desired resolution of data, (W,H,Amount Slices) e.g. 64,64,100.
            include_augmented_data (bool): True if augmented data should be included, else otherwise
            include_resizing (bool): True if resized data should be included, else cropping activated
            train_eval_test (string): If train, creates dataset from training set, if eval creates dataset from eval set, otherwise creates from test set.
        """
        resized_file_name_orig = 'data_orig_' + str(target_resolution[0]) + '_' + str(target_resolution[1]) + '_' + str(target_resolution[2])
        resized_file_name_mask = 'data_mask_' + str(target_resolution[0]) + '_' + str(target_resolution[1]) + '_' + str(target_resolution[2])

        if include_augmented_data:
            resized_file_name_orig += '_a'
            resized_file_name_mask += '_a'

        if not include_resizing:
            resized_file_name_orig += '_crop'
            resized_file_name_mask += '_crop'

        path_data_resized_orig = data_path + '/preprocessed/' + resized_file_name_orig + '.npy'
        path_data_resized_mask = data_path + '/preprocessed/' + resized_file_name_mask + '.npy'
        path_data_orig = data_path + '/preprocessed/data_orig.npy'

        print('INIT: Anerurysm Dataset, mode train_eval_test activated: ' + str(train_eval_test) + '...')

        if (os.path.isfile(path_data_resized_orig) and os.path.isfile(path_data_resized_mask)):
            print('Resized / cropped data ' + str(target_resolution) + ' found. Loading in progress...')
            self.data, self.mask = preprocessing.load_data_from_npy(data_path, resized_file_name_orig, resized_file_name_mask)
        else:
            if(os.path.isfile(path_data_orig)):
                print('No resized / cropped data ' + str(target_resolution) + ' found under the path: ' + path_data_resized_orig)
                print('Original npy data available. Loading in progress...')
                self.data, self.mask = preprocessing.load_data_from_npy(data_path, 'data_orig', 'data_mask', True)
                self.data = self.data.tolist()
                self.mask = self.mask.tolist()

            else:
                print('No original npy data available under the path: ' + path_data_orig)
                print('Preprocessing from niigz files in progress...')
                self.data, self.mask = preprocessing.load_niigz_as_npy(data_path)
                preprocessing.save_data_as_npy(data_path, self.data, self.mask, 'data_orig', 'data_mask')

            if include_resizing:
                self.data, self.mask = preprocessing.resize_width_height_skimage(self.data, self.mask,
                                                                             target_resolution=target_resolution)
                if include_augmented_data:
                    self.data, self.mask, self.locations = preprocessing.augment_data(self.data, self.mask)
            else:
                self.data, self.mask, self.locations = preprocessing.crop_data(self.data, self.mask, crop_size_xy=target_resolution[0], crop_size_z=target_resolution[2], overlap=overlap)  # augmenting / balancing included
            preprocessing.save_data_as_npy(data_path, self.data, self.mask, resized_file_name_orig, resized_file_name_mask)

        # todo : do not load them if data is self generated (augmented)
        self.locations = load_locations(data_path)

        # shuffle always with same order
        self.shuffle_data(data_path)

        # convert to torch tensors
        self.data = torch.tensor(self.data)
        print(self.data.shape)
        self.mask = torch.tensor(self.mask)
        self.locations = torch.tensor(self.locations)

        if train_eval_test == 'train':
            # todo
            self.data = self.data[:int(len(self.data) * 0.7)]
            self.mask = self.mask[:int(len(self.mask) * 0.7)]
            self.locations = self.locations[:int(len(self.locations) * 0.7)]
        elif train_eval_test == 'eval':
            # todo
            self.data = self.data[int(len(self.data) * 0.7): int(len(self.data) * 0.85)]
            self.mask = self.mask[int(len(self.data) * 0.7): int(len(self.mask) * 0.85)]
            self.locations = self.locations[int(len(self.locations) * 0.7): int(len(self.locations) * 0.85)]
        else:
            # todo
            self.data = self.data[int(len(self.data) * 0.85):]
            self.mask = self.mask[int(len(self.mask) * 0.85):]
            self.locations = self.locations[int(len(self.locations) * 0.85):]

    def shuffle_data(self, data_path):
        shuffled_file_path = data_path + '/preprocessed/shuffling_order_' + str(len(self.mask)) + '.npy'

        if (os.path.isfile(shuffled_file_path)): # search for shuffled data under the path above
            shuffling_order = np.load(data_path + '/preprocessed/shuffling_order_' + str(len(self.mask)) + '.npy')
        else:  # create new shuffling order and save as .npy
            shuffling_order = np.arange(len(self.mask))
            np.random.shuffle(shuffling_order)
            np.save(data_path + '/preprocessed/shuffling_order_' + str(len(self.mask)) + '.npy', shuffling_order)

        # shuffle with imported shuffled order
        self.data = np.array(self.data)[shuffling_order]
        self.mask = np.array(self.mask)[shuffling_order]
        self.locations = np.array(self.locations)[shuffling_order]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data.float()
        if data.max() > 0:
            data = (data - torch.min(data)) / torch.max(data - torch.min(data))  # normalize
        mask = self.mask[idx]
        if mask.max() > 0:
            mask = (mask - torch.min(mask)) / torch.max(mask - torch.min(mask))  # normalize
            mask[mask > 0.5] = 1.0  # convert to binary resized masks -> 0: no aneurysm vs 1: aneurysm
            mask[mask <= 0.5] = 0.0  # convert to binary resized masks -> 0: no aneurysm vs 1: aneurysm
        location = self.locations[idx]

        return data, mask, location


def load_locations(data_path):
    """
      Loads the ground truth locations from location.txt

      Parameters
      ----------
      data_path:  	    String
                      	Absolute path to the data folder.

      Returns
      -------
      data_orig:		    list
              			List of (different shapes) npy arrays containing TOF MRA data.
      data_mask:		    list
              			List of (different shapes) npy arrays containing binary masks corresponding to data_orig.
      """

    sub_dirs = [x[0] for x in os.walk(data_path)]
    if ((os.path.join(data_path, '10022') not in sub_dirs)):
        raise ValueError(
            "There is no data in the directory to work with! Check the data directory / data_path to data directory and try again...")
    gt_location = []

    print('Preprocessing from niigz files in progress...')
    for dir in sub_dirs:
        if (dir.endswith('orig')):
            loc_fname = os.path.join(dir.partition('orig')[0], 'location.txt')
            loc = np.atleast_2d(np.genfromtxt(loc_fname, delimiter=',', encoding='utf-8-sig'))
            # Reshape an empty result into a 0x4 array.
            if loc.size == 0:
                loc = loc.reshape(0, 4)
            # DEBUG: verify that the inner dimension size is 4.
            assert loc.shape[1] == 4
            gt_location.append(loc)

    print('DONE: Locations loaded')
    return gt_location
