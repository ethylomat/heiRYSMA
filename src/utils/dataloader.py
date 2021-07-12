import torch
import os
import numpy as np
from src.utils import preprocessing


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
                if target_resolution[2]==0:
                    self.data, self.mask = preprocessing.crop_data(self.data, self.mask,
                                                                   crop_size_xy=target_resolution[0],
                                                                   crop_size_z=8, overlap=overlap, include_augment=True)
                if include_augmented_data:
                    self.data, self.mask = preprocessing.augment_data(self.data, self.mask)
            else:
                self.data, self.mask = preprocessing.crop_data(self.data, self.mask, crop_size_xy=target_resolution[0], crop_size_z=target_resolution[2], overlap=overlap, include_augment=True)  # augmenting / balancing included
            preprocessing.save_data_as_npy(data_path, self.data, self.mask, resized_file_name_orig, resized_file_name_mask)

        # shuffle always with same order
        self.shuffle_data(data_path)

        # convert to torch tensors
        self.data = torch.tensor(self.data)
        self.mask = torch.tensor(self.mask)

        if train_eval_test == 'train':
            self.data = self.data[:int(len(self.data) * 0.7)]
            self.mask = self.mask[:int(len(self.mask) * 0.7)]
        elif train_eval_test == 'eval':
            self.data = self.data[int(len(self.data) * 0.7): int(len(self.data) * 0.85)]
            self.mask = self.mask[int(len(self.mask) * 0.7): int(len(self.mask) * 0.85)]
        else:
            self.data = self.data[int(len(self.data) * 0.85):]
            self.mask = self.mask[int(len(self.mask) * 0.85):]



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
        return data, mask