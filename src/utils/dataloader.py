import torch
import os
from utils import preprocessing


class AneurysmDataset(torch.utils.data.Dataset):
    """Aneurysm dataset."""

    def __init__(self, data_path, target_resolution, include_augmented_data=True, train=False):
        """
        Args:
            data_path (string): Path to dir with all the (not processed) data.
            train (bool, optional): If True, creates dataset from training set,
            otherwise creates from test set.
        """
        resized_file_name_orig = 'data_orig_' + str(target_resolution[0]) + '_' + str(target_resolution[1]) + '_' + str(target_resolution[2])
        resized_file_name_mask = 'data_mask_' + str(target_resolution[0]) + '_' + str(target_resolution[1]) + '_' + str(target_resolution[2])

        if include_augmented_data:
            resized_file_name_orig += '_a'
            resized_file_name_mask += '_a'

        path_data_resized_orig = data_path + '/preprocessed/' + resized_file_name_orig + '.npy'
        path_data_resized_mask = data_path + '/preprocessed/' + resized_file_name_mask + '.npy'
        path_data_orig = data_path + '/preprocessed/data_orig.npy'

        print('INIT: Anerurysm Dataset, mode train activated: ' + str(train) + '...')

        if (os.path.isfile(path_data_resized_orig) and os.path.isfile(path_data_resized_mask)):
            print('Resized data ' + str(target_resolution) + ' found. Loading in progress...')
            self.data, self.mask = preprocessing.load_data_from_npy(data_path, resized_file_name_orig, resized_file_name_mask)
        else:
            if(os.path.isfile(path_data_orig)):
                print('No resized data ' + str(target_resolution) + ' found under the path: ' + path_data_resized_orig)
                print('Original npy data available. Loading in progress...')
                self.data, self.mask = preprocessing.load_data_from_npy(data_path, 'data_orig', 'data_mask', True)
                self.data = self.data.tolist()
                self.mask = self.mask.tolist()

            else:
                print('No original npy data available under the path: ' + path_data_orig)
                print('Preprocessing from niigz files in progress...')
                self.data, self.mask = preprocessing.load_niigz_as_npy(data_path)
                preprocessing.save_data_as_npy(data_path, self.data, self.mask, 'data_orig', 'data_mask')

            self.data = preprocessing.resize_width_height_skimage(self.data, target_resolution = target_resolution)
            self.mask = preprocessing.resize_width_height_skimage(self.mask, target_resolution = target_resolution)

            if include_augmented_data:
                 self.data = preprocessing.augment_data(self.data)
                 self.mask = preprocessing.augment_data(self.mask)
            preprocessing.save_data_as_npy(data_path, self.data, self.mask, resized_file_name_orig, resized_file_name_mask)

        # convert to torch tensors
        self.data = torch.tensor(self.data)
        self.mask = torch.tensor(self.mask)

        if train:
            self.data = self.data[:int(len(self.data)*0.7)]
            self.mask = self.mask[:int(len(self.mask)*0.7)]
        else:
            self.data = self.data[int(len(self.data)*0.7):]
            self.mask = self.mask[int(len(self.mask)*0.7):]



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data.float()
        data = (data - torch.min(data)) / torch.max(data - torch.min(data))  # normalize
        mask = self.mask[idx]
        if mask.max() > 0:
            mask = (mask - torch.min(mask)) / torch.max(mask - torch.min(mask))  # normalize
            mask[mask > 0.5] = 1.0  # convert to binary resized masks -> 0: no aneurysm vs 1: aneurysm
            mask[mask <= 0.5] = 0.0  # convert to binary resized masks -> 0: no aneurysm vs 1: aneurysm
        return data, mask