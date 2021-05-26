import torch
import os
import preprocessing


class AneurysmDataset(torch.utils.data.Dataset):
    """Aneurysm dataset."""

    def __init__(self, data_path, target_resolution, train=False):
        """
        Parameters
        ----------
        data_path:          String
                  	        Absolute path to dir with all the (not processed) data.
        target_resolution:  tuple
                  	        Tuple for target resolution of data in form: (W, H, AmountSlices), e.g. (560, 560, 140).
        train:              bool
                            If True, creates dataset from training set,
                            otherwise creates from test set.
        """

        resized_file_name_orig = 'data_orig_' + str(target_resolution[0]) + '_' + str(target_resolution[1]) + '_' + str(target_resolution[2])
        resized_file_name_mask = 'data_mask_' + str(target_resolution[0]) + '_' + str(target_resolution[1]) + '_' + str(target_resolution[2])
        path_data_resized_orig = os.path.join(os.path.join(data_path, 'preprocessed'), resized_file_name_orig) + '.npy'
        path_data_resized_mask = os.path.join(os.path.join(data_path, 'preprocessed'), resized_file_name_mask) + '.npy'
        path_data_orig = os.path.join(os.path.join(data_path, 'preprocessed'), 'data_orig.npy')

        print('INIT: Anerurysm Dataset, mode train activated: ' + str(train) + '...')

        # check if resized data available, if yes -> take it
        # if not, check if original data as npy arrays available, if yes -> take it and resize
        # if not, take the niigz files from data dir and do the whole preprocessing
        if os.path.isfile(path_data_resized_orig) and os.path.isfile(path_data_resized_mask):
            print('Resized data ' + str(target_resolution) + ' found. Loading in progress...')
            self.data, self.mask = preprocessing.load_data_from_npy(data_path, resized_file_name_orig, resized_file_name_mask)
        else:
            if os.path.isfile(path_data_orig):
                print('No resized data ' + str(target_resolution) + ' found under the path: ' + path_data_resized_orig)
                print('Original npy data available. Loading in progress...')
                self.data, self.mask = preprocessing.load_data_from_npy(data_path, 'data_orig', 'data_mask')
            else:
                print('No original npy data available under the path: ' + path_data_orig)
                self.data, self.mask = preprocessing.load_niigz_as_npy(data_path)
                preprocessing.save_data_as_npy(data_path, self.data, self.mask, 'data_orig', 'data_mask')
                self.data = preprocessing.resize_width_height_skimage(self.data, target_resolution=target_resolution)
                self.mask = preprocessing.resize_width_height_skimage(self.mask, target_resolution=target_resolution)
                preprocessing.save_data_as_npy(data_path, self.data, self.mask, resized_file_name_orig, resized_file_name_mask)

        # convert to torch tensors
        self.data = torch.tensor(self.data).float()
        self.data = torch.unsqueeze(self.data, 1)
        self.mask = torch.tensor(self.mask).long()

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
        mask[mask > 0] = 1.0  # convert to binary resized masks -> 0: no aneurysm vs 1: aneurysm
        return data, mask