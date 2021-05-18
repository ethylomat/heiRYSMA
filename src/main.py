import torch
import os
from tqdm.auto import tqdm
from utils.dataloader import AneurysmDataset


if __name__ == "__main__":

    data_path = os.path.join('src', 'data')  # modify if data not stored in HeiRysma/src/data
    target_resolution = (64, 64, 64)  # modify here if other resolution needed
    batch_size = 6

    train = torch.utils.data.DataLoader(
            AneurysmDataset(
                data_path = data_path,
                target_resolution = target_resolution,
                train = True
            ),
            batch_size=batch_size,
            shuffle=True
        )

    test = torch.utils.data.DataLoader(
            AneurysmDataset(
                data_path = data_path,
                target_resolution = target_resolution,
                train = False
            ),
            batch_size=batch_size,
            shuffle=True
        )

    for epoch in tqdm(range(20), desc='Epoch'):

        # training
        for train_step, [train_ex, train_l] in enumerate(tqdm(train, desc='Train')):

            # TODO: train network
            print(train_ex.shape)
            print(train_l.shape)

        # test
        for test_step, [test_ex, test_l] in enumerate(tqdm(test, desc='Test')):

            # TODO: test network
            print(test_ex.shape)
            print(test_l.shape)