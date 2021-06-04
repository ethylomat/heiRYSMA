import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from utils.dataloader import AneurysmDataset
from utils.DenseSeg import DenseNetSeg3D
import numpy as np


if __name__ == "__main__":

    data_path = ...  # insert absolute path to the data directory
    target_resolution = (64, 64, 64)  # modify here if other resolution needed
    batch_size = 6
    include_augmented_data = False  # enable if flipped data (vertically + horizonatally) and rotated data (180 degrees) wanted

    train = torch.utils.data.DataLoader(
        AneurysmDataset(
            data_path=data_path,
            target_resolution=target_resolution,
            include_augmented_data=include_augmented_data,
            train=True
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    test = torch.utils.data.DataLoader(
        AneurysmDataset(
            data_path=data_path,
            target_resolution=target_resolution,
            include_augmented_data=include_augmented_data,
            train=False
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    device = torch.device(0 if torch.cuda.is_available() else "cpu")

    model = DenseNetSeg3D()
    model.to(device)
    # model = torch.nn.parallel.DataParallel(model, device_ids=[0,1]) -> enable for parallelism, device ids are gpus for calculation
    criterion = nn.BCEWithLogitsLoss()
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in tqdm(range(5000), desc='Epoch'):

        sum_aneurysm_truth_batch = 0
        sum_aneurysm_pred_batch = 0
        loss_batch = []
        # training
        for train_step, [train_ex, train_l] in enumerate(tqdm(train, desc='Train')):
            # forward

            train_ex = torch.unsqueeze(train_ex, 1)
            train_ex = train_ex.float()
            train_ex = train_ex.to(device)

            train_l = train_l.type(torch.LongTensor)
            train_l = train_l.to(device)

            scores = model(train_ex, device)
            scores = torch.squeeze(scores)
            loss = criterion(scores, train_l.float())

            loss_batch.append(loss.item())

            sum_aneurysm_truth = torch.sum(train_l)
            sum_aneurysm_truth_batch += sum_aneurysm_truth.item()

            sc = torch.zeros_like(scores)
            sc[scores < 0.5] = 0
            sc[scores >= 0.5] = 1
            sum_aneurysm_pred = torch.sum(sc)
            sum_aneurysm_pred_batch += sum_aneurysm_pred.item()

            # backward
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        print('Time: ' + str(datetime.datetime.now()))
        print('Train, epoch: ' + str(epoch))
        print("Amount pixel truth aneurym: " + str(int(sum_aneurysm_truth_batch)))
        print("Amount pixel predicted aneurym: " + str(int(sum_aneurysm_pred_batch)))
        print('Difference: ' + str(int(sum_aneurysm_pred_batch - sum_aneurysm_truth_batch)))
        print('CELoss: ' + str(np.mean(loss_batch)))
        print('')

        if (epoch + 1) % 10 == 0:

            sum_aneurysm_truth_batch = 0
            sum_aneurysm_pred_batch = 0
            loss_batch = []

            # test
            for test_step, [test_ex, test_l] in enumerate(tqdm(test, desc='Test')):
                test_ex = torch.unsqueeze(test_ex, 1)
                test_ex = test_ex.float()
                test_ex = test_ex.to(device)

                test_l = test_l.type(torch.LongTensor)
                test_l = test_l.to(device)

                scores = model(test_ex, device)
                scores = torch.squeeze(scores)
                loss = criterion(scores, test_l.float())

                loss_batch.append(loss.item())

                arg_max_scores = torch.argmax(scores, dim=1)

                sum_aneurysm_truth = torch.sum(test_l)
                sum_aneurysm_truth_batch += sum_aneurysm_truth.item()

                sc = torch.zeros_like(scores)
                sc[scores < 0.5] = 0
                sc[scores >= 0.5] = 1
                sum_aneurysm_pred = torch.sum(sc)
                sum_aneurysm_pred_batch += sum_aneurysm_pred.item()

            torch.save(model.state_dict(), 'MODEL_NAME_PLACEHOLDER')

            print('Time: ' + str(datetime.datetime.now()))
            print('TEST, epoch: ' + str(epoch))
            print("Amount pixel truth aneurym: " + str(int(sum_aneurysm_truth_batch)))
            print("Amount pixel predicted aneurym: " + str(int(sum_aneurysm_pred_batch)))
            print('Difference: ' + str(int(sum_aneurysm_pred_batch - sum_aneurysm_truth_batch)))
            print('CELoss: ' + str(np.mean(loss_batch)))
            print('')
