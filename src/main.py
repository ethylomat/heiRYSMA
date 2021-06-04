import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from utils.dataloader import AneurysmDataset
from utils.DenseSeg import DenseNetSeg3D
import numpy as np


def run_model_get_scores(example, label, target_resolution, sum_aneurysm_truth_batch, sum_aneurysm_pred_batch, loss_batch, train=True):
    label = label.type(torch.LongTensor)
    label = label.to(device)

    scores = model(example, device, target_resolution)
    scores = torch.squeeze(scores)
    loss = criterion(scores, label.float())
    loss_batch.append(loss.item())

    sum_aneurysm_truth = torch.sum(label)
    sum_aneurysm_truth_batch += sum_aneurysm_truth.item()

    # binarize -> hard decision -> if pixel > 0.5 -> aneurysm, else not
    sc = torch.zeros_like(scores)
    sc[scores < 0.5] = 0
    sc[scores >= 0.5] = 1
    sum_aneurysm_pred = torch.sum(sc)
    sum_aneurysm_pred_batch += sum_aneurysm_pred.item()

    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum_aneurysm_truth_batch, sum_aneurysm_pred_batch, loss_batch

if __name__ == "__main__":

    data_path = ...  # insert absolute path to the data directory
    target_resolution = (128, 128, 100)  # modify here if other resolution needed, currently available (64, 64, 64) and (128, 128, 100)
    batch_size = 16
    include_augmented_data = True  # enable if flipped data (vertically + horizonatally), rotated data (180 degrees) and brighter data 5% wanted
    model_name = 'model_128_128_10_a'  # enable if flipped data (vertically + horizonatally) and rotated data (180 degrees) wanted

    train = torch.utils.data.DataLoader(
        AneurysmDataset(
            data_path=data_path,
            target_resolution=target_resolution,
            include_augmented_data=include_augmented_data,
            train_eval_test='train'
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    eval = torch.utils.data.DataLoader(
        AneurysmDataset(
            data_path=data_path,
            target_resolution=target_resolution,
            include_augmented_data=include_augmented_data,
            train_eval_test='eval'
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    test = torch.utils.data.DataLoader(
        AneurysmDataset(
            data_path=data_path,
            target_resolution=target_resolution,
            include_augmented_data=include_augmented_data,
            train_eval_test='test'
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    device = torch.device(0 if torch.cuda.is_available() else "cpu")

    model = DenseNetSeg3D()
    model.to(device)
    model = torch.nn.parallel.DataParallel(model)  # -> enable for parallelism, device ids are gpus for calculation
    criterion = nn.BCEWithLogitsLoss()
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in tqdm(range(5000), desc='Epoch'):

        sum_aneurysm_truth_batch_train = 0
        sum_aneurysm_pred_batch_train = 0
        loss_batch_train = []
        # training
        for train_step, [train_ex, train_l] in enumerate(tqdm(train, desc='Train')):

            sum_aneurysm_truth_batch_train, sum_aneurysm_pred_batch_train, loss_batch_train = run_model_get_scores(train_ex, train_l, target_resolution,
                                                                                                                    sum_aneurysm_truth_batch_train, sum_aneurysm_pred_batch_train, loss_batch_train,
                                                                                                                    train=True)

        print('Train, epoch: ' + str(epoch))
        print("Amount pixel truth aneurym: " + str(int(sum_aneurysm_truth_batch_train)))
        print("Amount pixel predicted aneurym: " + str(int(sum_aneurysm_pred_batch_train)))
        print('Difference: ' + str(int(sum_aneurysm_pred_batch_train - sum_aneurysm_truth_batch_train)))
        print('BCEWithLogitsLoss: ' + str(np.mean(loss_batch_train)))
        print('')

        if (epoch + 1) % 10 == 0:

            sum_aneurysm_truth_batch_eval = 0
            sum_aneurysm_pred_batch_eval = 0
            loss_batch_eval = []

            # eval
            for eval_step, [eval_ex, eval_l] in enumerate(tqdm(eval, desc='Eval')):
                sum_aneurysm_truth_batch_eval, sum_aneurysm_pred_batch_eval, loss_batch_eval = run_model_get_scores(eval_ex, eval_l, target_resolution,
                                                                                                                    sum_aneurysm_truth_batch_eval, sum_aneurysm_pred_batch_eval, loss_batch_eval,
                                                                                                                    train=False)

            torch.save(model.state_dict(), model_name)

            print('EVAL, epoch: ' + str(epoch))
            print("Amount pixel truth aneurym: " + str(int(sum_aneurysm_truth_batch_eval)))
            print("Amount pixel predicted aneurym: " + str(int(sum_aneurysm_pred_batch_eval)))
            print('Difference: ' + str(int(sum_aneurysm_pred_batch_eval - sum_aneurysm_truth_batch_eval)))
            print('BCEWithLogitsLoss: ' + str(np.mean(loss_batch_eval)))
            print('')