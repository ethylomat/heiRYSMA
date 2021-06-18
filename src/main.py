import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from utils.dataloader import AneurysmDataset
from utils.DenseSeg import DenseNetSeg3D
import numpy as np


def run_model_get_scores(example, label, device, target_resolution, sum_aneurysm_truth_batch, sum_aneurysm_pred_batch, loss_batch, file, epoch, step, train=True):
    label = label.type(torch.LongTensor)
    label = label.to(device)
    example = example.to(device)
    example = example.double()

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
        file.write('TrainLossEpoch'+ str(epoch) + 'Step' + str(step) + ': ' + str(loss.item()) + '\n')
    else:
        file.write('EvalLossEpoch'+ str(epoch) + 'Step' + str(step) + ': ' + str(loss.item()) + '\n')

    return sum_aneurysm_truth_batch, sum_aneurysm_pred_batch, loss_batch

def create_loss_log_file(model_name):

    f = open('log/' + model_name + '_loss_log.txt', 'a')
    f.write('Log file start for the test: ' + model_name + '_loss_log.txt\n')

    return f

def write_stats_after_epoch(sum_aneurysm_truth_batch, sum_aneurysm_pred_batch, loss_batch, epoch, train_eval, file):
    print(train_eval + ', epoch: ' + str(epoch))
    print("Amount pixel truth aneurym: " + str(int(sum_aneurysm_truth_batch)))
    print("Amount pixel predicted aneurym: " + str(int(sum_aneurysm_pred_batch)))
    print('Difference: ' + str(int(sum_aneurysm_pred_batch - sum_aneurysm_truth_batch)))
    print('BCEWithLogitsLoss: ' + str(np.mean(loss_batch)))
    print('')
    file.write(train_eval + 'LossEpoch' + str(epoch) + ', Amount pixel truth aneurysm: ' + str(int(sum_aneurysm_truth_batch)) + '\n')
    file.write(train_eval + 'LossEpoch' + str(epoch) + ', Amount pixel predicted aneurysm: ' + str(int(sum_aneurysm_pred_batch)) + '\n')
    file.write(train_eval + 'LossEpoch' + str(epoch) + ', Difference: ' + str(int(sum_aneurysm_pred_batch - sum_aneurysm_truth_batch)) + '\n')
    file.write(train_eval + 'LossEpoch' + str(epoch) + ', BCEWithLogitsLoss Mean: ' + str(np.mean(loss_batch)) + '\n')

if __name__ == "__main__":

    data_path = ...  # insert absolute path to the data directory
    target_resolution = (64, 64, 64)  # modify here if other resolution needed, currently available (64, 64, 64) and (128, 128, 100)
    overlap = 10  # overlap for cropping
    batch_size = 8
    include_augmented_data = False  # enable if flipped data (vertically + horizonatally), rotated data (180 degrees) and brighter data 5% wanted
    include_resizing = False  # enable if resizing wanted, else cropping applied
    model_name = 'model_64_64_64_10_8_0001_crop'  # resolution_overlap_batchsize_learning_rate

    file = create_loss_log_file(model_name)

    train = torch.utils.data.DataLoader(
        AneurysmDataset(
            data_path=data_path,
            target_resolution=target_resolution,
            overlap=overlap,
            include_augmented_data=include_augmented_data,
            include_resizing=include_resizing,
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
            overlap=overlap,
            include_augmented_data=include_augmented_data,
            include_resizing=include_resizing,
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
            overlap=overlap,
            include_augmented_data=include_augmented_data,
            include_resizing=include_resizing,
            train_eval_test='test'
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    device = torch.device(0 if torch.cuda.is_available() else "cpu")

    model = DenseNetSeg3D()
    model.to(device)
    model = model.double()
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

            sum_aneurysm_truth_batch_train, sum_aneurysm_pred_batch_train, loss_batch_train = run_model_get_scores(train_ex, train_l, device, target_resolution,
                                                                                                                    sum_aneurysm_truth_batch_train, sum_aneurysm_pred_batch_train, loss_batch_train,
                                                                                                                    file, epoch, train_step, train=True)

        write_stats_after_epoch(sum_aneurysm_truth_batch_train, sum_aneurysm_pred_batch_train, loss_batch_train, epoch, 'Train', file)

        if (epoch + 1) % 10 == 0:

            sum_aneurysm_truth_batch_eval = 0
            sum_aneurysm_pred_batch_eval = 0
            loss_batch_eval = []

            # eval
            for eval_step, [eval_ex, eval_l] in enumerate(tqdm(eval, desc='Eval')):
                sum_aneurysm_truth_batch_eval, sum_aneurysm_pred_batch_eval, loss_batch_eval = run_model_get_scores(eval_ex, eval_l, device, target_resolution,
                                                                                                                    sum_aneurysm_truth_batch_eval, sum_aneurysm_pred_batch_eval, loss_batch_eval,
                                                                                                                    file, epoch, eval_step, train=False)

            torch.save(model.state_dict(), model_name)

            write_stats_after_epoch(sum_aneurysm_truth_batch_eval, sum_aneurysm_pred_batch_eval, loss_batch_eval,
                                    epoch, 'Eval', file)

    file.close()
