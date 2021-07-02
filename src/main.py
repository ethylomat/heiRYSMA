import argparse
import os
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np

from src.utils.dataloader import AneurysmDataset
from src.utils.DenseSeg import DenseNetSeg3D
from src.utils.Losses import DiceLoss

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
    parser = argparse.ArgumentParser(description='heiRYSMA')
    parser.add_argument('--data', dest='data_path', default=None, help='Absolute path of the data directory')
    parser.add_argument('--model-dir', dest='models_path', default=None, help='Absolute path of the model directory')
    parser.add_argument('--resolution', nargs=3, type=int, dest='resolution', default=[64, 64, 64], help='Dimension for cropping/resizing (e.g. 64 for dimension 64 x 64 x 64)')
    parser.add_argument('--overlap', dest='overlap', default=10, help='Overlap for cropping')
    parser.add_argument('--augmented', action='store_false', default=False, dest='augmented', help='Add flipped, rotated and brighter data')
    parser.add_argument('--batch-size', dest='batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--resizing', action='store_false', default=False, dest='resizing', help='Resizing (default is cropping)')
    parser.add_argument('--learning-rate', dest='learning_rate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--existing-model', action='store_true', default=True, dest='train_existing_model',
                        help='Training of existing model (if exist)')
    parser.add_argument('--dice', action='store_true', default=False, dest='dice',
                        help='Use dice loss')
    arguments = parser.parse_args()

    src_dir = pathlib.Path(__file__).parent.resolve()

    # Default data path: /heiRYSMA/data
    data_path = arguments.data_path
    if data_path is None:
        data_path = os.path.join(src_dir.parent.resolve(), "data")

    # Default models path: /heiRYSMA/models
    models_path = arguments.models_path
    if models_path is None:
        models_path = os.path.join(src_dir.parent.resolve(), "models")

    target_resolution = tuple(arguments.resolution)  # modify here if other resolution needed, currently available (64, 64, 64) and (128, 128, 100)
    overlap = arguments.overlap  # overlap for cropping
    batch_size = arguments.batch_size
    include_augmented_data = arguments.augmented  # enable if flipped data (vertically + horizonatally), rotated data (180 degrees) and brighter data 5% wanted
    include_resizing = arguments.resizing  # enable if resizing wanted, else cropping applied
    learning_rate = arguments.learning_rate

    # Model name format: resolution_overlap_batchsize_learning_rate
    model_name = f"model__{str(target_resolution).replace(', ', '_')[1:-1]}__o{str(overlap).zfill(2)}__b{str(batch_size).zfill(2)}__lr{str(learning_rate).replace('0.', '')}"
    if not include_resizing:
        model_name += "__crop"

    model_path = os.path.join(models_path, model_name)

    # Printing used settings (parsed or default)
    print("Using data path: ", data_path)
    print("Target resolution: ", target_resolution)
    print("Overlap: ", overlap)
    print("Batch size: ", batch_size)
    print("Include augmented data: ", include_augmented_data)
    print("Include resizing: ", include_resizing)
    print("Learning rate: ", learning_rate)
    print("Training existing model: ", arguments.train_existing_model)
    print("Model name: ", model_name, "\n\n")

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using", torch.cuda.device_count(), "GPUs")

    model = DenseNetSeg3D()
    model.to(device)
    model = model.double()

    device_count = torch.cuda.device_count()
    if device_count > 1:
        model = torch.nn.parallel.DataParallel(model, device_ids=list(range(device_count)))
    else:
        model = torch.nn.parallel.DataParallel(model)

    if os.path.isfile(model_path) and arguments.train_existing_model:
        print("Loading existing model from: ", model_path)
        model.load_state_dict(torch.load(model_path))
    else:
        print("Training new model at: ", model_path)

    if arguments.dice:
        print("Using Dice-Loss")
        criterion = DiceLoss()
    else:
        print("Using BinaryCrossEntropy Loss")
        criterion = nn.BCEWithLogitsLoss()

    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_benchmark = 1

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

        if (epoch + 1) % 1 == 0:

            sum_aneurysm_truth_batch_eval = 0
            sum_aneurysm_pred_batch_eval = 0
            loss_batch_eval = []

            # eval
            for eval_step, [eval_ex, eval_l] in enumerate(tqdm(eval, desc='Eval')):
                sum_aneurysm_truth_batch_eval, sum_aneurysm_pred_batch_eval, loss_batch_eval = run_model_get_scores(eval_ex, eval_l, device, target_resolution,
                                                                                                                    sum_aneurysm_truth_batch_eval, sum_aneurysm_pred_batch_eval, loss_batch_eval,
                                                                                                                    file, epoch, eval_step, train=False)

            write_stats_after_epoch(sum_aneurysm_truth_batch_eval, sum_aneurysm_pred_batch_eval, loss_batch_eval,
                                    epoch, 'Eval', file)

            if epoch > 0 and loss_benchmark > loss_batch_eval:
                print("Loss smaller than benchmark. Saving model to: ", model_path)
                torch.save(model.state_dict(), model_path)
            else:
                print("Loss greater than benchmark. Not saving model.")

            loss_benchmark = loss_batch_eval

    file.close()