import argparse
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from src.utils.DenseSeg import DenseNetSeg3D
from src.utils.dataloader import AneurysmDataset
from src.utils.dice_loss import BinaryDiceLoss
from src.utils.focal_loss import FocalLoss

from src.utils.evaluation import get_dsc, get_hausdorff, get_vs, get_images
import SimpleITK as sitk
from src.utils import evaluation
from src.utils import evaluation_detection


def get_metrics(scores, labels, lab_location, score_location):
    # binarize -> hard decision -> if pixel > 0.5 -> aneurysm, else not
    preds = np.zeros_like(scores)
    preds[scores < 0.5] = 0
    preds[scores >= 0.5] = 1

    # Prior to dilation (manipulation of prediction may be dangerous and changing the actual result?)
    # Assume for segmentation the dilation is not required
    test_image, result_image = get_images(sitk.GetImageFromArray(labels), sitk.GetImageFromArray(preds))

    sensitivity, false_positives = evaluation_detection.get_detection_metrics(lab_location, score_location, test_image)

    dsc = get_dsc(test_image, result_image)
    h95 = get_hausdorff(test_image, result_image)
    vs = get_vs(test_image, result_image)

    return h95, vs, dsc, sensitivity, false_positives


def run_model_get_scores(example, label, device, target_resolution, sum_aneurysm_truth_batch, sum_aneurysm_pred_batch, loss_batch, epoch, step, train=True):
    label = label.to(device)
    example = example.to(device)
    example = example.double()

    scores = model(example, target_resolution, loss_metric)
    scores = torch.squeeze(scores)
    loss = criterion(scores, label)
    loss_val = loss.item()
    loss_batch.append(loss_val)

    sum_aneurysm_truth = torch.sum(label)
    sum_aneurysm_truth_batch += sum_aneurysm_truth.item()

    batch_h95, batch_vs, batch_dsc, batch_sensitivity, batch_false_positives = 0, 0, 0, 0, 0
    h95_counter, vs_counter, dsc_counter, sensitivity_counter, false_positives_counter = 0, 0, 0, 0, 0

    # binarize -> hard decision -> if pixel > 0.5 -> aneurysm, else not
    if loss_metric == "FCL":  #  FCL has binary cross entropy with logits (sigmoid included in the loss, not in the net output)
        scores = torch.sigmoid(scores)
    sc = torch.zeros_like(scores)
    sc[scores < 0.5] = 0
    sc[scores >= 0.5] = 1
    sum_aneurysm_pred = torch.sum(sc)
    sum_aneurysm_pred_batch += sum_aneurysm_pred.item()

    lab_arr = []
    sc_arr = []
    for i in range(len(label)):
        print("LABEL:")
        label_center_of_mass = evaluation.get_center_of_mass(label[i])
        label_radius = evaluation.get_radius(label[i])
        if (label_radius == 0).all():
            label_final_location = np.ndarray((0, 4))
        else:
            label_final_location = np.append(label_center_of_mass, label_radius.reshape((label_radius.shape[0], 1)), axis=1)

        print("SCORES:")
        scores_center_of_mass = evaluation.get_center_of_mass(sc[i])

        lab_arr.append(label_final_location)
        sc_arr.append(scores_center_of_mass)

    for l, s, lab_location, score_location in zip(label.detach().cpu().numpy(), scores.detach().cpu().numpy(), lab_arr, sc_arr):
        h95, vs, dsc, sensitivity, false_positives = get_metrics(s.squeeze(), l.squeeze(), lab_location, score_location)
        if not np.isnan(h95):
            batch_h95 += h95
            h95_counter += 1
        if not np.isnan(vs):
            batch_vs += vs
            vs_counter += 1
        if not np.isnan(dsc):
            batch_dsc += dsc
            dsc_counter += 1
        if not np.isnan(sensitivity):
            batch_sensitivity += sensitivity
            sensitivity_counter += 1
        if not np.isnan(false_positives):
            batch_false_positives += false_positives
            false_positives_counter += 1

    mean_h95 = batch_h95 / h95_counter if h95_counter > 0 else np.nan
    mean_vs = batch_vs / vs_counter if vs_counter > 0 else np.nan
    mean_dsc = batch_dsc / dsc_counter if dsc_counter > 0 else np.nan
    mean_sensitivity = batch_sensitivity / sensitivity_counter if sensitivity_counter > 0 else np.nan
    mean_false_positives = batch_false_positives / false_positives_counter if false_positives_counter > 0 else np.nan



    return sum_aneurysm_truth_batch, sum_aneurysm_pred_batch, loss_batch,  {"loss": loss_val, "h95": mean_h95, "vs": mean_vs, "dsc": mean_dsc, "sensitivity": mean_sensitivity, "false_positives": mean_false_positives}


def create_loss_log_file(loss_log_file_name):
    f = open('log/' + loss_log_file_name + '_loss_log.txt', 'a')
    f.write('Log file start for the test: ' + loss_log_file_name + '_loss_log.txt\n')
    return f


def create_current_best_loss_file(best_loss_file_name):
    if (os.path.isfile('log/' + best_loss_file_name + '_best_loss_log.txt')):
        f = open('log/' + best_loss_file_name + '_best_loss_log.txt', "r+")
        lines = f.read().splitlines()
        try:
            best_loss = float(lines[-1])
        except:
            best_loss = 100.0
    else:
        f = open('log/' + best_loss_file_name + '_best_loss_log.txt', 'w')
        best_loss = 100.0
    return f, best_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='heiRYSMA')
    parser.add_argument('--data', dest='data_path', default=None, help='Absolute path of the data directory')
    parser.add_argument('--model-dir', dest='models_path', default=None, help='Absolute path of the model directory')
    parser.add_argument('--resolution', nargs=3, type=int, dest='resolution', default=[64, 64, 64],
                        help='Dimension for cropping/resizing (e.g. 64 for dimension 64 x 64 x 64)')
    parser.add_argument('--overlap', type=int, dest='overlap', default=10, help='Overlap for cropping')
    parser.add_argument('--augmented', action='store_false', default=False, dest='augmented',
                        help='Add flipped, rotated and brighter data')
    parser.add_argument('--batch-size', dest='batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--resizing', action='store_true', default=False, dest='resizing',
                        help='Resizing (default is cropping)')
    parser.add_argument('--learning-rate', dest='learning_rate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--existing-model', action='store_true', default=True, dest='train_existing_model',
                        help='Training of existing model (if exist)')
    parser.add_argument('--loss', default="DIC", dest='loss_metric',
                        help='Loss type: BCE - Binary Cross Entropy , DIC - Dice Loss, FOC - Focal Loss')

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

    target_resolution = tuple(
        arguments.resolution)  # modify here if other resolution needed, currently available (64, 64, 64) and (128, 128, 100)
    overlap = int(arguments.overlap)  # overlap for cropping
    batch_size = arguments.batch_size
    include_augmented_data = arguments.augmented  # enable if flipped data (vertically + horizonatally), rotated data (180 degrees) and brighter data 5% wanted
    include_resizing = arguments.resizing  # enable if resizing wanted, else cropping applied
    learning_rate = arguments.learning_rate

    loss_metric = arguments.loss_metric
    assert loss_metric in ["BCE", "DIC", "FOC"]

    # Model name format: resolution_overlap_batchsize_learning_rate
    model_name = f"model__{loss_metric}__{str(target_resolution).replace(', ', '_')[1:-1]}__o{str(overlap).zfill(2)}__b{str(batch_size).zfill(2)}__lr{str(learning_rate).replace('0.', '')}"
    if not include_resizing:
        pass
        # model_name += "__crop"

    model_name_optimizer = model_name + '_optim'

    model_path = os.path.join(models_path, model_name)
    model_optimizer_path = os.path.join(models_path, model_name_optimizer)

    last_model_path = os.path.join(models_path, "last_" + model_name)
    last_model_optimizer_path = os.path.join(models_path, "last_" + model_name_optimizer)

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

    test = torch.utils.data.DataLoader(
        AneurysmDataset(
            data_path=data_path,
            target_resolution=target_resolution,
            overlap=overlap,
            include_augmented_data=include_augmented_data,
            include_resizing=include_resizing,
            train_eval_test='test',
            challenge_mode_on = False
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using", torch.cuda.device_count(), "GPUs")

    device_count = torch.cuda.device_count()

    if device_count > 1:
        model = DenseNetSeg3D(device=device, device_ids=list(range(device_count)), target_resolution=target_resolution)
    else:
        model = DenseNetSeg3D(device=device, device_ids=[0], target_resolution=target_resolution)

    model = model.double()

    if os.path.isfile(last_model_path) and arguments.train_existing_model:
        print("Loading existing model from: ", last_model_path)
        model.load_state_dict(torch.load(last_model_path))
    else:
        print("Training new model at: ", model_path)

    model.to(device)

    if loss_metric == "DIC":
        print("Using Dice Loss")
        criterion = BinaryDiceLoss()
    elif loss_metric == "FOC":
        print("Using Focal Loss")
        criterion = FocalLoss(pos_weight=torch.tensor(10.), gamma=2, reduction='mean')  # pos_weight info: https://discuss.pytorch.org/t/how-to-apply-weighted-loss-to-a-binary-segmentation-problem/35317
    elif loss_metric == "BCE":
        print("Using Binary Cross Entropy Loss")
        criterion = nn.BCELoss()

    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if os.path.isfile(last_model_optimizer_path) and arguments.train_existing_model:
        print("Loading existing model optimizer from: ", last_model_optimizer_path)
        optimizer.load_state_dict(torch.load(last_model_optimizer_path))

    for epoch in tqdm(range(1), desc='Epoch'):

        model.train()

        sum_aneurysm_truth_batch_test = 0
        sum_aneurysm_pred_batch_test = 0
        sum_h95_test, sum_vs_test, sum_dsc_test, sum_sensitivity_test, sum_false_positives_test = 0, 0, 0, 0, 0
        loss_batch_test = []
        loss_batch_test_metrics = []
        test_h95_counter, test_vs_counter, test_dsc_counter, test_sensitivity_counter, test_false_positives_counter = 0, 0, 0, 0, 0

        # eval
        for test_step, [test_ex, test_l, data_shape, data_affine] in enumerate(tqdm(test, desc='Test')):
            sum_aneurysm_truth_batch_test, sum_aneurysm_pred_batch_test, loss_batch_test, metrics_test = run_model_get_scores(
                test_ex, test_l, device, target_resolution,
                sum_aneurysm_truth_batch_test, sum_aneurysm_pred_batch_test, loss_batch_test,
                epoch, test_step, train=False)

            loss_batch_test_metrics.append(metrics_test["loss"])
            if not np.isnan(metrics_test["h95"]):
                sum_h95_test += metrics_test["h95"]
                test_h95_counter += 1
            if not np.isnan(metrics_test["vs"]):
                sum_vs_test += metrics_test["vs"]
                test_vs_counter += 1
            if not np.isnan(metrics_test["dsc"]):
                sum_dsc_test += metrics_test["dsc"]
                test_dsc_counter += 1
            if not np.isnan(metrics_test["sensitivity"]):
                sum_sensitivity_test += metrics_test["sensitivity"]
                test_sensitivity_counter += 1
            if not np.isnan(metrics_test["false_positives"]):
                sum_false_positives_test += metrics_test["false_positives"]
                test_false_positives_counter += 1


        mean_test_h95 = sum_h95_test / test_h95_counter
        mean_test_vs = sum_vs_test / test_vs_counter
        mean_test_dsc = sum_dsc_test / test_dsc_counter
        mean_test_sensitivity = sum_sensitivity_test / test_sensitivity_counter
        mean_test_false_positives = sum_false_positives_test / test_false_positives_counter
        print(f"Mean Eval H95: {mean_test_h95}")
        print(f"Mean Eval VS: {mean_test_vs}")
        print(f"Mean Eval DSC: {mean_test_dsc}")
        print(f"Mean Eval SENSITIVITY: {mean_test_sensitivity}")
        print(f"Mean Eval FALSE POS: {mean_test_false_positives}")
        print()


