import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from utils.dataloader import AneurysmDataset
from utils.DenseSeg import DenseNetSeg3D
from utils.metrics import dice
from utils.evaluation import get_dsc, get_hausdorff, get_vs, get_images, get_center_of_mass
from utils.evaluation_detection import  get_locations, get_result_filename, get_result, get_treated_locations, get_detection_metrics
import SimpleITK as sitk


def get_metrics(scores, labels, locations, cropped_data_mask):
    # binarize -> hard decision -> if pixel > 0.5 -> aneurysm, else not
    preds = np.zeros_like(scores)
    preds[scores < 0.5] = 0
    preds[scores >= 0.5] = 1

    # Prior to dilation (manipulation of prediction may be dangerous and changing the actual result?)
    # Assume for segmentation the dilation is not required
    test_image, result_image = get_images(sitk.GetImageFromArray(labels), sitk.GetImageFromArray(preds))
    dsc = get_dsc(test_image, result_image)
    h95 = get_hausdorff(test_image, result_image)
    vs = get_vs(test_image, result_image)

    result_filename = get_result_filename(locations)
    result_locations = get_result(result_filename)

    test_locations = get_locations(os.path.join(locations, 'location.txt'))
    test_image = sitk.ReadImage(os.path.join(locations, 'aneurysms.nii.gz'))

    sensitivity, false_positive_count = get_detection_metrics(test_locations, result_locations, cropped_data_mask)

    print('Sensitivity: %.3f (higher is better, max=1)' % sensitivity)
    print('False Positive Count: %d (lower is better)' % false_positive_count)

    return h95, vs, dsc, sensitivity, false_positive_count


# location holen
# center of mass holen
# detection-Metrix ausrechnen


    # Dilate image
    # Maybe not necessary? -> maybe reinsert later
    # # dilated_preds = sitk.GetImageFromArray(preds)
    # # dilated_preds = sitk.BinaryDilate(dilated_preds, (2, 2, 2))
    # # dilated_preds = sitk.GetArrayFromImage(dilated_preds)

    # reinsert?
    # cluster_pattern = ndimage.generate_binary_structure(rank=dilated_preds.ndim, connectivity=dilated_preds.ndim)
    # aneurysm_clusters = ndimage.label(dilated_preds, cluster_pattern)[0]
    # cluster_indexes = np.unique(aneurysm_clusters)[1:]

    # take random  number of largest connected components
    # maximum number of detected aneurysms is 10, minimum is 1
    # # if len(cluster_indexes) > 10:
   # # num = random.randint(1, 10)
    ## else:
      ##  num = len(cluster_indexes)

    ## pred_locations = np.fliplr(ndimage.measurements.center_of_mass(dilated_preds,
       ##                                                            aneurysm_clusters,
        ##                                                           cluster_indexes[:num])).astype(int)

   ## pred_images = sitk.BinaryThreshold(sitk.GetImageFromArray(aneurysm_clusters),
     ##                                  lowerThreshold=1, upperThreshold=num)

def run_model_get_scores(example, label, location, device, target_resolution, file, epoch, step, train=True, cropped_data_mask):
    label = label.type(torch.LongTensor)
    label = label.to(device)
    example = example.to(device)
    example = example.double()

    scores = model(example, device, target_resolution)
    scores = torch.squeeze(scores)
    loss = criterion(scores, label.float())
    loss_val = loss.item()

    label_com = get_center_of_mass(label)
    scores_com = get_center_of_mass(scores)

    batch_h95, batch_vs, batch_dsc, batch_sens, batch_fpc = 0, 0, 0, 0, 0
    h95_counter, vs_counter, dsc_counter, sens_counter, fpc_counter = 0, 0, 0, 0, 0

    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        file.write('TrainLossEpoch' + str(epoch) + 'Step' + str(step) + ': ' + str(loss_val) + '\n')
    else:
        file.write('EvalLossEpoch' + str(epoch) + 'Step' + str(step) + ': ' + str(loss_val) + '\n')

    for l, s, loc in zip(label.detach().cpu().numpy(), scores.detach().cpu().numpy(), location.detach().cpu().numpy()):
        h95, vs, dsc, sensitivity, false_positive_count = get_metrics(s.squeeze(), l.squeeze(), loc.squeeze())
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
            batch_sens += sensitivity
            sens_counter += 1
        if not np.isnan(false_positive_count):
            batch_fpc += false_positive_count
            fpc_counter += 1

    # sum_aneurysm_truth = torch.sum(label)
    # sum_aneurysm_truth_batch += sum_aneurysm_truth.item()

    # sum_aneurysm_pred = torch.sum(sc)
    # sum_aneurysm_pred_batch += sum_aneurysm_pred.item()
    mean_h95 = batch_h95 / h95_counter if h95_counter > 0 else np.nan
    mean_vs = batch_vs / vs_counter if vs_counter > 0 else np.nan
    mean_dsc = batch_dsc / dsc_counter if dsc_counter > 0 else np.nan
    mean_sens = batch_sens / sens_counter if sens_counter > 0 else np.nan
    mean_fpc = batch_fpc / fpc_counter if fpc_counter > 0 else np.nan

    return {"loss": loss_val, "h95": mean_h95, "vs": mean_vs, "dsc": mean_dsc, "sensitivity": mean_sens, "false_positive_count": mean_fpc}


def create_loss_log_file(model_name):
    f = open('log/' + model_name + '_loss_log.txt', 'a')
    f.write('Log file start for the test: ' + model_name + '_loss_log.txt\n')

    return f


def write_stats_after_epoch(loss_batch, mean_h95, mean_vs, mean_dsc):
    print('BCEWithLogitsLoss: ' + str(np.mean(loss_batch)))
    print(f"Mean Hausdorff: {mean_h95}")
    print(f"Mean Volumetric Similarity: {mean_vs}")
    print(f"Mean Dice Coefficient: {mean_dsc}")

if __name__ == "__main__":

    data_path = r"C:/Users/D070310/Aneurysm_data"  # insert absolute path to the data directory
    target_resolution = (
        64, 64, 64)  # modify here if other resolution needed, currently available (64, 64, 64) and (128, 128, 100)
    overlap = 10  # overlap for cropping
    batch_size = 2
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
        batch_size=1,  # batch_size,
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

        sum_h95_train, sum_vs_train, sum_dsc_train = 0, 0, 0
        loss_batch_train = []
        train_h95_counter, train_vs_counter, train_dsc_counter = 0, 0, 0
        # training
        for train_step, [train_ex, train_l, train_loc] in enumerate(tqdm(train, desc='Train')):
            metrics_train = run_model_get_scores(
                train_ex, train_l, train_loc, device, target_resolution,
                file, epoch, train_step, train=True)

            loss_batch_train.append(metrics_train["loss"])
            if not np.isnan(metrics_train["h95"]):
                sum_h95_train += metrics_train["h95"]
                train_h95_counter += 1
            if not np.isnan(metrics_train["vs"]):
                sum_vs_train += metrics_train["vs"]
                train_vs_counter += 1
            if not np.isnan(metrics_train["dsc"]):
                sum_dsc_train += metrics_train["dsc"]
                train_dsc_counter += 1

        mean_train_h95 = sum_h95_train / train_h95_counter
        mean_train_vs = sum_vs_train / train_vs_counter
        mean_train_dsc = sum_dsc_train / train_dsc_counter

        write_stats_after_epoch(loss_batch_train, mean_train_h95, mean_train_vs, mean_train_dsc)

        if (epoch + 1) % 10 == 0:
            sum_h95_eval, sum_vs_eval, sum_dsc_eval = 0, 0, 0
            loss_batch_eval = []
            eval_h95_counter, eval_vs_counter, eval_dsc_counter = 0, 0, 0

            # evaluation
            for eval_step, [eval_ex, eval_l] in enumerate(tqdm(eval, desc='Eval')):
                metrics_eval = run_model_get_scores(eval_ex, eval_l, device, target_resolution,
                                                       file, epoch, eval_step, train=False)


            loss_batch_eval.append(metrics_train["loss"])
            if not np.isnan(metrics_train["h95"]):
                sum_h95_eval += metrics_train["h95"]
                eval_h95_counter += 1
            if not np.isnan(metrics_train["vs"]):
                sum_vs_eval += metrics_train["vs"]
                eval_vs_counter += 1
            if not np.isnan(metrics_train["dsc"]):
                sum_dsc_eval += metrics_train["dsc"]
                eval_dsc_counter += 1

            mean_eval_h95 = sum_h95_eval / eval_h95_counter
            mean_eval_vs = sum_vs_eval / eval_vs_counter
            mean_eval_dsc = sum_dsc_eval / eval_dsc_counter
            print(f"Mean Eval H95: {mean_eval_h95}")
            print(f"Mean Eval VS: {mean_eval_vs}")
            print(f"Mean Eval DSC: {mean_eval_dsc}")

            torch.save(model.state_dict(), model_name)

            write_stats_after_epoch(loss_batch_eval, mean_eval_h95, mean_eval_vs, mean_eval_dsc)

    file.close()
