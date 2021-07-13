import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from utils.dataloader import AneurysmDataset
from utils.DenseSeg import DenseNetSeg3D
from utils.metrics import dice
from utils.evaluation import get_dsc, get_hausdorff, get_vs, get_images
import SimpleITK as sitk


def get_metrics(scores, labels):
    # binarize -> hard decision -> if pixel > 0.5 -> aneurysm, else not
    preds = np.zeros_like(scores)
    preds[scores < 0.5] = 0
    preds[scores >= 0.5] = 1

    # Prior to dilation (manipulation of prediction may be dangerous and changing the actual result?)
    # Assume for segmentation the dilation is not required
    test_image, result_image = get_images(sitk.GetImageFromArray(labels), sitk.GetImageFromArray(preds))
    # dsc = get_dsc(test_image, result_image)
    # print("dsc")
    # print(dsc)
    h95 = get_hausdorff(test_image, result_image)
    vs = get_vs(test_image, result_image)

    return h95, vs


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


def run_model_get_scores(example, label, device, target_resolution, file, epoch, step, train=True):
    label = label.type(torch.LongTensor)
    label = label.to(device)
    example = example.to(device)
    example = example.double()

    scores = model(example, device, target_resolution)
    scores = torch.squeeze(scores)
    loss = criterion(scores, label.float())
    loss_val = loss.item()

    batch_h95, batch_vs = 0, 0
    h95_counter, vs_counter = 0, 0

    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        file.write('TrainLossEpoch' + str(epoch) + 'Step' + str(step) + ': ' + str(loss_val) + '\n')
    else:
        file.write('EvalLossEpoch' + str(epoch) + 'Step' + str(step) + ': ' + str(loss_val) + '\n')

    for l, s in zip(label.detach().cpu().numpy(), scores.detach().cpu().numpy()):
        h95, vs = get_metrics(s.squeeze(), l.squeeze())
        if not np.isnan(h95):
            batch_h95 += h95
            h95_counter += 1
        if not np.isnan(vs):
            batch_vs += vs
            vs_counter += 1

    # sum_aneurysm_truth = torch.sum(label)
    # sum_aneurysm_truth_batch += sum_aneurysm_truth.item()

    # sum_aneurysm_pred = torch.sum(sc)
    # sum_aneurysm_pred_batch += sum_aneurysm_pred.item()
    mean_h95 = batch_h95 / h95_counter if h95_counter > 0 else np.nan
    mean_vs = batch_vs / vs_counter if vs_counter > 0 else np.nan

    return {"loss": loss_val, "h95": mean_h95, "vs": mean_vs}


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
    file.write(train_eval + 'LossEpoch' + str(epoch) + ', Amount pixel truth aneurysm: ' + str(
        int(sum_aneurysm_truth_batch)) + '\n')
    file.write(train_eval + 'LossEpoch' + str(epoch) + ', Amount pixel predicted aneurysm: ' + str(
        int(sum_aneurysm_pred_batch)) + '\n')
    file.write(train_eval + 'LossEpoch' + str(epoch) + ', Difference: ' + str(
        int(sum_aneurysm_pred_batch - sum_aneurysm_truth_batch)) + '\n')
    file.write(train_eval + 'LossEpoch' + str(epoch) + ', BCEWithLogitsLoss Mean: ' + str(np.mean(loss_batch)) + '\n')


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

        sum_h95_train, sum_vs_train = 0, 0
        loss_batch_train = []
        train_h95_counter, train_vs_counter = 0, 0
        # training
        for train_step, [train_ex, train_l] in enumerate(tqdm(train, desc='Train')):
            metrics_train = run_model_get_scores(
                train_ex, train_l, device, target_resolution,
                file, epoch, train_step, train=True)

            loss_batch_train.append(metrics_train["loss"])
            if not np.isnan(metrics_train["h95"]):
                sum_h95_train += metrics_train["h95"]
                train_h95_counter += 1
            if not np.isnan(metrics_train["vs"]):
                sum_vs_train += metrics_train["vs"]
                train_vs_counter += 1

        print(f"Mean Train H95: {sum_h95_train / train_h95_counter}")
        print(f"Mean Train VS: {sum_vs_train / train_vs_counter}")

        # write_stats_after_epoch(sum_aneurysm_truth_batch_train, sum_aneurysm_pred_batch_train, loss_batch_train, epoch,
                                # 'Train', file)

        if (epoch + 1) % 10 == 0:
            sum_h95_eval, sum_vs_eval = 0, 0
            loss_batch_eval = []
            eval_h95_counter, eval_vs_counter = 0, 0

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
            print(f"Mean Eval H95: {sum_h95_eval / eval_h95_counter}")
            print(f"Mean Eval VS: {sum_vs_eval / eval_vs_counter}")

            torch.save(model.state_dict(), model_name)

            # write_stats_after_epoch(sum_aneurysm_truth_batch_eval, sum_aneurysm_pred_batch_eval, loss_batch_eval,
                                    # epoch, 'Eval', file)

    file.close()
