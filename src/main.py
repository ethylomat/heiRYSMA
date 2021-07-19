import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
from utils.dataloader import AneurysmDataset
from utils.DenseSeg import DenseNetSeg3D
import numpy as np
from utils.dice_loss import BinaryDiceLoss
from utils.focal_loss import FocalLoss
from utils import evaluation
from utils import evaluation_detection

from utils.metrics import dice
from utils.evaluation import get_dsc, get_hausdorff, get_vs, get_images
import SimpleITK as sitk

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


def run_model_get_scores(example, label, device, target_resolution, sum_aneurysm_truth_batch, sum_aneurysm_pred_batch, loss_batch, file, epoch, step, train=True):
    label = label.to(device)
    example = example.to(device)
    example = example.double()

    scores = model(example, target_resolution, loss_fct)
    scores = torch.squeeze(scores)
    loss = criterion(scores, label)
    loss_val = loss.item()
    loss_batch.append(loss_val)

    sum_aneurysm_truth = torch.sum(label)
    sum_aneurysm_truth_batch += sum_aneurysm_truth.item()

    batch_h95, batch_vs, batch_dsc, batch_sensitivity, batch_false_positives = 0, 0, 0, 0, 0
    h95_counter, vs_counter, dsc_counter, sensitivity_counter, false_positives_counter = 0, 0, 0, 0, 0

    # binarize -> hard decision -> if pixel > 0.5 -> aneurysm, else not
    if loss_fct == "FCL":  #  FCL has binary cross entropy with logits (sigmoid included in the loss, not in the net output)
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


    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        file.write('TrainLossEpoch'+ str(epoch) + 'Step' + str(step) + ': ' + str(loss.item()) + '\n')
    else:
        file.write('EvalLossEpoch'+ str(epoch) + 'Step' + str(step) + ': ' + str(loss.item()) + '\n')


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

    if (os.path.isfile('log/' +  best_loss_file_name + '_best_loss_log.txt')):
            f = open('log/' +  best_loss_file_name + '_best_loss_log.txt', "r+")
            lines = f.read().splitlines()
            best_loss = lines[-1]
            best_loss = float(best_loss)

    else:
        f = open('log/' + best_loss_file_name + '_best_loss_log.txt', 'w')
        best_loss = 100.0

    return f, best_loss

def write_stats_after_epoch(sum_aneurysm_truth_batch, sum_aneurysm_pred_batch, loss_batch, epoch, train_eval, file, mean_h95, mean_vs, mean_dsc):
    print(train_eval + ', epoch: ' + str(epoch))
    print("Amount pixel truth aneurym: " + str(int(sum_aneurysm_truth_batch)))
    print("Amount pixel predicted aneurym: " + str(int(sum_aneurysm_pred_batch)))
    print('Difference: ' + str(int(sum_aneurysm_pred_batch - sum_aneurysm_truth_batch)))
    print('Loss: ' + str(np.mean(loss_batch)))
    print(f"Mean Hausdorff: {mean_h95}")
    print(f"Mean Volumetric Similarity: {mean_vs}")
    print(f"Mean Dice Coefficient: {mean_dsc}")
    print('')
    file.write(train_eval + 'LossEpoch' + str(epoch) + ', Amount pixel truth aneurysm: ' + str(int(sum_aneurysm_truth_batch)) + '\n')
    file.write(train_eval + 'LossEpoch' + str(epoch) + ', Amount pixel predicted aneurysm: ' + str(int(sum_aneurysm_pred_batch)) + '\n')
    file.write(train_eval + 'LossEpoch' + str(epoch) + ', Difference: ' + str(int(sum_aneurysm_pred_batch - sum_aneurysm_truth_batch)) + '\n')
    file.write(train_eval + 'LossEpoch' + str(epoch) + ', BCEWithLogitsLoss Mean: ' + str(np.mean(loss_batch)) + '\n')
    file.write(train_eval + 'LossEpoch' + str(epoch) + ', Mean Hausdorff: ' + str(mean_h95) + '\n')
    file.write(train_eval + 'LossEpoch' + str(epoch) + ', Mean Volumetric Similarity: ' + str(mean_vs) + '\n')
    file.write(train_eval + 'LossEpoch' + str(epoch) + ', Mean Dice Coefficient: ' + str(mean_dsc) + '\n')


if __name__ == "__main__":

    data_path = ...  # insert absolute path to the data directory
    target_resolution = (256, 256, 0)  # modify here if other resolution needed, currently available (64, 64, 64) and (128, 128, 100)
    overlap = 1  # overlap for cropping
    batch_size = 2
    include_augmented_data = False  # enable if flipped data (vertically + horizonatally), rotated data (180 degrees) and brighter data 5% wanted
    include_resizing = True  # enable if resizing wanted, else cropping applied
    model_name = 'model__BCE__256_256_0__o01__b10__lr0001'  # resolution_overlap_batchsize_learning_rate
    model_name_optimizer = model_name + '_optim'  # resolution_overlap_batchsize_learning_rate
    loss_fct = "BCE"  # or FCL or DICE

    loss_log_file = create_loss_log_file(model_name)
    best_loss_log_file, curr_best_batch_loss = create_current_best_loss_file(model_name)

    train = torch.utils.data.DataLoader(
        AneurysmDataset(
            data_path=data_path,
            target_resolution=target_resolution,
            overlap=overlap,
            include_augmented_data=include_augmented_data,
            include_resizing=include_resizing,
            train_eval_test='train',
            challenge_mode_on=False
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
            train_eval_test='eval',
            challenge_mode_on=False
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # test = torch.utils.data.DataLoader(
    #     AneurysmDataset(
    #         data_path=data_path,
    #         target_resolution=target_resolution,
    #         overlap=overlap,
    #         include_augmented_data=include_augmented_data,
    #         include_resizing=include_resizing,
    #         train_eval_test='test',
    #          challenge_mode_on = False
    #
    #     ),
    #     batch_size=batch_size,
    #     shuffle=True,
    #     drop_last=True
    # )

    device = torch.device(0 if torch.cuda.is_available() else "cpu")

    model = DenseNetSeg3D(device=device)
    model = model.double()
    model.to(device)

    model.load_state_dict(torch.load(model_name), strict=False)  # enable if training continued

    # choose the loss function
    if loss_fct == 'FCL':
        criterion = FocalLoss(pos_weight=torch.tensor(10.), gamma=2, reduction='mean')  # pos_weight info: https://discuss.pytorch.org/t/how-to-apply-weighted-loss-to-a-binary-segmentation-problem/35317
    elif loss_fct == "BCE":
        criterion = nn.BCELoss()
    elif loss_fct == "DICE":
        criterion = BinaryDiceLoss()
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    #optimizer.load_state_dict(torch.load(model_name_optimizer))


    for epoch in tqdm(range(5000), desc='Epoch'):

        model.train()

        sum_h95_train, sum_vs_train, sum_dsc_train = 0, 0, 0
        train_h95_counter, train_vs_counter, train_dsc_counter = 0, 0, 0

        sum_aneurysm_truth_batch_train = 0
        sum_aneurysm_pred_batch_train = 0
        loss_batch_train = []
        loss_batch_train_metrics = []
        # training
        # for train_step, [train_ex, train_l, data_shape] in enumerate(tqdm(train, desc='Train')):
        #
        #     sum_aneurysm_truth_batch_train, sum_aneurysm_pred_batch_train, loss_batch_train, metrics_train = run_model_get_scores(train_ex, train_l, device, target_resolution,
        #                                                                                                            sum_aneurysm_truth_batch_train, sum_aneurysm_pred_batch_train, loss_batch_train,
        #                                                                                                            loss_log_file, epoch, train_step, train=True)
        #
        #     loss_batch_train_metrics.append(metrics_train["loss"])
        #     if not np.isnan(metrics_train["h95"]):
        #         sum_h95_train += metrics_train["h95"]
        #         train_h95_counter += 1
        #     if not np.isnan(metrics_train["vs"]):
        #         sum_vs_train += metrics_train["vs"]
        #         train_vs_counter += 1
        #     if not np.isnan(metrics_train["dsc"]):
        #         sum_dsc_train += metrics_train["dsc"]
        #         train_dsc_counter += 1
        #
        # mean_train_h95 = sum_h95_train / train_h95_counter
        # mean_train_vs = sum_vs_train / train_vs_counter
        # mean_train_dsc = sum_dsc_train / train_dsc_counter
        #
        # write_stats_after_epoch(sum_aneurysm_truth_batch_train, sum_aneurysm_pred_batch_train, loss_batch_train, epoch, 'Train', loss_log_file, mean_train_h95, mean_train_vs, mean_train_dsc)

        if (epoch + 1) % 1 == 0:
            #model.eval()

            sum_aneurysm_truth_batch_eval = 0
            sum_aneurysm_pred_batch_eval = 0
            sum_h95_eval, sum_vs_eval, sum_dsc_eval, sum_sensitivity_eval, sum_false_positives_eval = 0, 0, 0, 0, 0
            loss_batch_eval = []
            loss_batch_eval_metrics = []
            eval_h95_counter, eval_vs_counter, eval_dsc_counter, eval_sensitivity_counter, eval_false_positives_counter = 0, 0, 0, 0, 0

            # eval
            for eval_step, [eval_ex, eval_l, data_shape] in enumerate(tqdm(eval, desc='Eval')):
                sum_aneurysm_truth_batch_eval, sum_aneurysm_pred_batch_eval, loss_batch_eval, metrics_eval = run_model_get_scores(eval_ex, eval_l, device, target_resolution,
                                                                                                                    sum_aneurysm_truth_batch_eval, sum_aneurysm_pred_batch_eval, loss_batch_eval,
                                                                                                                    loss_log_file, epoch, eval_step, train=False)

                loss_batch_eval_metrics.append(metrics_eval["loss"])
                if not np.isnan(metrics_eval["h95"]):
                    sum_h95_eval += metrics_eval["h95"]
                    eval_h95_counter += 1
                if not np.isnan(metrics_eval["vs"]):
                    sum_vs_eval += metrics_eval["vs"]
                    eval_vs_counter += 1
                if not np.isnan(metrics_eval["dsc"]):
                    sum_dsc_eval += metrics_eval["dsc"]
                    eval_dsc_counter += 1
                if not np.isnan(metrics_eval["sensitivity"]):
                    sum_sensitivity_eval += metrics_eval["sensitivity"]
                    eval_sensitivity_counter += 1
                if not np.isnan(metrics_eval["false_positives"]):
                    sum_false_positives_eval += metrics_eval["false_positives"]
                    eval_false_positives_counter += 1


            mean_eval_h95 = sum_h95_eval / eval_h95_counter
            mean_eval_vs = sum_vs_eval / eval_vs_counter
            mean_eval_dsc = sum_dsc_eval / eval_dsc_counter
            mean_eval_sensitivity = sum_sensitivity_eval / eval_sensitivity_counter
            mean_eval_false_positives = sum_false_positives_eval / eval_false_positives_counter
            print(f"Mean Eval H95: {mean_eval_h95}")
            print(f"Mean Eval VS: {mean_eval_vs}")
            print(f"Mean Eval DSC: {mean_eval_dsc}")
            print(f"Mean Eval SENSITIVITY: {mean_eval_sensitivity}")
            print(f"Mean Eval FALSE POS: {mean_eval_false_positives}")

            # if (np.mean(loss_batch_eval) < curr_best_batch_loss):
            #     print("Current best batch loss: " + str(curr_best_batch_loss))
            #     print("New best batch loss: " + str(np.mean(loss_batch_eval)))
            #     print("Store model...")
            #     curr_best_batch_loss = np.mean(loss_batch_eval)
            #     torch.save(model.state_dict(), model_name)
            #     torch.save(optimizer.state_dict(), model_name_optimizer)
            #     best_loss_log_file.write(str(curr_best_batch_loss) + '\n')
            #     best_loss_log_file.flush()

            write_stats_after_epoch(sum_aneurysm_truth_batch_eval, sum_aneurysm_pred_batch_eval, loss_batch_eval,
                                    epoch, 'Eval', loss_log_file, mean_eval_h95, mean_eval_vs, mean_eval_dsc)

    loss_log_file.close()