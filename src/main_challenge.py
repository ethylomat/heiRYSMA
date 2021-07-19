import torch
from tqdm.auto import tqdm
import os
from utils.dataloader import AneurysmDataset
from utils.DenseSeg import DenseNetSeg3D
import numpy as np
from utils import cropper
from utils import preprocessing_challenge
import nibabel as nib


if __name__ == "__main__":

    #data_path = ...  # insert absolute path to the single TOF MRA data directory (which includes /input/orig and /output)
    data_path = 'C:/Users/marin/Desktop/DesktopLenovo/STUDIUM MASTER/1.Semester/DV_SV_Exercises_Project/ProseminarMedAnal/data/challenge_test_img'  # insert absolute path to the data directory
    target_resolution = (256, 256, 0)  # has to be (256, 256, 0)
    overlap = 1  # has to be 1
    batch_size = 1
    include_augmented_data = False  # not relevant for challenge
    include_resizing = True  # not relevant for challenge
    model_name = 'last_trained_model'  # model for challenge
    loss_fct = "BCE"  # or FCL or DICE, relevant -> forward method in DenseSeg

    test_challenge = torch.utils.data.DataLoader(
        AneurysmDataset(
            data_path=data_path,
            target_resolution=target_resolution,
            overlap=overlap,
            include_augmented_data=include_augmented_data,
            include_resizing=include_resizing,
            train_eval_test='test',
            challenge_mode_on=True
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    device = torch.device(0 if torch.cuda.is_available() else "cpu")

    model = DenseNetSeg3D(device=device)
    model = model.double()
    model.to(device)

    model.load_state_dict(torch.load(model_name), strict=False)

    scores_arr = []
    data_shape = (0,0,0)
    for test_challenge_step, [test_challenge_ex, test_challenge_l, data_shape, data_affine] in enumerate(tqdm(test_challenge, desc='Test Challenge')):
        test_challenge_l = test_challenge_l.to(device)
        test_challenge_ex = test_challenge_ex.to(device)
        test_challenge_ex = test_challenge_ex.double()

        scores = model(test_challenge_ex, target_resolution, loss_fct)
        if loss_fct == 'FCL':
            scores = torch.sigmoid(scores)
        scores = torch.squeeze(scores)
        scores[scores < 0.5] = 0
        scores[scores >= 0.5] = 1

        scores_arr.append(scores.cpu().detach().numpy())

    data_shape_reconstruction = (256, 256, data_shape[2].item())

    reconstructed_256x256xXimg, reconstructed_256x256xXimg_overlap_count = cropper.reconstruct_orig_img(scores_arr, data_shape_reconstruction, 256, 8, 1)

    reconstructed_256x256xXimg_final = reconstructed_256x256xXimg / reconstructed_256x256xXimg_overlap_count

    final_prediction = preprocessing_challenge.resize_width_height_skimage_challenge(reconstructed_256x256xXimg_final, data_shape[0].item())

    final_prediction[final_prediction < 0.5] = 0
    final_prediction[final_prediction >= 0.5] = 1

    nib_final_prediction = nib.Nifti1Image(final_prediction, affine=data_affine.detach().numpy().squeeze())
    nib.save(nib_final_prediction, os.path.join(data_path, 'output', 'result.nii.gz'))
