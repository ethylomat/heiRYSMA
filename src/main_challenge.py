import torch
from tqdm.auto import tqdm
import os
import re
import datetime
from src.utils.dataloader import AneurysmDataset
from src.utils.DenseSeg import DenseNetSeg3D
import numpy as np
from src.utils import cropper
from src.utils import preprocessing_challenge
import nibabel as nib

def logging(data_path, line):
    with open(os.path.join(data_path, "log.txt"), "a") as f:
        f.write(f"{datetime.datetime.now()}: {line} <br>\n")

def main(data=None, model=None, resolution=[256,256,0], overlap=1, loss=None, webapp=False):
    target_resolution = tuple(resolution)
    data_path = data  # insert absolute path to the single TOF MRA data directory (which includes /input/orig and /output)
    overlap = overlap  # has to be 1
    batch_size = 1
    include_augmented_data = False  # not relevant for challenge
    include_resizing = True  # not relevant for challenge
    model_name = model  # model for challenge

    if webapp:
        logging(data_path, "Started processing ...")

    if loss is not None:
        loss_fct = loss
    else:
        try:
            loss_fct = re.search(r"model__([A-Z]+)__", model_name).group(1)
        except AttributeError:
            raise AttributeError("No loss function indicated: Use --loss param.")

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

    model.load_state_dict(torch.load(model_name, map_location=device), strict=False)

    scores_arr = []
    data_shape = (0,0,0)
    for test_challenge_step, [test_challenge_ex, test_challenge_l, data_shape] in enumerate(tqdm(test_challenge, desc='Test Challenge')):
        if webapp:
            logging(data_path, f"Step Number: {test_challenge_step}")
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

    nib_final_prediction = nib.Nifti1Image(final_prediction, affine=np.eye(4))
    nib.save(nib_final_prediction, os.path.join(data_path, 'output', 'result.nii.gz'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='heiRYSMA')
    parser.add_argument('--data', dest='data_path', default=None, help='Absolute path of the data directory')
    parser.add_argument('--model', dest='model_path', default=None, help='Absolute path of the model directory')
    parser.add_argument('--resolution', nargs=3, type=int, dest='resolution', default=[256, 256, 0],
                        help='Dimension for cropping/resizing (e.g. 64 for dimension 64 x 64 x 64)')
    parser.add_argument('--overlap', type=int, dest='overlap', default=1, help='Overlap for cropping')
    parser.add_argument('--loss', dest='loss_fct', default=None, help='Loss function')
    arguments = parser.parse_args()
    main(data=arguments.data_path, model=arguments.model_path, resolution=arguments.resolution, overlap=arguments.overlap, loss=arguments.loss_fct)