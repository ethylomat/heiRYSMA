# heiRYSMA
## Project Seminar - Biomedical Image Analysis

### Preprocessing/Data loading
#### dataloader.py -> AneurysmDataset:
- checks if data in target resolution available (if all 113x datapoints are of the target resolution shape, e.g. 560x560x140 stored as npy arrays 
	in data_path/preprocessed/data_mask_W_H_AmoundSlices.npy (e.g. data_mask_560_560_140.npy) with corresponding masks (e.g. data_mask_560_560_140.npy)) or cropped in the target resolution (every datapoint cropped in e.g. n smaller 64x64x64 cubes without scaling)
  - if yes -> loads training data and test data (70/15/15 ratio, train-eval-test)
  - if no -> checks if original data (all 113x datapoints of different shapes is stored in data_path/preprocessed/data_orig.npy
    - if yes -> loads data and the corresponding masks -> resizes them to target resolution (and includes possibly augmentation) or cropps them as described above -> stores them under the path as described above -> loads training data and test data (70/15/15 ratio, train-eval-test)
    - if no -> loads data and corresponding masks from nii.gz files into npy arrays -> stores them -> resizes them to target resolution -> stores them -> loads training data and test data (70/30 ratio) 
- the data is shuffled before splitting into train/eval/test (shuffled always in the same order, such that the eval and test data are never included in the train data), shuffling order is stored as np array
- the train/eval/test data is normalized to range 0-1
- the train/eval/test masks are set to 1 if > 0.5 (through resizing, the masks are modified and are no longer of a binary form -> normalize again to 0 or 1) 

#### data directory structure: 
	- data dir 
		- 10001
			- orig
				- TOF.nii.gz
				...
			- pre
				...
			- aneurysms.nii.gz
			- location.txt
		- 10002
			...
		...
		- preprocessed 
			- data_mask.npy
			- data_mask_560_560_140.npy
			...
			- data_orig.npy
			- data_orig_560_560_140.npy
			...
