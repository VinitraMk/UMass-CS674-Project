# UMass-CS674-Project

This repository is to maintain code for the final project of UMass CS674 course. This project covers motion synthesis using text input.

## Instructions for setup

1. Install the python dependencies using the requirements.txt file.
2. You can download the dataset hosted in this drive link. This is processed by separate processing pipeline that's provided by the authors of the 
humanml3d data. Due to the distribution policy of the amass_data we can't directly distribute the processed data but for the sake of easy installation of
this project we are providing with a [drive link](https://drive.google.com/file/d/1bOmzxuH8xNk2XM4Onfde0tstDovgRowV/view?usp=sharing). You can directly download and unzip them to a folder called datasets/humanml3d/ or you can run the follow commands:
```
gdown https://drive.google.com/uc?id=1bOmzxuH8xNk2XM4Onfde0tstDovgRowV
unzip ./HumanML3D -qq -d ./datasets/humanml3d
```
3. Alternatively, you can also follow the detailed procedure to setup the dataset provided by the authors of HumanML3D. Download the dataset for this folder from amass data site. Navigate to this link of humanml3d repository (https://github.com/EricGuo5513/HumanML3D/tree/main)
4. Run the scripts for raw_pose_processing.ipynb, motion_representation.ipynb, cal_mean_variance.ipynb.
5. Follow the instructions of downloading datasets listed in the raw_pose_processing.ipynb. These datasets should be unzipped directly in a folder called amass_data/. For ex, the kitml dataset should be unzipped such that it follows this structure: amass_data/KIT/001/001.npy
6. Make sure all the other files such as license.txt are removed. The datasets folders will have also have to be renamed to some specific names described in the raw_pose_processing.ipynb file.
7. After running all 3 notebooks, you should end up with a folder called datasets/humanml3d which consists of npy files, text files, Mean.npy and Std.npy of the whole data. This should consists of motion representations collated from different data sources. It follows the SMPL skeleton structure of 22 joints.
8. Make sure to run the verification cells in the above scripts so there aren't any errors on dataset setup.
9. Make a folder called 'checkpoints' which will store all the necessary model checkpoints required to run this project and test it.
10. Use the appropriate config files based on the model stage as follows:

stage | config file path |
------|-------------------|
gan   | ./configs/config_gan_humanml3d.yaml |
wgan  | ./configs/config_wgan_humanml3d.yaml |
wgangp| ./configs/config_wgangp_humanml3d.yaml |

