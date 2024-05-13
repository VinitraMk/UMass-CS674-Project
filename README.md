# UMass-CS674-Project

This repository is to maintain code for the final project of UMass CS674 course. This project covers motion synthesis using text.

## Project structure
/body_models - body models to preprocess data into humanml3d joints
/common - common utilities
/configs - config yaml files to run the mld package scripts
/datautils - folder to store data preprocessing utilities
/data - folder to store all the source data zips as backup. Added in .gitignore
/deps - Folder that is .gitignore and stores all dependencies for the project such as smlpl models
/experiments
/helpers - Folder that containts all the package folders from mld that can be reused in the project
/human_body_prior - folder that consists of scripts to process the human body data
/joints - folder that stores all the joints data after preprocessing all data into humanml3d joints. Added in .gitignore
/models - models folder that has architecture
/motionml - the resultant humanml3d joint representations is saved in this folder. Added in .gitignore
/pose_data - pose data of all the different human body datasets is kept in this folder. Added in .gitignore
/prepare - preparation scripts to download and install all dependencies
/source-data - Extract folders of the datasets. Added in .gitignore

## Instructions for setup

1. Install the python dependencies using the requirements.txt file.
2. Start with running the notebook data-setup.ipynb. Follow the text instructions in the notebook properly. There are checks placed to ensure that you are on the right track. Download and unzip the humanact12 folder into ./pose_data whenever the notebook instructs you to do so.
3. Create /deps folder to download all dependencies for this project.
4. Run the scripts in prepare folder to setup the smpl models, clip models and the text-to-motion evaluators in the ./deps folder.
