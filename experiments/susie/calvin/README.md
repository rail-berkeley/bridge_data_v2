# Goal-conditioned policy training code for SuSIE paper, specific to CALVIN dataset

## To train goal reaching or language conditioned policies, you will first need to prepare the CALVIN dataset for training

1. Download the full CALVIN dataset following instructions in https://github.com/mees/calvin
2. Run the scripts in ```dataset_conversion_scripts``` to create two versions of the dataset in TFRecord format. Appropriately set path strings

## Training GCBC and LCBC policies

The GCBC policy, which is used as the low-level policy in SuSIE, and the baseline LCBC policy can be trained via the corresonding scripts in the ```scripts``` subdirectory. Make sure to first replace the placeholder path strings in the ```configs``` folder with the correct strings for your setup.
