# Explanation on how to run on Taurus

First you need to create a symlink from the scrath folder storing the code to your personal home folder.

The scripts assume that your scripts reside in /home/<USERNAME>/scratch

`ln -s /scratch/ws/0/cosi765e-python_virtual_environment scratch`

## idea for implementation of showing wether an image is augmented or not

1. generate another dataset of the same size as the unaugmented one but with augmentation
2. concat both sets
3. extend active learning dataset so it has a bool for each idx which will say augmented yes or no, similar to label map. maybe also implement in splittedataset
4. loop through the whole dataset and set the augmented value