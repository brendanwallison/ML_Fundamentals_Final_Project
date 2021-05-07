This code implements ProtoPNet for use with data from the kaggle competition. 

In current form, it required very little modification from the original code, but will require more later. Currently, it is using the default one-hot encoding of the Pytorch Dataset class ImageFolder, but this will have to be replaced later by the custom dataloader and dataset currently implemented in the competition_code. This will allow for soundscape testing, instead of just playing with Xeno Canto data.
