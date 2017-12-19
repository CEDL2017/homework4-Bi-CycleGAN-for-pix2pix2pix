"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    's2w_train': 1231,
    's2w_test': 309
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    's2w_train': '.jpg',
    's2w_test': '.jpg',
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    's2w_train': './input/s2w/s2w_train.csv',
    's2w_test': './input/s2w/s2w_test.csv',
}
