"""Split the SIGNS dataset into train/val/test and resize images to 64x64.
​
The SIGNS dataset comes into the following format:
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...
​
Original images have size (3024, 3024).
Resizing to (64, 64) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
makes training faster.
​
We already have a test set created, so we only need to split "train_signs" into train and val sets.
Because we don't have a lot of images and we want that the statistics on the val set be as
representative as possible, we'll take 20% of "train_signs" as val set.
"""

import argparse
import random
from pathlib import Path
from typing import (
    List, Iterator, Tuple,
    Dict, Union, Any
)
from PIL import Image


# from tqdm import tqdm
# import click

# parser = argparse.ArgumentParser()
# parser.add_argument('--config', default='conf.yaml', help="Path to the config file")


def resize_and_save(filename, output_dir, size):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(output_dir.joinpath(filename.name))


def _combined_image_labels(data_dir: Iterator[Path],
                           file_type: str,
                           labels: Dict[str, int] = None,
                           **kwargs) -> Tuple[List[Union[Path, Any]], ...]:
    """
    It combines the file ids and class labels according to `labels`.
    If labels are not None, we infer the labels from the dict or
    if None then we assume the labels are already existing.
    """
    data = []
    for d in data_dir:
        files = []
        for f in d.iterdir():
            if f.name.lower().endswith(file_type):
                if labels is None:
                    files.append(f)
                else:
                    # Define your own file name parser with the labels
                    # The accecpted format is {class_number}_{id}.{file_type}
                    parser = kwargs.get('parser')
                    files.append(parser(f, labels))
        data.append(files)
    return tuple(data)


def _csv_labeled_images(data):
    pass


def gather_data(dconf, lconf, **kwargs):
    """
    This function process the image data
    in the form of {class}_{id}.jpg in the
    train and test dataset.
    :param lconf: Config object for labels
    :param dconf: Config object for directories
    """
    # First create the directories
    base_path = Path(dconf.base_dir)
    sub_dirs = (dconf.train_dir, dconf.test_dir)
    data_paths = list(map(lambda x: base_path.joinpath(x), sub_dirs))
    # Validate if they exists or not
    for dp in data_paths:
        if not dp.exists():
            raise OSError(f'Can not find {dp}')
    parser = kwargs.get('parser')
    # labels = kwargs.get('labels', None)
    label_info, labels = lconf
    label_type = label_info['type']
    if label_type is None:
        # Labels are given None when the class label is present in the file name
        return _combined_image_labels(data_paths, dconf.img_type, labels=None, parser=parser)
    elif label_type.lower() == 'csv':
        pass
        # return _csv_labeled_images(
        #     data_paths, file_type,
        #     csv_file='data/label')


def split_and_store(dconf, oconf, datafiles):
    train_files, test_files = datafiles
    filenames = {
        'train': train_files,
        'test': test_files
    }
    output_dirs = ['train', 'test']
    if oconf.shuffle:
        random.seed(oconf.seed)
        random.shuffle(filenames['train'])
        random.shuffle(filenames['test'])
    if 0.0 < oconf.validation <= 1.0:
        if oconf.validation > 0.5:
            raise UserWarning('Validation frame is very large. Training may get affected')
        output_dirs.append('val')
        split_ratio = oconf.validation
        split_point = int((1-split_ratio) * len(filenames['train']))
        filenames['val'] = filenames['train'][split_point:]
        filenames['train'] = filenames['train'][:split_point]
    elif oconf.validation == 0.0:
        # It's user's choice not to use any validation
        pass
    else:
        raise ValueError('Prescribed validation range should be 0.0 to 0.5')
    if oconf.resize < 0:
        raise ValueError('Resize parameter can nor be zero')
    elif oconf.resize == 0:
        size = 'original'
    else:
        size = oconf.resize

    if dconf.output_dir is None:
        dir_name = f'data_{size}x{size}'
        p = Path(dconf.base_dir, dir_name)
    else:
        p = Path(oconf.output_dir)

    # Create the output directory
    p.mkdir()

    # for dirs in output_dirs:
    #     _p = p.joinpath(dirs)
    #     _p.mkdir()
    #     print(f"Processing {dirs} data, saving preprocessed data to {_p.resolve()}")
    #     for filename in tqdm(filenames[dirs]):
    #         #print(filename)
    #         resize_and_save(filename, _p, size=arg.dim)
    #         # print(filename)
    #         # break


if __name__ == '__main__':
    #
    #     # parse the config file from the argparser
    #     arg = parser.parse_args()
    from config import Config

    #     # Read the config file
    conf = Config(config_file='../config.yaml')
    datafiles = gather_data(conf.dirs, conf.labels)
    split_and_store(conf.dirs, conf.operations, datafiles)
#     print("Done building dataset")
