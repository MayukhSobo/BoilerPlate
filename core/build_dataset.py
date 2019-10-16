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

from tqdm import tqdm
import random
import csv
import numpy as np
from os import path
from pathlib import Path
from typing import (
    List, Iterator, Tuple,
    Dict, Union, Any
)
from PIL import Image

import sys
# import time

# import click

# parser = argparse.ArgumentParser()
# parser.add_argument('--config', default='conf.yaml', help="Path to the config file")


def _resize_and_save(input_file: Path, output_file: Path, size: int):
    """Resize the image contained in `input_file` and save it to the `output_file`"""
    image = Image.open(input_file)
    # Use bi-linear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(output_file)


def _get_image_paths(data_dir: Iterator[Path],
                     file_type: str) -> Tuple[List[Union[Path, Any]], ...]:
    """
    It gets all the images from the directories mentioned by ```data_dir```

    NOTE: It loads all the file names into the memory

    :param data_dir: Path of train and test dataset
    :param file_type: Type of images in the dataset
    """
    data = []
    for d in data_dir:
        files = []
        for f in d.iterdir():
            if f.name.lower().endswith(file_type):
                files.append(f)
        # print(files)
        data.append(files)
    return tuple(data)


def gather_data(dconf):
    """
    This function process the image data
    in the form of {class}_{id}.jpg in the
    train and test dataset.
    :param dconf: Config object for directories
    """
    # First create the directories
    base_path = Path(dconf.base_dir)
    sub_dirs = (dconf.train_dir, dconf.test_dir)
    data_paths = list(filter(None, map(lambda x: base_path.joinpath(x) if x is not None else x, sub_dirs)))
    # print(data_paths)
    # Validate if they exists or not
    for dp in data_paths:
        if not dp.exists():
            raise OSError(f'Can not find {dp}')
    return _get_image_paths(data_paths, dconf.img_type)

    # parser = kwargs.get('parser')
    # label_info, labels = lconf
    # labels = kwargs.get('labels', None)
    # label_type = label_info['type']
    # if label_type is None:
    #     # When the class labels are present in the file names itself
    #     return _combined_image_labels(data_paths, dconf.img_type)
    # elif label_type.lower() == 'csv':
    #     pass
    # return _csv_labeled_images(
    #     data_paths, file_type,
    #     csv_file='data/label')


def format_n_save(base_output, output_dirs, filenames, label_type, **kwargs):
    mapping = kwargs.get('map')
    resize = kwargs.get('resize')
    parser = kwargs.get('parser')
    img_type = kwargs.get('img_type')
    for dirs in output_dirs:
        # First create the sub-directories
        _p = base_output.joinpath(dirs)
        if not path.exists(_p):
            _p.mkdir()
        print(f"\nProcessing {dirs} data, saving preprocessed data to {_p.resolve()}")
        for input_file in tqdm(filenames[dirs], file=sys.stdout, desc='Processing ' + dirs + ' dataset'):
            if label_type is None:
                output_file = parser(input_file.name, img_type, mapping)
            elif label_type == 'csv':
                output_file = f"{mapping[input_file.stem]}_{input_file.name}"
            # _resize_and_save(input_file, _p.joinpath(output_file), size=resize)


def _get_id_label_mapping(base_dir, label_type):
    id_label_map = {}
    csvpath = Path(base_dir, label_type['path'])
    if not csvpath.exists():
        raise FileNotFoundError("CSV file can not be found")
    else:
        target = set()
        with open(csvpath, 'r', encoding='utf8') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                target.add(row[label_type['target_column']])
        target_label_map = dict(zip(target, np.arange(len(target))))

        with open(csvpath, 'r', encoding='utf8') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                id_label_map[row[label_type['id_column']]] = target_label_map[row[label_type['target_column']]]
    return id_label_map


def split_and_store(dconf, oconf, lconf, datafiles, parser=None, **kwargs):
    try:
        train_files, test_files = datafiles
    except ValueError:
        train_files = datafiles[0]
        test_files = []
    filenames = {
        'train': train_files,
        'test': test_files
    }
    output_dirs = ['train', 'test']
    if oconf.shuffle:
        random.seed(oconf.seed)
        random.shuffle(filenames['train'])
        random.shuffle(filenames['test'])

    # Check for the validation dataset
    if 0.0 < oconf.validation <= 1.0:
        if oconf.validation > 0.5:
            raise UserWarning('Validation frame is very large. Training may get affected')
        output_dirs.append('val')
        split_ratio = oconf.validation
        split_point = int((1 - split_ratio) * len(filenames['train']))
        filenames['val'] = filenames['train'][split_point:]
        filenames['train'] = filenames['train'][:split_point]
    elif oconf.validation == 0.0:
        # It's user's choice not to use any validation
        pass
    else:
        raise ValueError('Prescribed validation range should be 0.0 to 0.5')

    # Check for test dataset
    if filenames['test'] == [] and 0.0 < oconf.test <= 1.0:
        if oconf.test > 0.5:
            raise ValueError('The test data size is too large')
        else:
            # Make the split from the train dataset
            split_ratio = oconf.test
            split_point = int((1 - split_ratio) * len(filenames['train']))
            filenames['test'] = filenames['train'][split_point:]
            filenames['train'] = filenames['train'][:split_point]
    elif not filenames['test']:
        if oconf.test <= 0.0 or oconf.test > 1.0:
            raise ValueError('Test data is not present in the dataset')
    if oconf.resize < 0:
        raise ValueError('Resize parameter can nor be zero')
    elif oconf.resize == 0:
        size = 'original'
    else:
        size = oconf.resize

    if dconf.output_dir is None:
        dir_name = f'data_{size}x{size}_{dconf.train_dir.split("/")[0]}'
        p = Path(dconf.base_dir, dir_name)
    else:
        p = Path(oconf.output_dir)

    # Create the output directory if it doesn't exist
    if not path.exists(p):
        p.mkdir()

    # if parser is None:
    #     # The filename is in the right format already
    #     parser = lambda x: x

    ltype, mapping = lconf
    if ltype['type'] == 'csv':
        id_label_map = _get_id_label_mapping(dconf.base_dir, ltype)
        format_n_save(p, output_dirs, filenames, 'csv', map=id_label_map, resize=oconf.resize)
    elif ltype['type'] is None:
        format_n_save(p, output_dirs, filenames, None, map=mapping, resize=oconf.resize,
                      parser=parser, img_type=dconf.img_type)


if __name__ == '__main__':
    from core.config import Config
    from core import *
    # Read the config file
    conf = Config(config_file='../config.yaml')
    datafiles = gather_data(conf.dirs)
    split_and_store(conf.dirs, conf.operations, conf.labels, datafiles, parser=parser)
    print("\nDone building dataset")
