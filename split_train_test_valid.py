import pathlib
import pandas as pd
import shutil
from tqdm import tqdm
import os


SEED = 42


def split_train_valid(output, valid, add_backgrounds):
    labels_folder = output.joinpath('labels')
    images_folder = output.joinpath('images')
    all_files_list = list(labels_folder.glob('*.txt'))
    all_files_series = pd.Series(all_files_list)

    valid_files = all_files_series.sample(int(len(all_files_list) * valid), random_state=SEED)

    print('moving valid...')
    valid_folder = output.joinpath('valid')
    for valid_file in tqdm(valid_files, total=len(valid_files)):
        try:
            if add_backgrounds or (os.stat(valid_file).st_size > 0):
                shutil.move(valid_file, valid_folder.joinpath('labels', valid_file.name))
                img_file = images_folder.joinpath(valid_file.name.replace('.txt', '.png'))
                shutil.move(img_file, valid_folder.joinpath('images', img_file.name))
        except Exception as e:
            print(e)

    print('moving train...')
    train_folder = output.joinpath('train')
    for train_file in tqdm(all_files_series[~all_files_series.index.isin(valid_files.index)],
                           total=len(all_files_series) - len(valid_files)):
        try:
            if add_backgrounds or (os.stat(train_file).st_size > 0):

                shutil.move(train_file, train_folder.joinpath('labels', train_file.name))
                img_file = images_folder.joinpath(train_file.name.replace('.txt', '.png'))
                shutil.move(img_file, train_folder.joinpath('images', img_file.name))
        except Exception as e:
            print(e)


if __name__ == '__main__':
    output_folder = pathlib.Path(input('Where is the folder to split?:'))
    split_train_valid(output_folder, 0.2, True)
