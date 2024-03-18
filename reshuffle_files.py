import pathlib
import pandas as pd
import shutil
from tqdm import tqdm


def reshuffle_files(folder_name, class_type=''):
    test_img_files = pd.Series(list(folder_name.joinpath('test', 'images').glob('*.png')))
    valid_img_files = pd.Series(list(folder_name.joinpath('valid', 'images').glob('*.png')))
    train_img_files = pd.Series(list(folder_name.joinpath('train', 'images').glob('*.png')))

    test_labels_files = pd.Series(list(folder_name.joinpath('test', 'labels').glob('*.txt')))
    valid_labels_files = pd.Series(list(folder_name.joinpath('valid', 'labels').glob('*.txt')))
    train_labels_files = pd.Series(list(folder_name.joinpath('train', 'labels').glob('*.txt')))

    print(folder_name.joinpath('test', 'images').exists())
    print('moving test images...')
    for test_file in tqdm(test_img_files, total=len(test_img_files)):
        try:
            shutil.move(test_file, folder_name.joinpath('images', test_file.name))
        except Exception as e:
            print(e)

    print('moving test labels...')
    for test_file in tqdm(test_labels_files, total=len(test_labels_files)):
        try:
            shutil.move(test_file, folder_name.joinpath('labels%s' % class_type, test_file.name))
        except Exception as e:
            print(e)

    print('moving valid images...')
    for valid_file in tqdm(valid_img_files, total=len(valid_img_files)):
        try:
            shutil.move(valid_file, folder_name.joinpath('images', valid_file.name))

        except Exception as e:
            print(e)

    print('moving valid labels...')
    for valid_file in tqdm(valid_labels_files, total=len(valid_labels_files)):
        try:
            shutil.move(valid_file, folder_name.joinpath('labels%s' % class_type, valid_file.name))

        except Exception as e:
            print(e)

    print('moving train images...')
    for train_file in tqdm(train_img_files, total=len(train_img_files)):
        try:
            shutil.move(train_file, folder_name.joinpath('images', train_file.name))
        except Exception as e:
            print(e)

    print('moving train labels...')
    for train_file in tqdm(train_labels_files, total=len(train_labels_files)):
        try:
            shutil.move(train_file, folder_name.joinpath('labels%s' % class_type, train_file.name))
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main_folder = pathlib.Path(input('Where is the folder to reshuffle?'))
    class_name = input('class name?')

    if class_name != '':
        class_name = '_' + class_name

    confirm = input('press any key to start, reshuffling to: %s' % class_name)
    reshuffle_files(main_folder, class_name)
