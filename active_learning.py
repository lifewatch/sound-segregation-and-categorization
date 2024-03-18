
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pathlib
import pandas as pd
import seaborn as sns
import numpy as np
import shutil
import json
from tqdm import tqdm
import random
import os
import soundfile as sf

import dataset

random.seed(42)


def compute_overlap_map_detections(x, y, detections):
    iou_grid = np.zeros((len(y), len(x)))
    for _, d in detections.iterrows():
        mask = (x < d.width) & (y > d.min_freq) & (y <= d.max_freq)
        iou_grid[mask] += 1

    return iou_grid


unlabeled_config_path = input('Where is the unlabeled pool json config?')
f = open(unlabeled_config_path)
unlabeled_config = json.load(f)
unlabeled_ds = dataset.LifeWatchDataset(unlabeled_config)

labels_to_exclude = ['boat_sound', 'boat_noise', 'water_movement', 'boat_operation',
                     'electronic_noise', 'interference', 'voice', 'out_of_water', 'deployment']

active_learning_step = int(input('what is the active learning step?: '))
already_annotated = input('Did you already annotate the files? y/n: ') == 'y'

# Load a model
model_path = input('Where is the model? :')
model = YOLO(model_path)  # pretrained YOLOv8n model

previous_training_set_config_path = input('Where is the configuration of the previous training set? :')
f = open(previous_training_set_config_path)
previous_training_set_config = json.load(f)
training_ds = dataset.LifeWatchDataset(previous_training_set_config)

configs_folder = pathlib.Path(previous_training_set_config_path).parent

active_learning_folder = unlabeled_ds.dataset_folder.joinpath('active_learning/%s' % active_learning_step)

active_learning_config = unlabeled_config.copy()
active_learning_config.update({'wavs_folder': str(active_learning_folder.joinpath('wav_resampled')),
                               'dataset_folder': str(active_learning_folder)})
ds = dataset.LifeWatchDataset(active_learning_config)

if not already_annotated:
    overwrite = False

    # Predictions need to be done in the ENTIRE UNLABELED FOLDER
    if overwrite or (not unlabeled_ds.dataset_folder.joinpath('predictions_%s' % active_learning_step).exists()):
        print('predicting...')
        results = model(source=unlabeled_ds.images_folder, project=str(unlabeled_ds.dataset_folder),
                        name='predictions_%s' % active_learning_step, stream=True, save=False,
                        show=False, save_conf=True, save_txt=True, conf=0.1, save_crop=False, agnostic_nms=True)
        for r in results:
            pass

    # Get the files already selected on last steps
    wavs_to_exclude = []
    if active_learning_step > 0:
        for old_step in np.arange(active_learning_step):
            old_selection_folder = unlabeled_ds.dataset_folder.joinpath(
                'active_learning/%s/wav_resampled' % old_step)
            wavs_to_exclude = np.concatenate([wavs_to_exclude, list(old_selection_folder.glob('*.wav'))])

    print('converting training annotations to df...')
    training_foregrounds = training_ds.convert_raven_annotations_to_df(labels_to_exclude=labels_to_exclude,
                                                                       values_to_replace=0)

    unlabeled_predictions_folder = unlabeled_ds.dataset_folder.joinpath('predictions_%s' % active_learning_step)
    if not unlabeled_predictions_folder.joinpath('labels_df.csv').exists():
        print('converting detections to df...')
        detected_foregrounds, _ = unlabeled_ds.convert_detections_to_raven(unlabeled_predictions_folder)
        detected_foregrounds.to_csv(unlabeled_predictions_folder.joinpath('labels_df.csv'), index=False)
    else:
        detected_foregrounds = pd.read_csv(unlabeled_predictions_folder.joinpath('labels_df.csv'))

    # First compute the overlap with training set per each detection
    if 'iou' not in detected_foregrounds.columns:
        print('Getting overlap with training, and adding iou to each detection...')
        detected_foregrounds = unlabeled_ds.compute_detection_overlap_with_dataset(detected_foregrounds,
                                                                                   training_foregrounds)
        detected_foregrounds.to_csv(unlabeled_predictions_folder.joinpath('labels_df.csv'), index=False)

    # Compute how many interesting ones
    threshold = np.percentile(detected_foregrounds['iou'], 10)
    detected_foregrounds['interesting'] = 0
    detected_foregrounds.loc[(detected_foregrounds.iou <= threshold) | (detected_foregrounds.confidence < 0.25),
                             'interesting'] = 1

    # compute diversity of detections
    freq_array = np.arange(0, 1, 0.01)
    max_duration = np.percentile(detected_foregrounds['width'].values, 98)
    duration_array = np.linspace(0, max_duration, 100)
    grid_duration, grid_freq = np.meshgrid(duration_array, freq_array)
    detected_foregrounds.loc[detected_foregrounds['iou'] > 1, 'iou'] = 1
    detected_foregrounds['score'] = (1 - detected_foregrounds['iou']) * (1 - detected_foregrounds['confidence'])

    selected_wavs = pd.DataFrame(columns=['n_interesting', 'min_score', 'entropy', 'duration'],
                                 index=detected_foregrounds.wav.unique())
    print('computing entropy per wav...')

    for wav_name, wav_detections in detected_foregrounds.groupby('wav'):
        if wav_name not in wavs_to_exclude:
            wav_iou = compute_overlap_map_detections(grid_duration, grid_freq, wav_detections)
            wav_iou = wav_iou / len(wav_detections)
            nonzero_freqs = wav_iou[wav_iou.nonzero()]
            sparsity = -(nonzero_freqs * np.log(nonzero_freqs)).sum() / np.log(np.e)
            wav_file = sf.SoundFile(unlabeled_ds.wavs_folder.joinpath(wav_name))
            duration = wav_file.frames / wav_file.samplerate
            selected_wavs.loc[wav_name] = [wav_detections['interesting'].sum(), wav_detections['score'].max(), sparsity, duration]
        else:
            print('There is a problem and the last files where copied, not moved...! ')

    selected_wavs['total_score'] = selected_wavs['n_interesting'] / selected_wavs['duration'] * selected_wavs['min_score'] * selected_wavs['entropy']
    selected_wavs = selected_wavs.sort_values('total_score', ascending=False)
    selected_wavs.to_csv(active_learning_folder.joinpath('wavs_scores.csv'))

    if random.random() < 0.3:
        if random.random() < 0.5:
            replace = 0
        else:
            replace = 1
        randomly_selected = selected_wavs[2:].sample(1)
        selected_wavs.iloc[replace] = randomly_selected

    selected_wavs = selected_wavs[:2]

    all_spectrograms = list(unlabeled_ds.images_folder.glob('*.png'))
    all_spectrograms = pd.Series(all_spectrograms).astype(str)

    print('copying the wavs and the spectrograms to the dataset folder...')
    for wav_path, _ in selected_wavs.iterrows():
        wav_name = pathlib.Path(wav_path).name
        shutil.move(wav_path, str(active_learning_folder.joinpath('wav_resampled', wav_name)))
        wav_name_without_suffix = wav_name.split('.')[0]

        wav_sxx = all_spectrograms.loc[all_spectrograms.str.contains(wav_name_without_suffix)]
        for sxx_path in wav_sxx:
            sxx_name = pathlib.Path(sxx_path).name
            shutil.move(sxx_path, active_learning_folder.joinpath('images', sxx_name))
            label_name = sxx_name.replace('.png', '.txt')
            label_path = sxx_path.replace('images', 'predictions_%s\labels' % active_learning_step)
            label_path = label_path.replace('.png', '.txt')
            if os.path.exists(label_path):
                shutil.move(label_path, active_learning_folder.joinpath('labels', label_name))
            else:
                pass

    clean_detections, clean_detections_path = ds.convert_detections_to_raven(active_learning_folder)
    print('Now load the wav files to Raven together with %s, and then manually annotate' % clean_detections_path)

else:
    new_annotations = input('Did you finish? What is the path of the new annotations file?:')

    # Save the new config file for the RS addition
    ds['annotations_file'] = str(new_annotations)

    # Convert the manually corrected annotations to labels
    ds.convert_raven_annotations_to_yolo()

    training_datasets_parent = training_ds.dataset_folder.parent
    # Add the data to the training folder to create a dataset with RS + training
    new_path_dataset = training_datasets_parent.joinpath('/training_set_small_rgb_AL_%s' % active_learning_step)
    joined_annotations_path = dataset.join_datasets(previous_training_set_config,
                                                    ds.config, new_path_dataset,
                                                    join_annotations=True)
    RS_config_path = configs_folder.joinpath('/bpns_AL_selection_%s.json' % active_learning_step)
    ds.save_config(RS_config_path)

    ds['wavs_folder'] = new_path_dataset + '/wav_resampled'
    ds['dataset_folder'] = str(new_path_dataset)
    ds['annotations_file'] = str(joined_annotations_path)
    RS_config_path = configs_folder.joinpath('/bpns_AL_total_%s.json' % active_learning_step)
    ds.save_config(RS_config_path)
