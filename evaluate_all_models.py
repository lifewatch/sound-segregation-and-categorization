import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from ultralytics import YOLO

import dataset


def compute_overlaps(df, df_true, ds):
    print('computing overlap...')
    # Filter the ground truth
    df_true = df_true.loc[df_true['SNR NIST Quick (dB)'] > ds.MIN_SNR]
    df_true = df_true.loc[(df_true['End Time (s)'] - df_true['Begin Time (s)']) <= ds.MAX_DURATION]
    df_true = df_true.loc[(df_true['End Time (s)'] - df_true['Begin Time (s)']) >= ds.MIN_DURATION]

    df['max_iou'] = 0
    df['w'] = df['End File Samp (samples)'] - df['Beg File Samp (samples)']
    df['h'] = df['High Freq (Hz)'] - df['Low Freq (Hz)']

    df_true['detected'] = 0
    df_true['iou'] = 0
    df_true['w'] = df_true['End File Samp (samples)'] - df_true['Beg File Samp (samples)']
    df_true['h'] = df_true['High Freq (Hz)'] - df_true['Low Freq (Hz)']

    for wav_name, wav_detections in tqdm(df.groupby('Begin File')):
        or_annotations_wav = df_true.loc[df_true['Begin File'] == wav_name]

        for det_i, det in wav_detections.iterrows():

            # Get the closest groundtruth
            candidates = or_annotations_wav.loc[(or_annotations_wav['Beg File Samp (samples)'] >
                                                 (det['Beg File Samp (samples)'] - ds.desired_fs * 5)) & (
                                                        or_annotations_wav['End File Samp (samples)'] < (
                                                        det['End File Samp (samples)'] + ds.desired_fs * 5))]
            if len(candidates) > 0:
                inter = (np.minimum(det['End File Samp (samples)'], candidates['End File Samp (samples)']) -
                         np.maximum(det['Beg File Samp (samples)'], candidates['Beg File Samp (samples)'])).clip(0) * \
                        (np.minimum(det['High Freq (Hz)'], candidates['High Freq (Hz)']) -
                         np.maximum(det['Low Freq (Hz)'], candidates['Low Freq (Hz)'])).clip(0)

                # Union Area
                union = det['w'] * det['h'] + candidates['w'] * candidates['h'] - inter

                # IoU
                iou = inter / union
                df.loc[det_i, 'max_iou'] = iou.max()
                df_true.loc[candidates.index, 'iou'] = np.maximum(iou, candidates.iou)

    return df, df_true


def convert_to_grid(df, ds, max_duration):
    print('converting to grid...')
    freq_array = np.arange(0, ds.desired_fs / 2, step=ds.desired_fs / 2 / 640)
    duration_array = np.arange(0, max_duration, step=20 / 640)
    detected_grid = np.zeros((len(freq_array), len(duration_array)), dtype='uint8')
    df['start_grid'] = (df['Begin Time (s)'] / max_duration * len(duration_array)).astype(int)
    df['end_grid'] = np.ceil(df['End Time (s)'] / max_duration * len(duration_array)).astype(int)
    df['low_grid'] = (df['Low Freq (Hz)'] / (ds.desired_fs / 2) * len(freq_array)).astype(int)
    df['high_grid'] = np.ceil(df['High Freq (Hz)'] / (ds.desired_fs / 2) * len(freq_array)).astype(int)

    for det_i, det in tqdm(df.iterrows(), total=len(df)):
        # Add it to the mask
        detected_grid[det['low_grid']:det['high_grid'], det['start_grid']:det['end_grid']] += 1

    detected_grid[detected_grid > 1] = 1
    return detected_grid


def evaluate_models(ds):
    models_folder = pathlib.Path(input('Where is the folder with all the models?'))

    overwrite = False
    results_path = models_folder.joinpath('results.csv')

    if overwrite or not results_path.exists():
        results = pd.DataFrame(columns=['number_of_files', 'strategy', 'metric', 'value', 'conf', 'iou'])
        for m_folder in models_folder.glob('*'):
            if m_folder.is_dir():
                approach = m_folder.name
                for model_path in m_folder.glob('*.pt'):
                    model_name = model_path.name
                    strategy, n_files = model_name.split('_')
                    n_files = int(n_files.split('.')[0])
                    print('evaluating model %s...' % model_path)
                    model = YOLO(model_path)
                    predictions_folder = m_folder.joinpath('predictions_%s' % n_files)
                    if not predictions_folder.exists():
                        if ds.images_folder.exists():
                            results_list = model(source=str(ds.images_folder), project=str(m_folder),
                                            name=predictions_folder.name,
                                            stream=True,
                                            save=False, show=False, save_conf=True, save_txt=True, conf=0.1,
                                            save_crop=False, agnostic_nms=True)

                            for r in results_list:
                                pass
                        else:
                            os.mkdir(predictions_folder)
                            os.mkdir(predictions_folder.joinpath('labels'))
                            ds.create_spectrograms(overwrite=True, model=model, save_image=False,
                                                   labels_path=predictions_folder.joinpath('labels'))

                    if predictions_folder.joinpath('roi_detections_clean.txt').exists():
                        cleaned_detections = pd.read_table(predictions_folder.joinpath('roi_detections_clean.txt'))
                    else:
                        cleaned_detections, _ = ds.convert_detections_to_raven(
                            predictions_folder=predictions_folder, min_conf=0.1)

                    if 'max_iou' not in cleaned_detections.columns:
                        for _, original_annotations in ds.load_relevant_selection_table():
                            cleaned_detections, original_annotations = compute_overlaps(cleaned_detections,
                                                                                        original_annotations, ds)
                            original_annotations.to_csv(
                                predictions_folder.joinpath('annotations_with_iou.txt'),
                                sep='\t', index=False)
                            cleaned_detections.to_csv(predictions_folder.joinpath('roi_detections_clean.txt'),
                                                      sep='\t', index=False)
                    else:
                        original_annotations = pd.read_table(predictions_folder.joinpath(
                            'annotations_with_iou.txt'))

                    max_duration = max(original_annotations['Begin Time (s)'].max(),
                                       cleaned_detections['Begin Time (s)'].max())

                    grid_annot = convert_to_grid(df=original_annotations, ds=ds, max_duration=max_duration)
                    for conf in np.arange(0.1, 1, step=0.1):
                        cleaned_detections = cleaned_detections.loc[cleaned_detections.confidence >= conf]

                        grid_det = convert_to_grid(df=cleaned_detections, ds=ds, max_duration=max_duration)

                        grid_overlap = grid_det & grid_annot
                        detected = grid_overlap.sum() / grid_annot.sum()
                        fp_grid = grid_det.sum() - grid_overlap.sum()
                        fp_area = fp_grid / grid_det.sum()
                        grid_time_annot = grid_annot.sum(axis=0) >= 1
                        grid_time_det = grid_det.sum(axis=0) >= 1
                        time_overlap = grid_time_annot & grid_time_det
                        time_detected = time_overlap.sum() / grid_time_annot.sum()
                        fp_grid_time = grid_time_det.sum() - time_overlap.sum()
                        fp_time = fp_grid_time / grid_time_det.sum()

                        grid_tn = ~(grid_det | grid_annot)
                        tnr_area = grid_tn.sum() / (grid_tn.sum() + fp_grid)
                        grid_tn_time = ~(grid_time_det | grid_time_annot)
                        tnr_time = grid_tn_time.sum() / (grid_tn_time.sum() + fp_grid_time)

                        fpr_area = fp_grid / (fp_grid + grid_tn.sum())
                        fpr_time = fp_grid_time / (fp_grid_time + grid_tn_time.sum())

                        results.loc[len(results), ['number_of_files', 'strategy', 'metric', 'value', 'conf',
                                                   'iou']] = n_files, approach, 'det_area', detected, conf, 0
                        results.loc[len(results), ['number_of_files', 'strategy', 'metric', 'value', 'conf',
                                                   'iou']] = n_files, approach, 'fp_area', fp_area, conf, 0
                        results.loc[len(results), ['number_of_files', 'strategy', 'metric', 'value', 'conf',
                                                   'iou']] = n_files, approach, 'tnr_area', tnr_area, conf, 0
                        results.loc[len(results), ['number_of_files', 'strategy', 'metric', 'value', 'conf',
                                                   'iou']] = n_files, approach, 'det_time', time_detected, conf, 0
                        results.loc[len(results), ['number_of_files', 'strategy', 'metric', 'value', 'conf',
                                                   'iou']] = n_files, approach, 'fp_time', fp_time, conf, 0
                        results.loc[len(results), ['number_of_files', 'strategy', 'metric', 'value', 'conf',
                                                   'iou']] = n_files, approach, 'tnr_time', tnr_time, conf, 0
                        results.loc[len(results), ['number_of_files', 'strategy', 'metric', 'value', 'conf',
                                                   'iou']] = n_files, approach, 'fpr_area', fpr_area, conf, 0
                        results.loc[len(results), ['number_of_files', 'strategy', 'metric', 'value', 'conf',
                                                   'iou']] = n_files, approach, 'fpr_time', fpr_time, conf, 0

                        for iou_th in [0.1, 0.2, 0.3, 0.4, 0.5]:
                            tp = (cleaned_detections['max_iou'] >= iou_th).sum()
                            fp = (cleaned_detections['max_iou'] < iou_th).sum()

                            fn = (original_annotations.iou < iou_th).sum()

                            recall = tp / (tp + fn)
                            precision = tp / (tp + fp)
                            f1 = 2 * precision * recall / (precision + recall)
                            results.loc[len(results), ['number_of_files', 'strategy', 'metric', 'value', 'conf',
                                                       'iou']] = n_files, approach, 'precision', precision, conf, iou_th
                            results.loc[len(results), ['number_of_files', 'strategy', 'metric', 'value', 'conf',
                                                       'iou']] = n_files, approach, 'recall', recall, conf, iou_th
                            results.loc[len(results), ['number_of_files', 'strategy', 'metric', 'value', 'conf',
                                                       'iou']] = n_files, approach, 'f1', f1, conf, iou_th

        results[['value', 'conf', 'iou']] = results[['value', 'conf', 'iou']].astype(float)
        results[['strategy', 'metric']] = results[['strategy', 'metric']].astype(str)
        results['number_of_files'] = results['number_of_files'].astype(int)
        results.to_csv(results_path)

    else:
        results = pd.read_csv(results_path, index_col=0)

    model_base_al = results.loc[results.strategy == 'model_base'].copy()
    model_base_al['strategy'] = 'active_learning'

    model_base_rs = results.loc[results.strategy == 'model_base'].copy()
    model_base_rs['strategy'] = 'random_selection'

    results = pd.concat([results, model_base_al, model_base_rs])

    results = results.loc[results.strategy != 'model_base']
    results['number_of_files'] = results['number_of_files'] * 2
    # results = results.loc[results.strategy.isin(['random_selection', 'active_learning'])]

    for s in ['active_learning', 'random_selection']:
        for m in ['precision', 'recall', 'f1', 'det_area', 'fp_area', 'tnr_area', 'det_time', 'fp_time', 'tnr_time']:
            fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
            sns.lineplot(
                    results.loc[(results.strategy == s) & (results.metric == m) & (results.conf == 0.1)],
                    x='number_of_files', y='value', hue='iou', estimator=None, ax=ax[0][0])
            ax[0][0].set_title('0.1')
            sns.lineplot(
                    results.loc[(results.strategy == s) & (results.metric == m) & (results.conf == 0.2)],
                    x='number_of_files', y='value', hue='iou', estimator=None, ax=ax[0][1])
            ax[0][1].set_title('0.2')
            sns.lineplot(
                    results.loc[(results.strategy == s) & (results.metric == m) & (results.conf == 0.3)],
                    x='number_of_files', y='value', hue='iou', estimator=None, ax=ax[1][0])
            ax[1][0].set_title('0.3')
            sns.lineplot(
                    results.loc[(results.strategy == s) & (results.metric == m) & (results.conf == 0.4)],
                    x='number_of_files', y='value', hue='iou', estimator=None, ax=ax[1][1])
            ax[1][1].set_title('0.4')

            fig.supylabel('%s value' % m)
            fig.supxlabel('Number of files')
            plt.savefig(results_path.parent.joinpath('%s_iou_comparison_%s.png' % (m, s)))
            plt.close()

    results = results.loc[results.conf == 0.2]
    ax = sns.lineplot(results.loc[results.metric == 'fpr_time'], x='number_of_files', y='value', hue='strategy', estimator=None)
    ax.set_xlabel('Number of files')
    ax.set_ylabel('FPR')
    plt.savefig(results_path.parent.joinpath('fpr_comparison.png'))
    plt.close()

    ax = sns.lineplot(results.loc[results.metric == 'tnr_time'], x='number_of_files', y='value', hue='strategy', estimator=None)
    ax.set_xlabel('Number of files')
    ax.set_ylabel('TNR')
    plt.savefig(results_path.parent.joinpath('tnr_comparison.png'))
    plt.close()

    results = results.loc[results.iou == 0.1]
    ax = sns.lineplot(results.loc[results.metric == 'precision'], x='number_of_files', y='value', hue='strategy', estimator=None)
    ax.set_xlabel('Number of files')
    ax.set_ylabel('Precision')
    plt.savefig(results_path.parent.joinpath('precision_comparison.png'))
    plt.close()

    ax = sns.lineplot(results.loc[results.metric == 'recall'], x='number_of_files', y='value', hue='strategy', estimator=None)
    ax.set_xlabel('Number of files')
    ax.set_ylabel('Recall')
    plt.savefig(results_path.parent.joinpath('recall_comparison.png'))
    plt.close()

    ax = sns.lineplot(results.loc[results.metric == 'f1'], x='number_of_files', y='value', hue='strategy', estimator=None)
    ax.set_xlabel('Number of files')
    ax.set_ylabel('F1')
    plt.savefig(results_path.parent.joinpath('f1_comparison.png'))
    plt.close()


if __name__ == '__main__':
    config_path = input('Where is the config json file of the TEST dataset?: ')
    f = open(config_path)
    config = json.load(f)

    test_ds = dataset.LifeWatchDataset(config)
    test_ds.convert_raven_annotations_to_yolo()
    evaluate_models(test_ds)
