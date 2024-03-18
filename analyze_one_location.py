import json
import os

from ultralytics import YOLO

import dataset
import cluster

if __name__ == '__main__':
    config_path = input('Where is the config json file of the dataset?: ')
    f = open(config_path)
    config = json.load(f)

    ds = dataset.LifeWatchDataset(config)
    predictions_folder = ds.dataset_folder.joinpath('predictions')
    labels_path = predictions_folder.joinpath('labels')
    if not predictions_folder.joinpath('labels').exists():
        model_path = input('Where is the model to predict?')
        model = YOLO(model_path)
        os.mkdir(predictions_folder)
        os.mkdir(labels_path)
        ds.create_spectrograms(overwrite=True, model=model, save_image=False,
                               labels_path=labels_path)

    ds.convert_detections_to_raven(predictions_folder=predictions_folder)
    total_selection_table = cluster.generate_clusters(ds)

    ds.plot_clusters_polar_day(total_selection_table, selected_clusters=[0, 1, 2, 4, 5, 6, 14, 17, 19])
    ds.plot_clusters_polar_day(total_selection_table, selected_clusters=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
