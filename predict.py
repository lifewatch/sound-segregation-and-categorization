from ultralytics import YOLO
import pathlib
import json
import os
from tqdm import tqdm

import dataset


def predict(output_folder, spectrograms_folder):
    predictions_path = output_folder.joinpath('predictions')
    if not predictions_path.exists():
        # Load a model
        model_path = input('Where is the model? :')
        model = YOLO(model_path)  # pretrained YOLOv8n model
        results = model(source=spectrograms_folder, project=str(output_folder), name='predictions', stream=True,
                        save=False, show=False, save_conf=True, save_txt=True, conf=0.1,
                        save_crop=False, agnostic_nms=False, imgsz=640)
        for r in tqdm(results):
            pass
    else:
        print('This folder was already predicted, delete the files first if you want to re-predict them')
    return predictions_path


if __name__ == '__main__':
    config_path = input('Where is the dataset json config file? :')
    f = open(config_path)
    config = json.load(f)

    ds = dataset.LifeWatchDataset(config)

    images_folder = pathlib.Path(ds.images_folder)

    predictions_name = input('Predictions will be stored at the same location than the dataset, '
                             'under the predictions folder. How should we name the prediction? '
                             'Leave blank to directly store it there')
    if predictions_name == '':
        predictions_folder = ds.dataset_folder
    else:
        predictions_folder = ds.dataset_folder.joinpath(predictions_name)
    predictions_folder = pathlib.Path(predictions_folder)

    predictions_path = predict(predictions_folder, images_folder)
    ds.convert_detections_to_raven(predictions_path)


