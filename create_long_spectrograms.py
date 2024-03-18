import json
import dataset


if __name__ == '__main__':
    config_path = input('Where is the config json file of the dataset?: ')
    f = open(config_path)
    config = json.load(f)

    ds = dataset.LifeWatchDataset(config)
    ds.create_spectrograms(overwrite=True)
    if ds.annotations_file != '':
        labels_to_exclude = ['boat_sound', 'boat_noise', 'water_movement', 'boat_operation',
                             'electronic_noise', 'interference', 'voice', 'out_of_water', 'deployment']
        ds.convert_raven_annotations_to_yolo(labels_to_exclude=labels_to_exclude)
