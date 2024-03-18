import comet_ml

from ultralytics import YOLO
from ultralytics.data import utils
import yaml
import torch

import new_model

# YAML_FILE = '/mnt/fscompute_shared/roi/configs/custom_rtsys_rgb.yaml'
YAML_FILE = '/mnt/fscompute_shared/roi/configs/custom_rtsys_all.yaml'
# BEST_PARAMS = '/mnt/fscompute_shared/roi/runs/tune/best_config.yaml'


def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 0
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False
    print(f"{num_freeze} layers are frozen.")


def run():
    # Check if CUDA is available
    print('Is CUDA available?')
    print(torch.cuda.is_available())

    # Read the config file
    with open(YAML_FILE, 'r') as file:
        config = yaml.safe_load(file)

    # Start a comet ML experiment to log it online
    experiment = comet_ml.Experiment(
        api_key="DqVhaH0SLdHYx9z2ythE2gOcB",
        project_name="roi-bpns",
    )

    # Load a model
    # model = YOLO('/mnt/fscompute_shared/roi/models/model_base/model_0.pt')
    model = new_model.YOLO('yolov8n.pt')
    # model = YOLO('yolov8s.yaml')

    # Freeze the layers
    # model.add_callback("on_train_start", freeze_layer)

    # train the model
    # Before training and deciding which labels to run, MOVE THE FILES!
    # best_params['data'] = YAML_FILE
    best_params = {
        'mixup': 0.0,
        'copy_paste': 0.0,
        'iou': 0.3,
        'imgsz': 640,
        'mosaic': 0.0,
        'degrees': 0.0,
        'shear': 0.0,
        'perspective': 0.0,
        'scale': 0.0
    }
    model.train(epochs=200, batch=32, data=YAML_FILE,
                project=config['path'] + '/runs/detect/bpns', resume=False, **best_params)

    # single_cls
    # evaluate model performance on the validation and the tests
    # model.val(split='val', project=config['path'] + '/runs/detect/')
    # model.val(split='test', project=config['path'] + '/runs/detect/')
    experiment.end()


if __name__ == '__main__':
    run()
