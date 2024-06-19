# Machine learning for efficient segregation and labeling of potential biological sounds in long-term underwater recordings

This script collection is intended for re-use of the method presented in: 

Parcerisas, C.; Schall, E.; Te Velde, K.; Botteldooren, D.; Devos, P.; Debusschere, E.. 
Machine learning for efficient segregation and labeling of potential biological sounds in long-term underwater 
recordings. Submitted to Frontiers in Remote Sensing (2024). https://doi.org/10.3389/frsen.2024.1390687

All the data and the configuration files necessary to run the outputs can be found on IMIS https://www.vliz.be/nl/imis?module=dataset&dasid=8543 

The scripts are ready to be used in other locations. All the scripts are organized around "datasets", which basically 
means a folder with these subfolders: 


    dataset folder
    ├── images                      # segmented spectrograms for yolo prediction (optional)
    ├── labels                      # txt files yolo format (optional) obtained from manual annotations
    ├── predictions                 # txt files yolo format (optional) obtained from model predictions
    ├── wavs                        # raw wav files (optional)
    └── annotations.txt             # Raven selection table (optional)

As you see, all the components are optional, and this is because depends on what you want to run some parts will be 
necessary or not. 

For each of these folders (dataset), a config.json needs to be described. An example is provided in the file 
config_example.json. The parameters are: 
```json
{
    "duration" : 20, 
    "overlap" : 0.5,
    "desired_fs" : 24000,
    "channel" : 0,
    "color": true, 
    "log": false,
    
    "nfft" : 2048,
    "win_len" : 2048,
    "hop_ratio" : 8,
    
    "normalization_style": "noisy",
    
    "wavs_folder" : "path_to_the_wavs",
    "dataset_folder" : "path_to_dataset_folder",
    
    "annotations_file": "path_to_annotations.txt"
}
```

* duration: duration of each segment in seconds 
* overlap: 0 to 1, overlap between segments 
* desired_fs: sampling frequency to resample to, in Hz 
* channel: channel number 
* color: bool, true converts spectrograms to RBG values, otherwise grayscale 
* log: bool, true makes spectrograms with a logarithmic frequency scale 
* nfft: number of FFT bins to apply to the segment to generate spectrogram
* win_len: length of the window to apply the FFT to 
* hop_ratio: 0 to 100, percentage of hop between windows (1-overlap)
* normalization_style: "noisy" or "low_freq"
* wavs_folder: path to a folder where all the wavs to be analyzed are (they can be in subfolders!)
* dataset_folder: path to the main folder of the dataset, with the images and the labels folder (if applicable)
* annotations_file: path to the annotations file, can be from predictions or manual annotations. 
It does not necessarily need to be in the dataset folder


## Analyze a location 
To just analyze a new location using the existing model: 
1. Download the model weights (choose which one)
2. Run analyze_one_location.py and specify the path to the model and the path to the config file. 
The wavs folder does not necessarily need to be under the dataset folder.
The annotations file should be dataset_folder/roi_annotations_clean.txt. 
A folder names predictions/labels under the dataset folder will be created, but the images will not be saved to save
space.


## Train models 
To train models, you will have to: 
1. Manually create an annotations file using Raven (and set the correct path in the config.json file)
2. Create a dataset folder, and create the subfolders: "train", "valid", "images", "labels"
3. Create the spectrograms using create_long_spectrograms.py. This will also 
4. Run split_train_test_valid.py. This will split your images and your labels from images and labels folder to the train
and valid folders.
5. If for some reason you want to re-join all the images and labels, run reshuffle_files.py
6. Fill in the correct model_example.yaml with the paths to the valid and train paths (folders just created)
7. Run the retraining.py script and pass the model_example.yaml file path when prompt with the question


## Active Learning 
1. Create an unlabeled pool with an associated config.json file. This means create a dataset_folder with the necessary
subfolders, and one subfolder called active_learning
2. Run the create_long_spectrograms.py script for the unlabeled pool config (this will speed up things)
3. Run the active_learning.py script, it will guide you through the steps with an interactive console


## Other (maybe) useful scripts
* cluster.py -> Cluster manual (or predictions) from one location 
* predict.py -> Predict the events for one dataset


## Pre-trained models 
| Name               | Pretraining data            | Hours     | Link to pretrained weights   | 
| ------------------ | --------------------------- | --------- | ---------------------------- | 
| Model Final (MF)   | All annotated BPNS data     | 23.6h    | [Download](https://mda.vliz.be/directlink.php?fid=VLIZ_00000776_6662f3d66cec6504567484) |
| Model Base (MB)    | Initial training set        | 1.5h | [Download](https://mda.vliz.be/directlink.php?fid=VLIZ_00000776_6672fdd5df54b869504488)  |
| Active Learning (AL)    | Intial training set + 10 actively selected wavs  | 2.3h | [Download](https://mda.vliz.be/directlink.php?fid=VLIZ_00000776_6672fdd5df4ee563984337)  |
| Random Selection (RS)     | Intial training set + 10 randomly selected wavs  | 2.3h | [Download](	https://mda.vliz.be/directlink.php?fid=VLIZ_00000776_6672fdd5df553297088456)  |

If you're interested in weights of other of the trained models (2 more models per row available), please contact clea.parcerisas@vliz.be

## References
This work includes some scripts from to extract the CAE features. 
https://gitlab.lis-lab.fr/paul.best/repertoire_embedder
> Best P, Paris S, Glotin H, Marxer R (2023) Deep audio embeddings for vocalisation clustering. PLOS ONE 18(7): e0283396. https://doi.org/10.1371/journal.pone.0283396

This repository includes the filterbank.py script. Copyright (c) 2017 Jan Schlüter
https://github.com/f0k/ismir2015
> "Exploring Data Augmentation for Improved Singing Voice Detection with Neural Networks" by Jan Schlüter and Thomas Grill at the 16th International Society for Music Information Retrieval Conference (ISMIR 2015)
