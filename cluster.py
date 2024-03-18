import json
import pathlib
import pickle

import dbcv
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torchaudio
import umap
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
import soundfile as sf

import dataset

# from renumics import spotlight


torchaudio.set_audio_backend(backend='soundfile')


def get_all_metrics(x, clusters, labels):
    dbcv_score = dbcv.dbcv(x, clusters, n_processes=1, noise_id=-1)
    silhouette = metrics.silhouette_score(x, clusters)

    # Metrics with only labels
    check_df = pd.DataFrame({'label': labels, 'cluster': clusters})
    check_df = check_df.loc[~labels.isna()]

    nmi = metrics.normalized_mutual_info_score(check_df.label, check_df.cluster)
    homogeneity = metrics.homogeneity_score(check_df.label, check_df.cluster)
    completeness = metrics.completeness_score(check_df.label, check_df.cluster)
    v_measure = metrics.v_measure_score(check_df.label, check_df.cluster)

    return [silhouette,
               homogeneity,
               dbcv_score,
               nmi,
               v_measure,
               completeness]


def cluster_and_add_metrics_to_results(df, x, labels, features_name, strategy, metric, supervision,
                                       n_neighbours, min_dist):
    epsilon_hdbscan = 0.2
    min_cluster_size = 8
    hdbscan_model = hdbscan.HDBSCAN(cluster_selection_epsilon=epsilon_hdbscan,
                                    min_cluster_size=min_cluster_size)  # , prediction_data=True

    clusterer = hdbscan_model.fit(x)
    clusters = clusterer.labels_
    metrics_list = get_all_metrics(x, labels=labels, clusters=clusters)

    metadata_list = [features_name,
                     strategy,
                     metric,
                     supervision,
                     n_neighbours,
                     min_dist,
                     x.shape[1],
                     epsilon_hdbscan,
                     min_cluster_size]
    df.loc[len(df)] = metadata_list + metrics_list

    return df


def create_and_split_features(ds, features_name, labels_to_exclude=None, input_type=None, model_name='',
                              scaler_feat=None, scaler_box=None):
    # labels_to_exclude = None
    # text = ds.zero_shot_learning()
    if features_name == 'clap':
        features = ds.encode_clap(labels_to_exclude=labels_to_exclude)
    elif features_name == 'aves':
        features = ds.encode_aves(labels_to_exclude=labels_to_exclude, strategy=input_type)
    elif features_name == 'cae':
        features = ds.encode_ae(model_path=ds.dataset_folder.joinpath(model_name), nfft=2048,
                                sample_dur=3, n_mel=128, bottleneck=256,
                                labels_to_exclude=labels_to_exclude, input_type=input_type)
    else:
        raise Exception('%s is not implemented as a feature space' % features_name)

    labels = features['label']
    features = features.drop(columns=['label', 'snr'])
    extra_features = features[['max_freq', 'min_freq', 'bandwidth', 'duration']]
    features = features.drop(columns=['max_freq', 'min_freq', 'bandwidth', 'duration'])

    if scaler_feat is None:
        scaler_feat = RobustScaler()
        features = scaler_feat.fit_transform(features)
    else:
        features = scaler_feat.transform(features)
    if scaler_box is None:
        scaler_box = RobustScaler()
        extra_features = scaler_box.fit_transform(extra_features)
    else:
        extra_features = scaler_box.transform(extra_features)

    return features, extra_features, labels


def dimension_reduction_umap(n, umap_path, features):
    if not umap_path.exists():
        umap_transformer = umap.UMAP(n_components=n, n_neighbors=10, min_dist=0.0, metric='cosine')
        umap_transformer.fit(features)
        with open(umap_path, 'wb') as handle:
            pickle.dump(umap_transformer, handle)
    else:
        with open(umap_path, 'rb') as handle:
            umap_transformer = pickle.load(handle)
    umap_reduction = umap_transformer.transform(features)

    return umap_transformer, umap_reduction


def cluster(clusterer_path, x):
    if not clusterer_path.exists():
        hdbscan_model = hdbscan.HDBSCAN(cluster_selection_epsilon=0.5, min_cluster_size=10,
                                        prediction_data=True)  # , prediction_data=True

        clusterer_reduction = hdbscan_model.fit(x)

        with open(clusterer_path, 'wb') as handle:
            pickle.dump(clusterer_reduction, handle)
    else:
        with open(clusterer_path, 'rb') as handle:
            clusterer_reduction = pickle.load(handle)

    clusters_reduction = clusterer_reduction.labels_
    return clusterer_path, clusters_reduction


def plot_clusters(embedding, clusters, save_path):
    cmap = plt.get_cmap('tab20', len(np.unique(clusters)))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=clusters.astype('str'), s=1, cmap=cmap)
    plt.savefig(save_path)
    plt.show()


def generate_clusters(ds):
    RANDOM_SEED = 20210105
    labels_to_exclude = ['boat_sound', 'boat_noise', 'water_movement', 'boat_operations',
                         'electronic_noise', 'interference', 'voice', 'out_of_water', 'deployment']
    features = ds.encode_clap(labels_to_exclude=labels_to_exclude, max_duration=3)
    original_features = features.copy()

    # Cluster the features
    features = features.drop(columns=['label'])
    features = features.loc[features.duration > 0.3]

    features['max_freq'] = features['max_freq'] / 12000
    features['min_freq'] = features['min_freq'] / 12000
    features['bandwidth'] = features['bandwidth'] / 12000
    features['duration'] = features['duration'] / 10

    features = features.drop(columns=['max_freq', 'min_freq', 'bandwidth', 'duration'])

    # Dimension reduction
    umap_box = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=RANDOM_SEED)
    umap_box.fit(features)
    embedding = umap_box.transform(features)

    # Plot the embedding
    ax = sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1],
                         s=1, alpha=0.9,
                         legend=False)
    plt.xlabel('UMAP x')
    plt.ylabel('UMAP y')
    plt.savefig('umap2d.png')
    plt.show()

    # Clustering
    hdbscan_model = hdbscan.HDBSCAN(cluster_selection_epsilon=0.2, min_cluster_size=5, min_samples=100)
    clusterer = hdbscan_model.fit(embedding)
    clusters = clusterer.labels_

    # Plot the clusters
    noise_mask = clusters == -1
    clusters_array = np.arange(len(np.unique(clusters)) - 1)

    ax = sns.scatterplot(x=embedding[noise_mask, 0], y=embedding[noise_mask, 1],
                         s=1, alpha=0.9,
                         legend=False, color='gray')
    g = sns.scatterplot(x=embedding[~noise_mask, 0], y=embedding[~noise_mask, 1], s=8,
                        hue=clusters[~noise_mask].astype(str), hue_order=clusters_array.astype(str),
                        legend=True, ax=ax)
    # Plot the cluster number
    for c in clusters_array:
        embeddings_c = embedding[clusters == c]
        x, y = embeddings_c.mean(axis=0)
        plt.text(x, y, str(c))
    plt.xlabel('UMAP x')
    plt.ylabel('UMAP y')
    g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
    plt.savefig('clusters.png')
    plt.show()

    original_features['clusters'] = clusters.max() + 1
    original_features.loc[original_features.duration > 0.3, 'clusters'] = clusters
    pd.DataFrame(original_features).to_pickle(ds.dataset_folder.joinpath('features_with_clusters.pkl'))

    total_selection_table = pd.DataFrame()

    for selection_path, detected_foregrounds in ds.load_relevant_selection_table(labels_to_exclude=None):
        total_selection_table = pd.concat([total_selection_table, detected_foregrounds])

    total_selection_table.loc[original_features.index, 'clusters'] = original_features.clusters

    return total_selection_table


if __name__ == '__main__':
    # Get the dataset config
    config_path = pathlib.Path(input('Where is the config path of the dataset?'))

    # Transform the detections in features (adding also freq limits and duration)
    f = open(config_path)
    config = json.load(f)
    ds_test = dataset.LifeWatchDataset(config)
    generate_clusters(ds_test)
