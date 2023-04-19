import os
import pandas as pd
import numpy as np

LOCAL_DATABASE_DIR = 'local_database'
def load(base_dir):
    df_path = os.path.join(base_dir, LOCAL_DATABASE_DIR, 'MM-DatasetFallacies', 'no_duplicates', 'dataset.csv') # change if you want to use full, partial or full-no-duplicates dataset
    df = pd.read_csv(df_path, sep='\t')
    return df

# TODO: update this function if we want to load other inputs: dialogue sentences and argument
def load_audio(df, base_dir):
    audio_path = os.path.join(base_dir, LOCAL_DATABASE_DIR, 'MM-DatasetFallacies', 'audio_clips')
    train_audio_files = []
    val_audio_files = []
    test_audio_files = []
    # Loop through the whole dataframe that extracts the audio features of
    # the first and second sentences of the pair
    for index, row in df.iterrows():
        split = row['Split']
        debate_id = row['Dialogue ID']
        clip_id = row['idClipSnippet']
        snippet_path = os.path.join(audio_path, 'snippet', debate_id, clip_id + '.wav')
        if split == 'Train':
            train_audio_files.append(snippet_path)
        elif split == 'Validation':
            val_audio_files.append(snippet_path)
        elif split == 'Test':
            test_audio_files.append(snippet_path)

    return train_audio_files, val_audio_files, test_audio_files


def get_sentences(split, df):
    if split == 'train':
        df = df[df['Split'] == 'Train']
    elif split == 'val':
        df = df[df['Split'] == 'Validation']
    elif split == 'test':
        df = df[df['Split'] == 'Test']
    return df['SentenceSnippet'].values, df['Fallacy'].values

def get_unique_labels(y):
    return np.unique(y)

def get_num_labels(y):
    return len(get_unique_labels(y))
