from utils import alignment_utils as al_utils, clips_generation_utils as cg_utils, data_utils as d_utils, datasets_generation_utils as dg_utils
import os 
from pathlib import Path
import pandas as pd

sample_rate = 16000 # sample rate used for resampling. 16 kHz is used because wav2vec requires it


def generate_clips():
    # set project dir 
    project_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    # set current folder
    current_folder = os.getcwd()

    current_folder = os.getcwd()

    # set resources paths
    dict_download_links =os.path.join(project_dir, "download", "resources", "download_links.csv")
    dict_mapping_ids = os.path.join(project_dir, "download", "datasets", "link_ids.csv")

    # read the csv files
    df_mapping = pd.read_csv(dict_mapping_ids, sep=';')
    df = pd.read_csv(dict_download_links, sep=';')


    id_mapping = df_mapping.mm_id
    id_links = df.id
    link_df = df.link
    startMin_df = df.startMin
    startSec_df = df.startSec
    endMin_df = df.endMin
    endSec_df = df.endSec

    # get only the links that are in the id_mapping
    # we need to transform objects to strings because the comparison is not working

    tmp = []
    for x in id_links:
        tmp.append(str(x))

    id_links = tmp
    tmp = []
    for x in id_mapping:
        tmp.append(str(x))
    id_mapping = tmp

    id = []
    links = []
    startMin = []
    startSec = []
    endMin = []
    endSec = []

    for i in range(len(id_links)):
        if id_links[i] in id_mapping:
            # get index of id_links[i] in id_mapping
            index = id_mapping.index(id_links[i])
            id.append(id_links[i])
            links.append(link_df[i])
            startMin.append(startMin_df[i])
            startSec.append(startSec_df[i])
            endMin.append(endMin_df[i])
            endSec.append(endSec_df[i])

    # create folders for the clips
    if not os.path.exists('files/audio_clips'):
        os.makedirs('files/audio_clips')
    
    # add a folder for 21_1992
    if not os.path.exists('files/audio_clips/21_1992'):
        os.makedirs('files/audio_clips/21_1992')
    
    


   
    output_sentences_dataset_path_csv = 'datasets/trial_sentences_corrected.csv'
    output_sentences_dataset_path_json = 'datasets/trial_sentences_corrected.json'
    output_cleaned_dataset_path_json = 'datasets/trial_cleaned.json'
    output_cleaned_dataset_path_csv = 'datasets/trial_cleaned.csv'

    dataset_dial_sent_clip_csv = 'datasets/dataset_dial_sent_clip.csv'
    dataset_dial_sent_clip_json = 'datasets/dataset_dial_sent_clip.json'

    corrected_dial_sent_clip_csv = 'datasets/dataset_dial_sent_clip_corrected.csv'
    corrected_dial_sent_clip_json = 'datasets/dataset_dial_sent_clip_corrected.json'
   
    output_snippet_dataset_path_csv = 'datasets/trial_snippet.csv'
    output_snippet_dataset_path_json = 'datasets/trial_snippet.json'

    output_snippet_dataset_path_csv = 'datasets/trial_snippet.csv'
    output_snippet_dataset_path_json = 'datasets/trial_snippet.json'


    output_comp_dataset_path_csv = 'datasets/trial_comp.csv'
    output_comp_dataset_path_json = 'datasets/trial_comp.json'


    
    dataset_dial_sent_snippet_clip_csv = 'datasets/dataset_dial_sent_snippet_comp_clip.csv'
    dataset_dial_sent_snippet_clip_json = 'datasets/dataset_dial_sent_snippet_comp_clip.json'

    dataset_dial_sent_snippet_dialogues_clip_csv = 'datasets/dataset_dial_sent_snippet_comp_dialogues_clip.csv'
    dataset_dial_sent_snippet_dialogues_clip_json = 'datasets/dataset_dial_sent_snippet_comp_dialogues_clip.json'

    cleaned_dataset_dial_sent_snippet_dialogues_clip_csv = 'datasets/dataset_dial_sent_snippet_comp_dialogues_clip_clean.csv'
    cleaned_dataset_dial_sent_snippet_dialogues_clip_json = 'datasets/dataset_dial_sent_snippet_comp_dialogues_clip_clean.json'
   
    cg_utils.generate_clips_dialogue_sentences(ids, modality, output_cleaned_dataset_path_csv, sample_rate) 
  
    cg_utils.generate_clips_snippets(ids, modality, output_snippet_dataset_path_csv, sample_rate) 



    cg_utils.generate_clips_comps(ids, modality, output_comp_dataset_path_csv, sample_rate)
  

   
    cg_utils.generate_clips_dialogues(ids, modality, dataset_dial_sent_snippet_clip_csv, sample_rate)
    


    # create a copy of the dataset with only the clips for which id_map is in ids
    if modality == 'partial':
        final_dataset = pd.read_csv("datasets/MM-DatasetFallacies/dataset.csv", sep = '\t')
        final_dataset = final_dataset[final_dataset['id_map'].isin(ids)]
        final_dataset = final_dataset.rename(columns={'id_map': 'Dialogue ID'})
        final_dataset.to_csv("datasets/MM-DatasetFallacies/dataset_partial.csv", sep = '\t', index=False)
    else:
        final_dataset = pd.read_csv("datasets/MM-DatasetFallacies/dataset.csv", sep = '\t')
        final_dataset = final_dataset.rename(columns={'id_map': 'Dialogue ID'})
        final_dataset.to_csv("datasets/MM-DatasetFallacies/dataset_full.csv", sep = '\t', index=False)
"""
    # copy final dataset into MM-DatasetFallacies folder and if the folder does not exist, create it
    d_utils.copy_dataset_to_MM_DatasetFallacies_folder('datasets/dataset_dial_sent_snippet_comp_clip.csv', 'datasets/MM-DatasetFallacies') """

            