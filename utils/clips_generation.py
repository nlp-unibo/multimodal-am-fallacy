from utils import alignment_utils as al_utils, new_clips_generation_utils as cg_utils
import os 
from pathlib import Path
import pandas as pd

sample_rate = 16000 # sample rate used for resampling. 16 kHz is used because wav2vec requires it

# set project dir
project_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

# set audio clips folder path
AUDIO_CLIPS_PATH = os.path.join(project_dir, "local_database", "MM-DatasetFallacies", "audio_clips")
def generate_clips():

    modality = "full" # "full" or "partial"
    n_files = "all" # "all" or "10"


    # set project dir 
    project_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    # set current folder
    current_folder = os.getcwd()

    current_folder = os.getcwd()

    # set resources paths
    dict_download_links =os.path.join(project_dir, "resources", "download",   "download_links.csv")
    dict_mapping_ids = os.path.join(project_dir, "resources", "download",  "link_ids.csv")

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

    if modality == "full":
        ids = id

    elif modality == "partial":
        # if modality is partial, the user must specify the number of files to be downloaded
        # the first n_files in "dictionary.csv" will be downloaded
        if n_files != "all":
            n_files = int(n_files)
            ids = id[:n_files]


    # create folders for the clips
    try: 
        os.makedirs(AUDIO_CLIPS_PATH)
    except FileExistsError:
        print("Audio clips folder already exists")
    

    

    # generate clips only if AUDIO_CLIPS_PATH is empty
    if len(os.listdir(AUDIO_CLIPS_PATH)) == 0:
        base_dir_support_datasets = os.path.join(project_dir,  "resources", "clips_generation")

        output_cleaned_dataset_path_csv = os.path.join(base_dir_support_datasets, 'trial_cleaned.csv')
        output_snippet_dataset_path_csv = os.path.join(base_dir_support_datasets, 'trial_snippet.csv')
        output_comp_dataset_path_csv = os.path.join(base_dir_support_datasets, 'trial_comp.csv')

        dataset_dial_sent_snippet_clip_csv = os.path.join(base_dir_support_datasets, 'dataset_dial_sent_snippet_comp_clip.csv')

    
        # generate clips for sentences
    
        cg_utils.generate_clips("dial_sent", ids, modality, output_cleaned_dataset_path_csv, sample_rate) 
    
        cg_utils.generate_clips("snippet", ids, modality, output_snippet_dataset_path_csv, sample_rate) 

        #cg_utils.generate_clips("comp", ids, modality, output_comp_dataset_path_csv, sample_rate) #TODO: add datasets for clips components generation (from generation/ folder)
    
        # cg_utils.generate_clips("dial", ids, modality, dataset_dial_sent_snippet_clip_csv, sample_rate) #TODO: fix dialogues generation (from generation/ folder)
        

            