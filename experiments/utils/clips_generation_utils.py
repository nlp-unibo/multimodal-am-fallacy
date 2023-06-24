import pandas as pd
from tqdm import tqdm
import json 
from collections import defaultdict
from pydub import AudioSegment
import os
import shutil
# generate 1 dataset per id -> use the same function of MM-USElecDeb with modifications
# this new dataset contains 3 new columns for idclip-dialogue/snippet/component

#TODO: write only one function to generate clip for each kind of item (dialogue, snippet, component)

project_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

dataset_dir = os.path.join(project_dir, "local_database", "MM-DatasetFallacies")



def generate_clips_dialogue_sentences(ids, modality, dataset_path, sample_rate):

    df = pd.read_csv(dataset_path, sep='\t')
    if modality == 'full':
        ids = df.id_map.unique()

    print("IDS", ids)
    MAIN_FOLDER_PATH = 'files/audio_clips/dial_sent'
    if not os.path.exists(MAIN_FOLDER_PATH):
        os.mkdir(MAIN_FOLDER_PATH)
    else: 
        shutil.rmtree(MAIN_FOLDER_PATH)
        os.mkdir(MAIN_FOLDER_PATH)

    for i in tqdm(range(len(ids))):
        FOLDER_ID = ids[i]
        
        DATASET_PATH_CLIP_EMPTY = 'datasets/dial_sent/' + FOLDER_ID + '/dataset.csv'
        DATASET_CLIP_PATH = 'datasets/dial_sent/' + FOLDER_ID + '/dataset_clip.csv'
        FULL_AUDIO_PATH = 'files/debates_audio_recordings/' + FOLDER_ID + '/full_audio_trim.wav'
        
        AUDIO_CLIPS_PATH = 'files/audio_clips/dial_sent/' + FOLDER_ID
        if not os.path.exists(AUDIO_CLIPS_PATH):
            os.mkdir(AUDIO_CLIPS_PATH)
        else:
            shutil.rmtree(AUDIO_CLIPS_PATH)
            os.mkdir(AUDIO_CLIPS_PATH)

        # read dataframe with timestamps
        df = pd.read_csv(DATASET_PATH_CLIP_EMPTY, sep='\t')

        # unique dialogue rows
        unique_dialogue_rows = {}
        # generate clips
        sound = AudioSegment.from_file(FULL_AUDIO_PATH)
        sr = sound.frame_rate
        #print(sr)
        total_len = df.shape[0]
        for i, row in tqdm(df.iterrows(), total=total_len, position=0):
            timestamps_dial_begin = list(row['DialogueAlignmentBegin'][1:-1].strip().split(','))
            timestamps_dial_end = list(row['DialogueAlignmentEnd'][1:-1].strip().split(','))
            dialogue = row['Dialogue']

            idClipDialogues = []
            if dialogue not in unique_dialogue_rows.keys():
                for j in range(len(timestamps_dial_begin)):
                    if timestamps_dial_begin[j].strip() != 'NOT_FOUND' and timestamps_dial_end[j].strip() != 'NOT_FOUND':
                        start_time = float(timestamps_dial_begin[j].strip().replace('\'','')) * 1000 - 1005 # sottrazione per evitare che il clip inizi prima del tempo di inizio dial
                        #print(timestamps_dial_begin)
                        #print(start_time)
                        #print(timestamps_dial_end[j]) 
                        end_time = float(timestamps_dial_end[j].strip().replace('\'','')) * 1000 + 100 # aggiunta per evitare che il clip finisca dopo il tempo di fine dial
                        idClip = 'clip_' + str(i) + '_' + str(j)
                        clip_name = AUDIO_CLIPS_PATH + '/' + idClip + '.wav'
                        extract = sound[start_time:end_time]
                        extract = extract.set_frame_rate(sample_rate)
                        extract = extract.set_channels(1)
                        extract.export(clip_name, format="wav")
                        idClipDialogues.append(idClip)
                df.at[i, "idClipDialSent"] = idClipDialogues 
                unique_dialogue_rows[dialogue] = idClipDialogues
                        
            else: 
                df.at[i, "idClipDialSent"] = unique_dialogue_rows[dialogue]


        # save new csv only if not exists - added to deal with partial dataset clip generation
        if not os.path.exists(DATASET_CLIP_PATH):
            df.to_csv(DATASET_CLIP_PATH, sep = '\t')



def generate_clips_snippets(ids, modality, dataset_path, sample_rate):

    df = pd.read_csv(dataset_path, sep='\t')

    if modality == 'full':
        ids = df.id_map.unique()

    MAIN_FOLDER_PATH = 'files/audio_clips/snippet'
    if not os.path.exists(MAIN_FOLDER_PATH):
        os.mkdir(MAIN_FOLDER_PATH)
    else: 
        shutil.rmtree(MAIN_FOLDER_PATH)
        os.mkdir(MAIN_FOLDER_PATH)

    for i in tqdm(range(len(ids))):
        FOLDER_ID = ids[i]
        
        DATASET_PATH_CLIP_EMPTY = 'datasets/snippet/' + FOLDER_ID + '/dataset.csv'
        DATASET_CLIP_PATH = 'datasets/snippet/' + FOLDER_ID + '/dataset_clip.csv'
        FULL_AUDIO_PATH = 'files/debates_audio_recordings/' + FOLDER_ID + '/full_audio_trim.wav'
        
        AUDIO_CLIPS_PATH = 'files/audio_clips/snippet/' + FOLDER_ID
        if not os.path.exists(AUDIO_CLIPS_PATH):
            os.mkdir(AUDIO_CLIPS_PATH)
        else:
            shutil.rmtree(AUDIO_CLIPS_PATH)
            os.mkdir(AUDIO_CLIPS_PATH)


        # read dataframe with timestamps
        df = pd.read_csv(DATASET_PATH_CLIP_EMPTY, sep='\t')

        # unique snippet rows
        unique_snippet_rows = {}
        # generate clips
        sound = AudioSegment.from_file(FULL_AUDIO_PATH)
        sr = sound.frame_rate
        #print(sr)
        total_len = df.shape[0]
        for i, row in tqdm(df.iterrows(), total=total_len, position=0):
            start_time = row['BeginSnippet']
            end_time = row['EndSnippet']
            snippet = row['Snippet']
            if snippet not in unique_snippet_rows.keys():
                idClip = 'clip_' + str(i)
                if start_time != 'NOT_FOUND' and end_time != 'NOT_FOUND':
                    start_time = float(row['BeginSnippet'].strip().replace('\'','')) * 1000  - 1005 # sottrazione per evitare che il clip inizi prima del tempo di inizio snippet
                    end_time = float(row['EndSnippet'].strip().replace('\'','')) * 1000  + 100 # aggiunta per evitare che il clip finisca dopo il tempo di fine snippet 
                    clip_name = AUDIO_CLIPS_PATH + '/' + idClip + '.wav'
                    extract = sound[start_time:end_time]
                    extract = extract.set_frame_rate(sample_rate)
                    extract = extract.set_channels(1)
                    extract.export(clip_name, format="wav")
                    df.at[i, "idClipSnippet"] = idClip
                    unique_snippet_rows[snippet] = idClip
                    
            else: 
                df.at[i, "idClipSnippet"] = unique_snippet_rows[snippet]

        # save new csv only if not exists - added to deal with partial dataset clip generation
        if not os.path.exists(DATASET_CLIP_PATH):
            df.to_csv(DATASET_CLIP_PATH, sep='\t')





def generate_datasets_clip_comps(dataset_path):
    
    df = pd.read_csv(dataset_path, sep='\t')
    ids = df.id_map.unique()

    MAIN_FOLDER_PATH = 'datasets/comp/'
    if not os.path.exists(MAIN_FOLDER_PATH):
        os.mkdir(MAIN_FOLDER_PATH)
    else: 
        shutil.rmtree(MAIN_FOLDER_PATH)
        os.mkdir(MAIN_FOLDER_PATH)

    for i in tqdm(range(len(ids))):
        FOLDER_ID = ids[i]
        NEW_FILES_PATH = 'datasets/comp/' + FOLDER_ID + '/'
      
        if not os.path.exists(NEW_FILES_PATH):
            os.mkdir(NEW_FILES_PATH)
        else: 
            shutil.rmtree(NEW_FILES_PATH)
            os.mkdir(NEW_FILES_PATH)

        # count_rows of debate
        count_row_debate = 0
        for i, row in df.iterrows():
            if row['id_map'] == FOLDER_ID:
                count_row_debate += 1

        # generate new dataframe
        rows_new_df = []
        for i, row in df.iterrows():
            if row['id_map'] == FOLDER_ID:
                rows_new_df.append(row)
        
        new_df = pd.DataFrame(rows_new_df)
        new_col_id_clip_snippet = ['NOT_FOUND' for i in range(count_row_debate)]
        new_df['idClipCompText'] = new_col_id_clip_snippet

        # save new_df as csv
        new_df.to_csv(NEW_FILES_PATH + 'dataset.csv', sep = '\t')

def generate_clips_comps(ids, modality, dataset_path, sample_rate):

    df = pd.read_csv(dataset_path, sep='\t')
    if modality == 'full':
        ids = df.id_map.unique()

    MAIN_FOLDER_PATH = 'files/audio_clips/comp'
    if not os.path.exists(MAIN_FOLDER_PATH):
        os.mkdir(MAIN_FOLDER_PATH)
    else: 
        shutil.rmtree(MAIN_FOLDER_PATH)
        os.mkdir(MAIN_FOLDER_PATH)

    for i in tqdm(range(len(ids))):
        FOLDER_ID = ids[i]
        
        DATASET_PATH_CLIP_EMPTY = 'datasets/comp/' + FOLDER_ID + '/dataset.csv'
        DATASET_CLIP_PATH = 'datasets/comp/' + FOLDER_ID + '/dataset_clip.csv'
        FULL_AUDIO_PATH = 'files/debates_audio_recordings/' + FOLDER_ID + '/full_audio_trim.wav'
        
        AUDIO_CLIPS_PATH = 'files/audio_clips/comp/' + FOLDER_ID
        if not os.path.exists(AUDIO_CLIPS_PATH):
            os.mkdir(AUDIO_CLIPS_PATH)
        else:
            shutil.rmtree(AUDIO_CLIPS_PATH)
            os.mkdir(AUDIO_CLIPS_PATH)

        # read dataframe with timestamps
        df = pd.read_csv(DATASET_PATH_CLIP_EMPTY, sep='\t')

        # unique comp rows
        unique_comp_rows = {}
        # generate clips
        sound = AudioSegment.from_file(FULL_AUDIO_PATH)
        sr = sound.frame_rate
        #print(sr)
        total_len = df.shape[0]
        for i, row in tqdm(df.iterrows(), total=total_len, position=0):
            start_time = row['BeginCompText']
            end_time = row['EndCompText']
            comp = row['CompText']
            if comp not in unique_comp_rows.keys():
                idClip = 'clip_' + str(i)
                if start_time != 'NOT_FOUND' and end_time != 'NOT_FOUND':
                    start_time = float(row['BeginCompText'].strip().replace('\'','')) * 1000  - 1005 # sottrazione per evitare che il clip inizi prima del tempo di inizio CompText
                    end_time = float(row['EndCompText'].strip().replace('\'','')) * 1000  + 100 # aggiunta per evitare che il clip finisca dopo il tempo di fine CompText 
                    clip_name = AUDIO_CLIPS_PATH + '/' + idClip + '.wav'
                    extract = sound[start_time:end_time]
                    extract = extract.set_frame_rate(sample_rate)
                    extract = extract.set_channels(1)
                    extract.export(clip_name, format="wav")
                    df.at[i, "idClipCompText"] = idClip
                    unique_comp_rows[comp] = idClip
                    
            else: 
                df.at[i, "idClipCompText"] = unique_comp_rows[comp]

        # save new csv only if not exists - added to deal with partial dataset clip generation
        if not os.path.exists(DATASET_CLIP_PATH):
            df.to_csv(DATASET_CLIP_PATH, sep='\t')


def unify_datasets_comps_clips(dataset_path):
    df = pd.read_csv(dataset_path, sep='\t')
    ids = df.id_map.unique()

    # iterate over dial_sent/ids and generate a new dataset merging all dataset_clip.csv
    for i in tqdm(range(len(ids))):
        FOLDER_ID = ids[i]
        DATASET_DIALOGUE_SENT_PATH_CLIP = 'datasets/comp/' + FOLDER_ID + '/dataset_clip.csv'
        df_dial = pd.read_csv(DATASET_DIALOGUE_SENT_PATH_CLIP, sep='\t')
        #DATASET_OUTPUT = 'files/datasets/' + FOLDER_ID + '/dataset_dial_sent_clip.csv'
        break
    
    for i in tqdm(range(1, len(ids))):
        FOLDER_ID = ids[i]
        DATASET_DIALOGUE_SENT_PATH_CLIP = 'datasets/comp/' + FOLDER_ID + '/dataset_clip.csv'
        df_dial_1 = pd.read_csv(DATASET_DIALOGUE_SENT_PATH_CLIP, sep='\t')
        df_dial = pd.concat([df_dial, df_dial_1])

    # check if shape of concatenated dataframe is correct by comparing it to the original dataframe
    print("Actual shape: ", df_dial.shape, "Original shape: ", df.shape)

    df_dial = df_dial.loc[:, ~df_dial.columns.str.match('Unnamed')]
    # save
    FINAL_DATASET_PATH = 'datasets/dataset_dial_sent_snippet_comp_clip.csv'
    df_dial.to_csv(FINAL_DATASET_PATH, sep = '\t')

    # save a copy as json
    df_dial.to_json('datasets/dataset_dial_sent_snippet_comp_clip.json', orient='records', indent=4)



def generate_datasets_clip_dialogues(dataset_path):
    
    df = pd.read_csv(dataset_path, sep='\t')

    ids = df.id_map.unique()

    MAIN_FOLDER_PATH = 'datasets/dial/'
    if not os.path.exists(MAIN_FOLDER_PATH):
        os.mkdir(MAIN_FOLDER_PATH)
    else: 
        shutil.rmtree(MAIN_FOLDER_PATH)
        os.mkdir(MAIN_FOLDER_PATH)

    for i in tqdm(range(len(ids))):
        FOLDER_ID = ids[i]
        NEW_FILES_PATH = 'datasets/dial/' + FOLDER_ID + '/'
      
        if not os.path.exists(NEW_FILES_PATH):
            os.mkdir(NEW_FILES_PATH)
        else: 
            shutil.rmtree(NEW_FILES_PATH)
            os.mkdir(NEW_FILES_PATH)

        # count_rows of debate
        count_row_debate = 0
        for i, row in df.iterrows():
            if row['id_map'] == FOLDER_ID:
                count_row_debate += 1

        # generate new dataframe
        rows_new_df = []
        for i, row in df.iterrows():
            if row['id_map'] == FOLDER_ID:
                rows_new_df.append(row)
        
        new_df = pd.DataFrame(rows_new_df)
        new_col_id_clip_snippet = ['NOT_FOUND' for i in range(count_row_debate)]
        new_df['idClipDialogue'] = new_col_id_clip_snippet

        # save new_df as csv
        new_df.to_csv(NEW_FILES_PATH + 'dataset.csv', sep = '\t')

def generate_clips_dialogues(ids, modality, dataset_path, sample_rate):

    df = pd.read_csv(dataset_path, sep='\t')
    if modality == 'full':
        ids = df.id_map.unique()

    MAIN_FOLDER_PATH = 'files/audio_clips/dial'
    if not os.path.exists(MAIN_FOLDER_PATH):
        os.mkdir(MAIN_FOLDER_PATH)
    else: 
        shutil.rmtree(MAIN_FOLDER_PATH)
        os.mkdir(MAIN_FOLDER_PATH)

    for i in tqdm(range(len(ids))):
        FOLDER_ID = ids[i]
        
        DATASET_PATH_CLIP_EMPTY = 'datasets/dial/' + FOLDER_ID + '/dataset.csv'
        DATASET_CLIP_PATH = 'datasets/dial/' + FOLDER_ID + '/dataset_clip.csv'
        FULL_AUDIO_PATH = 'files/debates_audio_recordings/' + FOLDER_ID + '/full_audio_trim.wav'
        
        AUDIO_CLIPS_PATH = 'files/audio_clips/dial/' + FOLDER_ID
        if not os.path.exists(AUDIO_CLIPS_PATH):
            os.mkdir(AUDIO_CLIPS_PATH)
        else:
            shutil.rmtree(AUDIO_CLIPS_PATH)
            os.mkdir(AUDIO_CLIPS_PATH)

        # read dataframe with timestamps
        df = pd.read_csv(DATASET_PATH_CLIP_EMPTY, sep='\t')

        # unique comp rows
        unique_comp_rows = {}
        # generate clips
        sound = AudioSegment.from_file(FULL_AUDIO_PATH)
        sr = sound.frame_rate
        #print(sr)
        total_len = df.shape[0]
        for i, row in tqdm(df.iterrows(), total=total_len, position=0):
            start_time = row['DialogueBegin']
            end_time = row['DialogueEnd']
            if (type(start_time) == str): 
                start_time = start_time.strip().replace('\'','')
            if (type(end_time) == str):
                end_time = end_time.strip().replace('\'','')
            comp = row['Dialogue']
            if comp not in unique_comp_rows.keys():
                idClip = 'clip_' + str(i)
                if start_time != 'NOT_FOUND' and end_time != 'NOT_FOUND':
                    start_time = float(start_time) * 1000  - 1005 # sottrazione per evitare che il clip inizi prima del tempo di inizio CompText
                    end_time = float(end_time) * 1000  + 100 # aggiunta per evitare che il clip finisca dopo il tempo di fine CompText 
                    clip_name = AUDIO_CLIPS_PATH + '/' + idClip + '.wav'
                    extract = sound[start_time:end_time]
                    extract = extract.set_frame_rate(sample_rate)
                    extract = extract.set_channels(1)
                    extract.export(clip_name, format="wav")
                    df.at[i, "idClipDialogue"] = idClip
                    unique_comp_rows[comp] = idClip
                    
            else: 
                df.at[i, "idClipDialogue"] = unique_comp_rows[comp]


        # save new csv only if not exists - added to deal with partial dataset clip generation
        if not os.path.exists(DATASET_CLIP_PATH):
            df.to_csv(DATASET_CLIP_PATH, sep='\t')


def unify_datasets_dialogues_clips(dataset_path):
    df = pd.read_csv(dataset_path, sep='\t')
    ids = df.id_map.unique()

    # iterate over dial_sent/ids and generate a new dataset merging all dataset_clip.csv
    for i in tqdm(range(len(ids))):
        FOLDER_ID = ids[i]
        DATASET_DIALOGUE_SENT_PATH_CLIP = 'datasets/dial/' + FOLDER_ID + '/dataset_clip.csv'
        df_dial = pd.read_csv(DATASET_DIALOGUE_SENT_PATH_CLIP, sep='\t')
        #DATASET_OUTPUT = 'files/datasets/' + FOLDER_ID + '/dataset_dial_sent_clip.csv'
        break
    
    for i in tqdm(range(1, len(ids))):
        FOLDER_ID = ids[i]
        DATASET_DIALOGUE_SENT_PATH_CLIP = 'datasets/dial/' + FOLDER_ID + '/dataset_clip.csv'
        df_dial_1 = pd.read_csv(DATASET_DIALOGUE_SENT_PATH_CLIP, sep='\t')
        df_dial = pd.concat([df_dial, df_dial_1])

    # check if shape of concatenated dataframe is correct by comparing it to the original dataframe
    print("Actual shape: ", df_dial.shape, "Original shape: ", df.shape)

    df_dial = df_dial.loc[:, ~df_dial.columns.str.match('Unnamed')]
    # save
    FINAL_DATASET_PATH = 'datasets/dataset_dial_sent_snippet_comp_dialogues_clip.csv'
    df_dial.to_csv(FINAL_DATASET_PATH, sep = '\t')

    # save a copy as json
    df_dial.to_json('datasets/dataset_dial_sent_snippet_comp_dialogues_clip.json', orient='records', indent=4)