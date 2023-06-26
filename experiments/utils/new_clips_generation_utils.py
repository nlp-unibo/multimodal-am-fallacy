import pandas as pd
from tqdm import tqdm
import json 
from collections import defaultdict
from pydub import AudioSegment
import os
import shutil


project_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

dataset_dir = os.path.join(project_dir, "local_database", "MM-DatasetFallacies")

audio_clips_dir = os.path.join(dataset_dir, "audio_clips")

resources_dir = os.path.join(project_dir, "resources")

clips_generation_dir = os.path.join(resources_dir, "clips_generation")

def generate_clips(element, ids, modality, dataset_path, sample_rate):
    #  element can be sentences dialogue, snippets, components and dialogues 
    # ids is a list of ids of the elements to be processed
    # modality is the modality of the audio clips to be generated
    # dataset_path is the path to the dataset
    # sample_rate is the sample rate of the audio clips to be generated

    df = pd.read_csv(dataset_path, sep='\t')
    if modality == 'full':
        ids = df.id_map.unique()

    if element == 'dial_sent':
        MAIN_FOLDER_PATH = os.path.join(audio_clips_dir, 'dial_sent')
        
    elif element == 'snippet':
        MAIN_FOLDER_PATH = os.path.join(audio_clips_dir, 'snippet')
    
    elif element == 'comp':
        MAIN_FOLDER_PATH = os.path.join(audio_clips_dir, 'comp')

    elif element == 'dial':
        MAIN_FOLDER_PATH = os.path.join(audio_clips_dir, 'dial')

    
    if not os.path.exists(MAIN_FOLDER_PATH):
        os.makedirs(MAIN_FOLDER_PATH)
    else:
        shutil.rmtree(MAIN_FOLDER_PATH)
        os.makedirs(MAIN_FOLDER_PATH)
    
    for i in tqdm(range(len(ids))):
        FOLDER_ID = ids[i]


        DATASET_PATH_CLIP_EMPTY = os.path.join(clips_generation_dir, element, FOLDER_ID, 'dataset.csv')
        DATASET_CLIP_PATH = os.path.join(clips_generation_dir, element, FOLDER_ID, 'dataset_clip.csv')
        FULL_AUDIO_PATH = os.path.join(resources_dir, 'debates_audio_recordings', FOLDER_ID, 'full_audio_trim.wav')


        AUDIO_CLIPS_PATH = os.path.join(MAIN_FOLDER_PATH, FOLDER_ID)

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

        # set elements to read
        if element == 'dial_sent':
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

        elif element == 'snippet':
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
        elif element == 'component': 
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
        elif element == 'dialogue':
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
        # if not os.path.exists(DATASET_CLIP_PATH):
        #     df.to_csv(DATASET_CLIP_PATH, sep = '\t')
    





