import os
from pathlib import Path
from utils import download_recordings
import pandas as pd
from tqdm import tqdm
import shutil


current_folder = os.getcwd()

main_project_folder = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
dest_recordings_folder = os.path.join(main_project_folder, 'resources', 'debates_audio_recordings')
dict_download_links = os.path.join(main_project_folder, 'resources', 'download', 'download_links.csv')
dict_mapping_ids = os.path.join(main_project_folder, 'resources', 'download', 'link_ids.csv')


modality = "full"
n_files = "all"
sample_rate = 16000

if __name__ == '__main__':
    df = pd.read_csv(dict_download_links, sep=';')
    df_mapping = pd.read_csv(dict_mapping_ids, sep=';')
    df.columns = ['id', 'link', 'startMin', 'startSec', 'endMin', 'endSec']
    df_mapping.columns = ['merged_id', 'mm_id']
    id_mapping = df_mapping.mm_id
    id_links = df.id

    link_df = df.link
    startMin_df = df.startMin
    startSec_df = df.startSec
    endMin_df = df.endMin
    endSec_df = df.endSec

    # Debug print
    #print(id_mapping)
    # print(id_links)

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
    endSec =  []

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

    # Debug prints 
    # print(id)
    # print(links)



    if modality == "full":
        map_debate_link = dict(zip(id, links))

    elif modality == "partial":
        # if modality is partial, the user must specify the number of files to be downloaded
        # the first n_files in "dictionary.csv" will be downloaded
        if n_files != "all":
            n_files = int(n_files)
            id = id[:n_files]
            links = links[:n_files]
            # print(id)
            # print(links)
            map_debate_link = dict(zip(id, links))

    i = 0
    

    try: 
        os.mkdir(dest_recordings_folder)
    except FileExistsError as error:
        print('Debates audio recordings folder already exists')




    # Proceed with the download only if the folder is empty
    if not os.listdir(dest_recordings_folder):
        for doc, link in tqdm(map_debate_link.items()):
            print(doc, link)
            # call function to download only if dest_recordings_folder is empty or if the file is not already present
            
            download_recordings.download_and_trim(doc, link , startMin[i], startSec[i], endMin[i], endSec[i], dest_recordings_folder, sample_rate)
            i+=1
    
    # # add folder for 21_1992 since 20_1992 and 21_1992 are the same debate, so copy the content of 20_1992 to 21_1992
    # if not os.path.exists(dest_recordings_folder + '/' + '21_1992'):
    #     os.makedirs(dest_recordings_folder + '/' + '21_1992')
    #     shutil.copy(dest_recordings_folder + '/' + '20_1992' + '/full_audio_trim.wav', dest_recordings_folder + '/' + '21_1992')