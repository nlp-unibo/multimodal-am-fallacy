import os
from pathlib import Path

from pydub import AudioSegment
from tqdm import tqdm
import yt_dlp

def youtube_download(id: list, link: list, dest_folder: str, modality : str = "full", n_files: str  = "all") -> None:
    """
    :param id: list of strings representing debates IDs
    :param link: list of strings representing the urls to the YouTube videos of the debates
    :param dest_folder: string representing the path to the folder where the audio files will be saved
    :param modality: string representing the modality of the audio file to be downloaded. It can be 'full' or 'partial'
    :param n_files: string representing the number of files to be downloaded. It can be 'all' or a number
    :return: None. The function populates the folder 'resources/debates_audio_recordings' by creating a folder for each
             debate. Each folder contains the audio file extracted from the corresponding video
    """

    AUDIO_FILE_PATH = str(dest_folder)

    if modality == "full":
        map_debate_link = dict(zip(id, link))

    elif modality == "partial":
        # if modality is partial, the user must specify the number of files to be downloaded
        # the first n_files in "dictionary.csv" will be downloaded
        if n_files != "all":
            n_files = int(n_files)
            id = id[:n_files]
            link = link[:n_files]
            map_debate_link = dict(zip(id, link))

    for doc, link in tqdm(map_debate_link.items()):
        audio_path = AUDIO_FILE_PATH + '/' + doc
        if not os.path.exists('resources/debates_audio_recordings'):
            os.makedirs('resources/debates_audio_recordings')

        os.makedirs(audio_path, exist_ok=False)

        filename = AUDIO_FILE_PATH + '/' + doc + "/full_audio"
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': filename
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])
        #os.system("youtube-dl --rm-cache-dir")



def trim_audio(id: list, startMin: list, startSec: list, endMin: list, endSec: list, dest_folder: str,
               modality : str = "full", n_files: str  = "all") -> None:
    """

    :param id: list of strings representing debates IDs
    :param startMin: list of strings representing the number of minutes to be cut from the beginning of the file
    :param startSec: list of strings representing the number of seconds to be cut from the beginning of the file
    :param endMin: list of strings representing the number of minutes to be cut from the end of the file
    :param endSec: list of strings representing the number of seconds to be cut from the end of the file
    :return None: None. The function removes from the original audio file the portions of audio corresponding
                      to the specified seconds and minutes and saves a new version of the file '_trim.wav' in
                      'resources/debates_audio_recordings' (in the corresponding debate's sub folder).
    """
    base_path = str(dest_folder) +'/'
    if modality != "full":
        if n_files != "all":
            n_files = int(n_files)
            id = id[:n_files]

    print("Trimming audio files...")
    for i in tqdm(range(len(id))):
        FOLDER_ID = id[i]

        EXPORT_FILENAME = base_path + FOLDER_ID + "/full_audio_trim.wav"
        IMPORT_FILENAME = base_path + FOLDER_ID + '/' + 'full_audio.wav'

        if not Path(EXPORT_FILENAME).is_file():
        # importing file from location by giving its path
            sound = AudioSegment.from_file(IMPORT_FILENAME)

            # Selecting Portion we want to cut
            StrtMin = startMin[i]
            StrtSec = startSec[i]
            duration = sound.duration_seconds
            EndMin, EndSec = divmod(duration, 60)
            EndMin = EndMin - endMin[i]
            EndSec = EndSec - endSec[i]

            # Time to milliseconds conversion
            StrtTime = StrtMin * 60 * 1000 + StrtSec * 1000
            EndTime = EndMin * 60 * 1000 + EndSec * 1000
            # print(EndTime)

            # Opening file and extracting portion of it
            extract = sound[StrtTime:EndTime]
            # Saving file in required location

            extract.export(EXPORT_FILENAME, format="wav")  # wav conversion is faster than mp3 conversion


def download_and_trim(id: str, link: str,  startMin: str, startSec: int, endMin: int, endSec: int, dest_folder: str, sample_rate: int = 16000):

    # download audio file corresponding to debate with id "id"
    AUDIO_FILE_PATH = str(dest_folder)
    audio_path = AUDIO_FILE_PATH + '/' + id
    if not os.path.exists('resources/debates_audio_recordings'):
        os.makedirs('resources/debates_audio_recordings')

    if not os.path.exists(audio_path):
        os.makedirs(audio_path, exist_ok=False)

    filename = AUDIO_FILE_PATH + '/' + id + "/full_audio"
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': filename
    }

    print("Downloading audio file for debate ", id, "...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])

    # trim audio file

    base_path = str(dest_folder) + '/'
    print("Trimming audio files...")
    FOLDER_ID = id
    EXPORT_FILENAME = base_path + FOLDER_ID + "/full_audio_trim.wav"
    IMPORT_FILENAME = base_path + FOLDER_ID + '/' + 'full_audio.wav'

    if not Path(EXPORT_FILENAME).is_file():
        # importing file from location by giving its path
        sound = AudioSegment.from_file(IMPORT_FILENAME)

        # Selecting Portion we want to cut
        StrtMin = startMin
        StrtSec = startSec
        duration = sound.duration_seconds
        EndMin, EndSec = divmod(duration, 60)
        EndMin = EndMin - endMin
        EndSec = EndSec - endSec

        # Time to milliseconds conversion
        StrtTime = StrtMin * 60 * 1000 + StrtSec * 1000
        EndTime = EndMin * 60 * 1000 + EndSec * 1000
        # print(EndTime)

        # Opening file and extracting portion of it
        extract = sound[StrtTime:EndTime]
        # Resample
        extract = extract.set_frame_rate(sample_rate)
        # Saving file in required location
        extract.export(EXPORT_FILENAME, format="wav")  # wav conversion is faster than mp3 conversion

        # Delete original file
        os.remove(IMPORT_FILENAME)

