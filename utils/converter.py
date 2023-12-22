import tensorflow as tf
from transformers import BertTokenizer, RobertaTokenizerFast
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import librosa
import resampy
import utils.processor as processor
from tqdm import tqdm

maxBERTLen = 512
maxSBERTLen = 768

def prepare_text_data(X, y, text_model='bert', maxSentenceLen = maxBERTLen, is_train=False, q=0.99): # config = 'text_only' or 'audio_only' or 'text_audio'
    """

    :param X: a list of strings
    :param y: a list of strings
    :param text_model: a string
    :param maxSentenceLen: an integer
    :return: the encoded input data, the encoded labels
    """
    # Encode the labels
    y_enc = np.array(encode_labels(y))

    # Encode the input data
    if text_model == 'bert':
        pad = tf.keras.preprocessing.sequence.pad_sequences
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif text_model == 'roberta':
        pad = tf.keras.preprocessing.sequence.pad_sequences
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        #tokenizer.build_inputs_with_special_tokens
    elif text_model == 'sbert':
        encoded_inputs = []
        model = SentenceTransformer('all-mpnet-base-v2')
        maxSentenceLen = maxSBERTLen
        for i in tqdm(range(len(X))):
            encoded_input = model.encode(X[i])
            encoded_inputs.append(encoded_input)
        if is_train:
            return np.array(encoded_inputs), y_enc, maxSentenceLen
        else:
            return np.array(encoded_inputs), y_enc
        


    
    if text_model == 'bert':
        dataFields = {
            "input_ids_not_padded": [],
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
        }

    if text_model == 'roberta':
        dataFields = {
            "input_ids_not_padded": [],
            "input_ids": [],
            "attention_mask": [],
        }

    


    # store inputs_ids_not_padded to calculate the maxSentenceLen
    if is_train:
        for i in tqdm(range(len(X))):
            data = tokenizer.encode_plus(X[i].strip(), truncation=True)
            dataFields['input_ids_not_padded'].append(data['input_ids'])

        maxSentenceLen=get_max_sentence_len(dataFields['input_ids_not_padded'], q=q)

    for i in range(len(X)):
        #data = tokenizer(X[i], truncation=True)
        data = tokenizer.encode_plus(X[i].strip(), truncation=True)
        if text_model == 'bert':
            padded = pad([data['input_ids'], data['attention_mask'], data['token_type_ids']], padding = 'post', truncating = 'post', maxlen = maxSentenceLen)
            dataFields['input_ids'].append(padded[0])
            dataFields['attention_mask'].append(padded[1])
        
        elif text_model == 'roberta':
            input_ids = data['input_ids'] + [tokenizer.pad_token_id] * (maxSentenceLen - len(data['input_ids']))
            input_ids = input_ids[:maxSentenceLen]

            attention_mask = data['attention_mask'] + [0] * (maxSentenceLen - len(data['attention_mask']))
            attention_mask = attention_mask[:maxSentenceLen]

            dataFields['input_ids'].append(input_ids)
            dataFields['attention_mask'].append(attention_mask)


        
        if text_model == 'bert':
            dataFields['token_type_ids'].append(padded[-1])

    for key in dataFields:
        dataFields[key] = np.array(dataFields[key]) 


    

    if is_train:
        #maxSentenceLen=get_max_sentence_len(dataFields['input_ids_not_padded'], q=q)
        # cut the data to the maxSentenceLen
        # dataFields['input_ids'] = dataFields['input_ids'][:, :maxSentenceLen]
        # dataFields['attention_mask'] = dataFields['attention_mask'][:, :maxSentenceLen]
        # dataFields['token_type_ids'] = dataFields['token_type_ids'][:, :maxSentenceLen]
        
        if text_model == 'bert':
            return [dataFields["input_ids"], dataFields["token_type_ids"], dataFields["attention_mask"]], y_enc, maxSentenceLen
            
        elif text_model == 'roberta':
            return [dataFields["input_ids"], dataFields["attention_mask"]], y_enc, maxSentenceLen         


    else:
        # cut the data to the maxSentenceLen
        # dataFields['input_ids'] = dataFields['input_ids'][:, :maxSentenceLen]
        # dataFields['attention_mask'] = dataFields['attention_mask'][:, :maxSentenceLen]
        # dataFields['token_type_ids'] = dataFields['token_type_ids'][:, :maxSentenceLen]
        if text_model == 'bert':
            return [dataFields["input_ids"], dataFields["token_type_ids"], dataFields["attention_mask"]], y_enc
        elif text_model == 'roberta':
            return [dataFields["input_ids"], dataFields["attention_mask"]], y_enc
    



def prepare_audio_data(X, model_type = 'embedding',  audio_model='wav2vec', audio_model_sample_rate=16000, is_train=False):
    """

    :param X: a list of strings (audio file paths)
    :param y: a list of strings
    :param audio_model: a string
    :param config: a string
    :return: the encoded input data, the encoded labels
    """
    #TODO: add implementation for model_type 'features'. Need to change also the way in which max_frame_len is computed
    if model_type == 'embedding':
        encoded_audio_files = []
        for i in tqdm(range(len(X))):
            snippet_audio_file = X[i]
            snippet_audio_emb_file = snippet_audio_file.replace('.wav', '_' + audio_model + '_emb.npy') # TODO: check
            # # if snippet_audio_emb_file exist, remove it  # TODO: remove this
            # if os.path.isfile(snippet_audio_emb_file):
            #     os.remove(snippet_audio_emb_file)
        
            if not os.path.isfile(snippet_audio_emb_file):
                snippet_audio, sample_rate = librosa.load(snippet_audio_file, sr=None)

                snippet_audio = resampy.resample(snippet_audio,
                                                 sample_rate,
                                                 audio_model_sample_rate)

                
                if audio_model == 'wav2vec':
                    snippet_audio = processor.audio_processor(snippet_audio, audio_model=audio_model,
                                                audio_model_sample_rate=audio_model_sample_rate) # TODO: check if is correct to return model(input_state) or if input_state is an attribute of the mdoel
                    #snippet_audio = audio_model(snippet_audio[None, :]).last_hidden_state
                    snippet_audio = np.mean(snippet_audio.numpy().squeeze(axis=0), axis=0)
                else: 
                    snippet_audio = processor.audio_processor(snippet_audio, audio_model=audio_model,
                                                audio_model_sample_rate=audio_model_sample_rate)
                    
                    snippet_audio = np.mean(snippet_audio, axis=0)
                    snippet_audio = np.asarray(snippet_audio).astype('float32')                                               
                np.save(snippet_audio_emb_file, snippet_audio)

            else:
                
                snippet_audio = np.load(snippet_audio_emb_file)
            #print(snippet_audio.shape)

            encoded_audio_files.append(snippet_audio)

        if is_train:
            max_frame_len = snippet_audio.shape[0]
            return np.array(encoded_audio_files), max_frame_len
        else:
            return np.array(encoded_audio_files)

def encode_labels(y):
    encoded_labels = []

    lbls = {
        'AppealtoEmotion': 0,
        'AppealtoAuthority': 1,
        'AdHominem': 2,
        'FalseCause': 3,
        'Slipperyslope': 4,
        'Slogans': 5
    }
    for i in range(len(y)):
        encoded_labels.append(lbls[y[i]])

    return encoded_labels



def get_max_sentence_len(input_ids, q = 0.99):
    return min(int(np.quantile([len(x) for x in input_ids], q=q)), maxBERTLen)

