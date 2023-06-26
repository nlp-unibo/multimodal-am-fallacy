import pandas as pd 
from nltk.tokenize import sent_tokenize
from tqdm.auto import tqdm
import json 
import os 
import re 
from collections import defaultdict
from fuzzywuzzy import fuzz

def replace_new_line_with_dot(dial_sentences, multiple):
    if multiple: 
    
        new_dial_sentences = []
        
        for el in dial_sentences:
                trail_new_line = True if el[-2:] == '\n' else False
                el = el.replace('\n', '') if trail_new_line else el.replace('...    \n', '.').replace('... \n', '.').replace('...  \n', '.').replace('.... \n', '.').replace('. \n', '.').replace('.\t\n', '.').replace('.  \n', '.').replace('-\n','.').replace('  \n', '.').replace(' \n', '.') # single and double space before newline, we need to manage 2 cases 
                new_dial_sentences.append(el)
        return new_dial_sentences
    else:
        trail_new_line = True if dial_sentences[-2:] == '\n' else False
        dial_sentences = dial_sentences.replace('\n', '') if trail_new_line else dial_sentences.replace('...    \n', '.').replace('... \n', '.').replace('...  \n', '.').replace('.... \n', '.').replace('. \n', '.').replace('.\t\n', '.').replace('.  \n', '.').replace('-\n','.').replace('  \n', '.').replace(' \n', '.') # single and double space before newline, we need to manage 2 cases     
        return dial_sentences

def clean_pipeline(s):
    s = s.strip().replace('\'', '').replace('\"', '').replace('`', '').replace('’', '').replace('‘', '').replace('“', '').replace('”', '').replace('–', '').replace('—', '').replace('…', '').replace('.','').replace('-','').strip()
    return s 

def clean_list_of_sentences(list_of_sentences):
    new_list_of_sentences = []
    for el in list_of_sentences:
        new_list_of_sentences.append(clean_pipeline(el))
    return new_list_of_sentences

def get_dup_timestamps(id, alignment_results_path, dial_sentences):
    #get the alignments for the duplicate sentences
    dup_align_begin_time = []
    dup_align_end_time = []
    all_aligments = [] # 2. sentences debate 
    all_times_begin = [] # 3. begin times debate
    all_times_end = [] # 4. end times debate
    with open(os.path.join(alignment_results_path)) as json_file:
        sentences = json.load(json_file)
        for align_dict in sentences:
            all_aligments.append(clean_pipeline(align_dict['lines'][0])) # align_dict['lines'][0].strip())
            all_times_begin.append(float(align_dict['begin']))
            all_times_end.append(float(align_dict['end']))

    first_dial_sent = dial_sentences[0].strip()
    last_dial_sent = dial_sentences[-1].strip()
    interval_first_last = len(dial_sentences)

    # search for the first and last sentence of the dialogue in the debate 
    try: 
        
        first_dial_sent_index = all_aligments.index(first_dial_sent)
        last_dial_sent_index = all_aligments.index(last_dial_sent)
        if last_dial_sent_index < first_dial_sent_index:
            # store all indexes of the duplicated last sentence
            last_dial_sent_index = first_dial_sent_index + interval_first_last
        
        all_aligments = all_aligments[first_dial_sent_index:last_dial_sent_index]
        all_times_begin = all_times_begin[first_dial_sent_index:last_dial_sent_index]
        all_times_end = all_times_end[first_dial_sent_index:last_dial_sent_index]
    except ValueError:
        print('Error: sentence not found in debate')
        print('Debate: ', id)
        print('Sentence: ', first_dial_sent)
        print(all_aligments)
        
        

    # get the alignments for the duplicate sentences
    dup = {}

    for i in range(len(dial_sentences)):
        if dial_sentences.count(dial_sentences[i]) > 1:
            if dial_sentences[i] not in dup:
                dup[dial_sentences[i]] = [all_times_begin[i]], [all_times_end[i]]

            else:
                dup[dial_sentences[i]] = dup[dial_sentences[i]][0]+[all_times_begin[i]], dup[dial_sentences[i]][1]+[all_times_end[i]]
    return dup 

def get_occurrences(L): 
    D = {}
    for i in range(len(L)): 
        if L[i] not in D: 
            D[L[i]] = 1,[i]
        else:
            D[L[i]] = D[L[i]][0]+1, D[L[i]][1]+[i]
    

    R = {}
    for e in D: 
        if D[e][0] != 1: 
            R[e] = D[e]
    #print(R)
            
    return R

def get_alignment_duplicated_sentences(dial_sentences, alignment_results_path, id_debate, list_dialogue_alignment_begin, list_dialogue_alignment_end):
    dup = get_dup_timestamps(id_debate, alignment_results_path, dial_sentences) # get the timestamps of the duplicate sentences            
    ld = get_occurrences(dial_sentences)
    for k, v in ld.items():
        for key, align_time in dup.items(): 
            if k == key: 
                #print(v)
                for index in range(len(v[1])): 
                    list_dialogue_alignment_begin[v[1][index]] = float(align_time[0][index])
                    list_dialogue_alignment_end[v[1][index]] = float(align_time[1][index])
    return list_dialogue_alignment_begin, list_dialogue_alignment_end

def get_alignment_sentences(alignment_results_path, debate_id, dial_sentences, list_dialogue_alignment_begin, list_dialogue_alignment_end):
    with open(os.path.join(alignment_results_path)) as json_file:
        sentences = json.load(json_file)
        for align_dict in sentences:
            for s in dial_sentences:
                alignment = clean_pipeline((align_dict['lines'][0]))
                
                #if fuzz.ratio(clean_pipeline(align_dict['lines'][0]), clean_pipeline(s)) > 80: # 80% of similarity (e.g. "Interest" and "Interests ." have 84% similarity)
                if fuzz.ratio(alignment, s) > 80: # 80% of similarity (e.g. "Interest" and "Interests ." have 84% similarity)
                    list_dialogue_alignment_begin[dial_sentences.index(s)] = float(align_dict['begin'])
                    list_dialogue_alignment_end[dial_sentences.index(s)] = float(align_dict['end'])
                    # Debug code
                    # if list_dialogue_alignment_end[dial_sentences.index(s)] == list_dialogue_alignment_begin[dial_sentences.index(s)] and list_dialogue_alignment_begin[dial_sentences.index(s)] != 'NOT_FOUND' :
                    #     print('Debug 3')
                    #     print(debate_id)
                    #     print(float(align_dict['begin']), float(align_dict['end']))
                    #     print(list_dialogue_alignment_end[dial_sentences.index(s)], list_dialogue_alignment_begin[dial_sentences.index(s)])
    
    return list_dialogue_alignment_begin, list_dialogue_alignment_end

def check_if_dc_followed_by_new_line(s):
    s = s.replace('\n', ' NEW_LINE')
    s = re.sub('\s+',' ',s)
    s = s.split(' ')
    return s

def substitute_dc(s): 
    new_s = ''

    for i in range(len(s)-1):
        if s[i] == 'D.C.' and s[i+1] == 'NEW_LINE':
            tmp = 'DC.'
            new_s += tmp + ' '
        elif s[i] =='D.C.' and s[i+1] != 'NEW_LINE':
            tmp = 'DC'
            new_s += tmp + ' '
        elif s[i] == 'NEW_LINE':
            new_s += '\n' + ' '

        else:
            new_s += s[i] + ' '
    if s[-1] == 'NEW_LINE':
        new_s += '\n'
    else:
        new_s += s[-1]

        
    return new_s

def check_dc_in_string(s):


    split_s = s.split(' ')
    for i in range(len(split_s)):
        if split_s[i] == 'D.C.':
            #print(s)
            return True
        
    return False


def format_dial_sentences(row_dial_sentences):
    dial_sentences = list(row_dial_sentences[1:-1].strip().split('\''))
    # remove ',' from the list
    dial_sentences = [x for x in dial_sentences if x.strip() != ',']
    # first and last element are empty strings, we remove them
    dial_sentences = dial_sentences[1:-1]
    return dial_sentences

def format_comp_text(row_dial_sentences):
    dial_sentences = list(row_dial_sentences[1:-1].strip().split('\''))
    # remove ',' from the list
    dial_sentences = [x for x in dial_sentences if x.strip() != ',']
    # first and last element are empty strings, we remove them
    dial_sentences = dial_sentences[1:-1]
    if dial_sentences == []: 
        dial_sentences = list(row_dial_sentences[1:-1].strip().split('\"'))
        # remove ',' from the list
        dial_sentences = [x for x in dial_sentences if x.strip() != ',']
        # first and last element are empty strings, we remove them
        dial_sentences = dial_sentences[1:-1]

    #transform dial_sentences into string 
    dial_sentences = ' '.join(dial_sentences)
    return dial_sentences

def format_dial_sentences_raw(row_dial_sentences):
    dial_sentences = row_dial_sentences.replace('\"','\'')
    dial_sentences = list(dial_sentences[2:-2].strip().split('\', \''))
    # remove ',' from the list
    dial_sentences = [x for x in dial_sentences if x.strip() != ',']
    # first and last element are empty strings, we remove them
    #dial_sentences = dial_sentences[1:-1]
    return dial_sentences
  
def format_list_timestamps(row_timestamps):
    return list(row_timestamps[1:-1].strip().split(','))

def clean_dots_snippet(s):
    return s.replace('.', '')

def correct_errors_dialogues_timestamps(df_path):
    df = pd.read_csv(df_path, sep = '\t')

    # devo risalvare il dataframe in csv e in json 
    # first element in the tuple = first timestamp in list timestamps begin 
    # second element in the tuple = last time stamp in list timestamp end 
    # i.e. BeginDialogue and EndDialogue 
    dict_errors = {'5_1976': [(3741.64, 515.4)],
                    '9_1980': [(5029.12, 3408.6)], 
                    '12_1984': [(3872.68, 1955.28)],
                    '20_1992': [(1508.32, 1359.56)],
                    '22_1996': [(4992.84, 725.2)],
                    '31_2004': [(4132.8, 1349.0), (5627.48, 2211.8)],
                    '39_2012': [(3926.04, 2964.88), (5068.16, 3929.44)],
                    '3_1960': [(2853.2, 588.04)]}
    
    # first element in the tuple = begin of first sentence
    # second element in the tuple = end of first sentence
    corrections = {'5_1976': [(144.84, 145.16)],
                    '9_1980': [(2788.7200000000003, 2789.24)],
                    '12_1984': [(1382.44, 1382.92)],
                    '20_1992': [(1201.32, 1202.36)],
                    '22_1996': [(527.08, 529.36)],
                    '31_2004': [(1066.16, 1067.4), (1915.3600000000001, 1918.24)],
                    '39_2012': [(2638.2, 2638.72), (3394.7200000000003, 3395.96)],
                    '3_1960': [(327.76, 327.96)]}
    
    
    new_rows = []

    for i, row in df.iterrows():
        debate_id = row['id_map']
        timestamps_begin = format_list_timestamps(row['DialogueAlignmentBegin'])
        timestamps_end = format_list_timestamps(row['DialogueAlignmentEnd'])

        if debate_id in dict_errors.keys():
            if len(dict_errors[debate_id]) == 1:
                
                if timestamps_begin[0].strip() == str(dict_errors[debate_id][0][0]) and timestamps_end[-1].strip() == str(dict_errors[debate_id][0][1]):
                    timestamps_begin[0] = str(corrections[debate_id][0][0])
                    timestamps_end[0] = str(corrections[debate_id][0][1])
            else:
                if timestamps_begin[0].strip() == str(dict_errors[debate_id][0][0]) and timestamps_end[-1].strip() == str(dict_errors[debate_id][0][1]):
                    timestamps_begin[0] = corrections[debate_id][0][0]
                    timestamps_end[0] = corrections[debate_id][0][1]
                elif timestamps_begin[0].strip() == str(dict_errors[debate_id][1][0]) and timestamps_end[-1].strip() == str(dict_errors[debate_id][1][1]):
                    timestamps_begin[0] = corrections[debate_id][1][0]
                    timestamps_end[0] = corrections[debate_id][1][1]
        row['DialogueAlignmentBegin'] = timestamps_begin
        row['DialogueAlignmentEnd'] = timestamps_end
        #print(timestamps_begin[0].strip().replace('\'','').replace('\"', '').strip())
        row['DialogueBegin'] = timestamps_begin[0] #.strip()
        new_rows.append(row)

    return pd.DataFrame(new_rows) 

def clean_string_begin_end_timestamps(dataset_path, new_dataset_path_csv, new_dataset_path_json): 
    df = pd.read_csv(dataset_path, sep = '\t')
    new_rows = []
    for i, row in df.iterrows():
        row['BeginCompText'] = row['BeginCompText'].strip().replace('\'','').replace('\"', '').strip()
        row['EndCompText'] = row['EndCompText'].strip().replace('\'','').replace('\"', '').strip()
        row['BeginSnippet'] = row['BeginSnippet'].strip().replace('\'','').replace('\"', '').strip()
        row['EndSnippet'] = row['EndSnippet'].strip().replace('\'','').replace('\"', '').strip()
        new_rows.append(row)
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(new_dataset_path_csv, index=False, sep='\t') 
    new_df.to_json(new_dataset_path_json, orient='records',  indent = 4) 


def correct_wrong_snippet(dataset_path_csv, dataset_path_json, new_dataset_path_csv, new_dataset_path_json):
    # when performing EDA it has been discovered that one snippet is wrong
    # becuse it contains, as last sentence, a word that is present many times in the alignment results 
    # therefore, without removing it is not possible to understand to which sentence it refers
    # since it does not contain any meaningful information, we can remove it to get a more accurate alignment 

    # Error report: 
    #id Debate:  9_1980  
    # Snippet:  Mr. Stone, I have submitted an economic plan that I have worked out in concert with a number of fine economists in this country, all of whom approve it, and believe that over a five year projection, this plan can permit the extra spending for needed refurbishing of our defensive posture, that it can provide for a balanced budget by 1983 if not earlier, and that we can afford - along with the cuts that I have proposed in Government. spending
    # Sentence that should be removed: 'spending'
    df = pd.read_csv(dataset_path_csv, sep = '\t')
    new_rows = []
    for i, row in df.iterrows():
        snippet = row['Snippet']
        if snippet == 'Mr. Stone, I have submitted an economic plan that I have worked out in concert with a number of fine economists in this country, all of whom approve it, and believe that over a five year projection, this plan can permit the extra spending for needed refurbishing of our defensive posture, that it can provide for a balanced budget by 1983 if not earlier, and that we can afford - along with the cuts that I have proposed in Government. spending':
            sent_snippet = sent_tokenize(snippet)
            sent_snippet = sent_snippet[:-1]
            row['Snippet'] = ' '.join(sent_snippet)
    
            print(row['Snippet'])

        new_rows.append(row)
    new_df = pd.DataFrame(new_rows)

    
    new_df.to_csv(new_dataset_path_csv, index=False, sep='\t') 
    new_df.to_json(new_dataset_path_json, orient='records',  indent = 4) 

def remove_apexes_from_clip_name(s): 
    return s.replace('\'','').replace('\"', '').strip()