import pandas as pd
from utils import alignment_utils as al_utils, data_loader
from nltk import sent_tokenize
from collections import defaultdict
import os 

#filepath = 'datasets/MM-DatasetFallacies/dataset.csv'
#original_dataset = 'datasets/merged.csv'

# set project folder
project_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

# load dataset
project_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
df = data_loader.load(project_dir)


#df = pd.read_csv(filepath, sep='\t')

# original dataset path 
df_original_path = os.path.join(project_dir, 'local_database', 'USED-fallacy', 'original_no_duplicates_dial.csv')
df_original = pd.read_csv(df_original_path, sep='\t') # TODO: add df_original to the comparison
print(df_original.head())

# Check how many samples per fallacy type we have in the dataset

print("Number of samples per fallacy type in Original Dataset: ")
print(df_original['Fallacy'].value_counts())

print("Number of samples per fallacy type in MM-DatasetFallacies: ")
print(df['Fallacy'].value_counts())

# count unique ids in the original dataset and in the new dataset
print("Number of unique ids in Original Dataset: ", len(df_original['id_map'].unique()))
print("Number of unique ids in MM-DatasetFallacies: ", len(df['Dialogue ID'].unique()))
print()

# Check how many samples we had in original dataset and how many we have in the new dataset
print("Number of samples in Original Dataset: ", len(df_original))
print("Number of samples in MM-DatasetFallacies: ", len(df))
print("Number of samples in MM-DatasetFallacies that are not in Original Dataset: ", len(df_original) - len(df))
print()
# Check min, max and mean duration of the dialogues in the dataset in number of words 
# compute the number of words in each dialogue
df['num_words'] = df['Dialogue'].apply(lambda x: len(x.split(' ')))
print("Number of words in each dialogue: ")
print(df['num_words'].describe())
print()

# Check min, max and mean duration of the Snippet in the dataset in number of words
# compute the number of words in each snippet
df['num_words_snippet'] = df['Snippet'].apply(lambda x: len(x.split(' ')))
print("Number of words in each snippet: ")
print(df['num_words_snippet'].describe())
print()

# Check min, max and mean duration of the CompText in the dataset in number of words
# compute the number of words in each CompText
df['num_words_comptext'] = df['CompText'].apply(lambda x: len(x.split(' ')))
print("Number of words in each CompText: ")
print(df['num_words_comptext'].describe())
print()



df['num_sentences_snippet'] = df['SentenceSnippet'].apply(lambda x: len(sent_tokenize(text=x)))
# create SentenceCompText column by merging sentences of ReferenceSentencesCompText
df['SentenceCompText'] = df['ReferenceSentencesCompText'].apply(lambda x: al_utils.format_comp_text(x))
df['num_sentences_comptext'] = df['SentenceCompText'].apply(lambda x: len(sent_tokenize(x)))
print("Number of sentences in each snippet: ")
print(df['num_sentences_snippet'].describe())
print("Number of sentences in each CompText: ")
print(df['num_sentences_comptext'].describe())
print()

# compute mean and std of the number of sentences in each snippet and in each CompText
print("Mean and std of the number of sentences in each snippet: ")
print(df['num_sentences_snippet'].mean(), df['num_sentences_snippet'].std())
print("Mean and std of the number of sentences in each CompText: ")
print(df['num_sentences_comptext'].mean(), df['num_sentences_comptext'].std())
print()

# Check min, max and mean duration of the Dialogue in the dataset in number of sentences
# compute the number of sentences in each dialogue by counting the number of elements in the DialogueSentences column
df['num_sentences'] = df['DialogueSentences'].apply(lambda x: len(al_utils.format_dial_sentences(x)))
print("Number of sentences in each dialogue: ")
print(df['num_sentences'].describe())
print()

# Check min, max and mean duration of clip in Dialogue 
# compute the duration of each clip in Dialogue
# store durations 
for i, row in df.iterrows():
    df.at[i, 'duration_clip'] = float(row['DialogueEnd']) - float(row['DialogueBegin'])
    if df['duration_clip'][i] < 0: 
        print("Error: ", df['duration_clip'][i], i)
        print(row['DialogueEnd'], row['DialogueBegin'], row['Dialogue ID'])
print("Duration of each clip in Dialogue: ")
print(df['duration_clip'].describe())


# Check min, max and mean duration of clip in Dialogue 
# compute the duration of each clip in Dialogue
# store durations 
for i, row in df.iterrows():
    df.at[i, 'duration_clip_snippet'] = float(row['EndSnippet']) - float(row['BeginSnippet'])
    if df['duration_clip_snippet'][i] < 0: 
        print("Error: ", df['duration_clip_snippet'][i], i)
        print(row['EndSnippet'], row['BeginSnippet'], row['Dialogue ID'])
print("Duration of each clip in Snippet: ")
print(df['duration_clip_snippet'].describe())

# Check min, max and mean duration of clip in CompText
# compute the duration of each clip in CompText
# store durations 
for i, row in df.iterrows():
    df.at[i, 'duration_clip_comptext'] = float(row['EndCompText']) - float(row['BeginCompText'])
    if df['duration_clip_comptext'][i] < 0: 
        print("Error: ", df['duration_clip_comptext'][i], i)
        print(row['EndCompText'], row['BeginCompText'], row['Dialogue ID'])
print("Duration of each clip in CompText: ")
print(df['duration_clip_comptext'].describe())



errors = 0 
# check for how many dialogues the begin is greaters than the end
# quelli in cui è presente questo errore sono quelli che iniziano con una frase duplicata 
# quindi va corretto manualmente 
dict_errors = defaultdict(list)
for i, row in df.iterrows():
    if float(row['DialogueBegin']) > float(row['DialogueEnd']):
        errors += 1
        if (row['DialogueBegin'], row['DialogueEnd']) not in dict_errors[row['Dialogue ID']]:
            dict_errors[row['Dialogue ID']].append((row['DialogueBegin'], row['DialogueEnd']))

for el in dict_errors.items():
    print(el)
print(dict_errors)

# OUTPUT
# ('5_1976', [(3741.64, 515.4)])
# ('9_1980', [(5029.12, 3408.6)])
# ('12_1984', [(3872.68, 1955.28)])
# ('20_1992', [(1508.32, 1359.56)])
# ('22_1996', [(4992.84, 725.2)])
# ('31_2004', [(4132.8, 1349.0), (5627.48, 2211.8)])
# ('39_2012', [(3926.04, 2964.88), (5068.16, 3929.44)])
# ('3_1960', [(2853.2, 588.04)])


# check if all DialogueAlignemtnBegin are in sequence (ith-1 < ith) and if all DialogueAlignmentEnd are in sequence (ith < ith+1)
errors = 0
tot_num_sentences = 0 
errors_coincides_snippets = 0 
errors_one_sentence_context_snippet = 0 
error_two_sentences_context_snippet = 0

for i, row in df.iterrows():
    timestamps_begin = al_utils.format_list_timestamps(row['DialogueAlignmentBegin'])
    timestamps_end = al_utils.format_list_timestamps(row['DialogueAlignmentEnd'])
    dial_sentences = al_utils.format_dial_sentences_raw(row['DialogueSentences'])
    id_debate = row['Dialogue ID']
    clip_snippet = row['idClipSnippet']

    snippet_indexes = al_utils.format_list_timestamps(row['IndexReferenceSentencesSnippet'])


    first_index_snippet = int(float(snippet_indexes[0]))


    for j in range(len(timestamps_begin)-1):
        #if float(timestamps_begin[j]) > float(timestamps_begin[j+1]): # non è un controllo indicativo perché delle volte le frasi del dialogo sono riportate in ordine diverso rispetto a quelle dell'alignment
        if float(timestamps_begin[j].strip().replace('\'','')) > float(timestamps_end[j].strip().replace('\'','')):
            #print("Error: ", timestamps_begin[j], timestamps_begin[j+1], dial_sentences[j], id_debate)
            errors += 1
            if j == first_index_snippet: 
                errors_coincides_snippets += 1
                print("id Debate: ", id_debate, " Clip Snippet: ", clip_snippet, " Snippet: ", row['Snippet'])

            elif j != 0 and j == first_index_snippet-1: 
                errors_one_sentence_context_snippet += 1
            elif j != 0 and j == first_index_snippet-2:
                error_two_sentences_context_snippet += 1

        tot_num_sentences += 1
print("Number of errors in DialogueAlignmentBegin: ", errors)
print("Total number of sentences: ", tot_num_sentences)
print("Percentage of errors: ", errors/tot_num_sentences)
print("Number of errors that coincide with the snippet: ", errors_coincides_snippets)
print("Number of errors that coincide with the snippet and the previous sentence is the context: ", errors_one_sentence_context_snippet)
print("Number of errors that coincide with the snippet and the previous two sentences are the context: ", error_two_sentences_context_snippet)

# check how many snippets corresponds to comptext
count = 0
for i, row in df.iterrows():
    if row['Snippet'] == row['CompText']:
        count += 1
print("Number of snippets that corresponds to CompText: ", count)


# check how many duplicated snippets there are 
snippets = df['Snippet'].tolist()
snippets = [x.lower() for x in snippets]
snippets = [x.strip() for x in snippets]
count = 0
for el in snippets: 
    if snippets.count(el) > 1: 
        count += 1
print("Number of duplicated snippets: ", count)

# check how many duplicated comptexts there are
comptexts = df['CompText'].tolist()
comptexts = [x.lower() for x in comptexts]
comptexts = [x.strip() for x in comptexts]
count = 0
for el in comptexts:
    if comptexts.count(el) > 1:
        count += 1
print("Number of duplicated comptexts: ", count)


# create a dictionary to count number of sentences in snippets
counts = dict()
for i, row in df.iterrows():
    id_debate = row['Dialogue ID']
    ref_sentences = al_utils.format_list_timestamps(row['IndexReferenceSentencesSnippet'])
    len_snippet= len(ref_sentences)
    if len_snippet == 31:
        print("id Debate: ", id_debate, " Snippet: ", row['Snippet'])
    
    counts[len_snippet] = counts.get(len_snippet,0) + 1

# print the number of snippets for each number of sentences
print("Number of sentences in snippets: ")
for key, value in counts.items():
    print(key, value)

# create a dictionary to count number of sentences in comptexts
counts = dict()
for i, row in df.iterrows():
    comptext = row['CompText']
    sentences_comptext = sent_tokenize(comptext)
    len_comptext = len(sentences_comptext)
    counts[len_comptext] = counts.get(len_comptext,0) + 1

print("Number of sentences in comptexts: ")
# print the number of comptexts for each number of sentences
for key, value in counts.items():
    print(key, value)

# compare to IndexReferenceSentecesComptext to verify correctness
for i, row in df.iterrows():
    comptext = row['CompText']
    sentences_comptext = sent_tokenize(comptext)
    len_comptext = len(sentences_comptext)
    ref_sentences = al_utils.format_list_timestamps(row['IndexReferenceSentencesCompText'])
    len_ref_sentences = len(ref_sentences)
    if len_comptext != len_ref_sentences:
        print("Error: ", len_comptext, len_ref_sentences)

# create dictionary of dictionaries 
# key: fallacy
# value: dictionary of snippets and their number of sentences
fallacy_snippets = dict()
for i, row in df.iterrows():
    fallacy = row['Fallacy']
    snippet = row['Snippet']
    sentences_snippet = sent_tokenize(snippet)
    len_snippet = len(sentences_snippet)
    if fallacy not in fallacy_snippets: 
        fallacy_snippets[fallacy] = dict()
        fallacy_snippets[fallacy][len_snippet] = fallacy_snippets[fallacy].get(len_snippet,0) + 1
    else: 
        fallacy_snippets[fallacy][len_snippet] = fallacy_snippets[fallacy].get(len_snippet,0) + 1

print("Distribution of snippets lengths for each fallacy:")
print(fallacy_snippets)


# check ho many snippets are in the comptext
count = 0
for i, row in df.iterrows():
    comptext = row['CompText']
    snippet = row['Snippet']
    if snippet in comptext:
        count += 1

print("Number of snippets that are in the comptext: ", count)

# check how many comptexts are in the snippet
count = 0
for i, row in df.iterrows():
    comptext = row['CompText']
    snippet = row['Snippet']
    if comptext in snippet:
        count += 1

print("Number of comptexts that are in the snippet: ", count)

# check how manu snippets are in the comptext and viceversa
count = 0
for i, row in df.iterrows():
    comptext = row['CompText']
    snippet = row['Snippet']
    if comptext in snippet or snippet in comptext:
        count += 1

print("Number of comptexts that are in the snippet or viceversa: ", count, "out of ", len(df))



# check how many snippets contain less than 1 sentence by comparing the length of the snippet with the length of the reference sentences and store in a dict by fallacy
count = 0
fallacy_snippets = dict()
for i, row in df.iterrows():
    snippet = row['Snippet']
    ref_sentences = al_utils.format_dial_sentences_raw(row['ReferenceSentencesSnippet'])
    fallacy = row['Fallacy']
    if len(ref_sentences) == 1: 
        len_snippet = len(snippet)
        len_ref_sentences = len(ref_sentences[0])
        if len_snippet < len_ref_sentences:
            count += 1
            if fallacy not in fallacy_snippets:
                fallacy_snippets[fallacy] = 1
            else:
                fallacy_snippets[fallacy] += 1

print("Number of snippets that contain less than 1 sentence: ", count)
print("Number of snippets that contain less than 1 sentence by fallacy: ", fallacy_snippets)

# check how many comptexts contain less than 1 sentence by comparing the length of the comptexts with the length of the reference sentences and store in a dict by fallacy
count = 0
fallacy_comptexts = dict()
for i, row in df.iterrows():
    comptext = row['CompText']
    ref_sentences = al_utils.format_dial_sentences_raw(row['ReferenceSentencesCompText'])
    fallacy = row['Fallacy']
    if len(ref_sentences) == 1: 
        len_comptext = len(comptext)
        len_ref_sentences = len(ref_sentences[0])
        if len_comptext < len_ref_sentences:
            count += 1
            if fallacy not in fallacy_comptexts:
                fallacy_comptexts[fallacy] = 1
            else:
                fallacy_comptexts[fallacy] += 1

print("Number of comptexts that contain less than 1 sentence: ", count)
print("Number of comptexts that contain less than 1 sentence by fallacy: ", fallacy_comptexts)



# check if, when snippet contain more than one sentence, the first sentence is exactly the first reference sentence and the last sentence is the last reference sentence
count = 0
snippet_more_sentences = 0
for i, row in df.iterrows():
    snippet = row['Snippet']
    snippet = sent_tokenize(snippet)
    if len(snippet) >= 2:
        snippet_more_sentences += 1
        ref_sentences = al_utils.format_dial_sentences_raw(row['ReferenceSentencesSnippet'])
        if len(ref_sentences) >= 1:
            if ref_sentences[0] == snippet[0] and ref_sentences[-1] == snippet[-1]:
                count += 1

print("Number of snippets that contain more than one sentence: ", snippet_more_sentences)

print("Number of snippets that contain more than one sentence and the first and last sentence are the first and last reference sentence: ", count)

