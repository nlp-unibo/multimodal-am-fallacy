# look at each file presented in each sufolder of results/leave_one_out/official
# store the macro of each debate in a dictionary whose key is the concatenation of the subfolder name and each subfolder name 

import os
import sys
import numpy as np
import pandas as pd
import json
from scipy.stats import wilcoxon


# load dataset
project_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

results_dir = os.path.join(project_dir, 'results', 'leave_one_out', 'official')

# get the subfolders of results/leave_one_out/official
subfolders = [f.path for f in os.scandir(results_dir) if f.is_dir()]

# get the name of the subfolders
subfolders_names = [f.name for f in os.scandir(results_dir) if f.is_dir()]

# get the name of the subfolder of the subfolders
subsubfolders_names = []
for subfolder in subfolders:
    subsubfolders_names.append([f.name for f in os.scandir(subfolder) if f.is_dir()])

print(subsubfolders_names)
print(subfolders_names)

# for each results.json file in the subsubfolders, get the macro for each debate and store it in a dictionary whose key is the concatenation of the subfolder name and its parent subfolder name and the value is a list of the macro for each debate
results = {}
for i in range(len(subsubfolders_names)):
    for j in range(len(subsubfolders_names[i])):
        path = os.path.join(subfolders[i], subsubfolders_names[i][j], 'results.json')
        macro = []
        if os.path.exists(path):
           # read json as dictionary 
            with open(path) as json_file:
                data = json.load(json_file)
                for key, value in data.items():
                    if key != 'mean' and key != 'std':
                        macro.append(value["macro avg"])
                #print(data)
                #print(data['macro'])
                #print(subfolders_names[i] + subsubfolders_names[i][j])
                results[subfolders_names[i] + "_" +  subsubfolders_names[i][j]] = macro


# for each key in dictionary whose name starts with 'text_audio' that the third string and search for the corresponding element in the dictionary whose name starts with 'text_only' and whose the third string is equal to the other and print them 
wilcoxon_results = {}
for key in results.keys():
    if key.startswith('text_audio'):
        for key2 in results.keys():
            if key2.startswith('text_only'):
                if key.split('_')[2] == key2.split('_')[2]:
                    wilcoxon_results[key + "_" + key2] = wilcoxon(results[key], results[key2])


# print the results and say also that if the p_value is less than 0.05 then the difference is significant
for key, value in wilcoxon_results.items():
    print(key, value)
    if value.pvalue < 0.05:
        print("The difference is significant")
    else:
        print("The difference is not significant")
    print()


# for bert_text_audio and bert_text_only store the macro in a tuple containing the name of the debate and the macro by reading them from the results.json file
results_debates = {}
for i in range(len(subsubfolders_names)):
    for j in range(len(subsubfolders_names[i])):
        path = os.path.join(subfolders[i], subsubfolders_names[i][j], 'results.json')
        macro = []
        debate = []
        if os.path.exists(path):
           # read json as dictionary 
            with open(path) as json_file:
                data = json.load(json_file)
                for key, value in data.items():
                    if key != 'mean' and key != 'std':
                        macro.append(value["macro avg"])
                    debate.append(key)
                #print(data)
                #print(data['macro'])
                #print(subfolders_names[i] + subsubfolders_names[i][j])
                # pair the debate with the macro
                results_debates[subfolders_names[i] + "_" +  subsubfolders_names[i][j]] = list(zip(debate, macro))

print(results_debates)
# get the values of the macro for each debate for bert_text_audio and bert_text_only and compute the wilcoxon test for each debate 
# knowing that the first element of the tuple is the name of the debate and the second is the macro
wilcoxon_results_debates = {}
for key, value in results_debates.items():
    if key.startswith('text_audio_bert'):
        for key2, value2 in results_debates.items():
            if key2.startswith('text_only_bert'):
                if key.split('_')[2] == key2.split('_')[2]:
                    for tup in value:
                        for tup2 in value2:
                            if tup[0] == tup2[0]:
                                print(tup[1], tup2[1])
                                try:
                                    wilcoxon_results_debates[tup[0] + "_" + key + "_" + key2] = wilcoxon([tup[1]], [tup2[1]])
                                except ValueError:
                                    wilcoxon_results_debates[tup[0] + "_" + key + "_" + key2] = 1

    # print the results and say also that if the p_value is less than 0.05 then the difference is significant
    for key, value in wilcoxon_results_debates.items():
        print(key, value)
        if value != 1: 
            if value.pvalue < 0.05:
                print("The difference is significant")
            else:
                print("The difference is not significant")
        else:
            print("The difference is not significant")

        print()


# get the values of the macro for each debate for bert_text_audio and bert_text_only and generate a markdown table where the first column if the debate, the second is the macro for bert_text_only and the third is the macro for bert_text_audio
# knowing that the first element of the tuple is the name of the debate and the second is the macro
table = {}
for key, value in results_debates.items():
    if key.startswith('text_audio_bert'):
        for key2, value2 in results_debates.items():
            if key2.startswith('text_only_bert'):
                if key.split('_')[2] == key2.split('_')[2]:
                    for tup in value:
                        for tup2 in value2:
                            if tup[0] == tup2[0]:
                                print(tup[1], tup2[1])
                                table[tup[0]] = [tup[1], tup2[1]]

# export table as markdown
df = pd.DataFrame.from_dict(table, orient='index', columns=['bert_text_audio', 'bert_text_only'])
save_path = os.path.join(project_dir, 'results', 'leave_one_out', 'official', 'results_table.md')
df.to_markdown(save_path)

