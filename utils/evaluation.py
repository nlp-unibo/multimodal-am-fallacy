import tensorflow as tf
from sklearn.metrics import classification_report
import os
import numpy as np
import json
import pandas as pd

def toLabels(data):
    ypred = tf.argmax(data, axis = 1)
    return ypred

def evaluate_model(model, test_data, cross_val=False):
    Xtest, ytest = test_data
    y_pred = model.predict(Xtest)
    ypred = toLabels(y_pred)
    #print("YPred", ypred)
    #print("Ytest", ytest)
    #ytest = toLabels(ytest)
    # with open('/home/alex/data2/auditory-fallacies-test/auditory-fallacies/test_bert_text/results/reports_sentences_with_cw_rep' + str(repetition) +'encplus'+ '_pre_pre'+ '.txt', 'w') as f:
    #   f.write(classification_report(ytest, y_pred = ypred)+ "\n")
    if cross_val==True:
        cr = classification_report(ytest, y_pred=ypred, output_dict=True)
    else:
        cr = classification_report(ytest, y_pred=ypred, output_dict=False)
        print(cr + "\n")
    return cr


def save_results(run_path, cr):
    #path = os.path.join(project_dir, 'results')
    # save classification report as .json file

    results_path = os.path.join(run_path, 'metrics')
    results_filepath = os.path.join(results_path, 'classification_report.json')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with open(results_filepath, 'w') as f:
        f.write(cr)


def avg_results_cross_validation(results, project_dir, validation_strategy = 'cross_val', config = 'text_only', save_results = False):
    # label_0_f1 = []
    # label_1_f1 = []
    # label_2_f1 = []
    # label_3_f1 = []
    # label_4_f1 = []
    # label_5_f1 = []
    # accuracy = []
    # macro_avg = []
    # weighted_avg = []

    # compute the average precision for each label among the folds
    # build a dataframe where each row is a label (0, 1, 2, 3, 4, 5) and each column is a fold
    # compute the mean of each row

    labels = ['0', '1', '2', '3', '4', '5', 'accuracy', 'macro avg', 'weighted avg']
    int_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    folds = results.keys()
    df = pd.DataFrame(index = int_labels, columns=folds)
    df = df.rename(index={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5:'5', 6:'accuracy', 7:'macro avg', 8:'weighted avg'})
    # fill in the dataframe with the f1-score for each label in each fold when available
    for fold in folds:
        for label in labels:
            if label in results[fold].keys():
                print(label)
                if label != 'accuracy': 
                    #print(results[fold][label]['f1-score'])
                    df.loc[label, fold] = round(results[fold][label]['f1-score'],4)
                else:
                    #print(results[fold][label])
                    df.loc[label, fold] = round(results[fold][label], 4)

    #print(df.head())
    
    # compute the mean of each row
    df['mean'] = df.mean(axis=1)

    # compute the standard deviation of each row
    df['std'] = df.std(axis=1)

    #print(df.head())

    # print(results.items())
    # for k, v in results.items():
    #     label_0_f1.append(v['0']['f1-score'])
    #     label_1_f1.append(v['1']['f1-score'])
    #     label_2_f1.append(v['2']['f1-score'])
    #     label_3_f1.append(v['3']['f1-score'])
    #     label_4_f1.append(v['4']['f1-score'])
    #     label_5_f1.append(v['5']['f1-score'])
    #     accuracy.append(v['accuracy'])
    #     macro_avg.append(v['macro avg']['f1-score'])
    #     weighted_avg.append(v['weighted avg']['f1-score'])

    # print('Label 0 F1: ', np.mean(label_0_f1))
    # print('Label 1 F1: ', np.mean(label_1_f1))
    # print('Label 2 F1: ', np.mean(label_2_f1))
    # print('Label 3 F1: ', np.mean(label_3_f1))
    # print('Label 4 F1: ', np.mean(label_4_f1))
    # print('Label 5 F1: ', np.mean(label_5_f1))
    # print('Accuracy: ', np.mean(accuracy))
    # print('Macro Avg: ', np.mean(macro_avg))
    # print('Weighted Avg: ', np.mean(weighted_avg))

    # # Store all the means in a new dictionary
    # mean_results = {'label_0_f1': np.mean(label_0_f1),
    #                 'label_1_f1': np.mean(label_1_f1),
    #                 'label_2_f1': np.mean(label_2_f1),
    #                 'label_3_f1': np.mean(label_3_f1),
    #                 'label_4_f1': np.mean(label_4_f1),
    #                 'label_5_f1': np.mean(label_5_f1),
    #                 'accuracy': np.mean(accuracy),
    #                 'macro_avg': np.mean(macro_avg),
    #                 'weighted_avg': np.mean(weighted_avg)}
    
    # # Store the stdev of the results in a new dictionary
    # stdev_results = {'label_0_f1': np.std(label_0_f1),
    #                 'label_1_f1': np.std(label_1_f1),
    #                 'label_2_f1': np.std(label_2_f1),
    #                 'label_3_f1': np.std(label_3_f1),
    #                 'label_4_f1': np.std(label_4_f1),
    #                 'label_5_f1': np.std(label_5_f1),
    #                 'accuracy': np.std(accuracy),
    #                 'macro_avg': np.std(macro_avg),
    #                 'weighted_avg': np.std(weighted_avg)}
    
    if save_results == True:
        # save mean_results as a json file in the results folder
        # create a 'cross_validation' folder in the results folder if it doesn't exist
        results_path = os.path.join(project_dir, 'results')
        if validation_strategy == 'cross_val':
            cross_validation_path = os.path.join(results_path, 'cross_validation')
        elif validation_strategy == 'leave_one_out':
            cross_validation_path = os.path.join(results_path, 'leave_one_out')

        if not os.path.exists(cross_validation_path):
            os.makedirs(cross_validation_path)

        # create a folder for the current configuration if it doesn't exist
        config_path = os.path.join(cross_validation_path, config)
        if not os.path.exists(config_path):
            os.makedirs(config_path)
        
        results_filepath_json = os.path.join(config_path, 'results.json')
        results_filepath_csv = os.path.join(config_path, 'results.csv')
        # save df as a json file in the results folder
        df.to_json(results_filepath_json, indent=4)

        # save df as a csv file in the results folder
        df.to_csv(results_filepath_csv, index=True)


        # results_filepath_mean = os.path.join(config_path, 'mean_results.json') 
        # results_filepath_stdev = os.path.join(config_path, 'stdev_results.json')

        # with open(results_filepath_mean, 'w') as f:
        #     json.dump(mean_results, f, indent=4)
        
        # with open(results_filepath_stdev, 'w') as f:
        #     json.dump(stdev_results, f, indent=4)





