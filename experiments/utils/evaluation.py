import tensorflow as tf
from sklearn.metrics import classification_report
import os
import numpy as np
import json
import pandas as pd
from utils import converter 

def toLabels(data):
    ypred = tf.argmax(data, axis = 1)
    return ypred

def toEncodedLabels(data):
    encode_labels = []
    lbls = {
        'AppealtoEmotion': 0,
        'AppealtoAuthority': 1,
        'AdHominem': 2,
        'FalseCause': 3,
        'Slipperyslope': 4,
        'Slogans': 5

    }

    # check if y is equal to the value of lbls and add the key to the list
    for i in range(len(data)):
        for key, value in lbls.items():
            if data[i] == value:
                encode_labels.append(key)
    return encode_labels

def evaluate_model(model, test_data, cross_val=False):
    Xtest, ytest = test_data
    y_pred = model.predict(Xtest)
    ypred = toLabels(y_pred)

    if cross_val==True:
        cr = classification_report(ytest, y_pred=ypred, output_dict=True)
    else:
        cr = classification_report(ytest, y_pred=ypred, output_dict=False)
        print(cr + "\n")
    return cr


def evaluate_baseline(model, test_data, cross_val=False):
    Xtest, ytest = test_data
    ypred = model.predict(ytest)
    print(ypred)
    #ypred = toLabels(y_pred)
    print(ytest)

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


def avg_results_cross_validation(results, run_path, validation_strategy = 'cross_val', config = 'text_only', save_results = False):

    labels = ['0', '1', '2', '3', '4', '5', 'accuracy', 'macro avg', 'weighted avg']
    int_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    folds = results.keys()
    df = pd.DataFrame(index = int_labels, columns=folds)
    df = df.rename(index={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5:'5', 6:'accuracy', 7:'macro avg', 8:'weighted avg'})
    # fill in the dataframe with the f1-score for each label in each fold when available
    for fold in folds:
        for label in labels:
            if label in results[fold].keys():
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

    print(df.head())

    if save_results == True:

        results_filepath_json = os.path.join(run_path, 'results.json')
        results_filepath_csv = os.path.join(run_path, 'results.csv')
        # save df as a json file in the results folder
        df.to_json(results_filepath_json, indent=4)

        # save df as a csv file in the results folder
        df.to_csv(results_filepath_csv, index=True)





def make_predictions(model, test_data): 
    # make predictions on the test set
    Xtest, ytest = test_data
    y_pred = model.predict(Xtest)  
    ypred = toLabels(y_pred) 

    # convert the encoded labels to the original labels
    ypred_encoded = toEncodedLabels(ypred)
    ytest_encoded = toEncodedLabels(ytest)

    return ypred_encoded, ytest_encoded, ypred, ytest

def make_predictions_baseline(model, test_data): 
    # make predictions on the test set
    Xtest, ytest = test_data
    ypred = model.predict(ytest)  
    #ypred = toLabels(y_pred) 
    

    # convert the encoded labels to the original labels
    ypred_encoded = toEncodedLabels(ypred)
    ytest_encoded = toEncodedLabels(ytest)




    return ypred_encoded, ytest_encoded, ypred, ytest




def save_predictions(run_path, df_results, debate_id, test_snippet, test_sentence_snippet,  ypred_encoded, ytrue_encoded, ypred, ytrue):
    # add the predictions to the dataframe by saving also the snippet and the sentence snippet and the debate id

    #if df_results is empty, associate the columns to the dataframe, otherwise append the new predictions to the dataframe
    if df_results.empty:


        df_results['Snippet'] = test_snippet
        df_results['SentenceSnippet'] = test_sentence_snippet
        df_results['Dialogue ID'] = debate_id
        df_results['y_pred_encoded'] = ypred_encoded
        df_results['y_true_encoded'] = ytrue_encoded
        df_results['y_pred'] = ypred
        df_results['y_true'] = ytrue
    else:

        # create a new dataframe with the new predictions and append it to the existing one
        df_new = pd.DataFrame(columns=['Dialogue ID', 'SentenceSnippet', 'Snippet', 'y_pred', 'y_true', 'y_pred_encoded', 'y_true_encoded'])
        df_new['Snippet'] = test_snippet
        df_new['SentenceSnippet'] = test_sentence_snippet
        df_new['Dialogue ID'] = debate_id
        df_new['y_pred_encoded'] = ypred_encoded
        df_new['y_true_encoded'] = ytrue_encoded
        df_new['y_pred'] = ypred
        df_new['y_true'] = ytrue

        df_results = pd.concat([df_results, df_new], ignore_index=True)

    
    # save predictions as .json file
    results_path = os.path.join(run_path, 'predictions')
    results_filepath_json = os.path.join(run_path, 'predictions.json')
    results_filepath_csv = os.path.join(run_path, 'predictions.csv')

    if not os.path.exists(run_path):
        os.makedirs(run_path)

    print(df_results.head())
    df_results.to_json(results_filepath_json, indent=4)
    df_results.to_csv(results_filepath_csv, index=True)

    return df_results
