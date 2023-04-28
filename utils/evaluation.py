import tensorflow as tf
from sklearn.metrics import classification_report
import os
import numpy as np
import json

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


def avg_results_cross_validation(results, project_dir, save_results = False):
    label_0_f1 = []
    label_1_f1 = []
    label_2_f1 = []
    label_3_f1 = []
    label_4_f1 = []
    label_5_f1 = []
    accuracy = []
    macro_avg = []
    weighted_avg = []

    # compute the average precision for each label among the folds

    for k, v in results.items():
        label_0_f1.append(v['0']['f1-score'])
        label_1_f1.append(v['1']['f1-score'])
        label_2_f1.append(v['2']['f1-score'])
        label_3_f1.append(v['3']['f1-score'])
        label_4_f1.append(v['4']['f1-score'])
        label_5_f1.append(v['5']['f1-score'])
        accuracy.append(v['accuracy'])
        macro_avg.append(v['macro avg']['f1-score'])
        weighted_avg.append(v['weighted avg']['f1-score'])

    print('Label 0 F1: ', np.mean(label_0_f1))
    print('Label 1 F1: ', np.mean(label_1_f1))
    print('Label 2 F1: ', np.mean(label_2_f1))
    print('Label 3 F1: ', np.mean(label_3_f1))
    print('Label 4 F1: ', np.mean(label_4_f1))
    print('Label 5 F1: ', np.mean(label_5_f1))
    print('Accuracy: ', np.mean(accuracy))
    print('Macro Avg: ', np.mean(macro_avg))
    print('Weighted Avg: ', np.mean(weighted_avg))

    # Store all the means in a new dictionary
    mean_results = {'label_0_f1': np.mean(label_0_f1),
                    'label_1_f1': np.mean(label_1_f1),
                    'label_2_f1': np.mean(label_2_f1),
                    'label_3_f1': np.mean(label_3_f1),
                    'label_4_f1': np.mean(label_4_f1),
                    'label_5_f1': np.mean(label_5_f1),
                    'accuracy': np.mean(accuracy),
                    'macro_avg': np.mean(macro_avg),
                    'weighted_avg': np.mean(weighted_avg)}
    if save_results == True:
        # save mean_results as a json file in the results folder
        # create a 'cross_validation' folder in the results folder if it doesn't exist
        results_path = os.path.join(project_dir, 'results')
        cross_validation_path = os.path.join(results_path, 'cross_validation')
        if not os.path.exists(cross_validation_path):
            os.makedirs(cross_validation_path)

        results_filepath = os.path.join(cross_validation_path, 'mean_results.json')
        with open(results_filepath, 'w') as f:
            json.dump(mean_results, f)


