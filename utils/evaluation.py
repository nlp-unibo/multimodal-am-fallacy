import tensorflow as tf
from sklearn.metrics import classification_report
import os

def toLabels(data):
    ypred = tf.argmax(data, axis = 1)
    return ypred

def evaluate_model(model, test_data):
  Xtest, ytest = test_data
  y_pred = model.predict(Xtest)
  ypred = toLabels(y_pred)
  print("YPred", ypred)
  print("Ytest", ytest)
  #ytest = toLabels(ytest)
  # with open('/home/alex/data2/auditory-fallacies-test/auditory-fallacies/test_bert_text/results/reports_sentences_with_cw_rep' + str(repetition) +'encplus'+ '_pre_pre'+ '.txt', 'w') as f:
  #   f.write(classification_report(ytest, y_pred = ypred)+ "\n")
  cr = classification_report(ytest, y_pred=ypred)
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


