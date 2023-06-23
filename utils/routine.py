from utils import data_loader, converter, model_implementation, model_utils, reproducibility, evaluation
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier




def leave_one_out(dialogue_ids, df, project_dir, run_path, config, config_params, text_model, audio_model, sample_rate):
    results = {} # dictionary to store the results as fold : dict of results
    n_dial = 0 # counter for the number of dialogues
    # create empty dataframe predictions to store the predictions for each fold
    predictions = pd.DataFrame(columns=['Dialogue ID', 'SentenceSnippet', 'Snippet', 'y_pred', 'y_true', 'y_pred_encoded', 'y_true_encoded'])

    # iterate over dialogue IDs
    for dialogue_id in dialogue_ids:
        # if n_dial == 2:
        #      break
        print("Running cross validation for dialogue ID: {}".format(dialogue_id))
        # print some separation lines
        print("--------------------------------------------------")
        # mark the 'Split' column as 'Test' for the current dialogue ID
        df.loc[df['Dialogue ID'] == dialogue_id, 'Split'] = 'Test'
        # create a list of dialogue IDs that are not the current one
        train_dialogue_ids = [id for id in dialogue_ids if id != dialogue_id]
        # mark the 'Split' column as 'Train' for the remaining dialogue IDs
        df.loc[df['Dialogue ID'].isin(train_dialogue_ids), 'Split'] = 'Train'

        train_audio_files, indexes_train, val_audio_files, indexes_val, test_audio_files, indexes_test = data_loader.load_audio(df, project_dir)  # snippet audio files


        Xtrain, train_snippet, ytrain = data_loader.get_sentences(split='train', df=df)
        Xtest, test_snippet, ytest = data_loader.get_sentences(split='test', df=df)

        train_sentence_snippet = Xtrain
        test_sentence_snippet = Xtest

        encoded_Xtrain, y_train, max_sentence_len = converter.prepare_text_data(Xtrain, ytrain, text_model=text_model, is_train=True,
                                                                                q=0.99)  # 0.99 is the quantile for the max sentence length (99% of the sentences are shorter than this length
        if config != 'text_only':
            encoded_audio_train, max_frame_len = converter.prepare_audio_data(train_audio_files, model_type='embedding', audio_model=audio_model,
                                                                        audio_model_sample_rate=sample_rate, is_train=True)  # modify to have train, val and test audio files

    
        encoded_Xtest, y_test = converter.prepare_text_data(Xtest, ytest, text_model=text_model, q=0.99, maxSentenceLen=max_sentence_len)
        
        if config != 'text_only':
            encoded_audio_test = converter.prepare_audio_data(test_audio_files, model_type='embedding', audio_model=audio_model, audio_model_sample_rate=sample_rate)

        # get the number of labels
        n_labels = data_loader.get_num_labels(ytrain)


        # Instatiate the model
        if text_model == 'bert':
            model = model_implementation.bert_model(num_labels=n_labels,
                                                    config=config,
                                                    is_trainable= config_params['is_text_model_trainable'],
                                                    max_sentence_len=max_sentence_len,
                                                    max_frame_len=config_params['max_frame_len'],
                                                    )
        elif text_model == 'roberta':
            model = model_implementation.roberta_model(num_labels=n_labels,
                                                    config=config,
                                                    is_trainable=config_params['is_text_model_trainable'],
                                                    max_sentence_len=max_sentence_len,
                                                    max_frame_len=config_params['max_frame_len'],
                                                    )
        elif text_model == 'sbert':
            model = model_implementation.sbert_model(num_labels=n_labels,
                                                    config=config,
                                                    is_trainable=config_params['is_text_model_trainable'],
                                                    max_sentence_len=max_sentence_len,
                                                    max_frame_len=config_params['max_frame_len'],
                                                    )


        #print(model.summary())
        # Multimodal
        if config == 'text_audio':
            train_data = ([encoded_Xtrain, encoded_audio_train], y_train)
            #val_data = ([encoded_Xval, encoded_audio_val], y_val)
            test_data = ([encoded_Xtest, encoded_audio_test], y_test)

        # Text
        elif config == 'text_only':
            #print(encoded_Xtrain[0].shape)
            train_data = (encoded_Xtrain, y_train)
            #val_data = (encoded_Xval, y_val)
            test_data = (encoded_Xtest, y_test)

        # Audio
        else:
            train_data = (encoded_audio_train, y_train)
            #val_data = (encoded_audio_val, y_val)
            test_data = (encoded_audio_test, y_test)

        # Compile Model
        model_utils.compile_model(model, lr=config_params['lr'])
        seed = reproducibility.set_reproducibility()

        # Train Model
        trained_model = model_utils.train_model(model, train_data, train_data, #TODO: check if train_data, train_data is correct
                                epochs=config_params['epochs'],
                                batch_size=config_params['batch_size'],
                                callbacks=config_params['callbacks'],
                                use_class_weights=config_params['use_class_weights'])

        cr = evaluation.evaluate_model(trained_model, test_data, cross_val=True)
        print(cr)
        #print("Cross Validation Results for dialogue ID: {}".format(dialogue_id))
        #print(cr)
        results[dialogue_id] = cr
    
        # make predictions
        ypred_encoded, ytrue_encoded, ypred, ytrue= evaluation.make_predictions(trained_model, test_data)

        # save predictions
        predictions = evaluation.save_predictions(run_path, predictions, dialogue_id, test_snippet, test_sentence_snippet, ypred_encoded, ytrue_encoded, ypred, ytrue)

        n_dial+=1 # increment the counter
    return results # return the results dictionary





def leave_one_out_baselines(dialogue_ids, df, project_dir, run_path, config, config_params, model):
    results = {} # dictionary to store the results as fold : dict of results
    n_dial = 0 # counter for the number of dialogues
    # create empty dataframe predictions to store the predictions for each fold
    predictions = pd.DataFrame(columns=['Dialogue ID', 'SentenceSnippet', 'Snippet', 'y_pred', 'y_true', 'y_pred_encoded', 'y_true_encoded'])

    # iterate over dialogue IDs
    for dialogue_id in dialogue_ids:
        # if n_dial == 2:
        #      break
        print("Running cross validation for dialogue ID: {}".format(dialogue_id))
        # print some separation lines
        print("--------------------------------------------------")
        # mark the 'Split' column as 'Test' for the current dialogue ID
        df.loc[df['Dialogue ID'] == dialogue_id, 'Split'] = 'Test'
        # create a list of dialogue IDs that are not the current one
        train_dialogue_ids = [id for id in dialogue_ids if id != dialogue_id]
        # mark the 'Split' column as 'Train' for the remaining dialogue IDs
        df.loc[df['Dialogue ID'].isin(train_dialogue_ids), 'Split'] = 'Train'

        train_audio_files, indexes_train, val_audio_files, indexes_val, test_audio_files, indexes_test = data_loader.load_audio(df, project_dir)  # snippet audio files


        Xtrain, train_snippet, ytrain = data_loader.get_sentences(split='train', df=df)
        Xtest, test_snippet, ytest = data_loader.get_sentences(split='test', df=df)

        train_sentence_snippet = Xtrain
        test_sentence_snippet = Xtest

        ytrain = converter.encode_labels(ytrain)
        ytest = converter.encode_labels(ytest)

        test_data = (ytest, ytest)

        # Compile Model
        seed = reproducibility.set_reproducibility()
        dummy_clf = DummyClassifier(strategy=model, random_state=seed)
        

        # Train Model
        trained_model = dummy_clf.fit(ytrain, ytrain)
          

        cr = evaluation.evaluate_baseline(trained_model, test_data, cross_val=True)
        print(cr)
        #print("Cross Validation Results for dialogue ID: {}".format(dialogue_id))
        #print(cr)
        results[dialogue_id] = cr
    
        # make predictions
        ypred_encoded, ytrue_encoded, ypred, ytrue = evaluation.make_predictions_baseline(trained_model, test_data)

        # save predictions
        predictions = evaluation.save_predictions(run_path, predictions, dialogue_id, test_snippet, test_sentence_snippet, ypred_encoded, ytrue_encoded, ypred, ytrue)

        n_dial+=1 # increment the counter
    return results # return the results dictionary