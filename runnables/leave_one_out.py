from utils import data_loader, converter, model_implementation, model_utils, reproducibility, evaluation
import os
import numpy as np

# WARNING: comment the following line if you are not using CUDA
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# Reproducibility settings
seed = reproducibility.set_reproducibility()

#project_dir = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
project_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


df = data_loader.load(project_dir)

# get unique dialogue IDs
dialogue_ids = df['Dialogue ID'].unique()

results = {} # dictionary to store the results as fold : dict of results

# iterate over dialogue IDs
for dialogue_id in dialogue_ids:
    # mark the 'Split' column as 'Test' for the current dialogue ID
    df.loc[df['Dialogue ID'] == dialogue_id, 'Split'] = 'Test'
    # create a list of dialogue IDs that are not the current one
    train_dialogue_ids = [id for id in dialogue_ids if id != dialogue_id]
    # mark the 'Split' column as 'Train' for the remaining dialogue IDs
    df.loc[df['Dialogue ID'].isin(train_dialogue_ids), 'Split'] = 'Train'

    train_audio_files, indexes_train, val_audio_files, indexes_val, test_audio_files, indexes_test = data_loader.load_audio(df, project_dir)  # snippet audio files

    Xtrain, ytrain = data_loader.get_sentences(split='train', df=df)
    #Xval, yval = data_loader.get_sentences(split='val', df=df)
    Xtest, ytest = data_loader.get_sentences(split='test', df=df)

    encoded_Xtrain, y_train, max_sentence_len = converter.prepare_text_data(Xtrain, ytrain, is_train=True,
                                                                            q=0.99)  # 0.99 is the quantile for the max sentence length (99% of the sentences are shorter than this length
    # max_sentence_len = 512 # set the max sentence length to 512 TO BE REMOVED
    encoded_audio_train, max_frame_len = converter.prepare_audio_data(train_audio_files, model_type='embedding',
                                                                      is_train=True)  # modify to have train, val and test audio files

    #encoded_Xval, y_val = converter.prepare_text_data(Xval, yval, q=0.99, maxSentenceLen=max_sentence_len)
    #encoded_audio_val = converter.prepare_audio_data(val_audio_files, model_type='embedding')

    encoded_Xtest, y_test = converter.prepare_text_data(Xtest, ytest, q=0.99, maxSentenceLen=max_sentence_len)
    encoded_audio_test = converter.prepare_audio_data(test_audio_files, model_type='embedding')

    # get the number of labels
    n_labels = data_loader.get_num_labels(ytrain)

    # config_list = ['audio_only', 'text_only', 'text_audio']
    config_list = ['text_audio']
    config_params = {'epochs': 500,
                     'batch_size': 8,
                     'callbacks': 'early_stopping',
                     'use_class_weights': True,
                     'seed': seed,
                     'lr': 5e-05}

    # Build model input for each configuration
    for config in config_list:

        config_params['config'] = config

        # Instatiate the model
        model = model_implementation.bert_model(num_labels=n_labels,
                                                config=config,
                                                is_bert_trainable=False,
                                                max_sentence_len=max_sentence_len,
                                                max_frame_len=max_frame_len,
                                                )
        # Multimodal
        if config == 'text_audio':
            train_data = ([encoded_Xtrain, encoded_audio_train], y_train)
            #val_data = ([encoded_Xval, encoded_audio_val], y_val)
            test_data = ([encoded_Xtest, encoded_audio_test], y_test)

        # Text
        elif config == 'text_only':
            print(encoded_Xtrain[0].shape)
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
        model_utils.train_model(model, train_data, train_data, #TODO: check if train_data, train_data is correct
                                epochs=config_params['epochs'],
                                batch_size=config_params['batch_size'],
                                callbacks=config_params['callbacks'],
                                use_class_weights=config_params['use_class_weights'])

        cr = evaluation.evaluate_model(model, test_data, cross_val=True)
        results[dialogue_id] = cr

# Save Cross Validation Results
# TODO: save results of crossval for each config mode (text, audio, text_audio)
evaluation.avg_results_cross_validation(results, project_dir, validation_strategy= 'leave_one_out',  save_results=True)