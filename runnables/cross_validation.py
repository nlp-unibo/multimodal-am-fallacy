from utils import data_loader, converter, model_implementation, model_utils, reproducibility, evaluation
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold

# WARNING: comment the following line if you are not using CUDA
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# Reproducibility settings
seed = reproducibility.set_reproducibility()

#project_dir = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
project_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


df = data_loader.load(project_dir)
train_audio_files, val_audio_files, test_audio_files = data_loader.load_audio(df, project_dir) # snippet audio files

Xtrain, ytrain = data_loader.get_sentences(split = 'train', df=df)
Xval, yval= data_loader.get_sentences(split = 'val', df=df)
#Xtest,  ytest = data_loader.get_sentences(split = 'test', df=df)

# Concatenate train and val
X = np.concatenate((Xtrain, Xval), axis=0)
y = np.concatenate((ytrain, yval), axis=0)

# Cross validation
results = {} # dictionary to store the results as fold : dict of results
n_folds = 3
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
for i, (train, test) in enumerate(skf.split(X, y)):
    print("Running Fold", i+1, "/", n_folds)
    Xtrain, Xval = X[train], X[test]
    ytrain, yval = y[train], y[test]
    #print(np.unique(ytrain, return_counts=True), np.unique(ytest, return_counts=True))

    #print(Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape)

    encoded_Xtrain, y_train, max_sentence_len = converter.prepare_text_data(Xtrain, ytrain, is_train=True, q=0.99) # 0.99 is the quantile for the max sentence length (99% of the sentences are shorter than this length
    #max_sentence_len = 512 # set the max sentence length to 512 TO BE REMOVED
    encoded_audio_train, max_frame_len = converter.prepare_audio_data(train_audio_files, model_type='embedding', is_train=True) # modify to have train, val and test audio files

    encoded_Xval, y_val = converter.prepare_text_data(Xval, yval, q=0.99, maxSentenceLen=max_sentence_len)
    encoded_audio_val = converter.prepare_audio_data(val_audio_files, model_type='embedding')

    # encoded_Xtest, y_test = converter.prepare_text_data(Xtest, ytest, q=0.99, maxSentenceLen=max_sentence_len)
    # encoded_audio_test = converter.prepare_audio_data(test_audio_files, model_type='embedding')

    # get the number of labels
    n_labels = data_loader.get_num_labels(ytrain)


    #config_list = ['audio_only', 'text_only', 'text_audio']
    config_list = ['text_only']
    config_params = {'epochs': 1,
                    'batch_size': 8,
                    'callbacks': 'early_stopping',
                    'use_class_weights': True,
                    'seed': seed,
                    'lr':5e-05}

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
            train_data = ([encoded_Xtrain,encoded_audio_train], y_train)
            val_data  = ([encoded_Xval, encoded_audio_val], y_val)
            #test_data = ([encoded_Xtest, encoded_audio_test], y_test)

        # Text
        elif config == 'text_only':
            print(encoded_Xtrain[0].shape)
            train_data = (encoded_Xtrain, y_train)
            val_data = (encoded_Xval, y_val)
            #test_data = (encoded_Xtest, y_test)

        # Audio
        else:
            train_data = (encoded_audio_train, y_train)
            val_data = (encoded_audio_val, y_val)
            #test_data = (encoded_audio_test, y_test)

        # Compile Model
        model_utils.compile_model(model, lr=config_params['lr'])
        seed = reproducibility.set_reproducibility()

        # Train Model
        model_utils.train_model(model, train_data, val_data,
                                epochs=config_params['epochs'],
                                batch_size=config_params['batch_size'],
                                callbacks=config_params['callbacks'],
                                use_class_weights=config_params['use_class_weights'])

        #print(model.summary())

        # Evaluate Model
        cr = evaluation.evaluate_model(model, val_data, cross_val=True)
        results[i] = cr


# Save Cross Validation Results
evaluation.save_cross_validation_results(results, project_dir, save_results=True)






        # # Save Model
        # run_path = model_utils.save_model(project_dir, model, config)
        #
        # # Save Results
        # evaluation.save_results(run_path, cr)
        #
        # # Save Run Parameters
        # model_utils.save_config(run_path, config_params)












