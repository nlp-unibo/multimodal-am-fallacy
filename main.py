from utils import data_loader, converter, model_implementation, model_utils, reproducibility
import os
import numpy as np

# Reproducibility settings
reproducibility.set_reproducibility()

project_dir = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))

df = data_loader.load(project_dir)
train_audio_files, val_audio_files, test_audio_files = data_loader.load_audio(df, project_dir) # snippet audio files

Xtrain, ytrain = data_loader.get_sentences(split = 'train', df=df)
Xval, yval= data_loader.get_sentences(split = 'val', df=df)
Xtest,  ytest = data_loader.get_sentences(split = 'test', df=df)

encoded_Xtrain, y_train, max_sentence_len = converter.prepare_text_data(Xtrain, ytrain, is_train=True, q=0.99) # 0.99 is the quantile for the max sentence length (99% of the sentences are shorter than this length
encoded_audio_train, max_frame_len = converter.prepare_audio_data(train_audio_files, model_type='embedding', is_train=True) # modify to have train, val and test audio files

encoded_Xval, y_val = converter.prepare_text_data(Xtrain, ytrain, q=0.99, maxSentenceLen=max_sentence_len)
encoded_audio_val = converter.prepare_audio_data(train_audio_files, model_type='embedding')

encoded_Xtest, y_test = converter.prepare_text_data(Xtrain, ytrain, q=0.99, maxSentenceLen=max_sentence_len)
encoded_audio_test = converter.prepare_audio_data(train_audio_files, model_type='embedding')

# get the number of labels
n_labels = data_loader.get_num_labels(ytrain)

# Instatiate the model
config = 'text_audio'
model = model_implementation.bert_model(num_labels=n_labels,
                         config=config,
                         is_bert_trainable=False,
                         max_sentence_len=max_sentence_len,
                         max_frame_len=max_frame_len,
                         )
# Build model input
if config == 'text_audio':
    train_data = ([encoded_Xtrain,encoded_audio_train], y_train)
    val_data  = ([encoded_Xval, encoded_audio_val], y_val)
    test_data = ([encoded_Xtest, encoded_audio_test], y_test)

elif config == 'text_only':
    train_data = (encoded_Xtrain, y_train)
    val_data = (encoded_Xval, y_val)
    test_data = (encoded_Xtest, y_test)

else:
    train_data = (encoded_audio_train, y_train)
    val_data = (encoded_audio_val, y_val)
    test_data = (encoded_audio_test, y_test)

# Compile Model
model_utils.compile_model(model)

# Train Model
model_utils.train_model(model, train_data, val_data, epochs=100, batch_size=8, callbacks='early_stopping', use_class_weights=True)








