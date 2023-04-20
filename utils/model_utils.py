import tensorflow as tf
from utils.metrics import f1_m, precision_m, recall_m
from sklearn.utils.class_weight import compute_class_weight
import os
import json


def compile_model(model, lr=0.0001, loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy', f1_m]):
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr), loss=loss, metrics=metrics)
    return model

def train_model(model, train_dataset, val_dataset, epochs=100, batch_size=8, callbacks=None, use_class_weights=False):
    Xtrain = train_dataset[0]
    ytrain = train_dataset[1]

    if callbacks == 'early_stopping':
        callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5,
                                                    restore_best_weights=True)

    if use_class_weights:
        weights = compute_class_weight(class_weight='balanced', classes=[0, 1, 2, 3, 4, 5], y=train_dataset[1])  #TODO: change this to the actual labels or to a generalizable code
        class_weights = {0: weights[0], 1: weights[1], 2: weights[2], 3: weights[3], 4: weights[4], 5: weights[5]}
        model.fit(Xtrain, ytrain, validation_data=val_dataset, epochs=epochs, batch_size=batch_size, callbacks=callbacks, class_weight=class_weights)
    else:
        model.fit(Xtrain, ytrain, validation_data=val_dataset, epochs=epochs, batch_size=batch_size, callbacks=callbacks)


def save_model(project_dir, model, config):

    # create dir in results if it doesn't exist with the config name
    path = os.path.join(project_dir, 'results', config)
    if not os.path.exists(path):
        os.makedirs(path)

    # check if the folder with the name config contain a run_1 folder
    # if there isn't one, create it
    # if there is one, iterate over the folders and store the last char of the folder name
    # then create a new folder with the last char + 1

    # get the list of folders in the config folder
    folders = os.listdir(path)
    folders_indexs = []
    # if the list is empty, create a folder with name run_1
    if len(folders) == 0:
        run_path = os.path.join(path, 'run_1')
        os.makedirs(run_path)
    else:
        for i in range(len(folders)):
            folders[i] = folders[i].replace('run_', '')
            folders_indexs.append(int(folders[i]))
        folders_indexs.sort()
        run_path = os.path.join(path, 'run_' + str(folders_indexs[-1] + 1))
        os.makedirs(run_path)

    weights_path = os.path.join(run_path, 'weights')
    os.makedirs(weights_path)

    #model.save(run_path) # save the model
    model.save_weights(weights_path + '/weights.h5') # save the weights
    print("Model saved successfully!")

    return run_path

def save_config(run_path, param_dict):
    # create a folder named 'configuration'
    config_path = os.path.join(run_path, 'configuration')
    os.makedirs(config_path)

    # save the dictionary param_dict in a json file with indent = 4
    with open(config_path + '/config.json', 'w') as f:
        json.dump(param_dict, f, indent=4)
