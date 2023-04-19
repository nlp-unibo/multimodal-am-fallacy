import tensorflow as tf
from utils.metrics import f1_m, precision_m, recall_m
from sklearn.utils.class_weight import compute_class_weight


def compile_model(model, lr=0.0001, loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy', f1_m]):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=metrics)
    return model

def train_model(model, train_dataset, val_dataset, epochs=100, batch_size=8, callbacks=None, use_class_weights=False):
    Xtrain = train_dataset[0]
    ytrain = train_dataset[1]

    print(type(Xtrain), type(ytrain))
    if callbacks == 'early_stopping':
        callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5,
                                                    restore_best_weights=True)

    if use_class_weights:
        weights = compute_class_weight(class_weight='balanced', classes=[0, 1, 2, 3, 4, 5], y=train_dataset[1])  #TODO: change this to the actual labels or to a generalizable code
        class_weights = {0: weights[0], 1: weights[1], 2: weights[2], 3: weights[3], 4: weights[4], 5: weights[5]}
        model.fit(Xtrain, ytrain, validation_data=val_dataset, epochs=epochs, batch_size=batch_size, callbacks=callbacks, class_weight=class_weights)
    else:
        model.fit(Xtrain, ytrain, validation_data=val_dataset, epochs=epochs, batch_size=batch_size, callbacks=callbacks)