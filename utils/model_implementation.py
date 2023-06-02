import tensorflow as tf
from transformers import TFBertModel, TFRobertaModel

maxBERTLen = 512
maxFrameLen = 768 # embedding shape wav2vec
#TODO: add code for bert trainable

def bert_model( num_labels, config='text_only', is_trainable=False, max_sentence_len=maxBERTLen, max_frame_len = maxFrameLen,  dropout_text=0.1, answer_units=100, audio_units=64, audio_l2=0.0005, dropout_audio=0.1):
    # Text
    if config != 'audio_only':

        input_ids = tf.keras.layers.Input(shape=(max_sentence_len,), dtype=tf.int32)
        token_type_ids = tf.keras.layers.Input(shape=(max_sentence_len,), dtype=tf.int32)
        attention_mask = tf.keras.layers.Input(shape=(max_sentence_len,), dtype=tf.int32)
        bertModel = TFBertModel.from_pretrained("bert-base-uncased")(input_ids, token_type_ids=token_type_ids,
                                                                     attention_mask=attention_mask)[-1] #-1 is the pooled output

        text_emb = tf.keras.layers.Dropout(rate=dropout_text)(bertModel)


    # Audio
    if config != 'text_only':
        audio_input = tf.keras.layers.Input(shape=(max_frame_len, ))
        audio_emb = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(audio_input)
        audio_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=audio_units,
                                                                        kernel_regularizer=tf.keras.regularizers.l2(
                                                                            l2=audio_l2)))(audio_emb)


        #audio_emb = tf.keras.layers.Reshape( [audio_lstm.shape[0], -1])(audio_lstm)
        audio_emb = tf.keras.layers.Dropout(rate=dropout_audio)(audio_lstm)


    # Text-Audio
    if config == 'text_audio':
        stacked_features = tf.concat((text_emb, audio_emb), axis=-1)

        answer = tf.keras.layers.Dense(units=answer_units)(stacked_features)
        dense = tf.keras.layers.Dense(units=answer_units/2)(answer)
        dense = tf.keras.layers.Dense(units=num_labels)(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer) #TODO: eventually remove it
        bn = tf.keras.layers.BatchNormalization()(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer)  # TODO: eventually remove it
        #bn = tf.keras.layers.BatchNormalization()(answer)
        
        #bn = tf.keras.layers.BatchNormalization()(dense)
        out = tf.keras.layers.Dense(num_labels, activation=tf.nn.relu)(tf.keras.layers.Dropout(0.1)(bn))
        model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask, audio_input], outputs=out)
        return model

    # Text
    if config == 'text_only':
        answer = tf.keras.layers.Dense(units=answer_units)(text_emb)
        dense = tf.keras.layers.Dense(units=answer_units/2)(answer)
        dense = tf.keras.layers.Dense(units=num_labels)(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer) #TODO: eventually remove it
        bn = tf.keras.layers.BatchNormalization()(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer)  # TODO: eventually remove it
        #bn = tf.keras.layers.BatchNormalization()(answer)
        out = tf.keras.layers.Dense(num_labels, activation=tf.nn.relu)(tf.keras.layers.Dropout(0.1)(bn))
        model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=out)

        return model

    # Audio
    if config == 'audio_only':
        answer = tf.keras.layers.Dense(units=answer_units)(audio_emb)
        dense = tf.keras.layers.Dense(units=answer_units/2)(answer)
        dense = tf.keras.layers.Dense(units=num_labels)(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer) #TODO: eventually remove it
        bn = tf.keras.layers.BatchNormalization()(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer)  # TODO: eventually remove it
        #bn = tf.keras.layers.BatchNormalization()(answer)

        #dense = tf.keras.layers.Dense(units=num_labels)(answer)  # TODO: eventually remove it
        #bn = tf.keras.layers.BatchNormalization()(dense)
        out = tf.keras.layers.Dense(num_labels, activation=tf.nn.relu)(tf.keras.layers.Dropout(0.1)(bn))
        model = tf.keras.Model(inputs=[audio_input], outputs=out)

        return model




def roberta_model(num_labels, config='text_only', is_trainable=True, max_sentence_len=maxBERTLen, max_frame_len = maxFrameLen,  dropout_text=0.1, answer_units=100, audio_units=64, audio_l2=0.0005, dropout_audio=0.1):
    # Text
    if config != 'audio_only':

        input_ids = tf.keras.layers.Input(shape=(max_sentence_len,), dtype=tf.int32)
        #token_type_ids = tf.keras.layers.Input(shape=(max_sentence_len,), dtype=tf.int32)
        attention_mask = tf.keras.layers.Input(shape=(max_sentence_len,), dtype=tf.int32)
        # robertaModel = TFRobertaModel.from_pretrained("roberta-base")(input_ids, attention_mask=attention_mask)
        # print(robertaModel)
        # print(robertaModel[1].shape)
        # print(robertaModel[-1].shape)
        # print(robertaModel[1][0].shape)
    
        robertaModel = TFRobertaModel.from_pretrained("roberta-base")(input_ids, attention_mask=attention_mask)[-1] #-1 is the pooled output
        # #reshaped_tensor = tf.reshape(robertaModel, shape=[-1, robertaModel.shape[0]])
        # expanded_tensor = tf.expand_dims(robertaModel, axis=0)
        # shape_list = expanded_tensor.shape.as_list()
        # shape_list = [None if dim == 1 else dim for dim in shape_list]
        # reshaped_tensor = tf.TensorShape(shape_list)

        #text_emb = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(robertaModel)
        text_emb = tf.keras.layers.Dropout(rate=dropout_text)(robertaModel)
        


    # Audio
    if config != 'text_only':
        audio_input = tf.keras.layers.Input(shape=(max_frame_len, ))
        audio_emb = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(audio_input)
        audio_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=audio_units,
                                                                        kernel_regularizer=tf.keras.regularizers.l2(
                                                                            l2=audio_l2)))(audio_emb)


        #audio_emb = tf.keras.layers.Reshape( [audio_lstm.shape[0], -1])(audio_lstm)
        audio_emb = tf.keras.layers.Dropout(rate=dropout_audio)(audio_lstm)


    # Text-Audio
    if config == 'text_audio':
        stacked_features = tf.concat((text_emb, audio_emb), axis=-1)

        answer = tf.keras.layers.Dense(units=answer_units)(stacked_features)
        dense = tf.keras.layers.Dense(units=answer_units/2)(answer)
        dense = tf.keras.layers.Dense(units=num_labels)(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer) #TODO: eventually remove it
        bn = tf.keras.layers.BatchNormalization()(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer)  # TODO: eventually remove it
        #bn = tf.keras.layers.BatchNormalization()(answer)
        
        #bn = tf.keras.layers.BatchNormalization()(dense)
        out = tf.keras.layers.Dense(num_labels, activation=tf.nn.relu)(tf.keras.layers.Dropout(0.1)(bn))
        model = tf.keras.Model(inputs=[input_ids, attention_mask, audio_input], outputs=out)
        return model

    # Text
    if config == 'text_only':
        answer = tf.keras.layers.Dense(units=answer_units)(text_emb)
        dense = tf.keras.layers.Dense(units=answer_units/2)(answer)
        dense = tf.keras.layers.Dense(units=num_labels)(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer) #TODO: eventually remove it
        bn = tf.keras.layers.BatchNormalization()(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer)  # TODO: eventually remove it
        #bn = tf.keras.layers.BatchNormalization()(answer)
        out = tf.keras.layers.Dense(num_labels, activation=tf.nn.relu)(tf.keras.layers.Dropout(0.1)(bn))
        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=out)

        return model

    # Audio
    if config == 'audio_only':
        answer = tf.keras.layers.Dense(units=answer_units)(audio_emb)
        dense = tf.keras.layers.Dense(units=answer_units/2)(answer)
        dense = tf.keras.layers.Dense(units=num_labels)(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer) #TODO: eventually remove it
        bn = tf.keras.layers.BatchNormalization()(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer)  # TODO: eventually remove it
        #bn = tf.keras.layers.BatchNormalization()(answer)

        #dense = tf.keras.layers.Dense(units=num_labels)(answer)  # TODO: eventually remove it
        #bn = tf.keras.layers.BatchNormalization()(dense)
        out = tf.keras.layers.Dense(num_labels, activation=tf.nn.relu)(tf.keras.layers.Dropout(0.1)(bn))
        model = tf.keras.Model(inputs=[audio_input], outputs=out)

        return model




def sbert_model(num_labels, config='text_only', is_trainable=True, max_sentence_len=maxBERTLen, max_frame_len = maxFrameLen,  dropout_text=0.1, answer_units=100, audio_units=64, audio_l2=0.0005, dropout_audio=0.1):
    max_sentence_len = 768 # embedding shape sbert
    # Text
    if config != 'audio_only':

        encoded_input = tf.keras.layers.Input(shape=(max_sentence_len,), dtype=tf.float32)
        text_emb = tf.keras.layers.Dropout(rate=dropout_text)(encoded_input)
        


    # Audio
    if config != 'text_only':
        audio_input = tf.keras.layers.Input(shape=(max_frame_len, ))
        audio_emb = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(audio_input)
        audio_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=audio_units,
                                                                        kernel_regularizer=tf.keras.regularizers.l2(
                                                                            l2=audio_l2)))(audio_emb)


        #audio_emb = tf.keras.layers.Reshape( [audio_lstm.shape[0], -1])(audio_lstm)
        audio_emb = tf.keras.layers.Dropout(rate=dropout_audio)(audio_lstm)


    # Text-Audio
    if config == 'text_audio':
        stacked_features = tf.concat((text_emb, audio_emb), axis=-1)

        answer = tf.keras.layers.Dense(units=answer_units)(stacked_features)
        dense = tf.keras.layers.Dense(units=answer_units/2)(answer)
        dense = tf.keras.layers.Dense(units=num_labels)(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer) #TODO: eventually remove it
        bn = tf.keras.layers.BatchNormalization()(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer)  # TODO: eventually remove it
        #bn = tf.keras.layers.BatchNormalization()(answer)
        
        #bn = tf.keras.layers.BatchNormalization()(dense)
        out = tf.keras.layers.Dense(num_labels, activation=tf.nn.relu)(tf.keras.layers.Dropout(0.1)(bn))
        model = tf.keras.Model(inputs=[ encoded_input, audio_input], outputs=out)
        return model

    # Text
    if config == 'text_only':
        answer = tf.keras.layers.Dense(units=answer_units)(text_emb)
        dense = tf.keras.layers.Dense(units=answer_units/2)(answer)
        dense = tf.keras.layers.Dense(units=num_labels)(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer) #TODO: eventually remove it
        bn = tf.keras.layers.BatchNormalization()(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer)  # TODO: eventually remove it
        #bn = tf.keras.layers.BatchNormalization()(answer)
        out = tf.keras.layers.Dense(num_labels, activation=tf.nn.relu)(tf.keras.layers.Dropout(0.1)(bn))
        model = tf.keras.Model(inputs=[encoded_input], outputs=out)

        return model

    # Audio
    if config == 'audio_only':
        answer = tf.keras.layers.Dense(units=answer_units)(audio_emb)
        dense = tf.keras.layers.Dense(units=answer_units/2)(answer)
        dense = tf.keras.layers.Dense(units=num_labels)(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer) #TODO: eventually remove it
        bn = tf.keras.layers.BatchNormalization()(dense)
        #dense = tf.keras.layers.Dense(units=num_labels)(answer)  # TODO: eventually remove it
        #bn = tf.keras.layers.BatchNormalization()(answer)

        #dense = tf.keras.layers.Dense(units=num_labels)(answer)  # TODO: eventually remove it
        #bn = tf.keras.layers.BatchNormalization()(dense)
        out = tf.keras.layers.Dense(num_labels, activation=tf.nn.relu)(tf.keras.layers.Dropout(0.1)(bn))
        model = tf.keras.Model(inputs=[audio_input], outputs=out)

        return model
