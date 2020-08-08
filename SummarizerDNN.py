import pandas as pd
from nltk.corpus import re
from nltk.corpus import stopwords
import pickle
import nltk
nltk.download('stopwords')

from keras.models import Sequential
from keras.layers import Dense

batch_size = 64
epochs = 110
latent_dim = 256
num_samples = 10000

stories = pickle.load(open('review_dataset.pkl', 'rb'))
print('Loaded Stories %d' % len(stories))
print(type(stories))

#Vectorize
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
for story in stories:
    input_text = story['story']
    for highlight in story['highlights']:
        target_text = highlight# We use "tab" as the "start sequence" character# for the targets, and "\n" as "end sequence"
        target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
print(input_characters)
print(target_characters)
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)



def define_models(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,  initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model



# # Run training
# model = Sequential()
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# #model.fit([encoder_input_data, decoder_input_data], decoder_target_data,batch_size=batch_size,epochs=epochs,validation_split=0.2)
# model.fit([encoder_input_data, decoder_input_data], decoder_target_data,batch_size=batch_size,epochs=epochs,validation_split=0.2)
# # Save model
# model.save('model2.h5')



def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    encodestate = infenc.predict(source)# start of sequence input
    target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)# collect predictions
    output = list()
    for t in range(n_steps):# predict next char
        yhat, h, c = infdec.predict([target_seq] + state)# store prediction
        output.append(yhat[0,0,:])# update state
        state = [h, c]# update target sequence
        target_seq = yhat
        return array(output)