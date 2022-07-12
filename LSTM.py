# Install instructions: pip install -U tensorflow, python -m pip show tensorflow

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import numpy as np
import Resolver as res
import tensorflow as tf
import DBController as db
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop


class LSTMpy():

    def __init__(self, domain):
        self.domain = domain
        self.step_length = 1 # The step length we take to get our samples from our corpus
        self.epochs = 100 # Number of times we train on our full data
        self.batch_size = 32 # Data samples in each training step
        self.latent_dim = 32 # Size of our LSTM
        self.dropout_rate = 0.2 # Regularization with dropout
        self.model_path = os.path.realpath('./machina/sub_gen_model.h5') # Location for the model
        self.load_model = False # Enable loading model from disk
        self.store_model = True # Store model to disk after training
        self.verbosity = 1 # Print result for each epoch
        self.gen_amount = 500 # How many to generate
        self.input_path = os.path.realpath("./machina/data/names.txt")
        self.input_names = []
        self.char2idx = None
        self.idx2char = None
        self.concat_names = None
        self.max_sequence_length = 0
        self.chars = None
        self.num_chars = None
        self.sequences = []
        self.sequence = None
        self.dbController = db.DBController(self.domain)   # Custom Class for DB MGMT
        self.res = res.Resolver()   # Custom class for DNS utilities

    def main(self):
        print("[+] Starting LSTM Deep Learning Module..")
        self.readList()
        model = self.trainModel()
        self.generate(model)

    def readList(self):
        print('[+] Reading subdomains from DB:')
        dbSubs = self.dbController.selectSubs()
        for item in dbSubs:
            self.input_names.append(item)
        print('...')
        print("DB Items: " + str(self.input_names))

        #self.gen_amount = len(self.input_names)

        #==================================================
        # Make it all to a long string
        self.concat_names = '\n'.join(self.input_names).lower()

        # Find all unique characters by using set()
        self.chars = sorted(list(set(self.concat_names)))
        self.num_chars = len(self.chars)

        # Build translation dictionaries, 'a' -> 0, 0 -> 'a'
        self.char2idx = dict((c, i) for i, c in enumerate(self.chars))
        self.idx2char = dict((i, c) for i, c in enumerate(self.chars))

        # Use longest name length as our sequence window
        self.max_sequence_length = max([len(name) for name in self.input_names])

        print('[+] Total characters: {}'.format(self.num_chars))
        print('[+] Corpus length:', len(self.concat_names))
        print('[+] Number of names: ', len(self.input_names))
        print('[+] Longest name: ', self.max_sequence_length)


    def trainModel(self):
        next_chars = []

        # Loop over our data and extract pairs of sequences and next chars
        for i in range(0, len(self.concat_names) - self.max_sequence_length, self.step_length):
            self.sequences.append(self.concat_names[i: i + self.max_sequence_length])
            next_chars.append(self.concat_names[i + self.max_sequence_length])

        num_sequences = len(self.sequences) # number of sequences

        print('Number of sequences:', num_sequences)
        print('First 10 sequences and next chars:')
        for i in range(len(self.input_names)):
            print('X=[{}] y=[{}]'.replace('\n', ' ').format(self.sequences[i], next_chars[i]).replace('\n', ' '))

        #===================================================
        X = np.zeros((num_sequences, self.max_sequence_length, self.num_chars), dtype=bool)
        Y = np.zeros((num_sequences, self.num_chars), dtype=np.bool)

        for i, sequence in enumerate(self.sequences):
            for j, char in enumerate(sequence):
                X[i, j, self.char2idx[char]] = 1
                Y[i, self.char2idx[next_chars[i]]] = 1

        print('X shape: {}'.format(X.shape))
        print('Y shape: {}'.format(Y.shape))

        #===================================================

        model = Sequential()
        model.add(LSTM(self.latent_dim, input_shape=(self.max_sequence_length, self.num_chars), recurrent_dropout=self.dropout_rate))
        model.add(Dense(units=self.num_chars, activation='softmax'))

        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer)

        model.summary()

         #===================================================

        if self.load_model: # Load model from disk if setting is True
            model.load_weights(self.model_path)
        else:
            start = time.time()
            print('Start training for {} epochs'.format(self.epochs))
            history = model.fit(X, Y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbosity)
            end = time.time()
            print('Finished training - time elapsed:', (end - start)/60, 'min')
        if self.store_model:
            print('Storing model at:', self.model_path)
            model.save(self.model_path)

        # Start sequence generation from end of the input sequence
        self.sequence = self.concat_names[-(self.max_sequence_length - 1):] + '\n'

        return model

    def generate(self, model):
        new_names = []
        print('{} new names are being generated'.format(self.gen_amount))

        while len(new_names) < self.gen_amount:
            # Vectorize sequence for prediction
            x = np.zeros((1, self.max_sequence_length, self.num_chars))
            for i, char in enumerate(self.sequence):
                x[0, i, self.char2idx[char]] = 1

            # Sample next char from predicted probabilities
            probs = model.predict(x, verbose=0)[0]
            probs /= probs.sum()
            next_idx = np.random.choice(len(probs), p=probs)
            next_char = self.idx2char[next_idx]
            self.sequence = self.sequence[1:] + next_char

            # New line means we have a new name
            if next_char == '\n':
                gen_name = [name for name in self.sequence.split('\n')][1]
                
                # Never start name with two identical chars, could probably also
                if len(gen_name) > 2 and gen_name[0] == gen_name[1]:
                    gen_name = gen_name[1:]
                
                # Discard all names that are too short
                if len(gen_name) > 2:
                    # Only allow new and unique names
                    if gen_name not in self.input_names + new_names:
                        new_names.append(gen_name.capitalize())
                
                if 0 == (len(new_names) % (self.gen_amount/ 10)):
                    print('Generated {}'.format(len(new_names)))


        print_first_n = self.gen_amount
        results = []
        print('First {} generated names:'.format(print_first_n))
        for name in new_names[:print_first_n]:
            print("[LSTM][Generated][Unvalidated]: " + name)
        for name in new_names[:print_first_n]:
            if name not in results: # Weed out the dupes
                results.append(name)
                resolve = self.res.resolveHost(name)
                if "." + self.domain in name and resolve:   # ensure generated names are for the origin domain
                    print(resolve)
    

if __name__ == "__main__":
    generateSubs = LSTMpy(sys.argv[1])  # Pass domain as arg
    generateSubs.main()