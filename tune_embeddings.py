import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Add, Multiply, Dot, Concatenate
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer
from keras.utils import np_utils

from tqdm import tqdm
import numpy as np
import sys

################################
# Constant

words_per_sentence = 50
embedding_length = 300
alpha_change_margin = 0.1
alpha_hidden_layer_size = 300
sst_hidden_layer_size = 300



################################
# Read the data into S, Y

# Read words file
with open(sys.argv[1], 'r') as dist:
        words = []
        p = []
        for line in dist:
                words += [line.split()[0]]
                p += [int(line.split()[1])]
        p = np.array(p, dtype=np.float)
        num = np.sum(p)
        p /= num
        alpha = 0.001 / (0.001+p)

# Read word embedding
with open(sys.argv[2], 'r') as embeddings:
        lines = embeddings.readlines()
        V = []
        keys = []
        for (i, line) in enumerate(tqdm(lines, ncols=100)):
                line = line.split()
                v = []
                j = len(line) - embedding_length
                keys += [" ".join(line[0:j])]

                while j < len(line):
                        v += [float(line[j])]
                        j += 1
                V += [np.array(v)]

# Read training set
with open(sys.argv[3], 'r') as train_file:
        V_train = []
        a_train = []
        Y_train = []
        for line in train_file:
                a = []
                sentence = []
                line = line.split()
                Y_train += [int(line[-1])]
                i = 0
                for w in line[0:words_per_sentence]:
                        if w in keys and w in words:
                                sentence += [V[keys.index(w)]]
                                a += [alpha[words.index(w)]]
                                i += 1

                for j in range(words_per_sentence-i):
                        sentence += [np.zeros(embedding_length)]
                        a += [0]
                V_train += [np.array(sentence)]
                a_train += [a]
        V_train = np.array(V_train)
        a_train = np.array(a_train)
        Y_train = np.array(Y_train)




# Read test set
with open(sys.argv[4], 'r') as test_file:
        V_test = []
        a_test = []
        Y_test = []
        for line in test_file:
                a = []
                sentence = []
                line = line.split()
                Y_test += [int(line[-1])]
                i = 0
                for w in line[0:words_per_sentence]:
                        if w in keys and w in words:
                                sentence += [V[keys.index(w)]]
                                a += [alpha[words.index(w)]]
                                i += 1
                for j in range(words_per_sentence-i):
                        sentence += [np.zeros(embedding_length)]
                        a += [0]
                V_test += [np.array(sentence)]
                a_test += [a]
        V_test = np.array(V_test)
        a_test = np.array(a_test)
        Y_test = np.array(Y_test)


S_train = [a_train, V_train]
Y_train = np_utils.to_categorical(Y_train, 2)

S_test = [a_test, V_test]
Y_test = np_utils.to_categorical(Y_test, 2)



################################
# Create the model
words_per_sentence = 50
embedding_length = 300
alpha_change_margin = 0.1
alpha_hidden_layer_size = 300
sst_hidden_layer_size = 300


class Clip(Layer):
	def __init__(self, **kwargs):
		# self.output_dim = output_dim
	        super(Clip, self).__init__(**kwargs)

	def build(self, input_shape):
		super(Clip, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, x):
		return K.clip(x, -alpha_change_margin, alpha_change_margin)


alpha_static_input = Input(shape=(words_per_sentence,), name="alpha_input")
word_embedding_input = Input(shape=(words_per_sentence,embedding_length), name="embedding_input")

flattened_word_embeddings = Flatten(name="flattened_word_embedding")(word_embedding_input)

alpha_input = Concatenate(name="concatenated_alpha_input")([alpha_static_input, flattened_word_embeddings])
alpha_hidden = Dense(alpha_hidden_layer_size, activation='sigmoid', name="alpha_hidden")(alpha_input)
delta_alpha = Dense(words_per_sentence, activation='sigmoid', name="delta_alpha")(alpha_hidden)

clipped_delta = Clip(name="clipped_delta_alpha")(delta_alpha)
new_alpha = Add(name="new_alpha")([clipped_delta, alpha_static_input])


sentence_embedding = Dot(1, name="sentence_embedding")([word_embedding_input, new_alpha])

sentence_hidden = Dense(sst_hidden_layer_size, activation='sigmoid', name="sentence_hidden")(sentence_embedding)
output = Dense(2, activation='softmax', name="output")(sentence_hidden)
model = Model(inputs=[alpha_static_input, word_embedding_input], outputs=output)

for layer in model.layers:
        print layer.input_shape, layer.output_shape, layer.name

################################
# Train the network
batch_size = 32
num_epochs = 10

log_file = 'Tuned_SST_log.csv'
model_file = 'Tuned_SST_model.hdf5'


opt = keras.optimizers.Adagrad(0.01)
log = keras.callbacks.CSVLogger(log_file, separator=',', append=False)
cp = keras.callbacks.ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(S_train, Y_train,
          shuffle=True, batch_size=batch_size,
          epochs=num_epochs, verbose=1, callbacks=[cp, log],
          validation_data=(S_test, Y_test)
          )

