import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6,6)

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D
from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate
from keras.layers import Reshape, merge, Concatenate, Lambda, Average
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant
from keras.layers.merge import add

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

import pickle
import tensorflow as tf

filename = "News_Category_Dataset_v2.json"
df = pd.read_json(filename,lines=True)
df.head()

op=df.headline[1]+" "+df.short_description[1]

ctgrs = df.groupby('category')
print("total categories:", ctgrs.ngroups)

bias_array=ctgrs.size()
bias_array=dict(bias_array)

df.category = df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)

df['text']=df.headline+' '+df.short_description


tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.text)
X = tokenizer.texts_to_sequences(df.text)
df['words'] = X


df['word_length'] = df.words.apply(lambda i: len(i))
df = df[df.word_length >= 5]

maxlen = 50
X = list(sequence.pad_sequences(df.words, maxlen=maxlen))

# category to id

categories = df.groupby('category').size().index.tolist()
category_int = {}
int_category = {}
for i, k in enumerate(categories):
    category_int.update({k:i})
    int_category.update({i:k})

df['c2id'] = df['category'].apply(lambda x: category_int[x])


word_index = tokenizer.word_index

EMBEDDING_DIM = 100

embeddings_index = {}
f = open('glove.6B.100d.txt',encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s unique tokens.' % len(word_index))
print('Total %s word vectors.' % len(embeddings_index))


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index)+1,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=maxlen,
                            trainable=False)





X = np.array(X)
Y = np_utils.to_categorical(list(df.c2id))

#print (Y)
# and split to training set and validation set
#type(df)
#seed = 29
#df.category.value_counts().sort_index()
#x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=seed)
sss=StratifiedShuffleSplit(n_splits=4, test_size=0.4,  random_state=42)
# for train_index,test_index in sss:
#     X_train,X_test = X[train_index], X[test_index]
#     Y_train, Y_test = Y[train_index], Y[test_index]
for train_index,test_index in sss.split(df,df['category']):
    x_train,x_val = X[train_index], X[test_index]
    y_train, y_val = Y[train_index], Y[test_index]
#     strat_train_set=df.iloc[train_index]
#     strat_test_set=df.iloc[test_index]
#strat_test_set.head()
#strat_test_set.category.value_counts().sort_index()
#type(Y)
#Y_train.category.value_counts().sort_index()
#Y_train
#dir(X)



inp = Input(shape=(maxlen,), dtype='int32')
embedding = embedding_layer(inp)
stacks = []
for kernel_size in [2, 3, 4]:
    conv = Conv1D(64, kernel_size, padding='same', activation='relu', strides=1)(embedding)
    pool = MaxPooling1D(pool_size=3)(conv)
    drop = Dropout(0.5)(pool)
    stacks.append(drop)

merged = Concatenate()(stacks)
flatten = Flatten()(merged)
drop = Dropout(0.5)(flatten)
outp = Dense(len(int_category), activation='softmax')(drop)

TextCNN = Model(inputs=inp, outputs=outp)
TextCNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

TextCNN.summary()



textcnn_history = TextCNN.fit(x_train, 
                              y_train, 
                              batch_size=128, 
                              epochs=20, 
                              validation_data=(x_val, y_val))







filename = 'TextCnn_model.sav'
#pickle.dump(textcnn_history , open(filename, 'wb'))
#TextCNN.save('my_model_1.h5')
#textcnn_history.save('my_model.h5')
tf.keras.experimental.export_saved_model(TextCNN, 'my_model_1.h5')
