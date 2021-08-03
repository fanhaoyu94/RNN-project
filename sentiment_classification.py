import tarfile
import os
import random
import tensorflow as tf
from gensim.parsing.preprocessing import remove_stopwords
from typing import List
from string import punctuation
from collections import Counter
import numpy as np
from gensim.models import Word2Vec
from tensorflow.python.keras.layers import embeddings
from tensorflow.python.ops.init_ops_v2 import Constant, Initializer
# from tensorflow.python.keras.engine.input_layer import Input

save_path = './sentiment_data'
data_tgz = tarfile.open('review_polarity.tar.gz')
data_tgz.extractall(path=save_path)
data_tgz.close()

def split_review(folder_name:str):
    path = f"./sentiment_data/txt_sentoken/{folder_name}/"
    file_list = os.listdir(path)
    test_file_list = random.choices(file_list, k=len(file_list)//10)
    train_file_list = list(set(file_list) - set(test_file_list))
    return [path + file for file in train_file_list], [path + file for file in test_file_list]

def load_review(file:str):
    with open(file,'r') as f:
        review = f.read()
    return remove_stopwords(review.lower())

def clean_review(review:str):
    word_list = review.split()
    clean_word_list = []
    table = str.maketrans("","",punctuation)
    clean_word_list = [word.translate(table) for word in word_list 
                       if word.translate(table).isalpha() and len(word.translate(table))>1]
    return clean_word_list

def create_vocab(file_list:List[str]):
    vocab = Counter()
    for file in file_list:
        review = load_review(file)
        word_list = clean_review(review)
        vocab.update(word_list)

    vocab_final = [key for key,value in vocab.items() if value>=2]
    return vocab_final

def remove_oov_word(file:str,vocab: List[str]):
    review = load_review(file)
    clean_review_list = clean_review(review)
    clean_review_list_final = [word for word in clean_review_list if word in vocab]
    return clean_review_list_final

def get_tokenizer(vocab: List[str]):
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(vocab)
    return tokenizer

def create_sample(train_file_list:List[str], vocab: List[str], test_file_list:str=None):
    tokenizer = get_tokenizer(vocab)
    # when you fit train_sample, either ['there is a dog'], or [['there','is','a','dog']] can not be ['there','is','a','dog']
    train_sample = [
        tokenizer.texts_to_sequences([remove_oov_word(file,vocab)])[0]
        for file in train_file_list
    ]
    max_len = max([len(one_sample) for one_sample in train_sample])
    if test_file_list:
        test_sample = [
        tokenizer.texts_to_sequences([remove_oov_word(file,vocab)])[0]
        for file in test_file_list
        ]
        pad_sample = tf.keras.preprocessing.sequence.pad_sequences(test_sample, maxlen=max_len, padding='post')
    else:
        pad_sample = tf.keras.preprocessing.sequence.pad_sequences(train_sample, maxlen=max_len, padding='post')
    return pad_sample

def pretrained_embedding(train_file_list:List[str], vocab:List[str]):
    word_list = [remove_oov_word(file,vocab) for file in train_file_list]
    model = Word2Vec(word_list,min_count=1)
    return model.wv

def create_model(vocab_size:int, input_length:int, embedding_weights=None, model_type:str = None):
    if model_type == "conv":
        model = tf.keras.Sequential([
                                    tf.keras.layers.Embedding(vocab_size, 100, embedding_initializer= tf.keras.initializers.Constant(embedding_weights), input_length=input_length, trainable=False) 
                                    if np.any(embedding_weights)
                                    else tf.keras.layers.Embedding(vocab_size, 100, input_length=input_length),
                                    tf.keras.layers.Conv1D(filters=32, kernel_size=8, activation='relu'),
                                    tf.keras.layers.MaxPool1D(),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(10, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')
                                ])
    else:
        # the unit of lstm has to be smaller than embedding
        # when you do classification of entire sentence, it has to be better to use entire sequence instead of last step
        model = tf.keras.Sequential([
                                    tf.keras.layers.Embedding(vocab_size, 100, embeddings_initializer= tf.keras.initializers.Constant(embedding_weights), input_length=input_length, trainable=False) 
                                    # np.all() means there is no zero elements in the array
                                    if np.any(embedding_weights)
                                    else tf.keras.layers.Embedding(vocab_size, 100, input_length=input_length),
                                    tf.keras.layers.LSTM(96,return_sequences=True),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.LSTM(64,return_sequences=True),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(10,activation="relu",kernel_initializer="he_normal"),
                                    tf.keras.layers.Dense(1,activation="sigmoid")
        ])
    return model

train_file_list_neg, test_file_list_neg = split_review('neg')
train_file_list_pos, test_file_list_pos = split_review('pos')
train_file_total = train_file_list_neg + train_file_list_pos
test_file_total = test_file_list_neg + test_file_list_pos

vocab = create_vocab(train_file_total)
X_train = create_sample(train_file_total,vocab)
y_train = np.concatenate([np.zeros((len(train_file_list_neg))),np.ones((len(train_file_list_pos)))])
X_test = create_sample(train_file_total,vocab,test_file_list=test_file_total)
y_test = np.concatenate([np.zeros((len(test_file_list_neg))),np.ones((len(test_file_list_pos)))])

# get embedding metrix
pretrained_weight = pretrained_embedding(train_file_total, vocab)
token = get_tokenizer(vocab).index_word
embeddings_weight = np.zeros((len(vocab)+1,100))
for key,value in token.items():
    embeddings_weight[key,:] = pretrained_weight[value]

# model creation 
model1 = create_model(len(vocab)+1,X_train.shape[1],embedding_weights=embeddings_weight)

model1.compile(optimizer='adam',loss='binary_crossentropy',
               metrics = ['accuracy'])

earlystop_cb = tf.keras.callbacks.EarlyStopping(min_delta=0.01, patience=5)
reduceLR_cb = tf.keras.callbacks.ReduceLROnPlateau()
modelcheck_cb = tf.keras.callbacks.ModelCheckpoint(filepath="./models/model_weights.h5",save_best_only=True, save_weights_only=True)

model1.fit(x=X_train, y=y_train, 
           batch_size=32, 
           epochs=5,
           callbacks=[earlystop_cb,reduceLR_cb,modelcheck_cb],
           validation_split=0.1)

config = model1.get_config()
model2 = tf.keras.Sequential().from_config(config)
model2.load_weights("./models/model_weights.h5")
model2.compile(optimizer='adam',loss='binary_crossentropy',
               metrics = ['accuracy'])

# predict and model, both don't require status(loss, opitimizer), but evaluate require
model2.evaluate(X_test,y_test)

