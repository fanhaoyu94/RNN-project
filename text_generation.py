import tensorflow as tf
from tensorflow import keras
import string
from collections import Counter
import numpy as np
import re 
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pickle import dump
import torchtext


gpu = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpu[0],True)

policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
tf.keras.mixed_precision.experimental.set_policy(policy)

path = keras.utils.get_file('nietzsche.txt', origin = 'https://s3.amazonaws.com/text-datasets/nietzsche.txt')

####################### First approach for language model #############################

#creating corpse 
with open(path,"r",encoding="utf-8") as f_obj:
    text = f_obj.read()

text_clean = text.replace("--"," ").split()
clean_tokens = []

# create dictionary that key are punctuation and value are None
table = str.maketrans("","",string.punctuation)

for word in text_clean:
    word = word.translate(table)
    if word.isalpha():
        clean_tokens.append(word.lower())

word_dictionary = Counter(clean_tokens)
min_freq = 2
is_filterd = 0
unknown_word = "<unk>"

# wordindex_dic is an dictionary, use name to get index
wordindex_dic = {}
wordindex_dic[unknown_word] = 0

# indexword is a list, use index to get name
indexword = [unknown_word]

for word,count in word_dictionary.items():
    if count>=min_freq:
        wordindex_dic[word] = len(wordindex_dic)
        indexword.append(word)
    else:
        is_filterd+=1

# only do within an sentence
# don't remove stop words for language modeling, it is useful for classification
text_sentence = re.split('\.|\!|\?|\,',text.replace("\n"," ").replace("--"," "))
clean_text_sentence = [sentence.translate(table) for sentence in text_sentence]
len_array = np.array([len(sentence.split()) for sentence in clean_text_sentence])
sentence_array = np.array(clean_text_sentence)
clean_text_sentence_remove = sentence_array[len_array>=6].tolist()


max_len = 5
step = 1
x = []
y= []
for sentence in clean_text_sentence_remove:
    sentence_list = sentence.split()
    for i in range(0,len(sentence_list)-max_len,step):
        x_temp = sentence_list[i:i+max_len]
        y_temp = sentence_list[i+max_len]
        x.append([wordindex_dic.get(word.lower(),0) for word in x_temp])
        y.append(wordindex_dic.get(y_temp.lower(),0))
x_train = np.array(x)
y_train = keras.utils.to_categorical(y,num_classes=len(wordindex_dic))    

#creating models
embedding_size = 50
lstm_size = 256
inputs = keras.Input(shape=(max_len,))
embedding = keras.layers.Embedding(len(wordindex_dic),embedding_size,input_length=max_len)(inputs)
x = keras.layers.LSTM(lstm_size,return_sequences=True)(embedding)
x = keras.layers.LSTM(lstm_size)(x)
x = keras.layers.Dense(1024,activation="relu")(x)
x = keras.layers.Dense(2048,activation="relu")(x)
outputs = keras.layers.Dense(len(wordindex_dic),activation="softmax")(x)

LSTM = keras.Model(inputs=inputs,outputs=outputs)

LSTM.compile(optimizer="adam",
             loss="categorical_crossentropy",
             metrics="accuracy")

#training
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=1e-3,patience=10,restore_best_weights=True)
reduce_learing_cb = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",patience=4)

LSTM.fit(x_train, y_train,
         batch_size=64,
         epochs=30,
         validation_split=0.1,
         callbacks=[early_stopping_cb,reduce_learing_cb])

#prediction
string = "the real truth is hard"
str_list = string.split(" ")
index_list = [wordindex_dic.get(word,0) for word in str_list]
for i in range(5):
    input_ = np.array(index_list[i:i+max_len]).reshape(1,max_len)
    predicted_results = LSTM.predict(input_)
    index_list.append(np.argmax(predicted_results))
word_list = [indexword[index] for index in index_list]  
output = " ".join(word_list)
print(output)


####################### Second approach for language model #############################
with open(path,"r",encoding="utf-8") as f_obj:
    text = f_obj.read()

tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts([text])

text_sentence = re.split('\.|\!|\?|\,|\;',text.replace("\n"," ").replace("--"," "))
max_len = 10

input_sequence = []
for sentence in text_sentence:
    sentence_list = tokenizer.texts_to_sequences([sentence])[0]
    if len(sentence_list)<=max_len and len(sentence_list)>1:
        input_sequence.append(sentence_list)
    elif len(sentence_list)>max_len:
        for i in range(len(sentence_list)-max_len):
            input_sequence.append(sentence_list[i:i+max_len])

input_sequence_pad = np.array(pad_sequences(input_sequence,maxlen=10,padding="pre"))
x_train, y_train = input_sequence_pad[:,:-1],input_sequence_pad[:,-1]

y_train = keras.utils.to_categorical(y_train.tolist(),num_classes=len(tokenizer.word_index)+1) 

# create weights for embedding 
glove = torchtext.vocab.GloVe(name="6B", dim=50)

embedding_matrix = np.zeros((len(tokenizer.word_index)+1,50))
exist_word = 0
nonexist_word = 0
for word,i in tokenizer.word_index.items():
    if not np.all(glove[word].numpy() == np.zeros((1,50),dtype="float")):
        embedding_matrix[i] = glove[word].numpy()
        exist_word+=1
    else:
        nonexist_word+=1
print("exsit_word:",exist_word)
print("nonexsit_word:",nonexist_word)


#creating models
embedding_size = 50
lstm_size1 = 256
lstm_size2 = 128
inputs = keras.Input(shape=(max_len-1,))
embedding = keras.layers.Embedding(len(tokenizer.word_index)+1,
                                   embedding_size,
                                   embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                   input_length=max_len-1,
                                   trainable=False)(inputs)
x = keras.layers.Bidirectional(keras.layers.LSTM(lstm_size1,return_sequences=True))(embedding)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.LSTM(lstm_size2)(x)
x = keras.layers.Dense(2048,activation="relu")(x)
x = keras.layers.Dense(4096,activation="relu")(x)
outputs = keras.layers.Dense(len(tokenizer.word_index)+1,activation="softmax")(x)

LSTM = keras.Model(inputs=inputs,outputs=outputs)

LSTM.compile(optimizer="adam",
             loss="categorical_crossentropy",
             metrics="accuracy")

#training
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=1e-3,patience=10,restore_best_weights=True)
reduce_learing_cb = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",patience=4)

LSTM.fit(x_train, y_train,
         batch_size=64,
         epochs=30,
         validation_split=0.1,
         callbacks=[early_stopping_cb,reduce_learing_cb])

#prediction
string = "I really want to go to a beautiful place"
index_list = tokenizer.texts_to_sequences([string])[0]
for i in range(4):
    input_ = np.array(index_list[i:i+max_len-1]).reshape(1,max_len-1)
    predicted_results = LSTM.predict(input_)
    index_list.append(np.argmax(predicted_results))
word_list = tokenizer.sequences_to_texts([index_list])
output = " ".join(word_list)
print(output)

dump(tokenizer,open("tokenizer.pkl","wb"))



#creating training samples
max_len = 40
x = []
y = []
for i in range(0,len(clean_tokens)-max_len-1,8):
    x_temp = clean_tokens[i:i+max_len]
    x_temp2 = [wordindex_dic.get(word,0) for word in x_temp]
    y_temp = clean_tokens[i+1:i+max_len+1]
    y_temp2 = [wordindex_dic.get(word,0) for word in y_temp]
    x.append(x_temp2)
    y.append(y_temp2)

x_train = np.array(x)
y_train = np.zeros((x_train.shape[0],x_train.shape[1],len(wordindex_dic)))
for i in range(len(y)):
    y_train[i] = keras.utils.to_categorical(y[i], num_classes=len(wordindex_dic))

#creating models
embedding_size = 50
lstm_size = 256
inputs = keras.Input(shape=(max_len,))
embedding = keras.layers.Embedding(len(wordindex_dic),embedding_size,input_length=max_len)(inputs)
x = keras.layers.LSTM(lstm_size,return_sequences=True)(embedding)
Dense_layer1 = keras.layers.Dense(1024,activation="relu")
Dense_layer2 = keras.layers.Dense(2048,activation="relu")
Dense_layer3 = keras.layers.Dense(len(wordindex_dic),activation="softmax")
x = tf.keras.layers.TimeDistributed(Dense_layer1)(x)
x = tf.keras.layers.TimeDistributed(Dense_layer2)(x)
outputs = tf.keras.layers.TimeDistributed(Dense_layer3)(x)
LSTM = keras.Model(inputs=inputs,outputs=outputs)

LSTM.compile(optimizer="adam",
             loss="categorical_crossentropy",
             metrics="categorical_accuracy")

#training 
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=1e-3,patience=10,restore_best_weights=True)
reduce_learing_cb = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",patience=4)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

LSTM.fit(train_dataset,
         epochs=30,
         callbacks=[early_stopping_cb,reduce_learing_cb])


a = ['a','b','c']
b = [1,2,3]
dict(zip(a,b))

str1 = "we like to go to travel"
word,coefs = str1.split(maxsplit=1)

coef = "1.1 2.3 4.5 6.7"
np.array(coef.split(),dtype="float")
# This is a fast way to create a array from string
np.fromstring(coef,dtype="float",sep=" ")


