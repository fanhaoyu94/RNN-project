import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from pickle import dump
import torchtext
from typing import List
from abc import ABC,abstractmethod
import colorama


gpu = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpu[0],True)

policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
tf.keras.mixed_precision.experimental.set_policy(policy)

path = keras.utils.get_file('nietzsche.txt', origin = 'https://s3.amazonaws.com/text-datasets/nietzsche.txt')

colorama.init()
# preprocessing
class Preprocess_Data:
    def __init__(self, path, seq_length:int, seq_step:int, batch_size:int):
        self.path = path
        self.seq_length = seq_length
        self.seq_step = seq_step
        self.batch_size = batch_size
        self.text = self.load_file()
    
    def load_file(self) -> str:
        with open(self.path,"r",encoding="utf-8") as f_obj:
            text = f_obj.read()
            return text
    
    def create_tokenizer(self):
        tokenizer = Tokenizer(oov_token="<unk>")
        tokenizer.fit_on_texts([self.text])
        return tokenizer

    def vectorize_text(self) -> List[int]:
        return self.create_tokenizer().texts_to_sequences([self.text])[0]
    
    def create_sequence(self,vector:List[int]):
        total_sequence = []
        for i in range(0,self.seq_length,self.seq_step):
            sub_sequence = np.array(vector[i:])
            num_sequence = len(sub_sequence)//self.seq_length
            # resize can be not equal to num_squence*seq_length, reshape has to be
            # random.choice with retraction, random.sample without rectraction
            sub_array = np.resize(sub_sequence,(num_sequence,self.seq_length))
            total_sequence.append(sub_array)
        return np.concatenate(total_sequence)
    
    def batch_sort_for_stateful_rnn(self,vector:List[int]):
        sequence = self.create_sequence(vector)
        num_batches = len(sequence)//self.batch_size
        num_samples = num_batches * self.batch_size
        stateful_batch_sequence = np.zeros((num_samples,self.seq_length),dtype=np.int32)
        for i in range(self.batch_size):
            start_index = i * num_batches
            end_index = start_index + num_batches
            stateful_batch_sequence[i::self.batch_size,:] = sequence[start_index:end_index,:]
        return stateful_batch_sequence
    
    def create_samples(self):
        vector = self.vectorize_text()
        feature = vector[:-1]
        target = vector[1:]
        X = self.batch_sort_for_stateful_rnn(feature)
        y = self.batch_sort_for_stateful_rnn(target)
        X_train, X_test = X[:-self.batch_size,:], X[-self.batch_size:,:]
        y_train, y_test = y[:-self.batch_size,:], y[-self.batch_size:,:]
        return X_train, X_test, y_train, y_test
    
# create callbacks
class LivesampleTest(tf.keras.callbacks.Callback):
    def __init__(self, meta_model, sample_pool):
        super().__init__()
        self.meta_model = meta_model
        self.sample_pool = sample_pool
    
    def on_epoch_end(self, epoch, logs):
        self.meta_model.update_sample_model_weights()
        predictions = self.meta_model.sample(self.sample_pool)
        print("\n*****start_sampling*****")
        print(colorama.Fore.RED,predictions)
        print(colorama.Style.RESET_ALL)


class Model_Template(ABC):

    @abstractmethod
    def build_models(self, batch_size:int, embedding_size:int, rnn_size:int, num_layers:int):
        pass

    @abstractmethod
    def train(self, path, seq_length:int, seq_step:int, batch_size:int, embedding_size:int, 
              rnn_size:int, num_layers:int, num_epochs:int, live_sample:bool):
        pass

    @abstractmethod
    def update_sample_model_weights(self):
        pass

    @abstractmethod
    def sample(self):
        pass

# create model
class MetaModel(Model_Template):
    def __init__(self,tokenizer):
        self.train_model = None
        self.sample_model = None
        self.tokenizer = tokenizer

    def get_vocab_size(self):
        return len(self.tokenizer.word_index)

    def create_pretrained_metric(self,embedding_size:int):
        glove = torchtext.vocab.GloVe(name="6B", dim=embedding_size)
        pretrained_matrix = np.zeros((self.get_vocab_size()+1,embedding_size))
        for key,value in self.tokenizer.word_index.items():
            # convert tensor to numpy
            if np.any(glove[key].numpy()):
                pretrained_matrix[value,:] = glove[key].numpy()
        return pretrained_matrix

    def build_models(self, batch_size:int, embedding_size:int, rnn_size:int, num_layers:int):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(self.get_vocab_size()+1,embedding_size,
                                            embeddings_initializer=tf.keras.initializers.Constant(self.create_pretrained_metric(embedding_size)),
                                            batch_input_shape=(batch_size,None)))
        for _ in range(num_layers):
            model.add(tf.keras.layers.LSTM(rnn_size,return_sequences=True,stateful=True,dtype=tf.float32))
            model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.get_vocab_size(),activation='softmax')
        ))
        
        model.compile(optimizer='adam',
                      loss= 'sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy']
                     )
        self.train_model = model 
        config = model.get_config()
        config['layers'][0]['config']['batch_input_shape'] = (1,None)
        self.sample_model = tf.keras.Sequential().from_config(config)
        self.sample_model.trainable = False
        
    def train(self, path, seq_length:int, seq_step:int, batch_size:int, embedding_size:int, 
              rnn_size:int, num_layers:int, num_epochs:int, live_sample:bool):
        process_data_class = Preprocess_Data(path, seq_length, seq_step, batch_size)
        X_train, X_test, y_train, _ = process_data_class.create_samples()
        self.build_models(batch_size, embedding_size, rnn_size, num_layers)

        live_callbacks = LivesampleTest(self,X_test) if live_sample else None
        self.train_model.fit(X_train,y_train,
                             batch_size= batch_size,
                             epochs=num_epochs,
                             callbacks=[live_callbacks],
                             shuffle=False
        )

    def update_sample_model_weights(self):
        self.sample_model.set_weights(self.train_model.get_weights())
    
    def sample(self,sample_pool):
        self.sample_model.reset_states()
        live_test_sample = sample_pool[int(np.random.choice(len(sample_pool)))].reshape((1,sample_pool.shape[1]))
        output = self.sample_model.predict_classes(live_test_sample).tolist()
        return self.tokenizer.sequences_to_texts(output)[0]
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # delete dictionary item
        del state['train_model']
        del state['sample_model']
        return state

tokenizer = Preprocess_Data(path, 50, 25, 32).create_tokenizer()
model = MetaModel(tokenizer)

# the default dtype for floating-point values in the TensorFlow Python API—e.g. for tf.constant(37.0)—is tf.float32).
model.train(path,50,25,32,50,128,2,3,True)

model.train_model.layers[0].dtype

