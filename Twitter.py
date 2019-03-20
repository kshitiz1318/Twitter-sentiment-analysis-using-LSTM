import os
import pandas as pd
import numpy as np
import re
import keras

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Embedding,Activation,SpatialDropout1D,LSTM
from keras.losses import categorical_crossentropy
from keras.optimizers import adam
from keras.metrics import binary_accuracy
import sklearn
from sklearn.model_selection import train_test_split
os.chdir('C:\\Users\\91920\\Desktop')
data=pd.read_csv('sentiment.csv')


data_sentiment=data[['sentiment','text']]
#data_sentiment
data_sentiment=data_sentiment[data_sentiment.sentiment!='Neutral']
data_sentiment['text']=data_sentiment['text'].apply(lambda x: x.lower())
data_sentiment['text']=data_sentiment['text'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]','',x))
for idx,row in data_sentiment.iterrows():
    row[1]=row[1].replace('rt','')
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data_sentiment['text'].values)
X = tokenizer.texts_to_sequences(data_sentiment['text'].values)
X=pad_sequences(X,maxlen=28)
X.shape
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())



Y = pd.get_dummies(data_sentiment['sentiment']).values
Y.shape
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
batch_size = 32
model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)

score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))
