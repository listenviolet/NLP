from keras.preprocessing import sequence
from keras.models import Sequential 
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# max words to be used
max_features = 20000

# max len of rnn
maxlen = 80

batch_size = 32

(trainX, trainY), (testX, testY) = imdb.load_data(num_words = max_features)
print(len(trainX), 'train sequences') # 25000
print(len(testX), 'test sequences')   # 25000

trainX = sequence.pad_sequences(trainX, maxlen = maxlen)
testX= sequence.pad_sequences(testX, maxlen = maxlen)
print('trainX.shape:', trainX.shape) # (25000, 80)
print('testX.shape:', testX.shape)   # (25000, 80)

''' Build the model
'''
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(1, activation = 'sigmoid'))

''' Compile
''' 
model.compile(loss = 'binary_crossentropy',
			optimizer = 'adam',
			metrics = ['accuracy'])

''' Fit: train and evaluate
''' 
model.fit(trainX, trainY, 
		batch_size = batch_size, 
		epochs = 15, 
		validation_data = (testX, testY))

''' Test
''' 
score = model.evaluate(testX, testY, batch_size = batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])