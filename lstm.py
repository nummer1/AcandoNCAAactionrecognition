from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv2D, Reshape
from tensorflow.keras.layers import Bidirectional, ConvLSTM2D


rows = 360
cols = 490
colour_channels = 1
data_dim = rows * cols
timesteps = 20
num_classes = 11 + (timesteps * 2)

# LSTM input parameters
# tf.keras.layers.LSTM(units=units, activation='tanh', recurrent_activation='hard_sigmoid',
#         use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
#         bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
#         recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
#         kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
#         dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False,
#         return_state=False, go_backwards=False, stateful=False, unroll=False)

# ConvLSTM2D input parameters
# keras.layers.ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='valid',
#         data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid',
#         use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
#         bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
#         recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
#         kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
#         return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0)

model = Sequential()
# kernel_size: height width
model.add(TimeDistributed(Conv2D(32, (10, 10), strides=(3, 3), padding='same', activation='relu'),
        input_shape=(timesteps, rows, cols, colour_channels)))
model.add(TimeDistributed(Conv2D(32, (8, 8), strides=(3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(32, (5, 5), strides=(3, 3), padding='same', activation='relu')))
# model.add(MaxPooling2D)
# first number is output size
# model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))
model.add(Reshape((20, 14*19*32)))
model.add(LSTM(1000, return_sequence=True))
model.add(LSTM(200, return_sequence=True))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.load_weights("weights.hdf5", by_name=False)
model.summary()

model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_val, y_val))
model.save_weights("weights.hdf5")
score = model.evaluate(x_test, y_test, batch_size=1)

print(score)
