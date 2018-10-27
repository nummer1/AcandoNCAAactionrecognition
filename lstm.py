from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv2D, Reshape
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import optimizers
from functools import reduce
import dataImport
import Utility


rows = 360
cols = 490
colour_channels = 3
data_dim = rows * cols
timesteps = 20
num_classes = 11 + (timesteps * 2)
num_classes = 11

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
r_layer = TimeDistributed(Conv2D(32, (5, 5), strides=(3, 3), padding='same', activation='relu'))
model.add(r_layer)
# model.add(MaxPooling2D)
# first number is output size
# model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))
shape = reduce(lambda x, y: x*y, r_layer.output_shape[-3:])
model.add(Reshape((20, shape)))
model.add(LSTM(200, return_sequences=True))
model.add(LSTM(50, return_sequences=True))
model.add(Dense(num_classes, activation='sigmoid'))


rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.load_weights("weights.hdf5", by_name=False)
model.summary()


util = Utility.Utility()
train_c, train_e, _, _ = dataImport.readData("data", 0, numOfExamples=10)

print("read training set")
val_c, val_e, _, _ = dataImport.readData("data", 1, numOfExamples=1)
print("read validation set")
test_c, test_e, _, _ = dataImport.readData("data", 2, numOfExamples=1)
print("read test set")

# target_event = []
train_target = list(map(util.get_hot_in_1_from_label, train_e))
val_target = list(map(util.get_hot_in_1_from_label, val_e))
test_target = list(map(util.get_hot_in_1_from_label, test_e))
print("made target vectors")
# start_event = list(map(util.get_discret_event_timesstart_time()))
# for i, e in enumerate(events):
#     target_event.append(util.get_hot_in_1_from_label(e))

model.fit(train_c, train_target, batch_size=1, epochs=1, steps_per_epoch=10, validation_data=(val_c, val_target))
model.save_weights("weights.hdf5")
print("trained and saved data")
score = model.evaluate(test_c, test_target, batch_size=1)

print(score)
