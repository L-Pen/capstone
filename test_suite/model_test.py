from helper import load_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import keras
import math
from tensorflow.keras import layers, losses


class PrintAfterEachEpoch(keras.callbacks.Callback):
    def __init__(self, range):
        super(PrintAfterEachEpoch, self).__init__()
        self.range = range
    def on_epoch_end(self, epoch, logs=None):
        #save the model if it is the best so far
        value_avg_error = int(math.sqrt(logs["val_loss"])*1024/(self.range[1]-self.range[0]))
        print(
            "The average loss for epoch {} is {:7.4f} val loss: {:7.4f} giving avg error {}".format(
                epoch, logs["loss"],logs['val_loss'],value_avg_error
            )
        )
def run_model_test(params):
    X,y,times = load_data(params['group_size'],params['output_range'])
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['TEST_PROPORTION'], random_state=42)

    nodes_per_layer=[]
    for i in range(params['between_layers']):
        nodes_per_layer.append(max(int(params['first_layer_size']*params['layer_size_factor']**i),4))

    model_layers = []
    model_layers.append(layers.InputLayer(input_shape=params['num_features']))
    for size in nodes_per_layer:
        model_layers.append(size, activation='relu')
    model_layers.append(layers.Dense(params['bottleneck_size'], activation=params['bottleneck_activation'], name='Bottleneck'))
    for size in nodes_per_layer[::-1]:
        model_layers.append(size, activation='relu')
    model_layers.append(layers.Dense(params['num_features'], activation=params['output_activation']))
    model = Sequential(model_layers)

    model.compile(optimizer='adagrad', loss=losses.MeanSquaredError(), metrics=['mse'])
    checkpoint = ModelCheckpoint(f'checkpoints/best_model_{params["id"]}.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    # Train the model and print after each epoch but not verbose
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    model.fit(X_train, X_train,
                epochs=params['epochs'],
                shuffle=True,
                validation_data=(X_test, X_test),
                batch_size=params['batch_size'],
                callbacks=[PrintAfterEachEpoch(params['model']['output_range']),checkpoint,early_stop],
                verbose=0)
    return model.evaluate(X_test, X_test, verbose=0)[1]
