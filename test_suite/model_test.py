from helper import load_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
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
    X,y,times = load_data(params['group_size'],params['model']['output_range'])
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['TEST_PROPORTION'], random_state=42)


    model_layers = []
    for layer in params['model']['layers']:
        if layer['type'] == 'input':
            model_layers.append(layers.InputLayer(input_shape=layer['shape']))
        elif layer['type'] == 'dense':
            model_layers.append(layers.Dense(layer['units'], activation=layer['activation'], name=layer['name'] if 'name' in layer else None))

    model = Sequential(model_layers)

    model.compile(optimizer='adagrad', loss=losses.MeanSquaredError(), metrics=['mse'])
    checkpoint = ModelCheckpoint(f'checkpoints/best_model_{params["id"]}.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    # Train the model and print after each epoch but not verbose
    model.fit(X_train, X_train,
                epochs=params['epochs'],
                shuffle=True,
                validation_data=(X_test, X_test),
                batch_size=params['batch_size'],
                callbacks=[PrintAfterEachEpoch(params['model']['output_range']),checkpoint],
                verbose=0)
    return model.evaluate(X_test, X_test, verbose=0)[1]
