from helper import load_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow import keras


class PrintAfterEachEpoch(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} ".format(
                epoch, logs["loss"]
            )
        )
def run_model_test(params):
    X,y = load_data(params['group_size'])
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['TEST_PROPORTION'], random_state=42)

    model = Sequential(params['model']['layers'])

    model.compile(optimizer=params['model']['optimizer'], loss=params['model']['loss'], metrics=params['model']['metrics'])

    # Train the model and print after each epoch but not verbose
    model.fit(X_train, X_train,
                epochs=params['model']['epochs'],
                shuffle=True,
                validation_data=(X_test, X_test),
                batch_size=params['model']['batch_size'],
                callbacks=[PrintAfterEachEpoch()],
                verbose=0)
    return model.evaluate(X_test, X_test, verbose=0)[1]
