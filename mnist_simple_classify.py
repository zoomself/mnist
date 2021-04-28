import tensorflow as tf
import mnist_utils as utils


def create_model():
    seq = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(8, 3, 2, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(8, 3, 2, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])
    seq.summary()
    seq.compile("adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["acc"])
    return seq


if __name__ == '__main__':
    _save_model_path = "models/mnist_simple_classify"
    (train_x, train_y), (test_x, test_y) = utils.create_simple_data()
    if tf.io.gfile.exists(_save_model_path):
        _model = tf.saved_model.load(_save_model_path)
        print("load exist model")
    else:
        print("init model")
        _model = create_model()
        early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=3)
        _model.fit(train_x, train_y, batch_size=32, epochs=1000, validation_data=(test_x, test_y),
                   callbacks=[early_stop_callback])

        tf.saved_model.save(_model, _save_model_path)

    n = 5
    predict_y = _model(test_x[:n])
    real_y = test_y[:n]
    print("predict_y:{} ,real_y:{} ".format(tf.argmax(predict_y, axis=-1), real_y))
