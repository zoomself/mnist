import tensorflow as tf
import mnist_utils as utils


def create_model():
    seq = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(8, 3, 2, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(4, 3, 2, padding="same", activation="relu"),
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])
    seq.summary()
    seq.compile("adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["acc"])
    return seq


if __name__ == '__main__':
    model = create_model()
    (train_x, train_y), (test_x, test_y) = utils.create_simple_data()
    early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=3)
    model.fit(train_x, train_y, batch_size=32, epochs=1000, validation_data=(test_x, test_y),
              callbacks=[early_stop_callback])
