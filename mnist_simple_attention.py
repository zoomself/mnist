import tensorflow as tf
import mnist_utils as utils


def create_dnn_model():
    x = tf.keras.layers.Input(shape=(28, 28, 1))
    x1 = tf.keras.layers.Input(shape=(1,))
    # 使用 softmax 计算 attention 机率
    attention_prob = tf.keras.layers.Dense(28 * 28, activation="softmax")(x1)
    attention_prob = tf.keras.layers.Reshape(target_shape=(28, 28, 1), name="attention_prob")(attention_prob)
    attention_img = tf.keras.layers.Multiply(name="attention_img")([x, attention_prob])
    img = tf.keras.layers.Flatten()(attention_img)
    img = tf.keras.layers.Dense(16, "relu")(img)
    img = tf.keras.layers.Dense(16, "relu")(img)
    img = tf.keras.layers.Dropout(rate=0.2)(img)

    y = tf.keras.layers.Dense(10, "softmax")(img)
    m = tf.keras.models.Model([x, x1], y)
    m.summary()
    m.compile("adam", tf.keras.losses.SparseCategoricalCrossentropy(), ["acc"])
    tf.keras.utils.plot_model(m, to_file="model_images/mnist_simple_attention_dnn_model.png", show_shapes=True)
    return m


def create_cnn_model():
    x = tf.keras.layers.Input(shape=(28, 28, 1))
    attention_prob = tf.keras.layers.Conv2D(1, 2, 1, padding="same", activation=tf.keras.activations.softmax,
                                            name="attention_prob")(x)
    attention_img = tf.keras.layers.Multiply(name="attention_img")([x, attention_prob])

    img = tf.keras.layers.Conv2D(16, 3, 2, padding="same", activation=tf.keras.activations.relu)(attention_img)
    img = tf.keras.layers.BatchNormalization()(img)
    img = tf.keras.layers.Conv2D(8, 3, 2, padding="same", activation=tf.keras.activations.relu)(img)
    img = tf.keras.layers.BatchNormalization()(img)
    img = tf.keras.layers.GlobalMaxPooling2D()(img)
    y = tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)(img)
    m = tf.keras.models.Model(x, y)
    m.summary()
    m.compile("adam", tf.keras.losses.SparseCategoricalCrossentropy(), ["acc"])
    tf.keras.utils.plot_model(m, to_file="model_images/mnist_simple_attention_cnn_model.png", show_shapes=True)
    return m


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = utils.create_simple_data()
    epochs = 10000
    batch_size = 64
    early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=3)

    model_type = 1
    if model_type == 0:
        model = create_dnn_model()
        temp_train = tf.random.normal(shape=(len(train_x), 1))
        temp_test = tf.random.normal(shape=(len(test_x), 1))
        model.fit(x=[train_x, temp_train], y=train_y, epochs=epochs, batch_size=batch_size,
                  validation_data=([test_x, temp_test], test_y), callbacks=[early_stop_callback])

        # Trainable params: 14,570
        # Epoch 67/10000
        # loss: 0.5436 - acc: 0.8330 - val_loss: 0.3994 - val_acc: 0.8914

    else:
        model = create_cnn_model()
        model.fit(x=train_x, y=train_y, epochs=epochs, batch_size=batch_size,
                  validation_data=(test_x, test_y), callbacks=[early_stop_callback])

        # Trainable params： 1,463
        # Epoch 32/10000
        # loss: 0.2939 - acc: 0.9060 - val_loss: 0.2883 - val_acc: 0.9059
