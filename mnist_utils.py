import tensorflow as tf


def process_fn(x):
    x = tf.cast(x, tf.float32) / 255
    return x


def create_simple_data():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x = train_x[..., tf.newaxis]
    train_x = process_fn(train_x)

    test_x = test_x[..., tf.newaxis]
    test_x = process_fn(test_x)
    return (train_x, train_y), (test_x, test_y)


def create_dataset(batch_size):
    auto_tune = tf.data.experimental.AUTOTUNE
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x = train_x[..., tf.newaxis]  # conv2d 需要 （height,width,channel）
    test_x = test_x[..., tf.newaxis]

    ds_train_x = tf.data.Dataset.from_tensor_slices(train_x)
    ds_train_x = ds_train_x.shuffle(len(train_x)).map(process_fn, auto_tune).batch(batch_size).prefetch(auto_tune)
    ds_train_y = tf.data.Dataset.from_tensor_slices(train_y)
    ds_train = tf.data.Dataset.zip((ds_train_x, ds_train_y))

    ds_test_x = tf.data.Dataset.from_tensor_slices(test_x)
    ds_test_x = ds_test_x.shuffle(len(test_x)).map(process_fn, auto_tune).batch(batch_size).prefetch(auto_tune)
    ds_test_y = tf.data.Dataset.from_tensor_slices(test_y)
    ds_test = tf.data.Dataset.zip((ds_test_x, ds_test_y))

    return ds_train, ds_test


if __name__ == '__main__':
    _ds_train, _ds_test = create_dataset(32)
    print(next(iter(_ds_train)))
    print(next(iter(_ds_test)))
