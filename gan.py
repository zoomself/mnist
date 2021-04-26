import mnist_utils as utils
import tensorflow as tf
import matplotlib.pyplot as plt


def create_generator(latent_dim):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(latent_dim,)))
    model.add(tf.keras.layers.Dense(7 * 7 * 64, activation="relu"))
    model.add(tf.keras.layers.Reshape((7, 7, 64)))
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.Conv2D(1, kernel_size=3, padding="same", activation=tf.keras.activations.tanh))
    model.summary()
    return model


def create_discriminator():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()
    return model


class Gan(object):
    def __init__(self, epochs, batch_size, latent_dim, check_point_root, save_interval, log_dir):
        if not tf.io.gfile.exists(check_point_root):
            tf.io.gfile.makedirs(check_point_root)
        self.check_point_root = check_point_root

        if not tf.io.gfile.exists(log_dir):
            tf.io.gfile.makedirs(log_dir)
        self.log_dir = log_dir

        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.save_interval = save_interval

        ds_train, ds_test = utils.create_dataset(self.batch_size)
        self.ds_train = ds_train
        self.ds_test = ds_test

        self.generator = create_generator(self.latent_dim)
        self.discriminator = create_discriminator()
        self.loss_obj = tf.keras.losses.BinaryCrossentropy()
        self.generator_optimizer_obj = tf.keras.optimizers.Adam()
        self.discriminator_optimizer_obj = tf.keras.optimizers.Adam()
        self.metrics_obj = tf.keras.metrics.BinaryAccuracy()

        self.check_point = tf.train.Checkpoint(
            step=tf.Variable(0, dtype=tf.int32),
            generator=self.generator,
            discriminator=self.discriminator
        )

        self.check_point_manager = tf.train.CheckpointManager(checkpoint=self.check_point,
                                                              directory=self.check_point_root, max_to_keep=3)

        latest_checkpoint = self.check_point_manager.latest_checkpoint
        if latest_checkpoint:
            print("Initializing from check point : {}".format(latest_checkpoint))
            self.check_point.restore(latest_checkpoint)  # 从最近的检查点恢复数据
        else:
            print("Initializing from scratch...")

    def generator_loss_fn(self, dis_fake):
        return self.loss_obj(tf.ones_like(dis_fake), dis_fake)

    def discriminator_loss_fn(self, dis_fake, dis_real):
        loss_fake = self.loss_obj(tf.zeros_like(dis_fake), dis_fake)
        loss_real = self.loss_obj(tf.ones_like(dis_real), dis_real)
        return loss_fake + loss_real

    def create_noise_img(self, batch):
        return tf.random.normal(shape=(batch, self.latent_dim))

    def show_gen_img(self):
        noise = self.create_noise_img(self.batch_size)
        fake_img = self.generator(noise)
        plt.imshow(fake_img[0])
        plt.show()

    @tf.function
    def train_step(self, ds):
        with tf.GradientTape(persistent=True) as tape:
            x_true = ds[0]
            batch = x_true.shape[0]  # 动态计算每次迭代的batch_size
            noise = self.create_noise_img(batch)
            fake_img = self.generator(noise)

            dis_fake = self.discriminator(fake_img)
            dis_real = self.discriminator(x_true)

            loss_generator = self.generator_loss_fn(dis_fake)
            loss_discriminator = self.discriminator_loss_fn(dis_fake, dis_real)

        generator_grad = tape.gradient(loss_generator, self.generator.trainable_weights)
        self.generator_optimizer_obj.apply_gradients(zip(generator_grad, self.generator.trainable_weights))

        discriminator_grad = tape.gradient(loss_discriminator, self.discriminator.trainable_weights)
        self.discriminator_optimizer_obj.apply_gradients(zip(discriminator_grad, self.discriminator.trainable_weights))

        self.metrics_obj(tf.ones_like(dis_fake), dis_fake)  # 评价生成的图片与1之间的差距
        return loss_generator, loss_discriminator

    def train(self):
        file_writer = tf.summary.create_file_writer(logdir=self.log_dir)
        with file_writer.as_default():
            for epoch in range(self.epochs):
                for ds in self.ds_train:
                    loss_generator, loss_discriminator = self.train_step(ds=ds)
                    acc = self.metrics_obj.result() * 100
                    step = self.check_point.step.numpy()
                    log_info = "epoch:{} ,step:{}, loss_g:{} , loss_d:{} , acc:{} % ".format(epoch, step,
                                                                                             loss_generator,
                                                                                             loss_discriminator, acc)

                    print(log_info)
                    tf.summary.scalar(name="loss_generator", data=loss_generator, step=step)
                    tf.summary.scalar(name="loss_discriminator", data=loss_discriminator, step=step)
                    tf.summary.scalar(name="acc", data=acc, step=step)

                    if step % self.save_interval == 0:
                        self.check_point_manager.save()
                        noise_img = self.create_noise_img(self.batch_size)
                        gen_img = self.generator(noise_img)
                        tf.summary.image(name="gen_img", data=gen_img, step=step, max_outputs=5)

                    self.check_point.step.assign_add(1)

            self.show_gen_img()


if __name__ == '__main__':
    _check_point_root = "check_points/gan"
    _log_dir = "logs/gan"
    gan = Gan(epochs=10,
              latent_dim=100,
              batch_size=64,
              check_point_root=_check_point_root,
              save_interval=50,
              log_dir=_log_dir
              )
    print('Training ...')
    gan.train()
