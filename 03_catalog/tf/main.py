#https://www.tensorflow.org/tutorials/quickstart/advanced?hl=ko
import models.my_network as MyNetwork
import tensorflow as tf
import yaml

import argparse

@tf.function
def train_step(images, labels, model, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels, model, loss_object, optimizer, test_loss, test_accuracy):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
def pop_tail(tensor, pop_factor=1):
    sz = tensor.shape[0]
    return tensor[:sz//pop_factor]

def mk_dataset(cfg):
    batch_size = cfg['hyper_parameters']['batch_size']
    data_scale_factor = cfg['data_scale_factor']

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    x_train = pop_tail(x_train, data_scale_factor)
    x_test = pop_tail(x_test, data_scale_factor)
    y_train = pop_tail(y_train, data_scale_factor)
    y_test = pop_tail(y_test, data_scale_factor)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return train_ds, test_ds

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/cnn.yaml')
    return parser.parse_args()
def config_print(config, depth = 0):
    for k,v in config.items():
        prefix = ["\t"*depth,k,":"]

        if type(v)==dict:
            print(*prefix)
            config_print(v,depth+1)
        else:
            prefix.append(v)
            print(*prefix)

def main():
    parser = get_parser()
    with open(parser.config) as f:
        config = yaml.safe_load(f)
        config_print(config)

    batch_size = config['hyper_parameters']['batch_size']
    epochs = config['hyper_parameters']['epochs']
    learning_rate = config['hyper_parameters']['learning_rate']

    data_scale_factor = config['data_scale_factor']

    train_ds, test_ds = mk_dataset(config)

    model = MyNetwork.MyModel(config['network_parameters'])

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    EPOCHS = epochs

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels, model, loss_object, optimizer, train_loss, train_accuracy)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels, model, loss_object, optimizer, test_loss, test_accuracy)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
if __name__ == '__main__':
    main()
