import tensorflow as tf
import time
import numpy as np
import os
tf.enable_eager_execution()
'''
1) Mnist exampple for custom training
2) Writing tensor values into tensorboard
3) Make use of tf.data
'''


def preprocess_dataset(dataset, batch_size):
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.repeat(10)
    return dataset


def get_dataset(batch_size):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = tf.cast(x_train, 'float32')
    y_train = tf.cast(y_train, 'int32')
    x_test = tf.cast(x_test, 'float32')
    y_test = tf.cast(y_test, 'int32')

    # train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = preprocess_dataset(train_dataset, batch_size=batch_size)

    # test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = preprocess_dataset(test_dataset, batch_size=batch_size)
    return train_dataset, test_dataset


def create_model():
    model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
            ])
    return model


def grad(model, x, y):
    with tf.GradientTape() as tape:
        loss_val, accuracy = loss(model, x, y)
    return loss_val, accuracy, tape.gradient(loss_val, model.trainable_variables)


def loss(model, x, y):
    y_ = model(x)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
    y_ = tf.argmax(y_, axis=1, output_type=tf.int32)
    accuracy = np.mean(tf.math.equal(y, y_))
    return loss, accuracy


batch_size = 128
training_samples = 60000
no_batches = training_samples//batch_size

train_dataset, test_dataset = get_dataset(batch_size)

model = create_model()
NAME = f'mnist_simple_{time.strftime("%d%m%y_%H%M")}'
# NAME = 'mnist_simple_050319_1113'
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, f'./{NAME}', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
    global_step = tf.Variable(ckpt.step * no_batches)
else:
    print("Initializing from scratch.")
    global_step = tf.Variable(0)
    ckpt.__setattr__('path', NAME)

epochs = 100
train_writer = tf.contrib.summary.create_file_writer(f'{NAME}/logs')

for train_batch in train_dataset:
    input_img = train_batch[0]
    label = train_batch[1]
    # loss_val is the batch loss
    loss_val, accuracy, grads = grad(model, input_img, label)
    optimizer.apply_gradients(zip(grads, model.trainable_variables),
                              global_step)

    # completion of an epoch
    if tf.math.equal(global_step % no_batches, tf.constant(0)):
        print(f'the loss is {loss_val} and accuracy is {accuracy}')
        ckpt.step.assign(global_step // no_batches)
        save_path = manager.save()
        print("Saved checkpoint for epoch {}: {}".format(int(ckpt.step), save_path))
        with train_writer.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar("loss", loss_val,
                                      step=tf.cast(global_step // no_batches, 'int64'))
            tf.contrib.summary.scalar("accuracy", accuracy,
                                      step=tf.cast(global_step // no_batches, 'int64'))
            tf.contrib.summary.image("digits", input_img[..., np.newaxis], max_images=3,
                                     step=tf.cast(global_step // no_batches, 'int64'))
            print(global_step // no_batches)
            # Forces summary writer to send any buffered data to storage
            tf.contrib.summary.flush()
