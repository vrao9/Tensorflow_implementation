import tensorflow as tf
tf.enable_eager_execution()

'''
Mnist example whcih uses train_on_batch() API of the tf.keras.model
The visualisation is done in the tensorboard callback
'''


def preprocess_dataset(dataset, batch_size):
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.repeat(1)
    return dataset


def get_dataset(batch_size):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = tf.cast(x_train, 'float32')
    y_train = tf.cast(y_train, 'int64')
    x_test = tf.cast(x_test, 'float32')
    y_test = tf.cast(y_test, 'int64')

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

model = create_model()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

batch_size = 128
train_dataset, test_dataset = get_dataset(batch_size)

tensorboard = tf.keras.callbacks.TensorBoard(
  log_dir='./tmp/my_tf_logs',
  histogram_freq=0,
  batch_size=batch_size,
  write_graph=True,
  write_grads=True
)
tensorboard.set_model(model)


def named_logs(model, logs):
    result = {}
    for l in zip(model.metrics_names, logs):
        result[l[0]] = l[1]
    return result

batch_count = 0

for train_batch in train_dataset:
    batch_count += 1
    print(batch_count)
    input_img = train_batch[0]
    label = train_batch[1]
    logs = model.train_on_batch(input_img, label)
    tensorboard.on_batch_end(batch_count, named_logs(model, logs))
tensorboard.on_train_end(None)
