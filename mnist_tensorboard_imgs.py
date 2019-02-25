import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import time


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    # height, width, channel = tensor.shape
    height, width = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=1,
                         encoded_image_string=image_string)


class TensorBoardImage(tf.keras.callbacks.Callback):
    def __init__(self, tag):
        super().__init__() 
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        # Load image
        img_stack = self.validation_data[0][:3]
        summary_str = []
        for i in range(img_stack.shape[0]):
            img = img_stack[i,:,:]
            img = (255 * img).astype('uint8')
            summary_str.append(tf.Summary.Value(tag=self.tag + str(i),
                               image=make_image(img)))
            # multiple summaries can be appended
        writer = tf.summary.FileWriter(f'./logs/{NAME}')
        writer.add_summary(tf.Summary(value=summary_str), epoch)
        return  


NAME = f'mnist_simple_{time.strftime("%d%m%y_%H%M")}'

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[0:100]
y_train = y_train[0:100]

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callback_obj = TensorBoardImage('input_img')
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, callbacks=[callback_obj])
