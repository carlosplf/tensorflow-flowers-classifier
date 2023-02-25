import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class SequentialModel:

    def __init__(self):
        self.model = None

    def build(self, img_height, img_width, num_classes):
        # Need to run these layers on CPU, because it's not implemented yet on Apple M1 GPU
        # with tf.device('/CPU:0'):

            # Apply some random changes to images to avoid local optimizations.
            # data_augmentation = keras.Sequential(
            #    [
            #        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", input_shape = (img_height, img_width, 3)),
            #        layers.experimental.preprocessing.RandomRotation(0.2),
            #        layers.experimental.preprocessing.RandomZoom(0.2)
            #    ]
            # )

        self.model = Sequential([
            # data_augmentation,
            layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 2, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            # layers.Conv2D(64, 3, padding='same', activation='relu'),
            # layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def save(self, save_path):
        self.model.save_weights(save_path)
        print("Model saved.")

    def load(self, load_path):
        load_op = self.model.load_weights(load_path)
        print('Model loaded.')
        return load_op
