import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class SequentialModel:

    def __init__(self):
        self.model = None

    def build(self, img_height, img_width, num_classes):
        # Apple M1: Preprocessing Layers need to run these layers on CPU. It's not implemented yet on Apple M1 GPU.
        # with tf.device('/CPU:0'):

        # Apply some random changes to images to avoid local optimizations and overfit.
        data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomCrop(round(img_height*0.8), round(img_width*0.8)),
                layers.experimental.preprocessing.RandomFlip(mode="horizontal_and_vertical", seed=None, name=None)
            ]
        )

        self.model = Sequential([
            data_augmentation,
            layers.Conv2D(64, 5, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(64, activation='relu'),
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
