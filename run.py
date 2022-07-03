import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
import argparse

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# Some Definitions
TRAINING_EPOCHS = 2
BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
AUTOTUNE = tf.data.AUTOTUNE
MODEL_SAVE_PATH = "./model_save/weights"


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", type=int,
                    help="Train the model using N epochs.")
parser.add_argument("-p", "--predict", type=str,
                    help="Predict an image class. -p <IMG_PATH>")
args = parser.parse_args()


def download_dataset():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    return data_dir


def check_dataset(data_dir):
    print("Checking dataseti size...")
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print("Dataset size: ", image_count)


def create_train_dataset(data_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    return train_ds


def create_validation_dataset(data_dir):
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    return val_ds


def tune_models(train_ds, val_ds):
    tunned_train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    tunned_val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return tunned_train_ds, tunned_val_ds


def create_model(num_classes):

    # Need to run these layers on CPU, because it's not implemented yet on Appl M1 GPU
    with tf.device('/CPU:0'):
        data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)),
                layers.experimental.preprocessing.RandomRotation(0.2),
                layers.experimental.preprocessing.RandomZoom(0.2)
            ]
        )

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model


def train_model(model, train_ds, val_ds):
    epochs = TRAINING_EPOCHS
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    return history, model


def predict_from_file(model, img_filename):
    """
    Load an image and predict using the trained Model.
    Args:
        model: Tensorflow trained model
        img_filename: name of the img file to be loaded.
    Return: None.
    """

    img = tf.keras.utils.load_img(
        img_filename, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "PREDICT: This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


def save_model(model):
    model.save_weights(MODEL_SAVE_PATH)
    print("Model saved.")


def load_model(model):
    saved_model = model.load_weights(MODEL_SAVE_PATH)
    print('Model loaded.')
    return saved_model


if __name__ == "__main__":

    if args.train:
        TRAINING_EPOCHS = args.train
        print("Starting training...")
        data_dir = download_dataset()
        check_dataset(data_dir)
        train_ds = create_train_dataset(data_dir)
        class_names = train_ds.class_names
        val_ds = create_validation_dataset(data_dir)
        train_ds, val_ds = tune_models(train_ds, val_ds)

        num_classes = len(class_names)

        model = create_model(num_classes)

        history, model = train_model(model, train_ds, val_ds)
        
        save_model(model)
        
        print("Finished training.")

    if args.predict:
        print("Predicting images...")

        # TODO: Loading train_ds just to get number of classes. Need to change that.
        data_dir = download_dataset()
        train_ds = create_train_dataset(data_dir)
        class_names = train_ds.class_names
        num_classes = len(class_names)
        
        model = create_model(num_classes)
        load_model(model)
        
        predict_from_file(model, args.predict) 
        
        print("Finisihed predictions.")
