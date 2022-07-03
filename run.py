import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# Some Definitions
TRAINING_EPOCHS = 60
BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
AUTOTUNE = tf.data.AUTOTUNE


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

    # Need to run these layers on CPU, because it's not implemented yet
    # on Appl M1 GPU
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


def predict(model, img_url, img_filename):
    """
    Fetch image and predict using the trained Model.
    Args:
        model: Tensorflow trained model
        img_url: URL path.
        img_filename: Disk img filename to be saved.
    Return: None.
    """

    print("Predicting image {} from {}".format(img_filename, img_url))
    
    img_path = tf.keras.utils.get_file(img_filename, origin=img_url)

    img = tf.keras.utils.load_img(
        img_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "Predict: This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


if __name__ == "__main__":
    data_dir = download_dataset()
    check_dataset(data_dir)
    train_ds = create_train_dataset(data_dir)
    class_names = train_ds.class_names
    val_ds = create_validation_dataset(data_dir)
    train_ds, val_ds = tune_models(train_ds, val_ds)

    num_classes = len(class_names)

    model = create_model(num_classes)

    history, model = train_model(model, train_ds, val_ds)

    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    rose_url = "https://upload.wikimedia.org/wikipedia/commons/8/8b/Rose_flower.jpg"
    tulips_url = "https://www.ifpe.edu.br/campus/recife/noticias/tulips.jpg"
    tulips_2_url = "https://upload.wikimedia.org/wikipedia/commons/2/23/Red_Tulips.jpg"
    dandelion_url = "https://upload.wikimedia.org/wikipedia/commons/6/65/Ripe_fruits_by_Common_Dandelion.jpg"

    predict(model, sunflower_url, "sunflower_img")
    predict(model, rose_url, "rose_img")
    predict(model, tulips_url, "tulips_img")
    predict(model, tulips_2_url, "tulips_2_img")
    predict(model, dandelion_url, "dandelion_img")
