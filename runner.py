import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
import argparse

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from models.SequentialModel import SequentialModel


# Adjust TF log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Some Definitions
TRAINING_EPOCHS = 2
BATCH_SIZE = 64
IMG_HEIGHT = 256
IMG_WIDTH = 256
MODEL_SAVE_PATH = "./model_save/weights"


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", type=int,
                    help="Train the model using N epochs.")
parser.add_argument("--nosave",
                    help="Set no_save flag. Trained models won't be saved.",
                    action="store_true")
parser.add_argument("-p", "--predict", type=str,
                    help="Predict an image class. -p <IMG_PATH>")
args = parser.parse_args()

def download_dataset():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    return data_dir


def check_dataset(data_dir):
    print("Checking dataset size...")
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print("Dataset size: ", image_count)


def create_train_dataset(data_dir):
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=False,
    )

    train_ds = img_gen.flow_from_directory(data_dir, batch_size=BATCH_SIZE, shuffle=True, class_mode='sparse', subset="training", target_size=(IMG_HEIGHT, IMG_WIDTH))
    print(len(train_ds))

    return train_ds


def create_validation_dataset(data_dir):
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=False,
        validation_split=0.2
    )

    val_ds = img_gen.flow_from_directory(data_dir, batch_size=BATCH_SIZE, shuffle=True, class_mode='sparse', subset="validation", target_size=(IMG_HEIGHT, IMG_WIDTH))
    print(len(val_ds))
    return val_ds


def tune_models(train_ds, val_ds):
    tunned_train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    tunned_val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return tunned_train_ds, tunned_val_ds


def create_model(num_classes):
    seq_model = SequentialModel()
    seq_model.build(IMG_HEIGHT, IMG_WIDTH, num_classes)
    return seq_model


def train_model(n_epochs, seq_model, train_ds, val_ds):
    history = seq_model.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=n_epochs
    )
    return history


def predict_from_file(seq_model, img_filename, class_names):
    """
    Load an image and predict using the trained Model.
    Args:
        seq_model: SequentialModel class instance.
        img_filename: name of the img file to be loaded.
    Return: None.
    """

    img = tf.keras.preprocessing.image.load_img(
        img_filename, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = seq_model.model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "PREDICT: This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    return [class_names[np.argmax(score)], 100 * np.max(score)]


def run_training(n_epochs):
    
    print("Starting training...")
   
    data_dir = download_dataset()
    check_dataset(data_dir)
    train_ds = create_train_dataset(data_dir)
    # class_names = train_ds.class_names
    val_ds = create_validation_dataset(data_dir)

    num_classes = 5

    seq_model = create_model(num_classes)

    history = train_model(n_epochs, seq_model, train_ds, val_ds)
    
    if not args.nosave:
        seq_model.save(MODEL_SAVE_PATH) 

    print("Finished training.")

    return history.history


def run_predict(filename):

    print("Predicting images...")

    # TODO: Loading train_ds just to get number of classes. Need to change that.
    data_dir = download_dataset()
    train_ds = create_train_dataset(data_dir)
    class_names = list(train_ds.class_indices.keys())
    num_classes = len(class_names)

    seq_model = create_model(num_classes)

    # Load model weights from Tensorflow saving.
    seq_model.load(MODEL_SAVE_PATH)
    
    predict_from_file(seq_model, filename, class_names) 
    
    print("Finisihed predictions.")



if __name__ == "__main__":

    if args.train:
        run_training(args.train)

    if args.predict:
        run_predict(args.predict)
