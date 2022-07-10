# Tensorflow Flowers Classifier

Flower image classifier based on Tensorflow tutorial.

### How to run

Note: some requirements are for Apple M1 chip.

`pip install -r requirements.txt`

`python run.py --help` to access the Help menu.

`python run.py -t <N_EPOCHS>` to train the model and save the weights.
If you don't want to save the weights, add the `--nosave` option.
Note that when predicting, the system will try to load saved weights.

`python run.py -p <image_path>` to ask the Model to classify some flower image.
