# Tensorflow Flowers Classifier

This is a simple Tenforflow Convolutional Neural Network implementation to classify flowers.


## How to run

Note: the project contains two requirements files, one for ROCM and other for Apple M1 chip.

I managed to run TF with my RX580 GPU using [this guide](https://github.com/Grench6/RX580-rocM-tensorflow-ubuntu20.4-guide).

`pip install -r <REQUIREMENTS_FILE.txt>`


#### Running manually:

`python runner.py --help` to access the Help menu.

`python runner.py -t <N_EPOCHS>` to train the model and save the weights.
If you don't want to save the weights, add the `--nosave` option.
Note that when predicting, the system will try to load saved weights.

`python runner.py -p <image_path>` to ask the Model to classify some flower image.


## Performance

I trained the model with 100 epochs, and got the following results:

![This is an image](/screenshots/ml-flowers-accuracy.png)

The `test_images/` folder have some extra flower images to test the network.
