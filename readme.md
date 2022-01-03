# Drawining recognition project

## Requirements
- Python 3.9.7 64bits

## About the project

This project uses the [quickdraw data set from google](https://github.com/googlecreativelab/quickdraw-dataset/blob/master/categories.txt) through the quickdraw pytho library. If you want to create your training photos go to the model-with-ndjson.py file and run the script. This will gather the quickdraw dataset in `bin` format and convert it to randomize `.png` files which can be use to train your model.

To train your model head over to the `model.py` and run the entire script. This will create the neural network using the training photos with a 70-30 ratio (70 for training data and 30 as source of thruth)

### Other assets

- [Install python 3.9.7](https://www.python.org/downloads/release/python-397/)
- [Neural Networks Tutorial from Free codecamp](https://www.youtube.com/watch?v=tPYj3fFJGjk&t=16919s)