# Image Classifier Project

This project is part of the Udacity AI Programming with Python Nanodegree. In this project, I have built and trained a deep learning model to classify flower images. The model is trained on a dataset of 102 different flower species.

## Table of Contents
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Predicting with the Model](#predicting-with-the-model)
- [Author](#author)

## Dependencies

The project is written in Python 3 and uses the following libraries:

*   PyTorch
*   torchvision
*   NumPy
*   Matplotlib
*   Pillow (PIL)
*   Seaborn
*   argparse
*   json

You can install the required libraries using pip:

```bash
pip install torch torchvision numpy matplotlib pillow seaborn
```

## Dataset

The dataset used for this project is the [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). It consists of 102 different species of flowers, with each species having between 40 and 258 images. The dataset is split into training, validation, and testing sets.

The `cat_to_name.json` file contains a mapping from the category labels to the actual names of the flowers.

## Training the Model

The `train.py` script is used to train the image classifier. You can customize the training process using the following command-line arguments:

*   `--arch`: Choose the model architecture. The default is "vgg16".
*   `--save_dir`: Set the directory to save the checkpoint. The default is `./checkpoint.pth`.
*   `--learning_rate`: Set the learning rate for the optimizer. The default is 0.001.
*   `--hidden_units`: Set the number of hidden units in the classifier. The default is 120.
*   `--epochs`: Set the number of epochs for training. The default is 1.
*   `--gpu`: Use GPU for training. Default is "gpu".

Example usage:

```bash
python train.py --arch "vgg16" --hidden_units 512 --epochs 20 --gpu
```

## Predicting with the Model

The `predict.py` script is used to predict the class of an image using the trained model. You can use the following command-line arguments:

*   `--image`: Path to the image file for prediction. This is a required argument.
*   `--checkpoint`: Path to the checkpoint file. This is a required argument.
*   `--top_k`: Choose the top K most likely classes to display.
*   `--category_names`: Path to the JSON file that maps categories to real names. The default is `cat_to_name.json`.
*   `--gpu`: Use GPU for inference.

Example usage:

```bash
python predict.py --image /path/to/image.jpg --checkpoint checkpoint.pth --top_k 5 --gpu
```

## Author

Kumarvaishnav
