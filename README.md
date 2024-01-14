# Object Recognition App

This is a simple object recognition application built with Tkinter, Matplotlib, and Keras. The app allows users to select an image file and displays the image along with the predicted object category and a probability distribution chart.

Image database with faces: https://www.kaggle.com/datasets/olgabelitskaya/yale-face-database.
My face was additionally added to the dataset.

## Features

- Select an image file using the "Select Image" button.
- Display the selected image with the recognized object category and probability distribution.
- The filename is displayed below the image.
- Utilizes a pre-trained Keras model for object recognition.

- Confusion matrix created based on test dataset:

![Alt text](confusion_matrix.png?raw=true)

## Prerequisites

- Python 3.x
- Required Python packages: tkinter, pillow, matplotlib, keras, numpy

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/object-recognition-app.git
   cd object-recognition-app
