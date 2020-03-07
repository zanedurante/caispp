# Caispp

## About
This package allows for high level ML model creation.  It uses Keras with a Tensorflow backend, and was originally created to be used for the curriculum of USC's CAIS++ (Center for AI in Society, Student Branch).  

## Use Cases
The package currently supports Image Classification.

## Installation
To install run `pip install caispp`.  This package uses Tensorflow 2.0. 

## Example usage

You can see a jupyter notebook with ouputs in the `examples/` directory.  The notebook runs the code below:

```
from caispp import ImageDataset, ImageClassifier, Path

path = Path('example_dataset/') # Path to dataset
dataset = ImageDataset(path, show_distribution=True)

classifier = ImageClassifier(dataset)
classifier.train(epochs=10)

classifier.show_history()

classifier.test(show_distribution=True)
```
## Dataset directory structure
```
├── example_dataset         
│   ├── test
│   │   ├── class1      # Directory with images of class1
│   │   ├── class2      # Directory with images of class2
│   │   └── ...       
│   ├── train
│   │   ├── class1      # Directory with images of class1
│   │   ├── class2      # Directory with images of class2
│   │   └── ...         
│   ├── valid           # Optional validation set    
│   │   ├── class1
│   │   ├── class2
│   │   └── ... 
└──  
```
Each of the `test/`, `train/`, and `valid/` directories contain subdirectories for each class.  In those subdirectories, put the images files of that class.  

## Build the package

To build the package run the `build.sh` script in the directory.  The output is stored in `dist/`.
