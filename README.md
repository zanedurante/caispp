# Caispp

## About
This is package allows for high level ML model creation.  It uses Keras with a Tensorflow backend, and was originally created to be used for the curriculum of USC's CAIS++ (Center for AI in Society, Student Branch).  

## Use Cases
The package currently supports Image Classification.

## Example usage

You can see a jupyter notebook with ouputs in the `examples/` directory.  The notebook runs the code below:

```
from caispp import ImageDataset, ImageClassifier
from pathlib import Path

path = Path('example_dataset/') # Path to dataset
dataset = ImageDataset(path, show_distribution=True)

classifier = ImageClassifier(dataset)
classifier.train(epochs=10)

classifier.show_history()

classifier.test(show_distribution=True)
```

# Build the package

To build the package run the build.sh script in the directory.  The output is stored in `dist/`.