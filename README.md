# CAISPP

## About
This is package allows for high level ML model creation.  It uses Keras with a Tensorflow backend, and was originally created to be used for the curriculum of USC's CAIS++ (Center for AI in Society, Student Branch).  

## Use Cases
The package currently supports Image Classification.

## Example usage

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