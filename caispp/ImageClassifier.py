from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path, PosixPath
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import errno
import os



class ImageClassifier():
    
    # Argument takes an image dataset 
    def __init__(self, dataset, num_features=64, show_model=False):
        """ Creates model from dataset.
            dataset : ImageDataset class.  Dataset that the model will use.
            num_features : Number of features to use for classification.  This is
                           the number of features extracted from the inceptionV3 model.
                           If the number of classes > 16 consider increasing num_features.
            show_model : If True, outputs the model.summary() of the keras model used for classification.
        """
        self.dataset = dataset
        
        # Create model
        self.model = self.__createModel(num_features, show_model)
        self.history = None
        
    # use keras .fit() method on self.model
    def train(self, lr=0.0001, epochs=10, batch_size=8, 
              show_output=True):
        """ Trains model on ImageDataset.  Stops early if validation loss increases for > 4 epochs (uses best model).
            
            lr : learning rate used during training.
            epochs : number of epochs to train for.
            batch_size : number of examples used in each training batch.
            show_ouptput : Determines whether to show training output (progress bars + values).
        """
        if show_output:
            show_output = 1
        else:
            show_output = 0
        
        # Compile model
        self.model.compile(optimizer=tf.optimizers.SGD(lr=lr, momentum=0.5), 
                          loss=tf.losses.sparse_categorical_crossentropy,
                          metrics=['acc'])
        
        # Convert training data to numpy arrays
        x_data = []
        y_data = []
        for key in self.dataset.train_data.keys():
            x_data += [x for x in self.dataset.train_data[key]]
            y_data += [self.dataset.class2idx[key] for x in self.dataset.train_data[key]]
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        x_val_data = []
        y_val_data = []
        if self.dataset.valid_data:
            for key in self.dataset.valid_data.keys():
                x_val_data += [x for x in self.dataset.valid_data[key]]
                y_val_data += [self.dataset.class2idx[key] for x in self.dataset.valid_data[key]]
            
            x_val_data = np.array(x_val_data)
            y_val_data = np.array(y_val_data)
            
        # Fit model on training data (optionally use validation data)
        if self.dataset.valid_data:
            # Add early stopping if has validation data
            max_bad_epochs = 5 # The maximum number of 'bad' epochs before stops training (bad = val loss increases)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=max_bad_epochs, restore_best_weights=True)



            self.history = self.model.fit(x=x_data, y=y_data, batch_size=batch_size, epochs=epochs, 
                                        validation_data=(x_val_data, y_val_data), shuffle=True, 
                                        verbose=show_output, callbacks=[early_stopping])
        else:
            max_bad_epochs = 5 # The maximum number of 'bad' epochs before stops training (bad = loss increases)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=max_bad_epochs, restore_best_weights=True)
            self.history = self.model.fit(x=x_data, y=y_data, batch_size=batch_size, 
                                        epochs=epochs, shuffle=True, verbose=show_output,
                                        callbacks=[early_stopping])
        
    def test(self, show_distribution=False):
        """ Tests model on the test portion of the dataset that was given in the constructor.
            
            show_distribution : If True, shows distribution of model predictions compared to the 
                                distribution of the test set.  Also shows the distribution of 
                                incorrect classifications.
        """
        x_data = []
        y_data = []
        for key in self.dataset.test_data.keys():
            x_data += [x for x in self.dataset.test_data[key]]
            y_data += [self.dataset.class2idx[key] for x in self.dataset.test_data[key]]
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        
        preds = self.model.predict(x_data)
        preds = np.argmax(preds, axis=-1)
        print('Accuracy: ', round(accuracy_score(preds, y_data), 3), ', f1-score: ', 
              round(f1_score(preds, y_data, average='weighted'), 3))
        # Check if label not in output, otherwise use least common 
        all_labels_in_output = True
        
        for label in self.dataset.idx2class:
            num_val = self.dataset.class2idx[label]
            if num_val not in preds:
                all_labels_in_output = False
            
        if show_distribution:
            # Create side by side plots of the predictions vs the actual
            
            # Create dictionary of how often each class is used
            pred_counts = {label: 0 for label in self.dataset.idx2class}
            for pred in preds:
                pred_counts[self.dataset.idx2class[pred]] += 1
                
            actual_counts = {label: 0 for label in self.dataset.idx2class}
            for y in y_data:
                actual_counts[self.dataset.idx2class[y]] += 1
            
            plt.figure(figsize=(10, 3))
            plt.subplot(1, 3, 1)
            objects = self.dataset.idx2class
            x_pos = np.arange(len(objects))
            num_outputs = [pred_counts[obj] for obj in objects]

            plt.bar(x_pos, num_outputs, align='center')
            plt.xticks(x_pos, objects)
            plt.ylabel('Number of predictions')
            plt.title('Prediction distribution')
            
            plt.subplot(1, 3, 2)
            num_outputs = [actual_counts[obj] for obj in objects]
            plt.bar(x_pos, num_outputs, align='center')
            plt.xticks(x_pos, objects)
            plt.ylabel('Number of occurences')
            plt.title('Actual distribution')
            
            plt.subplot(1, 3, 3)
            incorrect_counts = {label: 0 for label in self.dataset.idx2class}
            for pred, y in zip(preds, y_data):
                if pred != y:
                    incorrect_counts[self.dataset.idx2class[y]] += 1
                    
            num_outputs = [incorrect_counts[obj] for obj in objects]
            plt.bar(x_pos, num_outputs, align='center')
            plt.xticks(x_pos, objects)
            plt.ylabel('Number of incorrect classifications')
            plt.title('Error distribution')
            
            plt.tight_layout()
            plt.show()
            
    def show_history(self, metric='loss'):
        """ Shows the training history of the model.
            
            metric : String that can be 'loss', 'acc', or 'all'.
                     Shows metric(s) across last training run.
                    'loss' : shows training loss.
                    'acc' : shows training accuracy.
                    'all' : shows all possible metrics [loss, acc].
        """
        # Supported metrics so far
        if metric not in ['loss', 'acc', 'all']:
            raise Exception('Metric not supported, supported types are [\'loss\', \'acc\', \'all\'].')
        
        if not self.history:
            raise Exception('Model must be trained before show_history can be called.')
        
        if metric == 'all':
            fig, ax1 = plt.subplots()
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss', color='r')
            ax1.plot(self.history.history['loss'], 'r--')
            ax1.tick_params(axis='y', labelcolor='r')

            ax2 = ax1.twinx()

            ax2.set_ylabel('Accuracy', color='b')
            ax2.plot(self.history.history['acc'], 'b-')
            ax2.tick_params(axis='y', labelcolor='b')
            
            fig.tight_layout()
            plt.title('Model loss and accuracy')
            
        else:
            if metric == 'acc':
                plt.plot(self.history.history[metric], 'b-')
            else:
                plt.plot(self.history.history[metric], 'r--')
                
            if metric == 'acc':
                metric = 'accuracy'

            plt.title('Model ' + metric)
            plt.ylabel(metric)
            plt.xlabel('epoch')
        
        plt.show()

    def __createModel(self, num_features, show_model):
        """ Creates model that will be used for transfer learning.
        
            num_features : number of features extracted from the final layer of the inceptionv3 model.
            show_model : if True outputs the model.summary() of the classification model.
            
        """
        # Create inceptionv3 model
        image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                       weights='imagenet',
                                                       input_shape = (299, 299, 3))
        
        # Make only the last few layers trainable
        num_trainable_layers = 3
        for layer in image_model.layers[:-num_trainable_layers]:
            layer.trainable = False
            
        # Add new head
        model = tf.keras.models.Sequential()
        model.add(image_model)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(num_features, activation='relu'))

        # Add layer of size self.dataset.num_outputs
        model.add(tf.keras.layers.Dense(self.dataset.num_outputs, activation='softmax'))
        if show_model:
            model.summary()
        return model


class ImageDataset():
  
    # Path: path to dataset, ignore=list of files to ignore
    # TODO: Make ignore work with file name instead of path from the running python script
    def __init__(self, path='', show_distribution=False, ignore=None):
        """ Creates an image classification dataset from 'path'.
            
            path : A pathlib Path variable that is the path to the dataset from the notebook.
            show_distribution : If True, shows bar graph of data distributions in 
                                test set, train set, and valid set (if it exists).
            ignore : List of files to ignore.  For example:
                     ImageDataset(path, ignore=['readme.txt', 'labels.csv'])
        """
        
        self.path = path
        self.ignore = ignore # Ignore these files
        self.img_size = 299 # used for inceptionV3
        
        # Required and optional files in path
        reqs, opts = ['train/', 'test/'], ['valid/']
    
        # Returns dict where flags[file] = True if the optional file exists
        self.flags = self.__check_files(reqs, opts)  
        
        self.train_data = self.__get_data('train')
        self.test_data = self.__get_data('test')
        self.valid_data = None
    
        if self.flags['valid/']:
            self.valid_data = self.__get_data('valid')
        
        if show_distribution:
            print('Dataset distribution:')
            self.__show_distribution(self.train_data, self.test_data, self.valid_data)
    
        # Ensure that train, test, and valid have the same keys for data
        if not set(self.train_data.keys()) == set(self.test_data.keys()):
            raise Warning("Classes in training data are not classes in test data")
        if self.valid_data:
            if not set(self.train_data.keys()) == set(self.valid_data.keys()):
                raise Warning("Warning: Classes in training data are not classes in validation data")
        
        self.num_outputs = len(list(self.train_data.keys()))
        
        # Create ids for training inputs
        self.idx2class = [key for key in self.train_data.keys()]
        self.class2idx = {key:val for val,key in enumerate(self.idx2class)}
        
    def __check_files(self, reqs, opts):
        """ Checks that all required files are found.  
            
            reqs : List of required directory names.
            opts : List of optional directory names.
            
            Returns dictionary where dict[filename] = True
            if the optional file exists.  
            
            For example:
                self.flags = __check_files(reqs, opts)
                has_valid_data = self.flags['valid']
        """
        files = list(self.path.glob('*'))
    
        # Ensure that path has required files
        for req in reqs:
            if self.path / PosixPath(req) not in files:
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), req)
    
        # Create flag dictionary (flags[filename] is True when it exists)
        opt_flags = [True if self.path / PosixPath(opt) in files else False for opt in opts]
        flags = {opts[idx]:opt_flags[idx] for idx in range(len(opts))}
    
        return flags
  
    # Returns dictionary of label: numpy_array of training data
    def __get_data(self, data_dir):
        """ Returns dataset from ['train/', 'test/', 'valid/'] directories in data_dir.
            
            The dataset is a dictionary where the keys are labels and the values are numpy arrays 
            of images of shape (?, 299, 299, 3) representing 299 x 299 RGB (3 channel) images. The 
            size of the first dimension is the number of images in the data.
            
            Examples: 
                data['class1'].shape = (100, 299, 299, 3) if there are 100 images of class1 in the dataset.
                data['class1'][0] is the first image in the class1 data.
        """
        dir_path = self.path / data_dir
        files_in_dir = list(dir_path.glob('*'))
                
        labels = [file.name for file in files_in_dir]
        if self.ignore:
            labels = [file.name for file in files_in_dir if file not in self.ignore ]
        
        data = {}
    
        # Open each image in the folder -- add any more extensions that you can think of 
        img_exts = ['.png', '.PNG', '.jpg', '.JPG', '.JPEG', '.jfif', '.JFIF', '.tiff', '.TIFF']
        
        for label in labels:
            arr = []
            nparr = None
            for img_ext in img_exts:
                for file in glob(str(dir_path) + '/' + label + '/*' + img_ext):
                    img = Image.open(file)
                    img = img.resize((self.img_size, self.img_size), Image.ANTIALIAS)
                    # Convert to numpy array
                    img = np.asarray(img)
                    img = tf.keras.applications.inception_v3.preprocess_input(img)
                    
                    arr.append(img)

            # Convert arr to numpy array (currently array of numpy arrays)
            nparr = np.ndarray((len(arr), self.img_size, self.img_size, 3))
            for i in range(len(arr)):
                nparr[i] = arr[i]
                
            data[label] = nparr
        
        return data
    
    def __show_distribution(self, train_data, test_data, valid_data):
        """Shows the distribution of the datasets.
            
            train_data : Training dataset.
            test_data : Test dataset
            valid_data : Validation dataset (None if doesn't exist)
        """
        num_plots = 2
        
        if valid_data:
            num_plots = 3
        plt.figure(figsize=(10, 3))
        plt.subplot(1, num_plots, 1)
        objects = train_data.keys()
        x_pos = np.arange(len(objects))
        num_examples = [len(train_data[obj]) for obj in objects]
        
        plt.bar(x_pos, num_examples, align='center')
        plt.xticks(x_pos, objects)
        plt.ylabel('Number of examples')
        plt.title('Training set distribution')
        
        plt.subplot(1, num_plots, 2)
        objects = test_data.keys()
        x_pos = np.arange(len(objects))
        num_examples = [len(test_data[obj]) for obj in objects]
        
        plt.bar(x_pos, num_examples, align='center')
        plt.xticks(x_pos, objects)
        plt.ylabel('Number of examples')
        plt.title('Test set distribution') 
        
        if valid_data:
            plt.subplot(1, num_plots, 3)
            objects = valid_data.keys()
            x_pos = np.arange(len(objects))
            num_examples = [len(valid_data[obj]) for obj in objects]

            plt.bar(x_pos, num_examples, align='center')
            plt.xticks(x_pos, objects)
            plt.ylabel('Number of examples')
            plt.title('Validation set distribution')
        
        plt.tight_layout()
        plt.show()  
        
        
    def train_data(self):
        """ Returns the training data of the model.
            
            Data is a dictionary where the keys are labels and the values are numpy arrays 
            of images of shape (?, 299, 299, 3) representing 299 x 299 RGB (3 channel) images. The 
            size of the first dimension is the number of images in the data.
            
            Examples: 
                data['class1'].shape = (100, 299, 299, 3) if there are 100 images in the dataset.
                data['class1'][0] is the first image in the data.
        """
        return self.train_data
    
    def test_data(self):
        """ Returns the testing data of the model.
            
            Data is a dictionary where the keys are labels and the values are numpy arrays 
            of images of shape (?, 299, 299, 3) representing 299 x 299 RGB (3 channel) images. The 
            size of the first dimension is the number of images in the data.
            
            Examples: 
                data['class1'].shape = (100, 299, 299, 3) if there are 100 images in the dataset.
                data['class1'][0] is the first image in the data.
        """
        return self.test_data
    
    def valid_data(self):
        """ Returns the validation data of the model.
            
            Data is a dictionary where the keys are labels and the values are numpy arrays 
            of images of shape (?, 299, 299, 3) representing 299 x 299 RGB (3 channel) images. The 
            size of the first dimension is the number of images in the data.
            
            Examples: 
                data['class1'].shape = (100, 299, 299, 3) if there are 100 images in the dataset.
                data['class1'][0] is the first image in the data.
        """
        return self.valid_data
    