import numpy as np
import tensorflow

from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from image_class.common.image_utils import extract_label

# def build_dataset(image_path_list,
#                   random_seed,
#                   test_size=0.2,
#                   batch_size = 32,
#                   data_augment=False):
#     images = []
#     labels = []
    
#     label = [extract_label(path) for path in image_path_list] 
    
#     classes = set(label)
#     class_mapping = {class_name: idx for idx, class_name in enumerate(classes)}
#     labels.append(class_mapping[label]) 
#     # Load images into memory and preprocess
  
#     for image_path in image_path_list:
#         image = load_img(image_path,target_size=(224, 224))
#         images.append(image)
    
#     images = np.array(images)
#     labels = np.array(labels) 
#     # Shuffle data (important before splitting)
#     images, labels = shuffle(images, labels, random_state=random_seed)
    
#     # Split data into train, validation, and test sets (80% train, 10% validation, 10% test)
#     X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=test_size, random_state=random_seed)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_seed)

#     if data_augment:
#         image_aug(X_train, y_train, X_val, y_val, X_test, y_test)
#     else: 
#         # No data augmentation (just rescale the images)
#         train_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling
        
#         val_test_datagen = ImageDataGenerator(rescale=1./255)
        
#         # Flow data from memory (since data is already loaded in memory)
#         train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
#         validation_generator = val_test_datagen.flow(X_val, y_val, batch_size=batch_size)
#         test_generator = val_test_datagen.flow(X_test, y_test, batch_size=batch_size)

#     return train_generator,validation_generator, test_generator, classes

def image_aug():
    
    train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Flow data from memory (since data is already loaded in memory)
    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
    validation_generator = val_test_datagen.flow(X_val, y_val, batch_size=32)
    test_generator = val_test_datagen.flow(X_test, y_test, batch_size=32)

    return train_generator, validation_generator, test_generator

def build_dataset_organized(image_dir_name, datapath, test_size):
    # # Assuming your images are organized in a directory, you can load them into a Dataset object.
    dataset = load_dataset(image_folder, data_dir=datapath)

    # Split the dataset into train, test, and validation (you can also use specific ratios)
    dataset = dataset["train"].train_test_split(test_size=0.2)  # Split into 80/20 train/test
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    # If you want to create a validation set from the training set:
    train_dataset = train_dataset.train_test_split(test_size=0.1)  # 10% for validation
    val_dataset = train_dataset["test"]
    train_dataset = train_dataset["train"]
    
    return train_dataset, val_dataset, test_dataset


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator

def build_dataset(image_path_list,
                  random_seed,
                  test_size=0.2,
                  batch_size=32,
                  data_augment=False):  
    
    class_names = [extract_label(path) for path in image_path_list] 
    classes = sorted(set(class_names))
    # Create a mapping of class names (strings) to integers
    class_mapping = {class_name: idx for idx, class_name in enumerate(classes)}
    # Load images into memory and preprocess
    images = []
    labels = []
    for image_path in image_path_list:
        image = load_img(image_path, target_size=(224, 224))  # Resize image to (224, 224)
        image = np.array(image)  # Convert image to numpy array
        images.append(image)
        
    # Extract label (assuming extract_label function returns the class name as a string)
    label = extract_label(image_path)  
    labels.append(class_mapping[label])  # Convert label to integer based on class_mapping
    
    images = np.array(images)
    labels = [class_mapping[extract_label(path)] for path in image_path_list]
    # labels = np.array(labels)  # Now labels are integers corresponding to class indices

    # Shuffle data (important before splitting)
    images, labels = shuffle(images, labels, random_state=random_seed)

    # Split data into train, validation, and test sets (80% train, 10% validation, 10% test)
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=test_size, random_state=random_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_seed)

    if data_augment:
        # Apply image augmentation function (assuming image_aug function does augmentation)
        image_aug(X_train, y_train, X_val, y_val, X_test, y_test)
    else:
        # No data augmentation (just rescale the images)
        train_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Flow data from memory (since data is already loaded in memory)
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        validation_generator = val_test_datagen.flow(X_val, y_val, batch_size=batch_size)
        test_generator = val_test_datagen.flow(X_test, y_test, batch_size=batch_size)
    

    return train_generator, validation_generator, test_generator, classes
