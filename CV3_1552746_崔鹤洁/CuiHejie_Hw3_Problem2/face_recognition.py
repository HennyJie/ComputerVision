
# coding: utf-8

# In[1]:


import yaml
import os
from os.path import join
import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

batch_size = 1000
nrof_train_images_per_class=100
min_nrof_images_per_class=10
image_size=160
threshold = 0.6

kernel='linear'
model_name='freeze_model'
path_train = 'cuihejie_train'
path_test = 'cuihejie_test'
path_other = 'cuihejie_others'
folder = 'recognition'

project_dir = os.path.dirname(os.getcwd())


# In[2]:


def embedding(path):
    
    with tf.Graph().as_default():

        with tf.Session() as sess:

            np.random.seed(seed=2)

            dataset = facenet.get_dataset(path)

            paths, labels = facenet.get_image_paths_and_labels(dataset)
            print(labels)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # Load the model
            print('Loading feature extraction model...')
            model_path = join(project_dir, folder, '%s.pb'%model_name)
            facenet.load_model(model_path)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images...')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            
            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
    
    return paths, labels, emb_array, dataset


# In[3]:


# Get the embedding results of train dataset
data_dir = join(project_dir, folder, path_train)
paths, labels, emb_array, dataset = embedding(data_dir)

# Use embedding results to train SVM classifiers
print('Training classifier...')
model = SVC(kernel=kernel, probability=True, random_state=666)
model.fit(emb_array, labels)

class_names = [cls.name.replace('_', ' ') for cls in dataset]


# In[4]:


# Get the embedding results of test dataset
data_dir = join(project_dir, folder, path_test)
paths, labels, emb_array, dataset = embedding(data_dir)

# Use SVW classifiers to classify testing embedding results
print('Testing classifier...')
predictions = model.predict_proba(emb_array)
best_class_indices = np.argmax(predictions, axis=1)
print(best_class_indices)
best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        
for i in range(len(best_class_indices)):
    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
    
accuracy = np.mean(np.equal(best_class_indices, labels))
print('Accuracy: %.3f' % accuracy)

# test samples that are not in the register dataset
data_dir = join(project_dir, folder, path_other)
paths, labels, emb_array, dataset = embedding(data_dir)
predictions = model.predict_proba(emb_array)
best_class_indices = np.argmax(predictions, axis=1)
best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
for i in range(len(labels)):
    if best_class_probabilities[i] < threshold:
        path = os.path.split(paths[i])[1]
        print('%s is not in the register dataset!'%path)

