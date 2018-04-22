#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/22 10:50
# @Author  : Aries
# @Site    : 
# @File    : train.py
# @Software: PyCharm Community Edition

from scipy import misc
import tensorflow as tf
import detect_face
import facenet
import cv2
import numpy as np
import os
import copy
import pickle


def main():

    data_dir = 'database\\'

    print(1)
    print(os.getcwd())
    print(os.listdir(data_dir))


    dataset = facenet.get_dataset(data_dir)


    # Check that there are at least one training image per class
    for cls in dataset:
        print(cls.image_paths)
        #assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

    paths, labels = facenet.get_image_paths_and_labels(dataset)

    print('Number of classes: %d' % len(dataset))
    print('Number of images: %d' % len(paths))

    # Load the model
    with tf.Graph().as_default():
        with tf.Session() as sess:
            print('Loading feature extraction model')
            facenet.load_model('20170512-110547')
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            #embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            #images = facenet.load_data(paths_batch, False, False, args.image_size)
            images = load_and_align_data(paths, 160, 44, 1.0)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)

            class_names = [cls.name.replace('_', ' ') for cls in dataset]

            with open('model.pkl', 'wb') as outfile:
                pickle.dump((labels, class_names, emb), outfile)
            print('Saved classifier model to file model.pkl')


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    tmp_image_paths = copy.copy(image_paths)
    print(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, remove ", image)
            continue
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

main()