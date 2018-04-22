#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/21 18:13
# @Author  : Aries
# @Site    : 
# @File    : mtcnn_detect.py
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
#import matplotlib.pyplot as plt



minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
gpu_memory_fraction = 1.0
image_size = 160
margin =44



with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)





with tf.Graph().as_default():
    with tf.Session() as sess:
        facenet.load_model('20170512-110547')
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        cv2.namedWindow("face")
        vc = cv2.VideoCapture(0)
        while 1:
            _, frame = vc.read()
            img = frame
            #print(frame)
            if frame is not None:
                img_size = np.asarray(frame.shape)[0:2]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                #cv2.putText(frame, str(len(bounding_boxes)), (50, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 2)
                if len(bounding_boxes)>= 1:
                    #print("can't detect face, remove ", img)
                    #continue
                    image_list = []
                    for det in bounding_boxes:
                        #det = np.squeeze(face_boxes[0, 0:4])
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(det[0] - margin / 2, 0)
                        bb[1] = np.maximum(det[1] - margin / 2, 0)
                        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                        cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
                        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                        prewhitened = facenet.prewhiten(aligned)
                        image_list.append(prewhitened)

                    crop_faces = []
                    for face_position in bounding_boxes:
                        face_position = face_position.astype(int)
                        cv2.rectangle(frame, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
                        crop = img[face_position[1]:face_position[3],
                               face_position[0]:face_position[2], ]
                    images = np.stack(image_list)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)

                    with open('model.pkl','rb') as infile:
                        (labels, class_names, embed) = pickle.load(infile)

                    classes = []
                    distance = []
                    for i in range(len(emb)):
                        distance = []
                        for j in range(len(embed)):
                            dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], embed[j, :]))))
                            distance.append(dist)
                        index = np.argmin(distance)
                        #print(distance)
                        #print(index)
                        #print(labels[index])
                        classes.append(class_names[labels[index]])
                    classes = str(classes)
                    cv2.putText(frame, classes, (50,100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0 ,0), 2)

                cv2.waitKey(1)
                cv2.imshow("face", frame)

vc.release()
cv2.destroyWindow("new")

