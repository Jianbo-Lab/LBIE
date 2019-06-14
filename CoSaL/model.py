from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import skimage.io as io
# from tensorflow.python.ops import array_ops
# from utils import *
from cells import *
from net import *
# from colornet_nyu import *
# try:
# 	from tensorflow.python.ops.rnn_cell_impl import _linear
# except:
# 	from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear 
# 	bias_ones = 1.0
 


EPS = 1e-12


def create_model(inputs, labels, captions, a):  

	with tf.variable_scope("generator") as scope: 
		generator = create_generator_colornet([inputs, captions], a) 

	if a.text_model != 'attention_reasonet': 
		with tf.name_scope("generator_loss"): 
			logits = generator

			boolean_masks = tf.not_equal(labels, tf.zeros_like(labels, dtype=tf.int64)) # 1: should include. 0: too difficult.
			masks = tf.cast(boolean_masks,tf.float32) * 9.0 + 1.0 # white areas are weighted by 1, the rest: 10.

			# reshaped_masks = tf.reshape(masks, [-1]) 
			gen_loss = tf.reduce_sum(masks * tf.nn.sparse_softmax_cross_entropy_with_logits(\
				labels=labels, logits=logits))/\
				 tf.maximum(tf.reduce_sum(masks),1.0)

			# gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

			predictions = tf.argmax(logits, axis=-1) 
			correct_labels = tf.equal(predictions, labels) 

			mean_iou, update_iou = tf.metrics.mean_iou(labels, 
				predictions, a.num_classes)

	elif a.text_model == 'attention_reasonet':  

		# logits: [T, batchsize, height, width, num_classes]
		# terminations: [T, batchsize, height, width, 1]
		with tf.name_scope("generator_loss"):
		  	logits, terminations = generator 
			_, _, height, width = [int(i) for i in terminations.get_shape()]
			terminations = tf.reshape(terminations, [a.T, a.batch_size, height, width])
			# non_terminate: [T, batchsize, height, width]
			log_non_terminate = tf.log(1 - terminations)
			# log_prob_terminate: [T, batchsize, height, width]
			log_prob_terminate = tf.cumsum(log_non_terminate, axis = 0, exclusive = True) + tf.log(terminations)
			# prob_terminate = tf.exp(log_prob_terminate)

			expanded_labels =tf.stack([labels for i in range(a.T)]) #  [T, batchsize, height, width, out_channels] 

			# [l, batchsize, height, width]
			true_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(\
				labels=expanded_labels, logits=logits)

			# average_logits: [l, batchsize, height, width]
			average_logits = log_prob_terminate - true_logprob

			# average_logits: [batchsize, height, width]
			average_logits = tf.reduce_logsumexp(average_logits, 0)

			boolean_masks = tf.not_equal(labels, tf.zeros_like(labels, dtype=tf.int64))
			masks = tf.cast(boolean_masks, tf.float32) * 9.0 + 1.0

			gen_loss = tf.reduce_sum(-average_logits * masks) / tf.reduce_sum(masks) 

			all_predictions = tf.argmax(logits, axis=-1) 
			all_correct_labels = tf.equal(all_predictions, expanded_labels) 

			termination_masks = tf.one_hot(tf.argmax(log_prob_terminate, axis=0),depth=a.T,axis=0, dtype=tf.int64) #0-1 vector at [a.T, a.batch_size, height, width]
			predictions = tf.cast(tf.reduce_sum(tf.argmax(logits, axis=-1) * termination_masks, axis=0), tf.int64)
			correct_labels = tf.equal(predictions, labels) 

			mean_iou, update_iou = tf.metrics.mean_iou(labels, 
				predictions, a.num_classes)



	with tf.name_scope("generator_train"): 

		gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
		gen_grads_and_vars = gen_optim.compute_gradients(gen_loss)
		gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

	ema = tf.train.ExponentialMovingAverage(decay=0.99, zero_debias=True)
	update_losses = ema.apply([gen_loss]) 
	# update_losses = ema.apply([gen_loss])

	global_step = tf.contrib.framework.get_or_create_global_step()
	incr_global_step = tf.assign(global_step, global_step+1)

	# Take the correct output as output.
	# [batchsize]
	# if a.text_model == 'attention_reasonet':
	# 	terminating_step = tf.argmax(log_prob_terminate, axis = 0) 
	# 	# termination = tf.one_hot(terminating_step, 4, axis = 0,on_value = 255, off_value = 0, dtype = tf.uint8) 

	# 	# outputs = outputs[3] #outputs[terminating_step[0]] 
	if a.text_model != 'attention_reasonet':
		termination_masks = None  
		all_predictions = None
	
	color_params = tf.constant([
		[255,255,255],
	    [0,0,255],##r
	    [0,255,0],##g
	    [255,0,0],##b
	    [0,0,0],##o
	    [128,128,128],##k
	    [255,255,0]##y
	    ], dtype = tf.uint8)

	predicted_reals = tf.nn.embedding_lookup(color_params, predictions)
	if a.text_model == 'attention_reasonet':
		all_predicted_reals = tf.nn.embedding_lookup(color_params, all_predictions)
	else:
		all_predicted_reals = None

	Model = collections.namedtuple("Model", "update_iou, batch_loss, gen_loss, mean_iou, correct_labels, train, termination, all_predictions, predictions") 

	return Model(
		mean_iou = mean_iou, 
		update_iou = update_iou,
		batch_loss = gen_loss,
		gen_loss =ema.average(gen_loss), 
		correct_labels = correct_labels,   
		all_predictions = all_predicted_reals,
		predictions = predicted_reals,
		train=tf.group(update_losses, incr_global_step, 
			gen_train, update_iou),
		termination=termination_masks 
	)

def create_summaries(reals, model, a):
	# summaries
	with tf.name_scope("predictions_summary"):
		tf.summary.image("predictions", model.predictions)

	with tf.name_scope("truths_summary"):
		tf.summary.image("truths", reals)

	with tf.name_scope("correct_labels_summary"):
		correct_labels = tf.cast(model.correct_labels, tf.uint8) * 255
		tf.summary.image("correct_labels", tf.expand_dims(correct_labels,-1)) 

	if a.text_model == 'attention_reasonet':
		for i in range(a.T):
			with tf.name_scope("predictions_at_stage_{}".format(i)):
				tf.summary.image("correct_labels", model.all_predictions[i]) 		

def save_images(fetches, image_count, a, caption_file, id2word, step = None):
	image_dir = os.path.join(a.output_dir, "images")
	if not os.path.exists(image_dir):
		os.makedirs(image_dir) 
	filesets = []

	# captions = fetches['captions']
	for i, _ in enumerate(fetches['real']):
		name = str(image_count)
		fileset = {"name": name, "step": step}

		# cap = captions[i]
		# word_caption = str(image_count) + ': ' + ' '.join([id2word[j] for j in cap]) + '\n'
		# caption_file.write(word_caption)
		kinds = ['real','predictions'] 
		if a.text_model == 'attention_reasonet':
			kinds += ['predictions_{}'.format(j) for j in range(a.T)] + ['termination_{}'.format(j) for j in range(a.T)]
		for kind in kinds:
			filename = name + "-" + kind + ".png"
			if step is not None:
				filename = "%08d-%s" % (step, filename)
			fileset[kind] = filename
			out_path = os.path.join(image_dir, filename)
			contents = fetches[kind][i] 
			# if kind != 'images':
			# 	contents = np.squeeze(contents, axis = -1) 

			io.imsave(out_path, contents) 

		filesets.append(fileset)
		image_count += 1

	return filesets, image_count










