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
# from tensorflow.python.ops import array_ops
from utils import *
from cells import *
from colornet import *
# try:
# 	from tensorflow.python.ops.rnn_cell_impl import _linear
# except:
# 	from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear 
# 	bias_ones = 1.0
 


EPS = 1e-12


def create_model(inputs, captions, sequence_lengths, targets, a): 
	def create_discriminator(discrim_inputs, text_embedding, discrim_targets):
		n_layers = 3
		layers = []

		# 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
		input = tf.concat([discrim_inputs, discrim_targets], axis=3)

		# layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
		with tf.variable_scope("layer_1"):
			convolved = conv(input, a.ndf, stride=2)
			rectified = lrelu(convolved, 0.2)
			layers.append(rectified)

		# layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
		# layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
		# layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
		for i in range(n_layers):
			with tf.variable_scope("layer_%d" % (len(layers) + 1)):
				out_channels = a.ndf * min(2**(i+1), 8)
				stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
				convolved = conv(layers[-1], out_channels, stride=stride)
				normalized = batchnorm(convolved)
				rectified = lrelu(normalized, 0.2)
				layers.append(rectified)
		if a.text_model.startswith('attention'):
			unrectified = conv(rectified, 2 * a.ndf, stride=stride)
		# layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
		with tf.variable_scope("layer_%d" % (len(layers) + 1)):
			# add language.
			
			if a.text_model.startswith('attention'):
				combined = build_merger(unrectified, text_embedding, a, reuse = None)
			else:
				combined = rectified

			convolved = conv(combined, out_channels=1, stride=1)
			output = tf.sigmoid(convolved)
			layers.append(output)

		return layers[-1]

	with tf.variable_scope("generator") as scope:
		out_channels = int(targets.get_shape()[-1])
		if a.model == 'pix2pix':
			outputs, emb_init, embedding_placeholder = create_generator_pix2pix([inputs, captions, sequence_lengths], out_channels, a)
		elif a.model == 'colornet':
			outputs, emb_init, embedding_placeholder, text_embedding = create_generator_colornet([inputs, captions, sequence_lengths], out_channels, a) 

	# create two copies of discriminator, one for real pairs and one for fake pairs
	# they share the same underlying variables
	with tf.name_scope("real_discriminator"):
		with tf.variable_scope("discriminator"):
			# 2x [batch, height, width, channels] => [batch, 30, 30, 1]
			predict_real = create_discriminator(inputs, text_embedding, targets)

	if a.text_model != 'attention_reasonet':
		with tf.name_scope("fake_discriminator"):
			with tf.variable_scope("discriminator", reuse=True):
				# 2x [batch, height, width, channels] => [batch, 30, 30, 1]
				predict_fake = create_discriminator(inputs, text_embedding, outputs)

		
		with tf.name_scope("discriminator_loss"):
			# minimizing -tf.log will try to get inputs to 1
			# predict_real => 1
			# predict_fake => 0
			discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

		with tf.name_scope("generator_loss"):
			# predict_fake => 1
			# abs(targets - outputs) => 0
			gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
			gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
			gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight
			# gen_loss = gen_loss_L1 * a.l1_weight 

	elif a.text_model == 'attention_reasonet':  
		outputs, termination_gates = outputs 
		# outputs: [T, Batchsize, a.crop_size, a.crop_size, out_channels]; 
		# termination_gates: [T, Batchsize, a.crop_size, a.crop_size, 1].  

		termination_gates = tf.reduce_mean(termination_gates, axis = [2,3]) # [T, batchsize]   
		# non_terminate: [T, batchsize]
		log_non_terminate = tf.log(1 - termination_gates)
		# log_prob_terminate: [T, batchsize]
		log_prob_terminate = tf.cumsum(log_non_terminate, axis = 0, exclusive = True) + tf.log(termination_gates)
		# prob_terminate: [T, batchsize]
		prob_terminate = tf.exp(log_prob_terminate)

		inputs =tf.reshape(tf.stack([inputs for i in range(a.T)]), [-1, a.crop_size, a.crop_size, int(inputs.get_shape()[-1])])
		targets =tf.stack([targets for i in range(a.T)]) #  [T, batchsize, a.cropsize, a.cropsize, out_channels] 

		with tf.name_scope("fake_discriminator"): 
			with tf.variable_scope("discriminator", reuse=True):
				# 2x [batch*T, height, width, channels] => [T*batch, 30, 30, 1]
				predict_fakes = create_discriminator(inputs, tf.reshape(outputs, [-1, a.crop_size, a.crop_size, out_channels])) 
				predict_fakes = tf.reshape(predict_fakes, [a.T, a.batch_size,
				int(predict_fakes.get_shape()[1]),int(predict_fakes.get_shape()[1]),1]) # [T, batch, 30, 30, 1]

		# with tf.name_scope("discriminator_loss"):
		# 	# minimizing -tf.log will try to get inputs to 1
		# 	# predict_real => 1
		# 	# predict_fake => 0
		# 	predict_fake_loss = tf.reduce_mean(-(tf.log(1 - predict_fakes + EPS)), axis = [-1,-2,-3]) - log_prob_terminate # [T, batch]
		# 	predict_fake_loss = tf.reduce_mean(tf.reduce_sum(predict_fake_loss, axis = 0)) 
		# 	discrim_loss = tf.reduce_mean(- tf.log(predict_real + EPS)) + predict_fake_loss

		# with tf.name_scope("generator_loss"):
		# 	# predict_fake => 1
		# 	# abs(targets - outputs) => 0
		# 	gen_loss_GAN =  tf.reduce_mean(-(tf.log(predict_fakes + EPS)), axis = [-1,-2,-3]) - log_prob_terminate # [T, batch]
		# 	gen_loss_GAN = tf.reduce_mean(tf.reduce_sum(gen_loss_GAN, axis = 0))  

		# 	gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs), axis = [-1,-2,-3]) #  [T, batchsize]
		# 	gen_loss_L1 = tf.reduce_mean(tf.reduce_sum(gen_loss_L1 * tf.exp(log_prob_terminate),axis = 0))
		# 	gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight 
		with tf.name_scope("discriminator_loss"):
			# minimizing -tf.log will try to get inputs to 1
			# predict_real => 1
			# predict_fake => 0
			predict_fake_loss = tf.reduce_mean(-(tf.log(1 - predict_fakes + EPS)), axis = [-1,-2,-3]) * prob_terminate # [T, batch]
			predict_fake_loss = tf.reduce_mean(tf.reduce_sum(predict_fake_loss, axis = 0)) 
			discrim_loss = tf.reduce_mean(- tf.log(predict_real + EPS)) + predict_fake_loss

		with tf.name_scope("generator_loss"):
			# predict_fake => 1
			# abs(targets - outputs) => 0
			gen_loss_GAN =  tf.reduce_mean(-(tf.log(predict_fakes + EPS)), axis = [-1,-2,-3]) * prob_terminate # [T, batch]
			gen_loss_GAN = tf.reduce_mean(tf.reduce_sum(gen_loss_GAN, axis = 0))  

			gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs), axis = [-1,-2,-3]) #  [T, batchsize]
			gen_loss_L1 = tf.reduce_mean(tf.reduce_sum(gen_loss_L1 * prob_terminate, axis = 0))
			gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight  

	with tf.name_scope("discriminator_train"):
		discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
		discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
		discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
		discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

	with tf.name_scope("generator_train"):
		# with tf.control_dependencies([discrim_train]):
		# if a.text_model != 'attention_fix' and a.text_model != 'attention_reasonet':
		# 	gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]

		# elif a.text_model == 'attention_fix' or a.text_model == 'attention_reasonet':
		# 	gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator") and \
		# 	not var.name.startswith("generator/{}/RNN/rnn/".format(a.model))]
		
		if a.text_model.startswith('attention'):
			gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator") and \
			not var.name.startswith("generator/{}/RNN/bidirectional_rnn".format(a.model))]

			var_list = [var.name for var in tf.trainable_variables() if var.name.startswith("generator") and \
			 var.name.startswith("generator/{}/RNN/bidirectional_rnn".format(a.model))]
		else:
			gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]

		gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
		gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
		gen_train = gen_optim.apply_gradients(gen_grads_and_vars)



	ema = tf.train.ExponentialMovingAverage(decay=0.99, zero_debias=True)
	update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])
	# update_losses = ema.apply([gen_loss])

	global_step = tf.contrib.framework.get_or_create_global_step()
	incr_global_step = tf.assign(global_step, global_step+1)


	# Take the correct output as output.
	# [batchsize]
	if a.text_model == 'attention_reasonet':
		terminating_step = tf.argmax(log_prob_terminate, axis = 0) 

		outputs = outputs[3]#outputs[terminating_step[0]]
		predict_fake = tf.reduce_sum(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.exp(log_prob_terminate),-1),-1),-1) * predict_fakes, axis = 0)

	else: 
		terminating_step = None

	Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, \
			gen_loss_L1, gen_grads_and_vars, discrim_train, train, embedding_placeholder, emb_init, termination") 

	return Model(
		predict_real=predict_real,
		predict_fake=predict_fake,
		discrim_loss=ema.average(discrim_loss),
		discrim_grads_and_vars=discrim_grads_and_vars,
		gen_loss_GAN=ema.average(gen_loss_GAN),
		gen_loss_L1=ema.average(gen_loss_L1),
		gen_grads_and_vars=gen_grads_and_vars,
		outputs=outputs,
		discrim_train = discrim_train,
		train=tf.group(update_losses, incr_global_step, gen_train),
		embedding_placeholder=embedding_placeholder,
		emb_init=emb_init,
		termination=terminating_step
	)

def create_summaries(converted_inputs, converted_targets, converted_outputs, model):
	# summaries
	with tf.name_scope("inputs_summary"):
		tf.summary.image("inputs", converted_inputs)

	with tf.name_scope("targets_summary"):
		tf.summary.image("targets", converted_targets)

	with tf.name_scope("outputs_summary"):
		tf.summary.image("outputs", converted_outputs)

	with tf.name_scope("predict_real_summary"):
		tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

	with tf.name_scope("predict_fake_summary"):
		tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

	tf.summary.scalar("discriminator_loss", model.discrim_loss)
	tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
	tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
	if model.termination != None:
		tf.summary.scalar("terminating_step", tf.reduce_mean(model.termination))
	# for var in tf.trainable_variables():
	# 	tf.summary.histogram(var.op.name + "/values", var)

	# for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
	# 	tf.summary.histogram(var.op.name + "/gradients", grad)