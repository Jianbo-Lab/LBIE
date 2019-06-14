from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np 
import tensorflow as tf
from tensorflow.python.ops import array_ops
from cells import *

try:
	from tensorflow.python.ops.rnn_cell_impl import _linear
except:
	from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear 
	bias_ones = 1.0
 
from utils import conv, lrelu, batchnorm, deconv
# from tensorflow.python.framework import ops 

class batch_norm(object):
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
		  self.epsilon  = epsilon
		  self.momentum = momentum
		  self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x,
		                  decay=self.momentum, 
		                  updates_collections=None,
		                  epsilon=self.epsilon,
		                  scale=True,
		                  is_training=train,
		                  scope=self.name) 

def conv2d(input_, output_dim, 
       k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
       name="conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
		      initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

		return conv

def atrous_conv2d(input_, output_dim, k_h=3, k_w=3, rate=2, stddev=0.02, name="atrous_conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('w',[k_h, k_w, input_.get_shape()[-1], output_dim],
			initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.atrous_conv2d(input_, w, rate, padding='SAME')

		biases = tf.get_variable('biases',[output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

		return conv




def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d"):
	with tf.variable_scope(name):
		# filter : [height, width, output_channels, in_channels]
		w = tf.get_variable('w', [k_h, k_w, output_shape[-1], int(input_.get_shape()[-1])],
		          initializer=tf.random_normal_initializer(stddev=stddev))

		try:
			deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
			        strides=[1, d_h, d_w, 1])

		# Support for verisons of TensorFlow before 0.7.0
		except AttributeError:
			deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
			        strides=[1, d_h, d_w, 1])

		biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
		deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

		return deconv
     
def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
	shape = input_.get_shape().as_list()

	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
		             tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", [output_size],
		  initializer=tf.constant_initializer(bias_start))
		if with_w:
			return tf.matmul(input_, matrix) + bias, matrix, bias
		else:
			return tf.matmul(input_, matrix) + bias 


def build_merger(image_encoder_output, lstm_output, a, reuse = None):

	h,w,num_feats = int(image_encoder_output.get_shape()[1]), int(image_encoder_output.get_shape()[2]), int(image_encoder_output.get_shape()[-1]) 


	if a.text_model.startswith('attention'):
		with tf.variable_scope("RNN/attention", reuse = reuse): 
			# hU: [batchsize, h, w, att_dim]
			hU = conv2d(image_encoder_output, a.att_dim, k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02, name='U_h') 
			# inputW: [batch_size, max_time, att_dim]
			inputW = slim.fully_connected(lstm_output, a.att_dim, activation_fn = None, scope = 'W_h') 
			# v_align: [batchsize, max_time, h, w, att_dim]
			v_align = tf.nn.tanh(tf.expand_dims(hU, 1) + tf.expand_dims(tf.expand_dims(inputW, 2),2))
			# logits: [batchsize, max_time, h, w, 1]
			logits = slim.fully_connected(v_align, 1, activation_fn = None, scope = 'logits_h')
			# weight: [batchsize, max_time, h, w, 1]
			weight = tf.nn.softmax(logits, dim=1)
			# weight: [batchsize, max_time, h, w, 1]
			x = weight * tf.expand_dims(tf.expand_dims(lstm_output,2),2)
			sentence_emb = tf.reduce_sum(x, axis = 1)  
	elif a.text_model == 'reed':
		sentence_emb = lstm_output

	with tf.variable_scope("gates", reuse = reuse):  
	# Reset gate and update gate.
		# We start with bias of 1.0 to not reset and not update.
		# value = tf.nn.sigmoid(_linear([mid_layer, sentence_emb], 2 * a.ngf * 4, True, bias_ones))  
		reshaped_mid_layer = tf.reshape(image_encoder_output, [-1, num_feats])
		reshaped_sentence_emb = tf.reshape(sentence_emb, [-1, num_feats]) 

		concated_input_ = tf.concat([reshaped_mid_layer, reshaped_sentence_emb], -1) 

		value = tf.nn.sigmoid(linear(concated_input_, 2*num_feats, bias_start = 1.0))  		

		r, u = array_ops.split(
			value=value,
			num_or_size_splits=2,
			axis=1) 

	with tf.variable_scope("candidate", reuse = reuse):
		if a.text_model == 'attention' or a.text_model == 'attention_fix' or a.text_model == 'reed':
			concated_input_2 = tf.concat([r*reshaped_mid_layer, reshaped_sentence_emb], -1)
		elif a.text_model == 'attention_reasonet':
			concated_input_2 = tf.concat([reshaped_mid_layer, r*reshaped_sentence_emb], -1)
		c = tf.nn.relu(linear(concated_input_2, num_feats)) 
		combined_layer = u * reshaped_mid_layer + (1 - u) * c 
		combined_layer = tf.reshape(combined_layer, [-1, h, w, num_feats])

	if a.text_model == 'attention_reasonet':
		with tf.variable_scope("termination", reuse = reuse):
			termination_gate = linear(tf.reshape(combined_layer,  [-1, h*w*num_feats]), 1)  
			termination_gate = tf.clip_by_value(tf.nn.sigmoid(termination_gate), clip_value_min=1e-6, clip_value_max=1-1e-6)
	 
	if a.text_model == 'attention_reasonet':
		return combined_layer, termination_gate # [Batchsize, h, w, num_feats], [Batchsize, 1]
	elif a.text_model == 'attention' or a.text_model == 'attention_fix' or a.text_model == 'reed':
		return combined_layer



def build_text_encoder(input_captions, sequence_lengths, image_feature_dims, a):
	h,w,mid_num_feats = image_feature_dims 

	if a.text_model == 'reed':
		
		sentence_emb = linear(input_captions, mid_num_feats, scope = 'fc1') 
			
		# sentence_emb = tf.nn.relu(sentence_emb) #??? 
		sentence_emb = tf.tile(tf.expand_dims(tf.expand_dims(sentence_emb,1), 1), [1,h,w,1]) 

		embedding_placeholder = emb_init = None 

		output = sentence_emb 

	elif a.text_model == 'attention' or a.text_model == 'attention_fix' or a.text_model == 'attention_reasonet':
		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [a.vocab_size, a.emb_dim], trainable= False)
			inputs_emb = tf.nn.embedding_lookup(embedding, input_captions) # inputs_emb: [batch, seq_len, emb_dim] 

			embedding_placeholder = tf.placeholder(tf.float32, [a.vocab_size, a.emb_dim])
			emb_init = embedding.assign(embedding_placeholder)

		with tf.variable_scope("RNN"): 

			# cell = BasicLSTMCell(mid_num_feats, forget_bias=0.0, state_is_tuple=True, 
			# 	reuse=tf.get_variable_scope().reuse)

			# output, state = tf.nn.dynamic_rnn(cell, inputs_emb, sequence_length = sequence_lengths, 
			# 	dtype = tf.float32, time_major = False) # output: [batch_size, T, lstm_dim], state: [batch_size, lstm_dim]  
			cell_fw = BasicLSTMCell(2*a.ngf, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
			cell_bw = BasicLSTMCell(2*a.ngf, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
			# added. 4 -> 2.
			output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw, inputs_emb, 
				sequence_length = sequence_lengths, 
				dtype = tf.float32,
				time_major = False) # output: [batch_size, T, lstm_dim], state: [batch_size, lstm_dim] 
			output = tf.concat(output, 2)   # added

		# output = tf.nn.relu(output)
		# output = slim.fully_connected(output, 4*a.ngf, activation_fn = None, scope='fc1') 
	return output, embedding_placeholder, emb_init


# def create_text_embedding(input_captions, sequence_lengths, mid_layer, a):
# 	"""
# 	mid_layer: [batchsize, h, w, nfeatures]
# 	"""
# 	h = w = int(mid_layer.get_shape()[1])
# 	mid_num_feats = int(mid_layer.get_shape()[-1])

# 	text_embedding, embedding_placeholder, emb_init = build_text_encoder(input_captions, sequence_lengths, image_feature_dims, a)
# 	combbuild_merger(image_encoder_output, lstm_output, a, reuse = None)
# 	if a.text_model == 'attention' or a.text_model == 'attention_fix':
# 		combined_layer = build_merger(mid_layer, text_embedding, a, reuse = None)
# 	# elif a.text_model == 'reasonet':
# 	# 	for i in range(a.T):

# 	return combined_layer, embedding_placeholder, emb_init

def build_encoder_colornet(input_images, ngf, reuse): 
	net = conv2d(input_images, ngf, k_h=3, k_w=3, d_h=1, d_w=1,stddev=0.02,name="conv1_1")
	net = tf.nn.relu(net, 'relu1_1')
	net = conv2d(net, ngf, k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, name="conv1_2")
	net = tf.nn.relu(net, 'relu1_2') 
	net = batch_norm(name='bn1')(net, True)#bn1(net, True)

	# 128,128

	net = conv2d(net,ngf*2, k_h=3, k_w=3, d_h=1, d_w=1,stddev=0.02,name="conv2_1")
	net = tf.nn.relu(net, 'relu2_1')
	net = conv2d(net,ngf*2, k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, name="conv2_2")
	net = tf.nn.relu(net, 'relu2_2') 
	net = batch_norm(name='bn2')(net, True) 
	# 64, 64

	# net = conv2d(net,256, k_h=3, k_w=3, d_h=1, d_w=1,stddev=0.02,name="conv3_1")
	# net = tf.nn.relu(net, 'relu3_1')
	# net = conv2d(net,256, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv3_2")
	# net = tf.nn.relu(net, 'relu3_2') 
	# net = conv2d(net,256, k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, name="conv3_3")
	# net = tf.nn.relu(net, 'relu3_3') 
	# net = bn3(net, True) 

	net = conv2d(net,ngf*4, k_h=3, k_w=3, d_h=1, d_w=1,stddev=0.02,name="conv4_1")
	net = tf.nn.relu(net, 'relu4_1')
	net = conv2d(net,ngf*4, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv4_2")
	net = tf.nn.relu(net, 'relu4_2') 
	net = conv2d(net,ngf*4, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv4_3")
	net = tf.nn.relu(net, 'relu4_3') 
	net = batch_norm(name='bn4')(net, True) #net = bn4(net, True) 

	net = atrous_conv2d(net,ngf*4, k_h=3, k_w=3, rate=2, stddev=0.02,name="conv5_1")
	net = tf.nn.relu(net, 'relu5_1')
	net = atrous_conv2d(net,ngf*4, k_h=3, k_w=3, rate=2, stddev=0.02, name="conv5_2")
	net = tf.nn.relu(net, 'relu5_2') 
	net = atrous_conv2d(net,ngf*4, k_h=3, k_w=3, rate=2, stddev=0.02, name="conv5_3")
	net = tf.nn.relu(net, 'relu5_3') 
	net = batch_norm(name='bn5')(net, True)# bn5(net, True) 

	net = atrous_conv2d(net,ngf*4, k_h=3, k_w=3, rate=2, stddev=0.02,name="conv6_1")
	net = tf.nn.relu(net, 'relu6_1')
	net = atrous_conv2d(net,ngf*4, k_h=3, k_w=3, rate=2, stddev=0.02, name="conv6_2")
	net = tf.nn.relu(net, 'relu6_2') 
	net = atrous_conv2d(net,ngf*4, k_h=3, k_w=3, rate=2, stddev=0.02, name="conv6_3")
	net = tf.nn.relu(net, 'relu6_3') 
	net = batch_norm(name='bn6')(net, True)#bn6(net, True) 


	net = conv2d(net,ngf*4, k_h=3, k_w=3, d_h=1, d_w=1,stddev=0.02,name="conv7_1")
	net = tf.nn.relu(net, 'relu7_1')
	net = conv2d(net,ngf*4, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv7_2")
	net = tf.nn.relu(net, 'relu7_2') 
	net = conv2d(net,ngf*4, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv7_3")
	# net = tf.nn.relu(net, 'relu7_3') 
	return net

def build_decoder_colornet(features, ngf, generator_outputs_channels, reuse):

	net = tf.nn.relu(features, 'relu7_3')

	net = batch_norm(name='bn7')(net, True)#bn7(net, True) 

	net_shape = [int(i) for i in net.get_shape()]
	net = deconv2d(net, [net_shape[0], net_shape[1]*2,net_shape[2]*2, ngf*2], 
		k_h=4, k_w=4, d_h=2, d_w=2,stddev=0.02, name="conv8_1")
	net = tf.nn.relu(net, 'relu8_1')
	net = conv2d(net,ngf*2, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv8_2")
	net = tf.nn.relu(net, 'relu8_2') 
	net = conv2d(net,ngf*2, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv8_3")
	net = tf.nn.relu(net, 'relu8_3')   
	net = batch_norm(name='bn8')(net, True)#bn8(net, True)  

	net_shape = [int(i) for i in net.get_shape()]
	net = deconv2d(net, [net_shape[0], net_shape[1]*2,net_shape[2]*2, ngf], 
		k_h=4, k_w=4, d_h=2, d_w=2,stddev=0.02, name="conv9_1")
	net = tf.nn.relu(net, 'relu9_1')
	net = conv2d(net,ngf, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv9_2")
	net = tf.nn.relu(net, 'relu9_2') 
	net = conv2d(net,ngf, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv9_3")
	net = tf.nn.relu(net, 'relu9_3')
	net = batch_norm(name='bn9')(net, True)#bn9(net, True)

	net = conv2d(net,generator_outputs_channels, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv9_color") 
	net = tf.nn.tanh(net)	

	return net

def colornet(input_, a, generator_outputs_channels = 2, reuse = None):
	ngf = a.ngf 
	input_images, input_captions, sequence_lengths = input_
	with tf.variable_scope('colornet', reuse = reuse):
		image_features = build_encoder_colornet(input_images, ngf, reuse)
		image_feature_dims = [int(image_features.get_shape()[1]), int(image_features.get_shape()[1]), int(image_features.get_shape()[-1])]
		if a.text_model != 'None':
			text_embedding, embedding_placeholder, emb_init = build_text_encoder(input_captions, sequence_lengths, image_feature_dims, a)

			combined_layer = build_merger(image_features, text_embedding, a, reuse = reuse)
		elif a.text_model == 'None': 
			combined_layer = image_features
			embedding_placeholder, emb_init, text_embedding = None, None, None

		net = build_decoder_colornet(combined_layer, ngf, generator_outputs_channels, reuse)

		return net, embedding_placeholder, emb_init, text_embedding  


def colorreasonet(input_, a, generator_outputs_channels = 2, reuse = None):  
	ngf = a.ngf 
	input_images, input_captions, sequence_lengths = input_
	with tf.variable_scope('colornet', reuse = reuse):
		image_features = build_encoder_colornet(input_images, ngf, reuse)
		image_feature_dims = [int(image_features.get_shape()[1]), int(image_features.get_shape()[1]), int(image_features.get_shape()[-1])] 

		text_embedding, embedding_placeholder, emb_init = build_text_encoder(input_captions, sequence_lengths, image_feature_dims, a)

		combined_layer = image_features

		combined_layers = []
		termination_gates = []

		for i in range(a.T):
			reuse_ = reuse if i == 0 else True

			combined_layer = build_merger(combined_layer, text_embedding, a, reuse = reuse_)  
			combined_layers.append(combined_layer)

		combined_layers = tf.stack(combined_layers, axis = 0) # [T, batchsize, h,w,num_feats] 

		h = int(combined_layers.get_shape()[2]); w = int(combined_layers.get_shape()[3]); num_feats = int(combined_layers.get_shape()[-1])
		combined_layers = tf.reshape(combined_layers, [-1, h, w, num_feats]) 
		outputs = build_decoder_colornet(combined_layers, a, generator_outputs_channels + 1, reuse = reuse)   
		outputs_shape = [int(i) for i in outputs.get_shape()]

		outputs = tf.reshape(outputs, [a.T, -1, outputs_shape[1], outputs_shape[2], generator_outputs_channels + 1])# [T, Batchsize, h,w, Num_feats+1] 

		termination_gates = tf.clip_by_value(tf.nn.sigmoid(outputs[:,:,:,:,0]), clip_value_min=1e-6, clip_value_max=1-1e-6) 
		outputs = tf.nn.tanh(outputs[:,:,:,:,1:])

		return [outputs, termination_gates], embedding_placeholder, emb_init                

def create_generator_colornet(generator_inputs, generator_outputs_channels, a):  
	# input_images, input_captions = generator_inputs 
	if a.text_model == 'attention' or a.text_model == 'attention_fix' or a.text_model == 'reed' or a.text_model == 'None':
		generator, embedding_placeholder, emb_init, text_embedding = colornet(generator_inputs, a, generator_outputs_channels, reuse = None)  
	elif a.text_model == 'attention_reasonet': 
		generator, embedding_placeholder, emb_init = colorreasonet(generator_inputs, a, generator_outputs_channels, reuse = None) 

	return generator, emb_init, embedding_placeholder, text_embedding


# input_ = tf.placeholder(tf.float32, [10,256,256,3])
# bns = [batch_norm(name='bn'+ str(i)) for i in range(1,9)]
# generator = colornet(input_, bns, is_training = True, reuse = None)
# sampler = colornet(input_, bns, is_training = False, reuse = True)
# print generator.get_shape()




def create_generator_pix2pix(generator_inputs, generator_outputs_channels, a):
	layers = []
	input_images, input_captions, sequence_lengths = generator_inputs
	with tf.variable_scope('pix2pix'):

		# encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
		with tf.variable_scope("encoder_1"):
			output = conv(input_images, a.ngf, stride=2)
			layers.append(output)

		layer_specs = [
			a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
			a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
			a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
			a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
			a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
			a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
			a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
		]

		for i, out_channels in enumerate(layer_specs):
			with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
				rectified = lrelu(layers[-1], 0.2)
				# [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
				convolved = conv(rectified, out_channels, stride=2)
				if i != 6:
					output = batchnorm(convolved)
				else:
					output = convolved
				layers.append(output)

		
		combined_layer, embedding_placeholder, emb_init = create_text_embedding(input_captions, sequence_lengths, layers[-1], a)
		layers[-1] = combined_layer
		output = batchnorm(convolved)
		mid_layer = lrelu(layers[-1], 0.2)

		layer_specs = [
			(a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
			(a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
			(a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
			(a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
			(a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
			(a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
			(a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
		]

		num_encoder_layers = len(layers)
		for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
			skip_layer = num_encoder_layers - decoder_layer - 1
			with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
				if decoder_layer == 0:
					# first decoder layer doesn't have skip connections
					# since it is directly connected to the skip_layer
					input = layers[-1]
				else:
					input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

				rectified = tf.nn.relu(input)
				# [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
				output = deconv(rectified, out_channels)
				output = batchnorm(output)

				if dropout > 0.0:
					output = tf.nn.dropout(output, keep_prob=1 - dropout)

				layers.append(output)

		# decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
		with tf.variable_scope("decoder_1"):
			input = tf.concat([layers[-1], layers[0]], axis=3)
			rectified = tf.nn.relu(input)
			output = deconv(rectified, generator_outputs_channels)
			output = tf.tanh(output)
			layers.append(output)

	return layers[-1], emb_init, embedding_placeholder


