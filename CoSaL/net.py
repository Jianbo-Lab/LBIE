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
 

# from FCN import vgg_net, FCN
# from tensorflow.python.framework import ops 
NUM_SENTS = 9
# class batch_norm(object):
# 	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
# 		with tf.variable_scope(name):
# 		  self.epsilon  = epsilon
# 		  self.momentum = momentum
# 		  self.name = name

# 	def __call__(self, x, train=True):
# 		return tf.contrib.layers.batch_norm(x,
# 		                  decay=self.momentum, 
# 		                  updates_collections=None,
# 		                  epsilon=self.epsilon,
# 		                  scale=True,
# 		                  is_training=train,
# 		                  scope=self.name) 

def batch_norm(x, is_training, name):
	return tf.contrib.layers.batch_norm(x, center=True, 
		scale=True, is_training=is_training, scope=name)

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
			# # hU: [batchsize, h, w, att_dim]
			# hU = conv2d(image_encoder_output, a.att_dim, k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02, name='U_h') 
			# # inputW: [batch_size, max_time, att_dim]
			# inputW = slim.fully_connected(lstm_output, a.att_dim, activation_fn = None, scope = 'W_h') 
			# # v_align: [batchsize, max_time, h, w, att_dim]
			# v_align = tf.nn.tanh(tf.expand_dims(hU, 1) + tf.expand_dims(tf.expand_dims(inputW, 2),2))
			# # logits: [batchsize, max_time, h, w, 1]
			# logits = slim.fully_connected(v_align, 1, activation_fn = None, scope = 'logits_h')
			# # weight: [batchsize, max_time, h, w, 1]
			# weight = tf.nn.softmax(logits, dim=1)

			# image_encoder_output: [B,h,w,num_feats]
			# lstm_output: [B,max_time,num_feats] 
			lstm_att = slim.fully_connected(lstm_output, 
				num_feats, activation_fn = None, scope = 'W_h') 

			weight = tf.einsum('bhwf,btf->bthw', 
				image_encoder_output, 
				lstm_att) / np.sqrt(num_feats)

			weight = tf.expand_dims(weight, -1)
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

		value = tf.nn.sigmoid(linear(concated_input_, 2*num_feats, bias_start = 0.5))  		

		r, u = array_ops.split(
			value=value,
			num_or_size_splits=2,
			axis=1) 

	with tf.variable_scope("candidate", reuse = reuse):
		concated_input_2 = tf.concat([reshaped_mid_layer, r*reshaped_sentence_emb], -1)
		c = tf.nn.relu(linear(concated_input_2, num_feats)) 
		combined_layer = u * reshaped_mid_layer + (1 - u) * c 
		combined_layer = tf.reshape(combined_layer, [-1, h, w, num_feats]) 
		
	return combined_layer

def build_contextual_merger(img_encoder, lstm_output, a, reuse = None):

	h,w,num_feats = int(img_encoder.get_shape()[1]), int(img_encoder.get_shape()[2]), int(img_encoder.get_shape()[-1]) 
	location_array = np.zeros((a.batch_size, h,w,2))
	for i in range(h):
		location_array[:,i,:,0] = i*1.0/h - 0.5
	for j in range(w):
		location_array[:,:,j,1] = j*1.0/w - 0.5 

	location_tensor = tf.constant(location_array, tf.float32)    # [batchsize, h, w, 2]
	image_encoder_output = tf.concat((img_encoder, location_tensor),-1)  # [batchsize, h, w, num_feats+2]
	contextual_encoder1 = tf.expand_dims(tf.expand_dims(image_encoder_output, 1), 1) # [batchsize, 1, 1, h, w, num_feats+2]
	contextual_encoder1 = tf.tile(contextual_encoder1, [1,h,w,1,1,1])
	contextual_encoder2 = tf.expand_dims(tf.expand_dims(image_encoder_output, -2), -2) # [batchsize, h, w,1,1, num_feats+2]
	contextual_encoder2 = tf.tile(contextual_encoder2, [1,1,1,h,w,1])
	contextual_encoder = tf.concat((contextual_encoder1, contextual_encoder2), -1) # [batchsize, h, w, h, w, 2* num_feats+4]

	reshaped_contextual_encoder = tf.reshape(contextual_encoder, [-1, h, w, 2* num_feats+4]) # [batchsize*h*w, h, w, 2* num_feats+4]

	with tf.variable_scope("RNN/attention", reuse = reuse): 
		# hU: [batchsize*h*w, h, w, att_dim]
		hU = conv2d(reshaped_contextual_encoder, a.att_dim, k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02, name='U_h')
		# hU: [batchsize, h, w, h, w, att_dim]
		hU = tf.reshape(hU, [a.batch_size, h, w, h, w, a.att_dim])

		# inputW: [batch_size, max_time, att_dim]
		inputW = slim.fully_connected(lstm_output, a.att_dim, activation_fn = None, scope = 'W_h') 
		# v_align: [batchsize, max_time, h, w, h, w, att_dim]
		v_align = tf.nn.tanh(tf.expand_dims(hU, 1) + tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(inputW, 2),2),2),2))
		
		# v_align: [batchsize*max_time*h*w, h, w, att_dim]
		v_align = tf.reshape(v_align, [-1, h,w,a.att_dim])

		# logits: [batchsize, max_time, h, w, h, w, 1]
		logits = conv2d(v_align, 1, k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02, name='logits_h')

		logits = tf.reshape(logits, [a.batch_size, NUM_SENTS, h, w, h, w, 1])


		# weight: [batchsize, max_time, h, w, h, w, 1]
		weight = tf.nn.softmax(logits, dim=1)
		# x: [batchsize, max_time, h, w, h, w, num_feats]
		x = weight * tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(lstm_output,2),2),2),2)

		# sentence_emb: [batchsize, h, w, h, w, num_feats]
		sentence_emb = tf.reduce_sum(x, axis = 1)  

		combined_embedding = tf.concat((contextual_encoder, sentence_emb), -1) # [batchsize, h, w, h, w, 3* num_feats+4] 
		combined_embedding = tf.reshape(combined_embedding, [-1,h,w,3* num_feats+4])

	with tf.variable_scope("MLP-function", reuse = reuse): 

		fc1 = tf.nn.relu(conv2d(combined_embedding, num_feats, k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02, name='fc1'))
		fc2 = tf.nn.relu(conv2d(fc1, num_feats, k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02, name='fc2'))
		# fc3 = tf.nn.relu(conv2d(fc2, 32, k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02, name='fc3')) 
		# fc1 = slim.fully_connected(combined_embedding, 256, activation_fn = tf.nn.relu, scope = 'fc1')
		# fc2 = slim.fully_connected(fc1, 256, activation_fn = tf.nn.relu, scope = 'fc2')
		# fc3 = slim.fully_connected(fc2, 128, activation_fn = tf.nn.relu, scope = 'fc3') # [batchsize, h, w, h, w, 128]
		fc3 = tf.reshape(fc2, [a.batch_size, h, w, h, w, num_feats])
		output = tf.reduce_mean(fc3, [3,4]) # [batchsize, h, w, 128]

	with tf.variable_scope("gates", reuse = reuse):  
		# Reset gate and update gate.
		# We start with bias of 1.0 to not reset and not update.
		# value = tf.nn.sigmoid(_linear([mid_layer, sentence_emb], 2 * a.ngf * 4, True, bias_ones))  
		reshaped_mid_layer = tf.reshape(image_encoder_output, [-1, num_feats+2])
		reshaped_sentence_emb = tf.reshape(lstm_output, 
			[-1, num_feats]) 

		concated_input_ = tf.concat([reshaped_mid_layer, reshaped_sentence_emb], -1) 

		value = tf.nn.sigmoid(linear(concated_input_, 2*num_feats, bias_start = 0.5))  		

		r, u = array_ops.split(
			value=value,
			num_or_size_splits=2,
			axis=1) 

	with tf.variable_scope("candidate", reuse = reuse):
		reshaped_output = tf.reshape(output, [-1, num_feats]) 
		concated_input_2 = tf.concat([reshaped_mid_layer, r*reshaped_output], -1)
		c = tf.nn.relu(linear(concated_input_2, num_feats)) 
		output = u * tf.reshape(img_encoder,
			[-1,num_feats]) + (1 - u) * c 
		output = tf.reshape(output, [-1, h, w, num_feats]) 

	return output


def build_text_encoder(input_captions, image_feature_dims, a):

	# input_captions: [batchsize, NUM_SENTS, max_length] 

	h,w, num_feats = image_feature_dims  

	reshaped_input_captions = tf.reshape(input_captions, [a.batch_size*NUM_SENTS, -1])  

	inputs_emb = tf.one_hot(reshaped_input_captions, a.vocab_size) # inputs_emb: [batchsize*num_sents, num_words, vocab_size]
	inputs_emb = tf.to_float(inputs_emb)

	with tf.variable_scope("RNN"): 

		cell = BasicLSTMCell(num_feats, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)

		output, state = tf.nn.dynamic_rnn(cell, inputs_emb, 
			dtype = tf.float32,
			time_major = False) # output: [batch_size*NUM_SENTS, T, lstm_dim], state: [batch_size*NUM_SENTS, lstm_dim] 


	with tf.variable_scope("Paragraph_RNN"): 
		reshaped_state = tf.reshape(state[1], [a.batch_size, NUM_SENTS, num_feats]) 

		cell = BasicLSTMCell(num_feats, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)

		output, state = tf.nn.dynamic_rnn(cell, reshaped_state, 
			dtype = tf.float32,
			time_major = False) 
		# output: [batch_size, NUM_SENTS, lstm_dim], state: [batch_size, lstm_dim] 

	return output  

def build_encoder_colornet(input_images, a, reuse): 

	ngf = a.ngf
	net = conv2d(input_images, ngf, k_h=3, k_w=3, d_h=1, d_w=1,stddev=0.02,name="conv1_1")
	net = tf.nn.relu(net, 'relu1_1')
	net = conv2d(net, ngf, k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, name="conv1_2")
	net = batch_norm(net, a.mode=='train', 'bn1')
	net = tf.nn.relu(net, 'relu1_2') 
	#bn1(net, True) 
	# 128,128

	net = conv2d(net,ngf*2, k_h=3, k_w=3, d_h=1, d_w=1,stddev=0.02,name="conv2_1")
	net = tf.nn.relu(net, 'relu2_1')
	net = conv2d(net,ngf*2, k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, name="conv2_2")
	net = batch_norm(net, a.mode=='train', 'bn2') 
	net = tf.nn.relu(net, 'relu2_2') 
	
	# 64, 64 
	net = conv2d(net,ngf*4, k_h=3, k_w=3, d_h=1, d_w=1,stddev=0.02,name="conv3_1")
	net = tf.nn.relu(net, 'relu3_1')
	net = conv2d(net,ngf*4, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv3_2")
	net = tf.nn.relu(net, 'relu3_2') 
	net = conv2d(net,ngf*4, k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, name="conv3_3")
	net = batch_norm(net, a.mode=='train', 'bn3')
	net = tf.nn.relu(net, 'relu3_3') 
	

	net = conv2d(net,ngf*8, k_h=3, k_w=3, d_h=1, d_w=1,stddev=0.02,name="conv4_1")
	net = tf.nn.relu(net, 'relu4_1')
	net = conv2d(net,ngf*8, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv4_2")
	net = tf.nn.relu(net, 'relu4_2') 
	net = conv2d(net,ngf*8, k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, name="conv4_3")
	net = batch_norm(net, a.mode=='train', 'bn4')
	net = tf.nn.relu(net, 'relu4_3') 
	 #net = bn4(net, True) 

	# net = tf.nn.relu(net, 'relu5_3') 
	# net = batch_norm(name='bn5')(net, True)# bn5(net, True) 

	# net = atrous_conv2d(net,ngf*4, k_h=3, k_w=3, rate=2, stddev=0.02,name="conv6_1")
	# net = tf.nn.relu(net, 'relu6_1')
	# net = atrous_conv2d(net,ngf*4, k_h=3, k_w=3, rate=2, stddev=0.02, name="conv6_2")
	# net = tf.nn.relu(net, 'relu6_2') 
	# net = atrous_conv2d(net,ngf*4, k_h=3, k_w=3, rate=2, stddev=0.02, name="conv6_3")
	# net = tf.nn.relu(net, 'relu6_3') 
	# net = batch_norm(name='bn6')(net, True)#bn6(net, True) 


	# net = conv2d(net,ngf*4, k_h=3, k_w=3, d_h=1, d_w=1,stddev=0.02,name="conv7_1")
	# net = tf.nn.relu(net, 'relu7_1')
	# net = conv2d(net,ngf*4, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv7_2")
	# net = tf.nn.relu(net, 'relu7_2') 
	# net = conv2d(net,ngf*4, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv7_3")
	# net = tf.nn.relu(net, 'relu7_3') 
	return net

def build_decoder_colornet(features, generator_outputs_channels, a, reuse):
	ngf = a.ngf
	net = batch_norm(features, a.mode=='train', 'bn5')
	net = tf.nn.relu(net, 'relu7_3')
	 #bn7(net, True) 

	net_shape = [int(i) for i in net.get_shape()]
	net = deconv2d(net, [net_shape[0], net_shape[1]*2,net_shape[2]*2, ngf*8], 
		k_h=4, k_w=4, d_h=2, d_w=2,stddev=0.02, name="conv7.5_1")
	net = tf.nn.relu(net, 'relu7.5_1')
	net = conv2d(net,ngf*8, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv7.5_2")
	net = tf.nn.relu(net, 'relu7.5_2') 
	net = conv2d(net,ngf*8, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv7.5_3")
	net = batch_norm(net, a.mode=='train', 'bn7.5')
	net = tf.nn.relu(net, 'relu7.5_3')   
	#bn8(net, True)  
		
	net_shape = [int(i) for i in net.get_shape()]
	net = deconv2d(net, [net_shape[0], net_shape[1]*2,net_shape[2]*2, ngf*4], 
		k_h=4, k_w=4, d_h=2, d_w=2,stddev=0.02, name="conv8_1")
	net = tf.nn.relu(net, 'relu8_1')
	net = conv2d(net,ngf*4, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv8_2")
	net = tf.nn.relu(net, 'relu8_2') 
	net = conv2d(net,ngf*4, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv8_3")
	net = batch_norm(net, a.mode=='train', 'bn8')
	net = tf.nn.relu(net, 'relu8_3')   
	#bn8(net, True)  

	net_shape = [int(i) for i in net.get_shape()]
	net = deconv2d(net, [net_shape[0], net_shape[1]*2,net_shape[2]*2, ngf*2], k_h=4, k_w=4, d_h=2, d_w=2,
		stddev=0.02, name="conv9_1")

	net = tf.nn.relu(net, 'relu9_1')
	net = conv2d(net,ngf*2, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv9_2")
	net = batch_norm(net, a.mode=='train', 'bn9')
	net = tf.nn.relu(net, 'relu9_2')  
	#bn9(net, True)

	net_shape = [int(i) for i in net.get_shape()]
	net = deconv2d(net, [net_shape[0], net_shape[1]*2,net_shape[2]*2, ngf], 
		k_h=4, k_w=4, d_h=2, d_w=2,stddev=0.02, name="conv10_1")
	net = tf.nn.relu(net, 'relu10_1')
	net = conv2d(net,ngf, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv10_2")
	net = batch_norm(net, a.mode=='train', 'bn10')
	net = tf.nn.relu(net, 'relu10_2')  
	#bn9(net, True)

	net = conv2d(net,generator_outputs_channels, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv10_color") 
	# if a.text_model == 'attention_reasonet':
	# 	net = net
	# else:
	# 	net = tf.nn.tanh(net)	 

	return net
 
def colornet(input_, a, generator_outputs_channels = 2, reuse = None): 
	build_merger_ = build_merger if a.merger == 'basic' else build_contextual_merger
	input_images, input_captions = input_
	with tf.variable_scope('colornet', reuse = reuse):
		image_features = build_encoder_colornet(input_images, a, reuse)
		image_feature_dims = [int(image_features.get_shape()[1]), int(image_features.get_shape()[1]), int(image_features.get_shape()[-1])]

		text_embedding = build_text_encoder(input_captions, image_feature_dims, a)

		combined_layer = build_merger_(image_features, text_embedding, a, reuse = reuse) 

		net = build_decoder_colornet(combined_layer, generator_outputs_channels, a, reuse)

		return net  

def colorreasonet(input_, a, generator_outputs_channels = 2, reuse = None):  
	build_merger_ = build_merger if a.merger == 'basic' else build_contextual_merger
	ngf = a.ngf 
	input_images, input_captions = input_
	with tf.variable_scope('colornet', reuse = reuse):
		image_features = build_encoder_colornet(input_images, a, reuse)
		image_feature_dims = [int(image_features.get_shape()[1]), int(image_features.get_shape()[1]), int(image_features.get_shape()[-1])] 

		text_embedding = build_text_encoder(input_captions, image_feature_dims, a)

		combined_layers = []
		termination_gates = []


		combined_layer = image_features

		for i in range(a.T):
			reuse_ = reuse if i == 0 else True

			combined_layer = build_merger_(combined_layer, text_embedding, a, reuse = reuse_)  
			combined_layers.append(combined_layer)

		combined_layers = tf.stack(combined_layers, axis = 0) # [T, batchsize, h,w,num_feats] 

		h = int(combined_layers.get_shape()[2]); w = int(combined_layers.get_shape()[3]); num_feats = int(combined_layers.get_shape()[-1])
		combined_layers = tf.reshape(combined_layers, [-1, h, w, num_feats]) 
		outputs = build_decoder_colornet(combined_layers, generator_outputs_channels + 1, a, reuse = reuse)   
		outputs_shape = [int(i) for i in outputs.get_shape()]

		outputs = tf.reshape(outputs, [a.T, -1, outputs_shape[1], outputs_shape[2], generator_outputs_channels + 1])# [T, Batchsize, h,w, Num_feats+1] 

		termination_gates = tf.clip_by_value(tf.nn.sigmoid(outputs[:,:,:,:,0]), clip_value_min=1e-6, clip_value_max=1-1e-6) 
		outputs = outputs[:,:,:,:,1:] 

		return [outputs, termination_gates]              

def create_generator_colornet(generator_inputs, a):  
	# input_images, input_captions = generator_inputs  

	generator_outputs_channels = a.num_classes

	if a.text_model != 'attention_reasonet':
		generator = colornet(generator_inputs, a, generator_outputs_channels, reuse = None)  
	elif a.text_model == 'attention_reasonet': 
		generator = colorreasonet(generator_inputs, a, generator_outputs_channels, reuse = None) 

	return generator 



