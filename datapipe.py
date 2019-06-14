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
from utils import *
def read_my_file_format(filename_queue, a):
	"""Sets up part of the pipeline that takes elements from the filename queue
	and turns it into a tf.Tensor of a batch of images.

	:param filename_queue:
		tf.train.string_input_producer object
	:param resize_shape:
		2 element list defining the shape to resize images to.
	"""
	reader = tf.TFRecordReader()
	key, serialized_example = reader.read(filename_queue)
	features, sequence_features = tf.parse_single_sequence_example(
		serialized_example, context_features={
			'image/encoded': tf.FixedLenFeature([], tf.string),
			'image/height': tf.FixedLenFeature([], tf.int64),
			'image/channels': tf.FixedLenFeature([], tf.int64),
			'image/width': tf.FixedLenFeature([], tf.int64),
			# 'image/caption_embedding': tf.FixedLenFeature([], tf.string),
			'image/sequence_length': tf.FixedLenFeature([], tf.int64)
			}, sequence_features={
			"caption": tf.FixedLenSequenceFeature([], dtype=tf.int64)
			})

	# inputs = [features, sequence_features]
	# queue = tf.RandomShuffleQueue(CAPACITY, MIN_AFTER_DEQUEUE, dtypes)
	# enqueue_op = queue.enqueue(inputs)
	# qr = tf.train.QueueRunner(queue, [enqueue_op] * NUM_THREADS)
	# tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)
	# features, sequence_features = queue.dequeue()
	# if a.text_model == 'reed':
	# 	captions = tf.decode_raw(features['image/caption_embedding'], tf.float32)
	# 	captions.set_shape([1024])
	if a.text_model == 'attention' or a.text_model == 'attention_fix' or a.text_model == 'attention_reasonet' or a.text_model == 'None':
		captions = sequence_features["caption"]

	sequence_length = features['image/sequence_length']

	raw_input = tf.image.decode_jpeg(features['image/encoded'], 3)

	raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

	assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
	with tf.control_dependencies([assertion]):
		raw_input = tf.identity(raw_input)

		raw_input.set_shape([None, None, 3])

		if True:
			#a.lab_colorization:
			# load color and brightness from image, no B image exists here
			lab = rgb_to_lab(raw_input)
			L_chan, a_chan, b_chan = preprocess_lab(lab)
			a_images = tf.expand_dims(L_chan, axis=2)
			b_images = tf.stack([a_chan, b_chan], axis=2)  
	# synchronize seed for image operations so that we do the same operations to both
	# input and output images
	seed = random.randint(0, 2**31 - 1)
	def transform(image):
		r = image 

		if a.flip:
			r = tf.image.random_flip_left_right(r, seed=seed)

		# area produces a nice downscaling, but does nearest neighbor for upscaling
		# assume we're going to be doing downscaling here
		r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

		offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - a.crop_size + 1, seed=seed)), dtype=tf.int32)
		if a.scale_size > a.crop_size:
			r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], a.crop_size, a.crop_size)
		elif a.scale_size < a.crop_size:
			raise Exception("scale size cannot be less than crop size")
		return r
		
	with tf.name_scope("input_images"):
		input_images = transform(a_images)

	with tf.name_scope("target_images"):
		target_images = transform(b_images)

	return input_images, target_images, captions, sequence_length

def batcher(filenames, a):
	"""Creates the batching part of the pipeline.

	:param filenames:
		list of filenames
	:param batch_size:
		size of batches that get output upon each access.
	:param resize_shape:
		for preprocessing. What to resize images to.
	:param num_epochs:
		number of epochs that define end of training set.
	:param min_after_dequeue:
		min_after_dequeue defines how big a buffer we will randomly sample
		from -- bigger means better shuffling but slower start up and more
		memory used.
		capacity must be larger than min_after_dequeue and the amount larger
		determines the maximum we will prefetch.  Recommendation:
		min_after_dequeue + (num_threads + a small safety margin) * batch_size
	"""
	Examples = collections.namedtuple("Examples", "inputs, targets, captions, sequence_lengths")

	filename_queue = tf.train.string_input_producer(
		filenames, num_epochs=a.max_epochs, shuffle=True)


	# min_queue_examples = 1
	# capacity = min_queue_examples + 100 * a.batch_size
	# filename_queue = tf.RandomShuffleQueue(
	#     capacity=capacity,
	#     min_after_dequeue=min_queue_examples,
	#     dtypes=[tf.string],
	#     name="random_") 


	## Test filename queue is correct.
	# filename_dequeue = filename_queue.dequeue()

	# sv = tf.train.Supervisor(logdir=a.output_dir, save_summaries_secs=0, saver=None)
	# with sv.managed_session() as sess:  

	# 	# initialize word embedding. 
	# 	# sess.run(model.emb_init, feed_dict = {model.embedding_placeholder: embedding_matrix})

	# 	# initialize filename queue.
	# 	coord = tf.train.Coordinator()
	# 	threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
	# 	for i in range(40):
	# 		print(sess.run(filename_dequeue), 'These are the filenames', i)

	input_images, target_images, captions, sequence_lengths = read_my_file_format(filename_queue, a)  
	# inputs_batch, targets_batch, captions_batch = tf.train.shuffle_batch(
	#   [input_images, target_images, captions], batch_size=batch_size, capacity=min_after_dequeue + 3 * batch_size,
	#   min_after_dequeue=min_after_dequeue)
	inputs_batch, targets_batch, captions_batch, sequence_lengths_batch = tf.train.batch(
	  [input_images, target_images, captions, sequence_lengths], batch_size=a.batch_size, capacity=32, dynamic_pad=True)

	return Examples( 
		inputs=inputs_batch,
		targets=targets_batch,
		captions = captions_batch,
		sequence_lengths = sequence_lengths_batch
	) 

def save_images(fetches, image_count, a, caption_file, id2word, step=None):
	image_dir = os.path.join(a.output_dir, "images")
	if not os.path.exists(image_dir):
		os.makedirs(image_dir) 
	filesets = []
	# captions = fetches['captions']
	for i, _ in enumerate(fetches["inputs"]):
		name = str(image_count)
		fileset = {"name": name, "step": step}

		cap = fetches["captions"][i]
		if a.text_model.startswith('attention'):
			word_caption = str(image_count) + ': ' + ' '.join([id2word[j] for j in cap]) + '\n'
			caption_file.write(word_caption)

		for kind in ["inputs", "outputs", "targets"]:
			filename = name + "-" + kind + ".png"
			if step is not None:
				filename = "%08d-%s" % (step, filename)
			fileset[kind] = filename
			out_path = os.path.join(image_dir, filename)
			contents = fetches[kind][i] 
			with open(out_path, "wb") as f: 
				f.write(contents)
		filesets.append(fileset)
		image_count += 1
	return filesets, image_count

def append_index(filesets, a, step=False):
	index_path = os.path.join(a.output_dir, "index.html")
	if os.path.exists(index_path):
		index = open(index_path, "a")
	else:
		index = open(index_path, "w")
		index.write("<html><body><table><tr>")
		if step:
			index.write("<th>step</th>")
		index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

	for fileset in filesets:
		index.write("<tr>")

		if step:
			index.write("<td>%d</td>" % fileset["step"])
		index.write("<td>%s</td>" % fileset["name"])

		for kind in ["inputs", "outputs", "targets"]:
			index.write("<td><img src='images/%s'></td>" % fileset[kind])

		index.write("</tr>")
	return index_path

def convert_to_normal(model, examples):
	# undo colorization splitting on images that we use for display/output 

	# inputs is brightness, this will be handled fine as a grayscale image
	# need to augment targets and outputs with brightness
	targets = augment(examples.targets, examples.inputs)
	outputs = augment(model.outputs, examples.inputs)
	# inputs can be deprocessed normally and handled as if they are single channel
	# grayscale images
	inputs = deprocess(examples.inputs)  

	def convert(image):
		# if a.aspect_ratio != 1.0:
		# 	# upscale to correct aspect ratio
		# 	size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
		# 	image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

		return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

	# reverse any processing on images so they can be written to disk or displayed to user
	with tf.name_scope("convert_inputs"):
		converted_inputs = convert(inputs)

	with tf.name_scope("convert_targets"):
		converted_targets = convert(targets)

	with tf.name_scope("convert_outputs"):
		converted_outputs = convert(outputs)

	with tf.name_scope("encode_images"):
		display_fetches = {
			# "paths": examples.paths,
			"inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
			"targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
			"outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
			"captions": examples.captions
		}
	return 	converted_inputs, converted_targets, converted_outputs, display_fetches 

from extract_cap import tokenize_caption, extract_token_id
from PIL import Image
def create_test_batcher(filenames, caption_strs, embedding, a):
	# model = gensim.models.KeyedVectors.load_word2vec_format('flower/GoogleNews-vectors-negative300.bin', binary = True)
	caption_strs = {'test': caption_strs}
	tokens = tokenize_caption(caption_strs)

	token_ids = extract_token_id(tokens, embedding['word2id'].keys(), embedding['word2id'])['test'] # list of caption ids. 

	sequence_lengths_array = [len(token) for token in token_ids]
	


	image_matrices = []
	for filename in filenames:
		image_matrices.append(np.array(Image.open(filename)))

	raw_caption_id = tf.placeholder(tf.int64, [None], name = 'caption')
	raw_image_ = tf.placeholder(tf.uint8, [None, None, 3], name = 'image')
	raw_sequence_lengths = tf.placeholder(tf.int64, name = 'sequence_length')

	raw_image = tf.image.convert_image_dtype(raw_image_, dtype=tf.float32) 

	assertion = tf.assert_equal(tf.shape(raw_image)[2], 3, message="image does not have 3 channels")
	with tf.control_dependencies([assertion]):
		raw_image = tf.identity(raw_image)

		raw_image.set_shape([None, None, 3])

		if True:#a.lab_colorization:
			# load color and brightness from image, no B image exists here
			lab = rgb_to_lab(raw_image)
			L_chan, a_chan, b_chan = preprocess_lab(lab)
			a_images = tf.expand_dims(L_chan, axis=2)
			b_images = tf.stack([a_chan, b_chan], axis=2)  

	# synchronize seed for image operations so that we do the same operations to both
	# input and output images
	seed = random.randint(0, 2**31 - 1)
	def transform(image):
		r = image
		if a.flip:
			r = tf.image.random_flip_left_right(r, seed=seed)

		# area produces a nice downscaling, but does nearest neighbor for upscaling
		# assume we're going to be doing downscaling here


		r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

		offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - a.crop_size + 1, seed=seed)), dtype=tf.int32)
		if a.scale_size > a.crop_size:
			r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], a.crop_size, a.crop_size)
		elif a.scale_size < a.crop_size:
			raise Exception("scale size cannot be less than crop size")
		return r
		
	with tf.name_scope("input_images"):
		input_images = transform(a_images)

	with tf.name_scope("target_images"):
		target_images = transform(b_images)

	Examples = collections.namedtuple("Examples", "images, token_ids, sequence_lengths_array, raw_caption_id, raw_image, raw_sequence_lengths, inputs, targets, captions, sequence_lengths")

	return Examples(
		images=image_matrices,
		token_ids=token_ids,
		sequence_lengths_array=sequence_lengths_array,
		raw_caption_id=raw_caption_id,
		raw_image=raw_image_,
		raw_sequence_lengths = raw_sequence_lengths,
		inputs=tf.expand_dims(input_images,axis = 0),
		targets=tf.expand_dims(target_images,axis = 0),
		captions = tf.expand_dims(raw_caption_id, axis = 0),
		sequence_lengths = tf.expand_dims(raw_sequence_lengths, axis = 0)
	 )