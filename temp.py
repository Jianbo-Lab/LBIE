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
from cells import * 
from model2 import *
from datapipe import *
import cPickle as pkl
parser = argparse.ArgumentParser()
# parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", default = 'test', choices=["train", "test", "export", "test2"])

parser.add_argument("--color", default = 'white', type = str)
parser.add_argument("--model", default='colornet', choices=['pix2pix','colornet'])
parser.add_argument("--text_model", default='attention', choices=['None','reed','attention','attention_fix','attention_reasonet'])
# parser.add_argument("--rnn", default='None', choices=['None','reed','attention','attention_fix','reasonet'])
parser.add_argument("--dataset",default='flower', choices = ['flower','bird'], help="path to folder containing images")

parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default='models', help="directory with checkpoint to resume training from or use for testing") 

parser.add_argument("--summary_freq", type=int, default=0, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=0, help="display progress every progress_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")   
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)

parser.add_argument("--max_epochs", type=int, default=40, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--ngf", type=int, default=32, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--T", type=int, default=1, help="number of steps for reasonet")

parser.add_argument("--att_dim", type=int, default=128, help="hidden dimension of lstm.") 
# parser.add_argument("--add_attention", type=int, default=1, help="if we add attention")

# parser.add_argument("--lstm_dim", type=int, default=128, help="hidden dimension of lstm.")
parser.add_argument("--scale_size", type=int, default=256, help="scale images to this size before cropping to 256x256")
parser.add_argument("--min_after_dequeue", type=int, default=100, help="min_after_dequeue")
parser.add_argument("--crop_size", type=int, default=256, help="crop images to this size before cropping to 256x256")
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient") 
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])

# python main.py --dataset flower --model colornet --max_epochs 40 --batch_size 1 --crop_size 64 --lr 0.0002 --scale_size 72 --text_model reed --mode test --checkpoint flower/colornet_reed64 

parser.add_argument("--input_dir", default='bw_images')
parser.add_argument("--text_file", default="texts.txt") 
parser.add_argument("--output_dir", default="color_images")
a = parser.parse_args() 
dict_a = vars(a) 
# Load in embedding.
with open('{}/embedding.p'.format(a.dataset), 'rb') as f:
	embedding = pkl.load(f) 
	embedding_matrix = embedding['embedding matrix']
	vocab_size, emb_dim = embedding_matrix.shape
dict_a.update({'vocab_size': vocab_size, 'emb_dim': emb_dim}) 

mode = 'train' if a.mode == 'train' else 'test'

# Compute steps per epoch.
input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size)) #* 10


from extract_cap import tokenize_caption, extract_token_id
from PIL import Image
def create_test_batcher(filenames, caption_strs, embedding, a):
	tokens = tokenize_caption(caption_strs) 
	for i in tokens:
		tokens[i][0][3] = a.color
	token_ids_ = extract_token_id(tokens, embedding['word2id'].keys(), embedding['word2id'])#['test'] # list of caption ids.  
	sequence_lengths_array = [len(tokens[key][0]) for key in tokens] 

	image_matrices = []
	token_ids = []
	for filename in filenames:
		image_matrices.append(np.array(Image.open(filename)))
		token_ids.append(token_ids_[filename][0]) 
	raw_caption_id = tf.placeholder(tf.int64, [None], name = 'caption')

	raw_image_ = tf.placeholder(tf.uint8, [None, None, 3], name = 'image')
	raw_sequence_lengths = tf.placeholder(tf.int64, 
		name = 'sequence_length')

	raw_image = tf.image.convert_image_dtype(raw_image_, 
		dtype=tf.float32) 

	assertion = tf.assert_equal(tf.shape(raw_image)[2], 3, 
		message="image does not have 3 channels")
	with tf.control_dependencies([assertion]):
		raw_image = tf.identity(raw_image)

		raw_image.set_shape([None, None, 3])

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

	Examples = collections.namedtuple("Examples", "images, token_ids, sequence_lengths_array, raw_caption_id, raw_image, raw_sequence_lengths, inputs, targets, captions, sequence_lengths, filenames") 

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
		sequence_lengths = tf.expand_dims(raw_sequence_lengths, 
			axis = 0),
		filenames = filenames
	 )

def save_images(fetches, image_count, a, id2word, filename_, step=None):
	image_dir = a.output_dir  
	if not os.path.exists(image_dir):
		os.makedirs(image_dir) 
	filesets = []
	# captions = fetches['captions']
	for i, _ in enumerate(fetches["inputs"]):
		name = str(image_count)
		fileset = {"name": name, "step": step} 

		for kind in ["inputs", "outputs", "targets"]:
			# kind = "outputs"
			start_name = 'color' if kind == 'outputs' else "bw"
			filename = start_name + '-' + filename_[:-4] + ".png"
			if step is not None:
				filename = "{}-{}-{}.png".format(a.color,step,kind) if kind != "targets" else "{}-{}.png".format(step,kind)
			fileset[kind] = filename
			out_path = os.path.join(image_dir, filename)
			contents = fetches[kind][i] 
			with open(out_path, "wb") as f: 
				f.write(contents)

		filesets.append(fileset)
		image_count += 1
	return filesets, image_count

def main():  
	if a.seed is None:
		a.seed = random.randint(0, 2**31 - 1)

	tf.set_random_seed(a.seed)
	np.random.seed(a.seed)
	random.seed(a.seed)

	if not os.path.exists(a.output_dir):
		os.makedirs(a.output_dir)

	# disable these features in test mode
	a.scale_size = a.crop_size
	a.flip = False

	for k, v in a._get_kwargs():
		print(k, "=", v) 


	with tf.variable_scope('input_pipe'), tf.device('/cpu:0'):
		# filenames = tf.train.match_filenames_once('{}/{}-*.tfrecord'.format(a.dataset,  'train' if a.mode == 'train' else 'test'))
		# examples = batcher(filenames, a) 

		input_filenames =[a.input_dir + '/' + i for i in os.listdir(a.input_dir)]
		# for i in os.listdir(a.input_dir):
		# 	print(i)
		with open(a.text_file,'rb') as f:
			# texts = f.readlines()
			texts = f.read().splitlines()
			input_caption_strs = {a.input_dir + '/' + text.split(':')[0]:[text.split(':')[1]] for text in texts}

		examples =  create_test_batcher(input_filenames, input_caption_strs, embedding, a)
	
	# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


	# inputs and targets are [batch_size, height, width, channels] 
	model = create_model(examples.inputs, examples.captions, examples.sequence_lengths, examples.targets, a) 

	converted_inputs, converted_targets, converted_outputs, display_fetches = convert_to_normal(model, examples) 

	saver = tf.train.Saver(max_to_keep=1,var_list = {v for v in tf.trainable_variables()}) 

	init_op = tf.global_variables_initializer()
	# sv = tf.train.Supervisor(logdir=a.output_dir, save_summaries_secs=0, saver=None)
	with tf.Session() as sess:  
		sess.run(init_op)
		sess.run(model.emb_init, feed_dict = {model.embedding_placeholder: embedding_matrix}) 


		#  loading model from checkpoint 
		print("loading model from checkpoint")
		checkpoint = 'models/model-245000'

		saver.restore(sess, checkpoint)


		max_steps = steps_per_epoch * a.max_epochs   

		# testing
		# at most, process the test data once
		max_steps = len(examples.token_ids)
		image_count = 0

		id2word = embedding['id2word']
		for step in range(max_steps): 
			results = sess.run(display_fetches, feed_dict = {examples.raw_caption_id: examples.token_ids[step],
				examples.raw_image: examples.images[step],
				examples.raw_sequence_lengths: examples.sequence_lengths_array[step]}) 
			current_filename=examples.filenames[step].split('/')[1]  
			filesets, image_count = save_images(results, image_count, a, id2word, current_filename, step)
			for i, f in enumerate(filesets):
				print("evaluated image", f["name"])  

main()

