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
parser.add_argument("--mode", required=True, choices=["train", "test", "export", "test2"])
parser.add_argument("--model", default='pix2pix', choices=['pix2pix','colornet'])
parser.add_argument("--text_model", default='None', choices=['None','reed','attention','attention_fix','attention_reasonet'])
parser.add_argument("--rnn", default='None', choices=['None','reed','attention','attention_fix','reasonet'])
parser.add_argument("--dataset",default='flower', choices = ['flower','bird'], help="path to folder containing images")

# parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing") 

parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")   
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--num_layers", type=int, default=2, help="number of layers in decoder")

parser.add_argument("--max_epochs", type=int, default=40, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--T", type=int, default=4, help="number of steps for reasonet")

parser.add_argument("--att_dim", type=int, default=128, help="hidden dimension of lstm.") 
# parser.add_argument("--add_attention", type=int, default=1, help="if we add attention")

parser.add_argument("--lstm_dim", type=int, default=128, help="hidden dimension of lstm.")
parser.add_argument("--scale_size", type=int, default=72, help="scale images to this size before cropping to 256x256")
parser.add_argument("--min_after_dequeue", type=int, default=100, help="min_after_dequeue")
parser.add_argument("--crop_size", type=int, default=64, help="crop images to this size before cropping to 256x256")
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")  

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args() 
dict_a = vars(a) 
# Load in embedding.
with open('{}/embedding.p'.format(a.dataset), 'rb') as f:
	embedding = pkl.load(f) 
	embedding_matrix = embedding['embedding matrix']
	vocab_size, emb_dim = embedding_matrix.shape
dict_a.update({'vocab_size': vocab_size, 'emb_dim': emb_dim}) 

mode = 'train' if a.mode == 'train' else 'test'
dict_a.update({'input_dir': '{}/{}_data'.format(a.dataset,mode),'output_dir': '{}/{}_{}{}{}'.format(a.dataset, a.model, 
	a.text_model, a.crop_size, '' if mode =='train' else '_test')})

# Compute steps per epoch.
input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size)) * 10

def main():  
	if a.seed is None:
		a.seed = random.randint(0, 2**31 - 1)

	tf.set_random_seed(a.seed)
	np.random.seed(a.seed)
	random.seed(a.seed)

	if not os.path.exists(a.output_dir):
		os.makedirs(a.output_dir)

	if a.mode == "test" or a.mode == "test2":
		if a.checkpoint is None:
			raise Exception("checkpoint required for test mode")

		# load some options from the checkpoint
		options = {"which_direction", "ngf", "ndf", "lab_colorization"}
		with open(os.path.join(a.checkpoint, "options.json")) as f:
			for key, val in json.loads(f.read()).items():
				if key in options:
					print("loaded", key, "=", val)
					setattr(a, key, val)
		# disable these features in test mode
		a.scale_size = a.crop_size
		a.flip = False

	for k, v in a._get_kwargs():
		print(k, "=", v)

	with open(os.path.join(a.output_dir, "options.json"), "w") as f:
		f.write(json.dumps(vars(a), sort_keys=True, indent=4))


	with tf.variable_scope('input_pipe'), tf.device('/cpu:0'):
		filenames = tf.train.match_filenames_once('{}/{}-*.tfrecord'.format(a.dataset, 'train' if a.mode == 'train' else 'test'))
		examples = batcher(filenames, a) 

		if a.mode == 'test2':
			if a.dataset == 'flower':
				input_filenames = ['flower/test_data/'+ i for i in ['image_03132.jpg','image_06686.jpg','image_06740.jpg','image_03324.jpg',
				'image_05208.jpg','image_06172.jpg','image_05660.jpg','image_04324.jpg','image_04308.jpg','image_04320.jpg']] # + ['flower/train_data/image_00029.jpg']
				input_caption_strs = [['the flower has red petals'] for i in range(10)]   #[['yellow yellow yellow'] for i in range(5)]##
				examples =  create_test_batcher(input_filenames, input_caption_strs, embedding, a)
	
	# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


	# inputs and targets are [batch_size, height, width, channels] 
	
	inputs = tf.placeholder(tf.float32, shape = [a.batch_size, a.crop_size, a.crop_size, 1])
	if a.text_model != 'reed':
		captions = tf.placeholder(tf.int64, shape = [a.batch_size, None])
	else:
		captions = tf.placeholder(tf.float32, shape = [a.batch_size, 1024])
	sequence_lengths = tf.placeholder(tf.int64, shape = [a.batch_size])
	targets = tf.placeholder(tf.float32, shape = [a.batch_size, a.crop_size, a.crop_size, 2])


	model = create_model(inputs, captions, sequence_lengths, targets, a)
	Examples = collections.namedtuple("Examples", "inputs, targets, captions, sequence_lengths")
	placeholder_example = Examples(inputs=inputs,
								targets=targets,
								captions = captions,
								sequence_lengths = captions, sequence_lengths)

	converted_inputs, converted_targets, converted_outputs, display_fetches = convert_to_normal(model, placeholder_example)

	create_summaries(converted_inputs, converted_targets, converted_outputs, model)

	with tf.name_scope("parameter_count"):
		parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

	saver = tf.train.Saver(max_to_keep=1,var_list = {v for v in tf.trainable_variables() \
			if 'ExponentialMovingAverage' not in v.name})

	if a.text_model.startswith('attention') and a.mode == 'train': 
		saver_pretrained_attention = tf.train.Saver(var_list = {v.op.name[19:]: v for v in tf.trainable_variables() \
			if v.name.startswith('generator/colornet/RNN') and "attention" not in v.name})

	sv = tf.train.Supervisor(logdir=a.output_dir, save_summaries_secs=0, saver=None)
	with sv.managed_session() as sess:  
		if a.text_model.startswith('attention'):
			# initialize word embedding. 
			sess.run(model.emb_init, feed_dict = {model.embedding_placeholder: embedding_matrix})

		# initialize filename queue.
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord) 

		print("parameter_count =", sess.run(parameter_count))

		if (a.text_model.startswith('attention')) and a.mode == 'train':
			print("loading text embedding from checkpoint")
			checkpoint = tf.train.latest_checkpoint('{}/pretrained_64/'.format(a.dataset))
			saver_pretrained_attention.restore(sess, checkpoint)


		#  loading model from checkpoint
		if a.checkpoint is not None:
			print("loading model from checkpoint")
			checkpoint = tf.train.latest_checkpoint(a.checkpoint)
			saver.restore(sess, checkpoint)

		max_steps = steps_per_epoch * a.max_epochs 

		if a.mode == "test":
			caption_file = open(os.path.join(a.output_dir, 'caption_list.txt'), 'wb') 

			# testing
			# at most, process the test data once
			max_steps = min(steps_per_epoch, max_steps) 
			image_count = 0

			id2word = embedding['id2word'] 
			for step in range(max_steps):
				
				if a.text_model == 'attention_reasonet':
					inputs_val, captions_val, sequence_lengths_val, targets_val = sess.run([examples.inputs, 
						examples.captions, examples.sequence_lengths, examples.targets]) 

					results, termination = sess.run([display_fetches, model.termination], 
						feed_dict = {inputs: inputs_val, captions: captions_val, sequence_lengths: sequence_lengths_val, targets: targets_val})
				else: 

					inputs_val, captions_val, sequence_lengths_val, targets_val = sess.run([examples.inputs, 
						examples.captions, examples.sequence_lengths, examples.targets]) 

					results = sess.run(display_fetches, 
						feed_dict = {inputs: inputs_val, captions: captions_val, sequence_lengths: sequence_lengths_val, targets: targets_val})
				
				filesets, image_count = save_images(results, image_count, a, caption_file, id2word)
				for i, f in enumerate(filesets):
					print("evaluated image", f["name"])
					# if a.text_model == 'attention_reasonet':
					# 	print(termination)
				index_path = append_index(filesets, a)

			caption_file.close()
			print("wrote index at", index_path)

		elif a.mode == 'test2':
			caption_file = open(os.path.join(a.output_dir, 'caption_list.txt'), 'wb') 

			# testing
			# at most, process the test data once
			max_steps = len(examples.token_ids)
			image_count = 0

			id2word = embedding['id2word']
			for step in range(max_steps):  
				results = sess.run(display_fetches, feed_dict = {examples.raw_caption_id: examples.token_ids[step], 
					examples.raw_image: examples.images[step],
					examples.raw_sequence_lengths: examples.sequence_lengths_array[step]}) 

				filesets, image_count = save_images(results, image_count, a, caption_file, id2word)
				for i, f in enumerate(filesets):
					print("evaluated image", f["name"])
				index_path = append_index(filesets, a)

			caption_file.close()
			print("wrote index at", index_path)

		else:
			# training
			start = time.time()
			image_count = 0
			for step in range(max_steps):
				def should(freq):
					return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1) 
				discrim_train = model.discrim_train
				
				fetches = {
					"train": model.train, 
					"global_step": sv.global_step,
				}

				if should(a.progress_freq):
					fetches["discrim_loss"] = model.discrim_loss
					fetches["gen_loss_GAN"] = model.gen_loss_GAN
					fetches["gen_loss_L1"] = model.gen_loss_L1

				if should(a.summary_freq):
					fetches["summary"] = sv.summary_op

				if should(a.display_freq):
					fetches["display"] = display_fetches

				inputs_val, captions_val, sequence_lengths_val, targets_val = sess.run([examples.inputs, 
					examples.captions, examples.sequence_lengths, examples.targets]) 

				sess.run(discrim_train, feed_dict = {inputs: inputs_val, captions: captions_val, sequence_lengths: sequence_lengths_val, targets: targets_val})
				results = sess.run(fetches, feed_dict = {inputs: inputs_val, captions: captions_val, sequence_lengths: sequence_lengths_val, targets: targets_val}) 
 
				# # Add one more training step for generator. 
				# sess.run(fetches["train"])

				if should(a.summary_freq):
					print("recording summary")
					sv.summary_writer.add_summary(results["summary"], results["global_step"])

				if should(a.display_freq):
					print("saving display images")
					filesets, image_count = save_images(results["display"], image_count, a, step=results["global_step"])
					append_index(filesets, a, step=True)

				if should(a.progress_freq):
					# global_step will have the correct step count if we resume from a checkpoint 
					train_epoch = math.ceil(results["global_step"] / steps_per_epoch)
					train_step = (results["global_step"] - 1) % steps_per_epoch + 1
					rate = (step + 1) * a.batch_size / (time.time() - start)
					remaining = (max_steps - step) * a.batch_size / rate
					print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
					print("discrim_loss", results["discrim_loss"])
					print("gen_loss_GAN", results["gen_loss_GAN"])
					print("gen_loss_L1", results["gen_loss_L1"])

				if should(a.save_freq):
					print("saving model")
					saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

				if sv.should_stop():
					break


main()

