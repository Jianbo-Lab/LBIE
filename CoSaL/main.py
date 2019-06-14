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
# from utils import *
from cells import * 
from model import *
from make_data import * 
import cPickle as pkl
parser = argparse.ArgumentParser()
# parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--model", default='pix2pix', choices=['pix2pix','colornet']) 
parser.add_argument("--text_model", default='attention', choices=['attention','attention_reasonet'])
parser.add_argument("--merger", default='basic', choices=['basic','relational'])

# parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing") 

parser.add_argument("--summary_freq", type=int, default=10, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=5, help="display progress every progress_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=500, help="save model every save_freq steps, 0 to disable")  

parser.add_argument("--img_size", type=int, default=48, help="number of training epochs")
parser.add_argument("--max_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--ngf", type=int, default=16, help="number of generator filters in first conv layer") 
parser.add_argument("--T", type=int, default=4, help="number of steps for reasonet")
parser.add_argument("--att_dim", type=int, default=16, help="hidden dimension of lstm.")   
parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")  
parser.add_argument("--num_classes", type=int, default=7)
parser.add_argument("--num_train", type=int, default=2000) 
parser.add_argument("--num_abs", type=int, default=1) 

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

num_train = a.num_train

num_test = 1000 
if a.mode == 'train':
	np.random.seed(0)
	random.seed(0)
	num_test = 1 

# Generate data for extracting id2word.
dataset = Dataset(1,1, a.img_size, num_abs = a.num_abs) 

# Get vocab size from dataset. 
dict_a = vars(a)  
dict_a.update({'vocab_size': len(dataset.id2word.keys())}) 

# In the infinite data setting.
dict_a.update({'update_data_freq': a.num_train/a.batch_size})


# Name checkpoint with routine. 
if a.checkpoint:
	dict_a['checkpoint'] = 'models/{}_{}_{}'.format(a.text_model,a.merger,a.num_abs)
	print('loading checkpoint...')


dict_a.update({
	'output_dir': 'models/{}_{}_{}'.format(a.text_model,a.merger,a.num_abs)})

# Compute steps per epoch.
# input_paths = glob.glob(os.path.join(a.input_dir, "*.p"))
steps_per_epoch = int(math.ceil(num_train / a.batch_size)) if a.mode == 'train' else  int(math.ceil(num_test / a.batch_size))

def main():  
	if a.seed is None:
		a.seed = random.randint(0, 2**31 - 1)



	tf.set_random_seed(a.seed)
	np.random.seed(a.seed)
	random.seed(a.seed)
	# Generate data. 
	# dataset = Dataset(num_train, num_test, a.img_size, num_abs = a.num_abs) 

	with open('models/dataset_7.p','rb') as f:
		dataset = pkl.load(f)

	if not os.path.exists(a.output_dir):
		os.makedirs(a.output_dir)

	if a.mode == "test":
		if a.checkpoint is None:
			raise Exception("checkpoint required for test mode") 

	for k, v in a._get_kwargs():
		print(k, "=", v)

	with open(os.path.join(a.output_dir, "options.json"), "w") as f:
		f.write(json.dumps(vars(a), sort_keys=True, indent=4))


	
	# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
 
	inputs = tf.placeholder(tf.uint8, [a.batch_size, a.img_size, a.img_size, 3])
	reals = tf.placeholder(tf.uint8, [a.batch_size, a.img_size, a.img_size, 3])
	labels = tf.placeholder(tf.int64, [a.batch_size, a.img_size, a.img_size])

	normalized_inputs = tf.to_float(inputs) / 255.0 - 0.5

	captions = tf.placeholder(tf.int64, [a.batch_size, 9, 3])
	model = create_model(normalized_inputs, labels, captions, a) 

	display_fetches = {'real': reals, 'predictions': model.predictions}
	if a.text_model == 'attention_reasonet':
		all_predictions = tf.cast(tf.cumsum(tf.to_int64(model.all_predictions) * tf.expand_dims(model.termination,-1), axis = 0),tf.uint8)
		display_fetches.update({'predictions_{}'.format(i): all_predictions[i] for i in range(a.T)})
		display_fetches.update({'termination_{}'.format(i): 255 * model.termination[i] for i in range(a.T)})
		# display_fetches.update({'all_predictions': model.all_predictions, 'termination': model.termination})

	create_summaries(reals, model, a)
	tf.summary.scalar("loss", model.gen_loss)
	tf.summary.scalar("iou", model.mean_iou)

	with tf.name_scope("parameter_count"):
		parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

	saver = tf.train.Saver(max_to_keep=1,var_list = {v for v in tf.trainable_variables()})

	sv = tf.train.Supervisor(logdir=a.output_dir, save_summaries_secs=0, saver=None)
	with sv.managed_session() as sess:      

		print("parameter_count =", sess.run(parameter_count))
		# print('saving a test image ... ')
		# io.imsave('./test.png',sess.run(test_image)[0])

		# if (a.text_model.startswith('attention')) and a.mode == 'train':
		# 	print("loading text embedding from checkpoint")
		# 	checkpoint = tf.train.latest_checkpoint('nyu/pretrained/')
		# 	saver_pretrained_attention.restore(sess, checkpoint)


		#  loading model from checkpoint
		if a.checkpoint is not None:
			print("loading model from checkpoint")
			checkpoint = tf.train.latest_checkpoint(a.checkpoint)
			# saver.restore(sess, 'models/attention_reasonet_relational/model-20000')
			saver.restore(sess, checkpoint)

		max_steps = steps_per_epoch * a.max_epochs 

		if a.mode == "test":

			caption_file = open(os.path.join(a.output_dir, 'caption_list.txt'), 'wb') 

			# testing
			# at most, process the test data once
			max_steps = min(steps_per_epoch, max_steps)  
			image_count = 0 
			iou,loss = 0.0,0.0
			for step in range(max_steps):
				batch_inputs, batch_reals, batch_labels, batch_captions = dataset.train.next_batch(a.batch_size)
				feed_dict = {inputs: batch_inputs,reals: batch_reals,labels: batch_labels,captions:batch_captions}	

				results = sess.run(display_fetches,feed_dict = feed_dict) 
				filesets, image_count = save_images(results, image_count, a, caption_file, dataset.id2word)

				sess.run(model.update_iou,
					feed_dict=feed_dict) 
				iou += sess.run(model.mean_iou,
					feed_dict=feed_dict) 
				loss += sess.run(model.batch_loss,
					feed_dict=feed_dict)

				
				# for i, f in enumerate(filesets):
				# 	print("evaluated image", f["name"])
					# if a.text_model == 'attention_reasonet':
					# 	print(termination)
				# index_path = append_index(filesets, a)

			iou = iou / max_steps
			loss = loss / max_steps
			print('The mean IOU is {}.'.format(iou))
			print('The mean loss is {}'.format(loss))
			caption_file.close()
			# print("wrote index at", index_path)

		else:
			# training
			start = time.time()
			image_count = 0 
			for step in range(max_steps):

				batch_inputs, batch_reals, batch_labels, batch_captions = dataset.train.next_batch(a.batch_size)
				feed_dict = {inputs: batch_inputs,reals: batch_reals,labels: batch_labels,captions:batch_captions}	

				def should(freq):
					return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)  
				
				fetches = {
					"train": model.train, 
					"global_step": sv.global_step,
				}

				if should(a.progress_freq):
					fetches["gen_loss"] = model.gen_loss

				if should(a.summary_freq):
					fetches["summary"] = sv.summary_op

				if should(a.display_freq):
					fetches["display"] = display_fetches


				results = sess.run(fetches, feed_dict = feed_dict) 

				if should(a.summary_freq):
					print("recording summary")
					sv.summary_writer.add_summary(results["summary"], results["global_step"])

				if should(a.display_freq):
					print("saving display images")
					caption_file = open(os.path.join(a.output_dir, 'caption_list.txt'), 'wb')
					filesets, image_count = save_images(results["display"], image_count, a, caption_file, dataset.id2word)
					
					# append_index(filesets, a, step=True)

				if should(a.progress_freq):
					# global_step will have the correct step count if we resume from a checkpoint 
					train_epoch = math.ceil(results["global_step"] / steps_per_epoch)
					train_step = (results["global_step"] - 1) % steps_per_epoch + 1
					rate = (step + 1) * a.batch_size / (time.time() - start)
					remaining = (max_steps - step) * a.batch_size / rate
					print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
					print("gen_loss", results["gen_loss"])

				# if should(a.update_data_freq):
				# 	dataset = Dataset(num_train, num_test, 
				# 		a.img_size, num_abs = a.num_abs)

				if should(a.save_freq):
					print("saving model")
					saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

				if sv.should_stop():
					break


main()

