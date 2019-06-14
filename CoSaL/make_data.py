from __future__ import print_function

import numpy as np 
import scipy

import cv2
import os 
import random
import cPickle as pkl 

colors = [
    (0,0,255),##r
    (0,255,0),##g
    (255,0,0),##b
    (0,0,0),##o
    (128,128,128),##k
    (255,255,0)##y
]
num_colors = len(colors)  

def rectangle(input_img, real_img, small_img_size):
    original = np.copy(real_img) 
    pt1 = (int(np.random.uniform(0.1,0.3)*small_img_size), int(np.random.uniform(0.1,0.3)*small_img_size))
    pt2 = (int(np.random.uniform(0.7,.9)*small_img_size), int(np.random.uniform(0.7,0.9)*small_img_size))
    choice_color = np.random.randint(num_colors)
    cv2.rectangle(input_img,pt1,pt2, [0,0,0],thickness=-1)
    cv2.rectangle(real_img,pt1,pt2,colors[choice_color],thickness=-1)
    labels = np.zeros((small_img_size,small_img_size),np.int64)
    
    labels = np.not_equal(np.sum(real_img, -1),np.sum(original, -1)) * (choice_color+1)
    return input_img, real_img, labels, choice_color+1
    
    
def half_ellipse(input_img, real_img, small_img_size):
    original = np.copy(real_img)
    center = (int(np.random.uniform(0.45,0.55)*small_img_size), int(np.random.uniform(0.45,0.55)*small_img_size))
    axes = (small_img_size/3,small_img_size/6)
    choice_color = np.random.randint(num_colors)
    cv2.ellipse(input_img,center,axes,0,0,180,[0,0,0],thickness= -1)
    cv2.ellipse(real_img,center,axes,0,0,180,colors[choice_color],thickness=-1)
    labels = np.zeros((small_img_size,small_img_size),np.int64)
    
    labels = np.not_equal(np.sum(real_img, -1),np.sum(original, -1)) * (choice_color+1)
    return input_img, real_img, labels, choice_color+1
    
        
def ellipse(input_img, real_img, small_img_size): 
    original = np.copy(real_img)
    center = (int(np.random.uniform(0.45,0.55)*small_img_size), int(np.random.uniform(0.45,0.55)*small_img_size))
    axes = (small_img_size/3,small_img_size/6)
    choice_color = np.random.randint(num_colors)
    cv2.ellipse(input_img,center,axes,0,0,360, [0,0,0],thickness=-1)
    cv2.ellipse(real_img,center,axes,0,0,360, colors[choice_color],thickness=-1)
    labels = np.zeros((small_img_size,small_img_size),np.int64)
    
    labels = np.not_equal(np.sum(real_img, -1),np.sum(original, -1)) * (choice_color+1)
    return input_img, real_img, labels, choice_color+1
    
        
def triangle(input_img, real_img, small_img_size): 
    original = np.copy(real_img)
    pt1 = [int(np.random.uniform(0.2,0.4)*small_img_size), int(np.random.uniform(0.7,0.8)*small_img_size)]
    pt2 = [int(np.random.uniform(0.7,0.8)*small_img_size), int(np.random.uniform(0.7,0.8)*small_img_size)]
    pt3 = [int(np.random.uniform(0.4,0.6)*small_img_size), int(np.random.uniform(0.2,0.3)*small_img_size)]
    choice_color = np.random.randint(num_colors)
    cv2.fillConvexPoly(real_img, np.array([pt1,pt2,pt3]), color = colors[choice_color])
    cv2.fillConvexPoly(input_img, np.array([pt1,pt2,pt3]), color = [0,0,0])
    # cv2.polylines(input_img, [np.array([pt1,pt2,pt3])], color = [0,0,0],isClosed=True)
    labels = np.zeros((small_img_size,small_img_size),np.int64)
    
    labels = np.not_equal(np.sum(real_img, -1),np.sum(original, -1)) * (choice_color+1)
    return input_img, real_img, labels, choice_color+1
    
    
def reverse_triangle(input_img, real_img, small_img_size): 
    original = np.copy(real_img)
    pt1 = [int(np.random.uniform(0.2,0.4)*small_img_size), int(np.random.uniform(0.2,0.3)*small_img_size)]
    pt2 = [int(np.random.uniform(0.7,0.8)*small_img_size), int(np.random.uniform(0.2,0.3)*small_img_size)]
    pt3 = [int(np.random.uniform(0.4,0.6)*small_img_size), int(np.random.uniform(0.7,0.8)*small_img_size)]
    choice_color = np.random.randint(num_colors) 
    cv2.fillConvexPoly(real_img, np.array([pt1,pt2,pt3]), color = colors[choice_color])
    cv2.fillConvexPoly(input_img, np.array([pt1,pt2,pt3]), color = [0,0,0])
    # cv2.polylines(input_img, [np.array([pt1,pt2,pt3])], color = [0,0,0],isClosed=True)  
    labels = np.zeros((small_img_size,small_img_size),np.int64)
    
    labels = np.not_equal(np.sum(real_img, -1),np.sum(original, -1)) * (choice_color+1)
    return input_img, real_img, labels, choice_color+1
    
            
def thin_ellipse(input_img, real_img, small_img_size):
    original = np.copy(real_img)
    center = (int(np.random.uniform(0.45,0.55)*small_img_size), int(np.random.uniform(0.45,0.55)*small_img_size))
    axes = (small_img_size/6,small_img_size/3)
    choice_color = np.random.randint(num_colors)
    cv2.ellipse(input_img,center,axes,0,0,360, [0,0,0],thickness=-1) 
    cv2.ellipse(real_img,center,axes,0,0,360,colors[choice_color],thickness=-1) 
    labels = np.zeros((small_img_size,small_img_size),np.int64)
    
    labels = np.not_equal(np.sum(real_img, -1),np.sum(original, -1)) * (choice_color+1)
    return input_img, real_img, labels, choice_color+1
    
        
def thin_half_ellipse(input_img, real_img, small_img_size): 
    original = np.copy(real_img)
    center = (int(np.random.uniform(0.45,0.55)*small_img_size), int(np.random.uniform(0.45,0.55)*small_img_size))
    axes =  (small_img_size/6,small_img_size/3)
    choice_color = np.random.randint(num_colors)
    cv2.ellipse(input_img,center,axes,0,0,180,[0,0,0],thickness=-1)
    cv2.ellipse(real_img,center,axes,0,0,180,colors[choice_color],thickness=-1)
    labels = np.zeros((small_img_size,small_img_size),np.int64)
    
    labels = np.not_equal(np.sum(real_img, -1),np.sum(original, -1)) * (choice_color+1)
    return input_img, real_img, labels, choice_color+1
    
        
def diamond(input_img, real_img, small_img_size):
    original = np.copy(real_img)
    pt1 = [int(np.random.uniform(0.2,0.3)*small_img_size), int(np.random.uniform(0.45,0.55)*small_img_size)]
    pt2 = [int(np.random.uniform(0.7,0.8)*small_img_size), int(np.random.uniform(0.45,0.55)*small_img_size)]
    pt3 = [int(np.random.uniform(0.45,0.55)*small_img_size),int(np.random.uniform(0.2,0.3)*small_img_size)]
    pt4 = [int(np.random.uniform(0.45,0.55)*small_img_size),int(np.random.uniform(0.7,0.8)*small_img_size)]
    choice_color = np.random.randint(num_colors)
    # cv2.polylines(input_img, [np.array([pt1,pt3,pt2,pt4])], color = [0,0,0],isClosed=True)
    cv2.fillConvexPoly(real_img, np.array([pt1,pt3,pt2,pt4]), color = colors[choice_color]) 
    cv2.fillConvexPoly(input_img, np.array([pt1,pt3,pt2,pt4]), color = [0,0,0]) 
    labels = np.zeros((small_img_size,small_img_size),np.int64)
    
    labels = np.not_equal(np.sum(real_img, -1),np.sum(original, -1)) * (choice_color+1)
    return input_img, real_img, labels, choice_color+1
    
        
def circle(input_img, real_img, small_img_size):
    original = np.copy(real_img)
    center = (int(np.random.uniform(0.45,0.55)*small_img_size), int(np.random.uniform(0.45,0.55)*small_img_size))
    radius = int(np.random.uniform(0.3,0.4)*small_img_size)
    choice_color = np.random.randint(num_colors)
    cv2.circle(input_img, center, radius, [0,0,0],thickness=-1) 
    cv2.circle(real_img, center, radius, colors[choice_color],thickness=-1) 
    labels = np.zeros((small_img_size,small_img_size),np.int64)
    
    labels = np.not_equal(np.sum(real_img, -1),np.sum(original, -1)) * (choice_color+1)
    return input_img, real_img, labels, choice_color+1
    
        
        
def reshape(image, small_img_size,label = True): 
    if not label:
        image = np.reshape(image, [3,3,small_img_size,small_img_size,3])
        image = np.transpose(image, [0,2,1,3,4])
        image = np.reshape(image, [3*small_img_size,3*small_img_size,3])
        return image
    else:
        image = np.reshape(image, [3,3,small_img_size,small_img_size])
        image = np.transpose(image, [0,2,1,3])
        image = np.reshape(image, [3*small_img_size,3*small_img_size])
        return image        
    
image_generators = np.array([rectangle, half_ellipse, ellipse, triangle, reverse_triangle, thin_ellipse, thin_half_ellipse, diamond,circle])  
generator_names = np.array(["rectangle", "half fat ellipse", 'fat ellipse', 'triangle', 
                    'reverse triangle', 'thin ellipse', 'half thin ellipse',
                   'diamond','circle'])
color_names = np.array(['blue','light green','red','black','grey','yellow',])

all_words = list(generator_names) + list(color_names) + ['is','above','below','left','right']

id2word = {i: all_words[i] for i in range(len(all_words))}
word2id = {all_words[i]: i for i in range(len(all_words))}

def generate_single_image(image_generators, img_size):
    small_img_size = img_size / 3
    

    input_image = np.zeros((9, small_img_size,small_img_size,3),np.uint8) + 255
    real_image = np.copy(input_image)
    labels = np.zeros((9, small_img_size,small_img_size),np.int64)
    choice_colors = np.zeros(9,np.int64)
    
    permute_shape_ids = np.random.permutation(9) 
    image_generators = image_generators[permute_shape_ids]
    for i in range(9):
        input_image[i],real_image[i], labels[i], choice_colors[i] = image_generators[i](input_image[i],real_image[i],small_img_size)
        
        
    
    
    
    input_image = reshape(input_image,small_img_size,False) 
    real_image = reshape(real_image,small_img_size,False)
    labels = reshape(labels,small_img_size,True)
        
    return input_image, real_image, labels, choice_colors, permute_shape_ids
        
def generate_pos_info(colors, shapes): 
    sentences = [[shapes[i],'is',color] for i, color in enumerate(colors)] 
    return zip(sentences, [(i,i) for i in range(9)]) 

def generate_rel_pos_info(colors, shapes):  
    colors = np.reshape(colors,[3,3])
    shapes = np.reshape(shapes, [3,3])
    sentences = []
    rel_locs = []
    for i in range(3):
        for j in range(3):
            if i != 0:
                sentences.append([colors[i,j],'below',shapes[i-1,j]])
                rel_locs.append([3*i+j,3*(i-1)+j])
            if j != 0:
                sentences.append([colors[i,j],'right',shapes[i,j-1]])
                rel_locs.append([3*i+j,3*(i)+j-1])
            if i != 3 - 1:
                sentences.append([colors[i,j],'above',shapes[i+1,j]])
                rel_locs.append([3*i+j,3*(i+1)+j])
            if j != 3 - 1:
                sentences.append([colors[i,j],'left',shapes[i,j+1]])
                rel_locs.append([3*i+j,3*(i)+j+1])

    return zip(sentences, rel_locs) 
def serialize_caption(sents_and_locs):  
    existing_locs = {}
    ordered_sents_locs = []
    while len(ordered_sents_locs) < 9:
        if existing_locs == {}:
            ordered_sents_locs = [(sent,loc) for sent,loc in sents_and_locs if 'is' in sent]  
            ordered_sents, locs = zip(*ordered_sents_locs) 
            existing_locs = set(locs[0])
            sents_and_locs = [sent_and_loc for sent_and_loc in sents_and_locs if sent_and_loc[1][0] not in existing_locs]
        else:  
            new_sents_and_locs = [(sent,loc) for sent,loc in sents_and_locs if loc[1] in existing_locs and loc[0] not in existing_locs]
            i = np.random.randint(len(new_sents_and_locs))
            new_sent_and_loc = new_sents_and_locs[i]
            new_loc = new_sent_and_loc[1][0]
            ordered_sents_locs.append(new_sent_and_loc)
            sents_and_locs = [sent_and_loc for sent_and_loc in sents_and_locs if sent_and_loc[1][0] != new_sent_and_loc]
            existing_locs.update({new_loc}) 

    return list(zip(*ordered_sents_locs)[0]) 

def generate_single_datum(img_size, num_abs = 1):
	input_image, real_image, label, choice_colors, permute_shape_ids = generate_single_image(image_generators, img_size)

	colors = color_names[choice_colors-1] 
	shapes = generator_names[permute_shape_ids] 

	abs_sents_locs = generate_pos_info(colors, shapes)
	rel_sents_locs = generate_rel_pos_info(colors, shapes)

	abs_ids = np.random.choice(range(9), size = num_abs,replace = False)
	sentences = serialize_caption([abs_sents_locs[abs_id] for abs_id in abs_ids]+rel_sents_locs)

	token_ids = np.array([[word2id[word] for word in sent] for sent in sentences])

	return input_image, real_image, label, token_ids

def generate_data(num_train, num_test, img_size, num_abs = 1):	
	for stage in ['train','test']:
		num = num_train if stage == 'train' else num_test

		print('building {} datasets...'.format(stage))
		input_images = np.zeros([num,img_size,img_size,3],np.uint8)
		real_images = np.zeros([num,img_size,img_size,3],np.uint8)
		labels = np.zeros([num,img_size,img_size],np.int64)
		captions = np.zeros([num, 9, 3],np.int64)
		for i in range(num):
			input_image, real_image, label, token_ids = generate_single_datum(img_size, num_abs = num_abs)
			input_images[i] = input_image
			real_images[i] = real_image
			labels[i] = label 
			captions[i] = token_ids

		if stage == 'train':
			train_data = [input_images, real_images, labels, captions]
		else:
			test_data = [input_images, real_images, labels, captions]

	return train_data, test_data
	# print('saving datasets...')
	# if not os.path.isdir('data/'):
	# 	os.mkdir('data')
	# filename = os.path.join('data','sort-of-clevr.pickle')
	# with  open(filename, 'wb') as f:
	# 	pkl.dump({"train_data":train_data, 
	# 		'test_data':test_data,
	# 		'id2word': id2word,
	# 		'word2id': word2id}, 
	# 		f)
	# print('datasets saved at {}'.format(filename))



	# Generate 9 different colors. 
	# Order them randomly. 

class SingleDataset:
	def __init__(self, input_image, real_images, labels, token_ids):  
		self.inputs = input_image 
		self.labels = labels
		self.captions = token_ids
		self.reals = real_images
		self.num_samples = len(input_image)        
		self.batch_id = 0
		
	def next_batch(self, batch_size):
		""" Return a batch of data. When dataset end is reached, start over.
		""" 

		if self.batch_id + batch_size > self.num_samples: 

			self.batch_id = 0 
			permutation = np.random.permutation(self.num_samples)
			self.inputs = self.inputs[permutation]
			self.reals = self.reals[permutation]
			self.labels = self.labels[permutation]
			self.captions = self.captions[permutation]
			

		batch_inputs = self.inputs[self.batch_id:min(self.batch_id + batch_size, self.num_samples)]  

		batch_reals = self.reals[self.batch_id:min(self.batch_id + batch_size, self.num_samples)]

		batch_labels = self.labels[self.batch_id:min(self.batch_id + batch_size, self.num_samples)]

		batch_captions = self.captions[self.batch_id:min(self.batch_id + batch_size, self.num_samples)] 

		self.batch_id = min(self.batch_id + batch_size, self.num_samples)

		return batch_inputs, batch_reals, batch_labels, batch_captions 

class Dataset:
	def __init__(self, num_train, num_test, img_size, num_abs = 1):
		train_data, test_data = generate_data(num_train, num_test, img_size, num_abs = num_abs)
		input_images, real_images, labels, token_ids = train_data 
		self.train = SingleDataset(input_images, real_images, labels, token_ids)

		input_images, real_images, labels, token_ids = test_data 
		self.test = SingleDataset(input_images, real_images, labels, token_ids)

		self.id2word = id2word
		self.word2id = word2id

















