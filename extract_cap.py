import json
import os
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse 
import h5py
import time
import json
import os
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse  
import time
import nltk
from nltk.tokenize import word_tokenize
import itertools
import gensim
import cPickle as pkl
def extract_captions(img_dir):  
    image_files = [f for f in os.listdir(img_dir) if 'jpg' in f]  
    image_captions = { img_file : [] for img_file in image_files }    

    caption_dir = 'flower/text_c10' 
    class_dirs = []
    for i in range(1, 103):
        class_dir_name = 'class_%.5d'%(i)
        class_dirs.append( join(caption_dir, class_dir_name))

    for class_dir in class_dirs:
        caption_files = [f for f in os.listdir(class_dir) if 'txt' in f]
        for cap_file in caption_files:
            with open(join(class_dir,cap_file)) as f:
                captions = f.read().split('\n')
            img_file = cap_file[0:11] + ".jpg"
            # 5 captions per image
            if img_file in image_captions:
                image_captions[img_file] += [cap for cap in captions if len(cap) > 0]#[0:5] 
    
    return image_captions


def tokenize_caption(image_captions):
    """
    image_captions: a dictionary with keys being the file names and values being the strings of their captions. 
    """
    image_tokens = {}
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for i, img in enumerate(image_captions):
        all_tokens = []
        for t in image_captions[img]: 
            # replace - by space:
            t.replace('-',' ')
            
            tokens = []
            sents = sent_detector.tokenize(t) 
            for sent in sents:
                tokens += word_tokenize(sent)
            all_tokens.append(tokens)
        image_tokens[img] = all_tokens
    return image_tokens



def extract_embedding(image_tokens, image_tokens_test, model): 
    all_image_tokens = dict(image_tokens.items() + image_tokens_test.items())
    lists_of_captions = all_image_tokens.values() 
    list_of_captions = list(itertools.chain.from_iterable(lists_of_captions)) 
    list_caption_words = list(itertools.chain.from_iterable(list_of_captions)) 
    dict_words = list(set(list_caption_words)) 
    dict_words = ['']+[word for word in dict_words if word in model]

    word2id = {word: i for i, word in enumerate(dict_words)}
    id2word = {i: word for i, word in enumerate(dict_words)}

    vocab_size = len(dict_words)
    emb_dim = 300
    embedding_matrix = np.zeros([vocab_size, emb_dim])
    # wrong_words = []
    for i in range(vocab_size): 
        if i != 0:
            embedding_matrix[i] = model[id2word[i]]
        else:
            embedding_matrix[i] = np.zeros(emb_dim)

        
    return embedding_matrix, word2id, id2word

def extract_token_id(image_tokens, model, word2id):
    image_tokens_ids = {}
    for img in image_tokens:
        id_lists = []
        for token_list in image_tokens[img]:
            id_list = [word2id[token] if token in model else word2id[''] for token in token_list]
            id_list = np.array(id_list, dtype = int)
            id_lists.append(id_list)

        image_tokens_ids[img] = id_lists
        
    return image_tokens_ids

def extract_all(model):
    print 'Extracting captions from text files...'
    image_captions_train = extract_captions('flower/train_data')
    image_captions_test = extract_captions('flower/test_data')
    
    print 'Tokenizing captions...'
    image_tokens_train = tokenize_caption(image_captions_train)
    image_tokens_test = tokenize_caption(image_captions_test)
    
#     print 'Extracting GoogleNews Model...'
#     model = gensim.models.KeyedVectors.load_word2vec_format('flower/GoogleNews-vectors-negative300.bin', 
#                                                             binary = True)
    
    print 'Extracting embedding matrix...'
    embedding_matrix, word2id, id2word = extract_embedding(image_tokens_train, image_tokens_test, model)
    
    print 'Extracting token ids...'
    image_tokens_ids_train = extract_token_id(image_tokens_train, model, word2id)
    image_tokens_ids_test = extract_token_id(image_tokens_test, model, word2id)
    
    embedding = {'embedding matrix': embedding_matrix,
                'word2id': word2id,
                'id2word': id2word,
                "image_tokens_ids_train": image_tokens_ids_train,
                "image_tokens_ids_test": image_tokens_ids_test}

    print 'Saving everything to a pickle file...'
    with open('flower/embedding.p', 'wb') as f:
        pkl.dump(embedding, f)     


def main():
    print 'Openning Google News Word2Vec...'
    model = gensim.models.KeyedVectors.load_word2vec_format('flower/GoogleNews-vectors-negative300.bin', 
                                                                binary = True)

    extract_all(model)

if __name__ == '__main__':
    main()