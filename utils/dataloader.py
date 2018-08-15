import numpy as np
import csv
import json
import os
from utils.text_process import *

class TextDataLoader():
    def __init__(self, batch_size, seq_length, end_token=0):
        self.batch_size = batch_size
        self.end_token = end_token
        self.seq_length = seq_length
        self.num_batch = 0
        self.image_batch = np.array([])
        self.sequence_batch = np.array([])

    def create_batches(self):
        # Load data
        
        iw_loc = './data/small/iw.npy'
        wi_loc = './data/small/wi.npy'
        data_loc = './data/small/image_coco.txt'
    
        if os.path.exists(wi_loc) == False or os.path.exists(iw_loc) == False:
            tokens = get_tokenlized(data_loc)
            word_set = get_word_list(tokens, 5)
            [wi_dict, iw_dict] = get_dict(word_set)
            np.save(wi_loc, wi_dict)
            np.save(iw_loc, iw_dict)
        else:
            tokens = get_tokenlized(data_loc)
            wi_dict = np.load(wi_loc).item()
            iw_dict = np.load(iw_loc).item()
        
        # Load text data
        self.sentences = text_to_array(tokens, wi_dict, self.seq_length)

        # make batches
        self.num_batch = len(self.sentences) // self.batch_size
        
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        # shuffle the data
        shuffle_indices = np.random.permutation(np.arange(int(self.num_batch * self.batch_size)))
        self.sentences = self.sentences[shuffle_indices]
        
        # make batches
        self.sequence_batch = np.split(self.sentences, self.num_batch, 0)
        self.pointer = 0

        return wi_dict, iw_dict     

    def next_train_batch(self):
        ret_seq = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return None, ret_seq, None

    def next_test_batch(self):
        return

    def reset_pointer(self):
        self.pointer = 0



class FlickrDataLoader():
    def __init__(self, batch_size, feature_dim, seq_length, end_token=0):
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.end_token = end_token
        self.seq_length = seq_length
        self.num_batch = 0
        self.image_batch = np.array([])
        self.sequence_batch = np.array([])

    def create_batches(self):
        # Load data
        img_file = './data/flickr30/fc7.npy'
        count_file = './data/flickr30/count.txt'
        meta_file = './data/flickr30/flickr30.csv'
        
        iw_loc = './data/flickr30/iw.npy'
        wi_loc = './data/flickr30/wi.npy'
        data_loc = './data/flickr30/flickr30.txt'
    
        if os.path.exists(wi_loc) == False or os.path.exists(iw_loc) == False:
            tokens = get_tokenlized(data_loc)
            word_set = get_word_list(tokens, 5)
            [wi_dict, iw_dict] = get_dict(word_set)
            np.save('./data/flickr30/wi.npy', wi_dict)
            np.save('./data/flickr30/iw.npy', iw_dict)
        else:
            wi_dict = np.load(wi_loc).item()
            iw_dict = np.load(iw_loc).item()
        
        # self.sequence_length = len(max(tokens, key=len))
        # self.sequence_length = 30
        # self.vocab_size = len(self.wi_dict) + 1
        
        # if os.path.exists(self.oracle_file) == False:
        #     with open(self.oracle_file, 'w') as outfile:
        #         outfile.write(text_to_code(tokens, self.wi_dict, self.sequence_length))

        self.meta = []
        with open(meta_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for _, line in enumerate(reader):
                self.meta.append(line)
        
        # load image data
        image_data = np.load(img_file)
        count_file = open(count_file, 'r')
        count_lines = count_file.readlines()
        if self.feature_dim != np.shape(image_data)[2]:
            print ('Image feature does not match\n')
            return
        shape = np.shape(image_data)
        image_data = np.reshape(image_data, [shape[0]*shape[1], shape[2]])
        count_lines = [int(x) for x in count_lines]
        max_len = min(len(count_lines), shape[0]*shape[1])
        # ad-hoc
        train_num = 29000
        if train_num > max_len:
            import pdb; pdb.set_trace()
        self.test_image_data = image_data[train_num:max_len]
        self.test_meta = self.meta[train_num:max_len]

        count_lines = count_lines[:train_num]
        image_data = image_data[:train_num]
        image_data = np.repeat(image_data, count_lines, axis=0)
        self.meta = self.meta[:train_num]
        self.meta = [self.meta[x] for x in range(len(self.meta)) for _ in range(count_lines[x])]

        # Load text data
        # import pdb; pdb.set_trace()
        self.sentences = text_to_array(tokens, wi_dict, self.seq_length)
        self.sentences = self.sentences[:len(image_data)]

        # self.meta = np.array(self.meta)
        # self.meta = self.meta[:len(image_data)]
        # if len(image_data) != len(self.sentences):
        #     print ('Image and sentences does not algined\n')
        #     return
        # make batches
        num_data = min(len(image_data), len(self.sentences))
        self.num_batch = int(num_data) // self.batch_size
        self.num_train_batch = self.num_batch
        image_data = image_data[: self.num_batch * self.batch_size]
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.meta = self.meta[:self.num_batch * self.batch_size]
        # shuffle the data
        shuffle_indices = np.random.permutation(np.arange(int(self.num_batch * self.batch_size)))
        image_data = image_data[shuffle_indices]
        self.sentences = self.sentences[shuffle_indices]
        self.meta = [self.meta[i]["image_path"] for i in shuffle_indices]
        
        # make batches
        self.image_batch = np.split(image_data, self.num_batch, 0)
        self.sequence_batch = np.split(self.sentences, self.num_batch, 0)
        self.meta_batch = [self.meta[i * self.batch_size:(i + 1)*self.batch_size] for i in range(self.num_batch)]
        self.pointer = 0
        
        self.test_batch_num = (max_len - train_num) // self.batch_size
        
        self.test_image_data = self.test_image_data[:self.test_batch_num*self.batch_size]
        self.test_meta = self.test_meta[:self.test_batch_num*self.batch_size]
        self.test_meta_image = [meta["image_path"] for meta in self.test_meta]
        self.test_meta_image_batch = [self.test_meta_image[i * self.batch_size:(i + 1)*self.batch_size]
                                        for i in range(self.test_batch_num)]
        self.test_image_batch = np.split(self.test_image_data, self.test_batch_num, 0)
        self.test_meta_text = [meta["text_path"] for meta in self.test_meta]
        self.test_meta_text_batch = [self.test_meta_text[i * self.batch_size:(i + 1)*self.batch_size]
                                        for i in range(self.test_batch_num)]
        self.test_pointer = 0

        return wi_dict, iw_dict     

    def next_train_batch(self):
        ret_img, ret_seq, ret_meta = self.image_batch[self.pointer], self.sequence_batch[self.pointer], self.meta_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret_img, ret_seq, ret_meta

    def next_test_batch(self):
        ret_img, ret_meta_img, ret_meta_img_seq = self.test_image_batch[self.test_pointer], self.test_meta_image_batch[self.test_pointer], self.test_meta_text_batch[self.test_pointer]
        self.test_pointer = (self.test_pointer + 1) % self.test_batch_num
        return ret_img, ret_meta_img_seq, ret_meta_img

    def reset_pointer(self):
        self.pointer = 0

    def disc_batch(self, negative_file):
        positive_examples = self.sequence_batch[self.pointer]
        negative_examples = []
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.seq_length:
                    negative_examples.append(parse_line)
        negative_examples = np.array(negative_examples[:self.batch_size])
        sentences = np.concatenate([positive_examples, negative_examples], 0)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        labels = np.concatenate([positive_labels, negative_labels], 0)

        images = np.concatenate([self.image_batch[self.pointer], self.image_batch[self.pointer]], 0)
        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        sentences = sentences[shuffle_indices]
        labels = labels[shuffle_indices]
        images = images[shuffle_indices]

        # Split batches
        return sentences[:self.batch_size], labels[:self.batch_size], images[:self.batch_size]

class MSCOCODataLoader():
    def __init__(self, batch_size, feature_dim, seq_length, end_token=0):
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.end_token = end_token
        self.seq_length = seq_length

    def create_batches(self, reload=False):
        self.train_meta = []
        self.val_meta = []
        self.test_meta = []
        self.train_img = []
        self.val_img = []
        self.test_img = []
        self.train_seq = []
        self.test_seq = []
        self.val_seq = []
        self.tokens = []

        train_img_loc = './data/mscoco/train_img.npy'
        val_img_loc = './data/mscoco/val_img.npy'
        test_img_loc = './data/mscoco/test_img.npy'
        
        train_seq_loc = './data/mscoco/train_seq.npy'
        val_seq_loc = './data/mscoco/val_seq.npy'
        test_seq_loc = './data/mscoco/test_seq.npy'

        train_meta_loc = './data/mscoco/train_meta.npy'
        val_meta_loc = './data/mscoco/val_meta.npy'
        test_meta_loc = './data/mscoco/test_meta.npy'

        wi_loc = './data/mscoco/wi.npy'
        iw_loc = './data/mscoco/iw.npy'

        if reload == False:
            self.train_img = np.load(train_img_loc)
            self.val_img = np.load(val_img_loc)
            self.test_img = np.load(test_img_loc)

            self.train_seq = np.load(train_seq_loc)
            self.val_seq = np.load(val_seq_loc)
            self.test_seq = np.load(test_seq_loc)

            self.train_meta = np.load(train_meta_loc)
            self.val_meta = np.load(val_meta_loc)
            self.test_meta = np.load(test_meta_loc)

            wi_dict = np.load(wi_loc).item()
            iw_dict = np.load(iw_loc).item()
        else:
            # Load data
            with open('./data/mscoco/dataset_coco.json') as jsonfile:
                data = json.loads(jsonfile.read())

            print("json file loaded")
            for line in data['images']:
                if line['split'] == 'train':
                    for sentence in line['sentences']:
                        self.tokens.append(sentence['tokens'])
                        self.train_meta.append('./data/mscoco/' + line['filepath'] + '/' + line['filename'])
                        self.train_seq.append(sentence['tokens'])
                        self.train_img.append(np.load('./data/mscoco/features/' + str(line['cocoid']) + '.npy'))
                if line['split'] == 'val':
                    sent_list = []
                    for sentence in line['sentences']:
                        sent_list.append(sentence['tokens'])
                        self.tokens.append(sentence['tokens'])
                    self.val_seq.append(sent_list)
                    self.val_meta.append('./data/mscoco/' + line['filepath'] + '/' + line['filename'])
                    self.val_img.append(np.load('./data/mscoco/features/' + str(line['cocoid']) + '.npy'))
                if line['split'] == 'test':
                    sent_list = []
                    for sentence in line['sentences']:
                        sent_list.append(sentence['tokens'])
                        self.tokens.append(sentence['tokens'])
                    self.test_seq.append(sent_list)
                    self.test_meta.append('./data/mscoco/' + line['filepath'] + '/' + line['filename'])
                    self.test_img.append(np.load('./data/mscoco/features/' + str(line['cocoid']) + '.npy'))
            
            print("data loaded!")

            word_set = get_word_list(self.tokens, 5)
            [wi_dict, iw_dict] = get_dict(word_set)
            np.save(wi_loc, wi_dict)
            np.save(iw_loc, iw_dict)

            self.train_img = np.array(self.train_img)
            self.val_img = np.array(self.val_img)
            self.test_img = np.array(self.test_img)

            import pdb; pdb.set_trace()
            self.train_seq = text_to_array(self.train_seq, wi_dict, self.seq_length)
            self.val_seq = np.array(self.val_seq)
            self.test_seq = np.array(self.test_seq)

            self.train_meta = np.array(self.train_meta)
            self.val_meta = np.array(self.val_meta)
            self.test_meta = np.array(self.test_meta)
            
            np.save(train_seq_loc, self.train_seq)
            np.save(test_seq_loc, self.test_seq)
            np.save(val_seq_loc, self.val_seq)

            np.save(train_img_loc, self.train_img)
            np.save(test_img_loc, self.test_img)
            np.save(val_img_loc, self.val_img)

            np.save(train_meta_loc, self.train_meta)
            np.save(test_meta_loc, self.test_meta)
            np.save(val_meta_loc, self.val_meta)
        
        # import pdb; pdb.set_trace()
        print("data loaded!")
        self.num_train_batch = len(self.train_meta) // self.batch_size
        self.train_img = self.train_img[: self.num_train_batch * self.batch_size]
        self.train_seq = self.train_seq[:self.num_train_batch * self.batch_size]
        self.train_meta = self.train_meta[:self.num_train_batch * self.batch_size]

        self.num_test_batch = len(self.test_meta) // self.batch_size
        self.test_img = self.test_img[: self.num_test_batch * self.batch_size]
        self.test_seq = self.test_seq[:self.num_test_batch * self.batch_size]
        self.test_meta = self.test_meta[:self.num_test_batch * self.batch_size]   
        
        self.num_val_batch = len(self.val_meta) // self.batch_size
        self.val_img = self.val_img[: self.num_val_batch * self.batch_size]
        self.val_seq = self.val_seq[:self.num_val_batch * self.batch_size]
        self.val_meta = self.val_meta[:self.num_val_batch * self.batch_size]   

        # shuffle the index
        shuffle_indices = np.random.permutation(np.arange(int(self.num_train_batch * self.batch_size)))
        self.train_img = self.train_img[shuffle_indices]
        self.train_seq = self.train_seq[shuffle_indices]
        self.train_meta = self.train_meta[shuffle_indices]

        # make batches
        self.train_img_batch = np.split(self.train_img, self.num_train_batch, 0)
        self.train_seq_batch = np.split(self.train_seq, self.num_train_batch, 0)
        self.train_meta_batch = np.split(self.train_meta, self.num_train_batch, 0)

        # make batches
        self.val_img_batch = np.split(self.val_img, self.num_val_batch, 0)
        self.val_seq_batch = np.split(self.val_seq, self.num_val_batch, 0)
        self.val_meta_batch = np.split(self.val_meta, self.num_val_batch, 0)

        # make batches
        self.test_img_batch = np.split(self.test_img, self.num_test_batch, 0)
        self.test_seq_batch = np.split(self.test_seq, self.num_test_batch, 0)
        self.test_meta_batch = np.split(self.test_meta, self.num_test_batch, 0)

        self.train_pointer = 0
        self.test_pointer = 0
        self.val_pointer = 0

        return wi_dict, iw_dict


    def next_train_batch(self):
        ret_img, ret_seq, ret_meta = self.train_img_batch[self.train_pointer],\
            self.train_seq_batch[self.train_pointer], \
            self.train_meta_batch[self.train_pointer]
        self.train_pointer = (self.train_pointer + 1) % self.num_train_batch
        return ret_img, ret_seq, ret_meta

    def next_test_batch(self):
        ret_img, ret_seq, ret_meta = self.test_img_batch[self.test_pointer],\
            self.test_seq_batch[self.test_pointer], \
            self.test_meta_batch[self.test_pointer]
        self.test_pointer = (self.test_pointer + 1) % self.num_test_batch
        return ret_img, ret_seq, ret_meta

    def next_val_batch(self):
        ret_img, ret_seq, ret_meta = self.val_img_batch[self.val_pointer],\
            self.val_seq_batch[self.val_pointer], \
            self.val_meta_batch[self.val_pointer]
        self.val_pointer = (self.val_pointer + 1) % self.num_val_batch
        return ret_img, ret_seq, ret_meta

    def reset_pointer(self):
        self.train_pointer = 0
        self.test_pointer = 0
        self.val_pointer = 0

    def disc_batch(self, negative_file):
        return
        