# coding=utf-8
import nltk
import numpy as np


def text_to_code(tokens, dictionary, seq_len):
    code_str = ""
    eof_code = len(dictionary)
    for sentence in tokens:
        index = 0
        code_str += ('0 ')
        for word in sentence:
            if word in dictionary and index < seq_len:
                code_str += (str(dictionary[word]) + ' ')
                index += 1
        while index < seq_len:
            code_str += (str(eof_code) + ' ')
            index += 1
        code_str += (str(eof_code) + ' \n')
    return code_str

def text_to_array(tokens, dictionary, seq_len):
    codes = []
    eof_code = len(dictionary)
    for sentence in tokens:
        tmp = []
        tmp.append(0)
        for word in sentence:
            if len(tmp) < seq_len:
                if word in dictionary:
                    tmp.append(dictionary[word])
                else:
                    tmp.append(dictionary['#UNK#'])
        while len(tmp)  < seq_len:
            tmp.append(eof_code)
        codes.append(tmp)
    return np.array(codes, dtype=int)

def code_to_text(codes, dictionary, seq_len=None):
    paras = ""
    eof_code = len(dictionary)
    # sentence = codes.split('\n')
    for line in codes:
        # numbers = [int(s) for s in line.split() if s.isdigit()]
        numbers = map(int, line)
        for number in numbers:
            if number == 0:
                continue
            if number == eof_code or (not (str(number) in dictionary)):
                # continue
                # paras += '\n'
                break
            # paras += (dictionary[str(number)] + ' ')
            paras += (dictionary[str(number)] + ' ')
        paras += '\n'
    return paras


def get_tokenlized(file):
    tokenlized = list()
    with open(file, encoding='utf-8') as raw:
        for text in raw:
            text = nltk.word_tokenize(text.lower())
            tokenlized.append(text)
    return tokenlized


def get_word_list(tokens, word_count_threshold=5):
    word_set = list()
    word_counts = {}
    for sentence in tokens:
        for word in sentence:
            word_counts[word] = word_counts.get(word, 0) + 1
    vocab =  [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from %d to %d' % (len(word_counts), len(vocab)))
    return vocab

def get_dict(word_set):
    word_index_dict = dict()
    index_word_dict = dict()
    index = 1
    word_index_dict['#START#'] = 0
    index_word_dict['0'] = '#START#'
    for word in word_set:
        word_index_dict[word] = str(index)
        index_word_dict[str(index)] = word
        index += 1
    word_index_dict['#UNK#'] = str(index)
    index_word_dict[str(index)] = '#START#'
    return word_index_dict, index_word_dict

def text_precess(train_text_loc, word_count_threshold=5):
    train_tokens = get_tokenlized(train_text_loc)
    word_set = get_word_list(train_tokens, word_count_threshold)
    [word_index_dict, index_word_dict] = get_dict(word_set)

    sequence_len = len(max(train_tokens, key=len))

    return sequence_len, len(word_index_dict) + 1
