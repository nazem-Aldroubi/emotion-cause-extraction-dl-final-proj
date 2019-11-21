import numpy as np
import tensorflow as tf
import numpy as np
import re
import string

##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
# ##########DO NOT CHANGE#####################

####################### IO and CLEANING ################
def read_data(file_name):
    """
    DO NOT CHANGE

  Load text data from file

    :param file_name:  string, name of data file
    :return: list of sentences, each a list of words split on whitespace
  """
    text = []
    with open(file_name, 'rt', encoding='latin') as data_file:
        for line in data_file: text.append(line)
    return text

def clean_sentences(text):
    """
    Clean the text and return a list of sentences

    :param text
    :return: np array of sentences, a sentence is a list of words
    """
    sentences = []
    for i, line in enumerate(text):
        sentence = re.sub('<[^<]+>', "", line)

        # Remove punctuations from causes
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        # Remove digits from causes
        sentence = sentence.translate(str.maketrans('', '', string.digits))

        # Split sentence into a list of words
        sentence = sentence.split()
        sentences.append(sentence)

    return sentences        


def extract_and_clean_emotions(text):

    emotions = []
    for i, line in enumerate(text):
        # Extracts all strings within tags. First one is the emotion of the sentence
        cur_emotion = re.findall('<(.*?)>', line)[0]

        # Remove emotion tags
        text[i] = text[i][len(cur_emotion) + 2:-len(cur_emotion) - 4]

        emotions.append(cur_emotion)

    return emotions

def extract_and_clean_causes(text):

    causes = []
    for i, line in enumerate(text):
        # Extracts the cause from a sentence. Returns a list so index into first and only element
        cur_cause = re.findall('<cause>(.*?)<\\\cause>', line)[0]

        # Remove tags from line
        re.sub('<cause>', '', text[i])
        re.sub('<\\\cause>', '', text[i])

        # Remove punctuations from causes
        cur_cause = cur_cause.translate(str.maketrans('', '', string.punctuation))
        # Remove digits from causes
        cur_cause = cur_cause.translate(str.maketrans('', '', string.digits))

        # Split sentence into a list of words
        cur_cause = cur_cause.split()
        causes.append(cur_cause)

    return causes
    
####################### IO and CLEANING ################

###################### PADDING #########################
def pad_sentences_and_causes(sentences, causes):
    """
    Pad the sentences and causes to same lengths
    
    :param sentences: a np array of sentences
    :param causes: a np array of causes
    :return: np array of padded sentences, np array of padded causes
    """

    sentence_padding_size = get_padding_size(sentences)
    cause_padding_size = get_padding_size(causes)

    padded_sentences = pad_corpus(sentences, sentence_padding_size)
    padded_causes = pad_corpus(causes, cause_padding_size)

    return padded_sentences, padded_causes

def get_padding_size(sentences):
    """
    Determine the padding size given all the causes

    :param sentences: list of sentences, each a list of words
    :return: int, the length of longest sentence
    """
    max_length = 0
    for sentence in sentences:
        if len(sentence) > max_length:
            max_length = len(sentence)
    
    return max_length

def pad_corpus(sentences, padding_size):
    """
    Pad all the sentences

    :param sentenses: a np array of sentences
    :param padding_size: the padding size, equals to the length of the longest sentence
    :return: a np array of padded sentences
    """
    padded_sentences = []

    for line in sentences:
        padded_sentence = line[:padding_size]
        padded_sentence += [STOP_TOKEN] + [PAD_TOKEN] * (padding_size - len(padded_sentence)-1)
        padded_sentences.append(padded_sentence)

    padded_sentences = np.array(padded_sentences)
    return padded_sentences

###################### PADDING #########################

################### BUILD VOCAB ########################

def build_vocab(sentences):
    """
    Builds vocab from list of sentences

    :param sentences:  list of sentences, each a list of words
    :return: tuple of (dictionary: word --> unique index, pad_token_idx)
    """
    tokens = []
    for s in sentences: tokens.extend(s)
    all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

    vocab =  {word:i for i,word in enumerate(all_words)}

    return vocab,vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
    """
    Convert sentences to indexed 

    :param vocab:  dictionary, word --> unique index
    :param sentences:  list of lists of words, each representing padded sentence
    :return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
    """
    return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])

################### BUILD VOCAB ########################

################## MAIN INTERFACES #####################
def get_data(file_name):

    text = read_data(file_name)

    sentences = clean_sentences(text)

    emotions = extract_and_clean_emotions(text)

    causes = extract_and_clean_causes(text)

    #### By end of this line, text is cleaned from tags and emotion-cause pairs are extracted ####
    sentences = np.array(sentences)
    emotions = np.array(emotions)
    causes = np.array(causes)
    
    # Padding
    padded_sentences, padded_causes = pad_sentences_and_causes(sentences, causes)

    return padded_sentences, emotions, padded_causes
################## MAIN INTERFACES #####################
