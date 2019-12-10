import numpy as np
import operator
from collections import defaultdict
import re
from random import shuffle
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
    Load text data from file

    :param file_name:  string, name of data file
    :return: list of sentences, each a list of words split on whitespace
    """
    text = []
    with open(file_name, 'rt', encoding='latin') as data_file:
        for line in data_file: text.append(line)
    return text

def extract_clauses_and_labels(text, emotion_seeds):
    document_clauses = []
    emotion_labels = []
    cause_labels = []
    
    emotion_cause_pairs = []

    for i, line in enumerate(text):
        clauses = re.split("[.,!;:\"]+", line)
        emotion_clauses = []
        cause_clauses = []

        for clause in clauses:
            cleaned_clause = clean_clause(clause)
            clause_words = cleaned_clause.split()
            
            if len(clause_words) == 0:
                continue
            
            document_clauses.append(clause_words)

            if "<cause>" in clause:
                cause_labels.append(1)
                cause_clauses.append(clause_words)
            else:
                cause_labels.append(0)
            
            has_seed = False
            for word in clause_words:
                if word.lower() in emotion_seeds:
                    emotion_labels.append(1)
                    emotion_clauses.append(clause_words) 
                    has_seed = True
                    break
            if not has_seed:
                emotion_labels.append(0)

        for e_clause in emotion_clauses:
            for c_clause in cause_clauses:
                emotion_cause_pairs.append((e_clause, c_clause))

    return np.array(document_clauses), np.array(emotion_labels), np.array(cause_labels), emotion_cause_pairs

def clean_clause(clause):
    clause = re.sub('<[^<]+>', "", clause)

    # Remove punctuation.
    clause = clause.translate(str.maketrans('', '', string.punctuation))
    # Remove digits.
    clause = clause.translate(str.maketrans('', '', string.digits))

    return clause
    

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
def pad_clauses(clauses):
    """
    Pad the clauses to the same lengths.
    
    :param sentences: a np array of sentences
    :param causes: a np array of causes
    :return: np array of padded sentences, np array of padded causes
    """

    clause_padding_size = get_padding_size(clauses) + 1
    padded_clauses = pad_corpus(clauses, clause_padding_size)
    return padded_clauses

def get_padding_size(clauses):
    """
    Determine the padding size given all the clauses.

    :param clauses: List of clauses, each a list of words
    :return: int, the length of the longest clause
    """
    max_length = 0
    for clause in clauses:
        if len(clause) > max_length:
            max_length = len(clause)

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

def calculate_vocab_frequency(sentences):
    vocab_frequency = defaultdict(int)
    for sentence in sentences:
        for word in sentence:
            vocab_frequency[word] += 1

    return sorted(vocab_frequency.items(), key=operator.itemgetter(1))

def is_complete(emotion_seeds, sentences):
    emotion_seeds_complete = True
    for sentence in sentences:
        has_seed = False
        for word in sentence:
            if word.lower() in emotion_seeds:
                has_seed = True
                break
        if not has_seed:
            print("Sentence does not contain an emotion seed word: ", sentence)
            emotion_seeds_complete = False
    return emotion_seeds_complete

def convert_clauses_to_id(vocab, clauses):
    """
    Convert clauses to indices

    :param vocab:  dictionary, word --> unique index
    :param clauses:  list of lists of words, each representing a padded clause
    :return: numpy array of integers, with each row representing the word indices in the corresponding clauses
    """
    return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in clause] for clause in clauses])

def pad_and_convert_pairs_to_id(vocab, emotion_cause_pairs, padding_size):
    for i in range(len(emotion_cause_pairs)):
        pair = emotion_cause_pairs[i]
        pair = pad_corpus(pair, padding_size)
        emotion_cause_pairs[i] = convert_clauses_to_id(vocab, pair)


################### BUILD VOCAB ########################

################## MAIN INTERFACES #####################
def get_data(file_name):

    text = read_data(file_name)

    sentences = clean_sentences(text)

    word2id, pad_index = build_vocab(sentences)

    emotion_seeds = set(["ashamed", "delighted", "pleased", "concerned", "delight", "happy", "embarrassed", "furious", "nervous", 
                     "miffed", "angry", "mad", "anger", "excitement", "horror", "resentful", "astonished", "revulsion", 
                     "frightened", "cross", "sad", "down", "astonishment", "miserable", "worried", "sorrow", "overjoyed",
                     "dismay", "grief", "annoyance", "alarmed", "astounded", "anguish", "despair", "infuriated", 
                     "embarrassment", "peeved", "amused", "disgruntled", "indignant", "thrilled", "anxious", "excited",
                     "exasperation", "petrified", "heartbroken", "saddened", "depressed", "dismayed", "frustrated", "fedup", "livid",
                     "revulsion", "bewildered", "flabbergasted", "happier", "ecstatic", "elation", "exhilarated", "exhilaration",
                     "glee", "gleeful", "crestfallen", "sadness", "amusement", "dejected", "desolate", "despondency", "horrors",
                     "agitated", "disquiet", "horrified", "exasperated", "irked", "disgruntlement", "sickened", "revolted",
                     "devastated", "heartbreak", "inconsolable", "bewilderment", "nonplussed", "puzzlement", "disquieted",
                     "glum", "downcast", "griefstricken", "startled"])
    
    assert is_complete(emotion_seeds, sentences)
    
    shuffle(text)
    train_split = 0.9
    num_sentences = len(text)
    num_train_sentences = int(train_split*num_sentences)
    train_text = text[:num_train_sentences]
    test_text = text[num_train_sentences:]

    train_clauses, train_emotion_labels, train_cause_labels, train_emotion_cause_pairs = extract_clauses_and_labels(train_text, emotion_seeds)
    test_clauses, test_emotion_labels, test_cause_labels, test_emotion_cause_pairs = extract_clauses_and_labels(test_text, emotion_seeds)

    padding_size = max(get_padding_size(train_clauses), get_padding_size(test_clauses)) + 1

    train_clauses = pad_corpus(train_clauses, padding_size)
    train_clauses = convert_clauses_to_id(word2id, train_clauses)

    test_clauses = pad_corpus(test_clauses, padding_size)
    test_clauses = convert_clauses_to_id(word2id, test_clauses)

    pad_and_convert_pairs_to_id(word2id, train_emotion_cause_pairs, padding_size)
    pad_and_convert_pairs_to_id(word2id, test_emotion_cause_pairs, padding_size)

    return train_clauses, test_clauses, train_emotion_labels, test_emotion_labels, \
           train_cause_labels, test_cause_labels, train_emotion_cause_pairs, \
           test_emotion_cause_pairs, word2id, pad_index, padding_size

################## MAIN INTERFACES #####################

if __name__ == "__main__":
    train_clauses, test_clauses, train_emotion_labels, test_emotion_labels, \
    train_cause_labels, test_cause_labels, train_emotion_cause_pairs, \
    test_emotion_cause_pairs, word2id, pad_index, padding_size = get_data("data.txt")

