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
    """
    Extracts all clauses, emotion labels, cause labels, 
    and emotion-cause pairs from the given text.

    :param text: string of text
    :param emotion_seeds: set of "seed" words that designate whether or not a clause is an emotion clause
    :return: 2D np.array of clauses, 1D np.array of emotion labels, 1D np.array of cause labels, list of tuples of lists of emotion-cause pairs
    """
    document_clauses = []
    emotion_labels = []
    cause_labels = []
    
    emotion_cause_pairs = []

    for i, line in enumerate(text):
        # Determine clauses by splitting on punctuation.
        clauses = re.split("[.,!;:\"]+", line)
        emotion_clauses = []
        cause_clauses = []

        for clause in clauses:
            # Remove punctuation, digits, and XML tags from the clause.
            cleaned_clause = clean_clause(clause)
            # Split into words.
            clause_words = cleaned_clause.split()
            
            if len(clause_words) == 0:
                continue
            
            document_clauses.append(clause_words)
            
            # Cause clauses contain the 'cause' XML tag.
            if "<cause>" in clause:
                cause_labels.append(1)
                cause_clauses.append(clause_words)
            else:
                cause_labels.append(0)
            
            # Emotion clauses contain a word in the set of emotion seed words.
            has_seed = False
            for word in clause_words:
                if word.lower() in emotion_seeds:
                    emotion_labels.append(1)
                    emotion_clauses.append(clause_words) 
                    has_seed = True
                    break
            if not has_seed:
                emotion_labels.append(0)
        
        # Obtain the emotion-cause pairs from a sentence.
        for e_clause in emotion_clauses:
            for c_clause in cause_clauses:
                emotion_cause_pairs.append((e_clause, c_clause))

    return np.array(document_clauses), np.array(emotion_labels), np.array(cause_labels), emotion_cause_pairs

def clean_clause(clause):
    """
    Removes punctuation, digits, and XML tags from a clause.

    :param clause: String representing a clause.
    :return: Cleaned clause String.
    """
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
    """
    Extracts the actual emotions (e.g. "happy", "sad") from the emotion XML tag of each sentence.
    :param text: String of text.
    :return: List of Strings representing emotions.
    """
    emotions = []
    for i, line in enumerate(text):
        # Extracts all strings within tags. First one is the emotion of the sentence
        cur_emotion = re.findall('<(.*?)>', line)[0]

        # Remove emotion tags
        text[i] = text[i][len(cur_emotion) + 2:-len(cur_emotion) - 4]

        emotions.append(cur_emotion)

    return emotions

def extract_and_clean_causes(text):
    """
    Extracts the words between the cause XML tags in the data.
    :param: String of text.
    :return: List of Strings representing causes.
    """
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
    """
    Calculates the frequency of each vocab word in a list of sentences.
    Helpful in determining the "emotion seed" words of the data.

    :param sentences: List of sentences.
    :return: Sorted dictionary mapping vocab words to their frequencies.
    """
    vocab_frequency = defaultdict(int)
    for sentence in sentences:
        for word in sentence:
            vocab_frequency[word] += 1

    return sorted(vocab_frequency.items(), key=operator.itemgetter(1))

def is_complete(emotion_seeds, sentences):
    """
    Checks whether or not the set of emotion seeds is complete for
    the given sentences, meaning that each sentence contains an 
    emotion seed word.

    :param emotion_seeds: Set of emotion seed words.
    :param sentences: List of sentences.
    :return: Boolean representing whether or not the set of emotion seeds is complete.
    """
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
    """
    Pads each clause in the list of emotion-cause pairs, and converts all words into ids.

    :param vocab: Dictionary mapping vocab words to unique ids.
    :param emotion_cause_pairs: List of tuples of lists representing emotion-cause pairs.
    :param padding size: Int representing the size to pad all clauses to.
    """
    for i in range(len(emotion_cause_pairs)):
        pair = emotion_cause_pairs[i]
        pair = pad_corpus(pair, padding_size)
        emotion_cause_pairs[i] = convert_clauses_to_id(vocab, pair)


################### BUILD VOCAB ########################

################## MAIN INTERFACES #####################
def get_data(file_name):
    # Reads the data as a list of sentences.
    text = read_data(file_name)
    
    # Clean the list of sentences.
    sentences = clean_sentences(text)
    
    # Builds the vocab dictionary.
    word2id, pad_index = build_vocab(sentences)
    
    # Manually determined emotion seeds in the data.
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
    
    # Split into train and test text.
    shuffle(text)
    train_split = 0.9
    num_sentences = len(text)
    num_train_sentences = int(train_split*num_sentences)
    train_text = text[:num_train_sentences]
    test_text = text[num_train_sentences:]
    
    # Obtain clauses, emotion labels, cause labels, and emotion-cause pairs from the train and test text.
    train_clauses, train_emotion_labels, train_cause_labels, train_emotion_cause_pairs = extract_clauses_and_labels(train_text, emotion_seeds)
    test_clauses, test_emotion_labels, test_cause_labels, test_emotion_cause_pairs = extract_clauses_and_labels(test_text, emotion_seeds)
    
    # Pad all clauses and convert words to ids.
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

