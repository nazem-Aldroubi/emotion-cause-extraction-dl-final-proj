import numpy as np
import tensorflow as tf
import numpy as np
import re

# ##########DO NOT CHANGE#####################
# PAD_TOKEN = "*PAD*"
# STOP_TOKEN = "*STOP*"
# START_TOKEN = "*START*"
# UNK_TOKEN = "*UNK*"
# FRENCH_WINDOW_SIZE = 14
# ENGLISH_WINDOW_SIZE = 14
# ##########DO NOT CHANGE#####################

# def pad_corpus(french, english):
# 	"""
# 	DO NOT CHANGE:

# 	arguments are lists of FRENCH, ENGLISH sentences. Returns [FRENCH-sents, ENGLISH-sents]. The
# 	text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
# 	the end.

# 	:param french: list of French sentences
# 	:param english: list of English sentences
# 	:return: A tuple of: (list of padded sentences for French, list of padded sentences for English)
# 	"""
# 	FRENCH_padded_sentences = []
# 	FRENCH_sentence_lengths = []
# 	for line in french:
# 		padded_FRENCH = line[:FRENCH_WINDOW_SIZE]
# 		padded_FRENCH += [STOP_TOKEN] + [PAD_TOKEN] * (FRENCH_WINDOW_SIZE - len(padded_FRENCH)-1)
# 		FRENCH_padded_sentences.append(padded_FRENCH)

# 	ENGLISH_padded_sentences = []
# 	ENGLISH_sentence_lengths = []
# 	for line in english:
# 		padded_ENGLISH = line[:ENGLISH_WINDOW_SIZE]
# 		padded_ENGLISH = [START_TOKEN] + padded_ENGLISH + [STOP_TOKEN] + [PAD_TOKEN] * (ENGLISH_WINDOW_SIZE - len(padded_ENGLISH)-1)
# 		ENGLISH_padded_sentences.append(padded_ENGLISH)

# 	return FRENCH_padded_sentences, ENGLISH_padded_sentences

# def build_vocab(sentences):
# 	"""
# 	DO NOT CHANGE

#   Builds vocab from list of sentences

# 	:param sentences:  list of sentences, each a list of words
# 	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
#   """
# 	tokens = []
# 	for s in sentences: tokens.extend(s)
# 	all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

# 	vocab =  {word:i for i,word in enumerate(all_words)}

# 	return vocab,vocab[PAD_TOKEN]

# def convert_to_id(vocab, sentences):
# 	"""
# 	DO NOT CHANGE

#   Convert sentences to indexed 

# 	:param vocab:  dictionary, word --> unique index
# 	:param sentences:  list of lists of words, each representing padded sentence
# 	:return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
#   """
# 	return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


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

		causes.append(cur_cause)

	return causes

def get_data(file_name):

	text = read_data(file_name)

	emotions = extract_and_clean_emotions(text)

	causes = extract_and_clean_causes(text)

	#### By end of this line, text is cleaned from tags and emotion-cause pairs are extracted ####

get_data('data.txt')
	