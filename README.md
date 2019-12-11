# Emotion-Cause Pair Extraction (ECPE)

Implements a model for ECPE based on the paper "Emotion-Cause Pair Extraction: A New Task to Emotion Analysis in Texts" by Rui Xia, Zixiang Ding, accepted by ACL 2019. 

# Data
Emotion-stimulus data to accompany this paper (http://www.site.uottawa.ca/~diana/resources/emotion_stimulus_data/):

Diman Ghazi, Diana Inkpen & Stan Szpakowicz (2015). “Detecting Emotion Stimuli in Emotion-Bearing Sentences”. Proceedings of the 16th International Conference on Intelligent Text Processing and Computational Linguistics (CICLing 2015), Cairo, Egypt, to appear.

We use the Emotion Cause dataset, which contains 820 sentences, each containing an emotion and its corresponding cause, which are annotated by XML tags. We manually altered the data (e.g.
added more words and punctuation to each sentence), so that the causes and emotions would be better split into clauses during preprocessing.

