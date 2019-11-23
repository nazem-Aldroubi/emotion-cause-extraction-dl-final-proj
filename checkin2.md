# Project Check in #2

**Title**: Learning Emotional Intelligence

**Who**:

Daphne Li-Chen (dlichen)

Nazem Aldroubi (naldroub)

Sophie Yang (syang55)

Ang Li (ali4)

## Introduction

​	The general objective is to extract the potential causes behind certain emotions in text. One common approach is to identify emotions from text and then try to extract the corresponding causes. We are re-implementing the paper “Emotion-Cause Pair Extraction: A New Task to Emotion Analysis in Texts”, which describes a new approach for this task (Emotion-cause pair extraction). Emotions and causes are first extracted to form all potential pairs, and then the pairs are filtered to obtain the valid ones (i.e. the pairs where the cause properly corresponds with the emotion). This can be viewed as a classification problem because we are determining the types of clauses (emotion/cause/neither) within a document of text, and then trying to filter out the significant emotion-cause pairs (thereby classifying the pairs as valid/invalid). The original implementation of the paper is available online, but we intend to re-implement using a different framework (PyTorch vs. TensorFlow) and a different dataset. 

## Chanllenges

​	We are mainly working with preprocessing. The hardest part has been finding the proper dataset and extracting/labeling clauses from the sentences. We need to divide sentences into clauses and then label each clause to indicate whether it is an "emotion clause" or a "cause clause" or neither. The causes are given by the dataset and we use "emotion seeds" to determine whether a clause is "emotional".

## Insights

​	We are still building the model.

## Plan

We think we are on track with the project. We might need to spend more time building the model. 