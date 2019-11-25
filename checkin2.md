# Project Check in #2

**Title**: Learning Emotional Intelligence

**Who**:

Daphne Li-Chen (dlichen)

Nazem Aldroubi (naldroub)

Sophie Yang (syang55)

Ang Li (ali4)

## Introduction

​	The general objective is to extract the potential causes behind certain emotions in text. One common approach is to identify emotions from text and then try to extract the corresponding causes. We are re-implementing the paper “Emotion-Cause Pair Extraction: A New Task to Emotion Analysis in Texts”, which describes a new approach for this task (Emotion-cause pair extraction). Emotions and causes are first extracted to form all potential pairs, and then the pairs are filtered to obtain the valid ones (i.e. the pairs where the cause properly corresponds with the emotion). This can be viewed as a classification problem because we are determining the types of clauses (emotion/cause/neither) within a document of text, and then trying to filter out the significant emotion-cause pairs (thereby classifying the pairs as valid/invalid). The original implementation of the paper is available online, but we intend to re-implement using a different framework (TensorFlow 2.0  vs. TensorFlow 1.8) and a different dataset (English vs. Chinese). 

## Challenges

​	We mainly worked with preprocessing, which presented multiple challenges. The hardest part was finding the proper dataset and extracting/labeling clauses from the sentences. We were only able to find one English-language dataset (the Emotion Stimulus dataset consisting of 820 sentences) that had both emotions and causes annotated (other datasets were either in Chinese, or did not have cause annotations), and the dataset was not in a convenient format. Although the original paper uses clauses as data inputs, the dataset only provided full sentences and the emotion/cause annotations were not done on the clause level. Thus, we had to figure out a way to divide sentences into clauses and then label each clause to indicate whether it was an "emotion clause" or a "cause clause" or neither. We decided to label the cause clauses as the clauses containing words surrounded by the 'cause' tag annotation in the dataset, and we manually determined "emotion seeds" (i.e. emotion words that specify whether or not a clause is an emotion clause) to extract all of the emotion clauses from the dataset. 

## Insights

​	At this point, we have finished preprocessing the data into clauses, emotion labels, cause labels, and all of the valid emotion-cause pairs from the dataset. We also performed the training/testing split on the data. We are still building the model based on the architecture in the original paper, and we hope that the model will perform similarly (in terms of the computed F1 metric score) to the model from the paper, though there may be different results due to the vastly different dataset/preprocessing we used. 

## Plan

We think we are on track with the project. We need to dedicate more time to building and training the model so we can test the model, compute our accuracy metrics, and compare the results to that of the original paper. We may change how we choose cause/emotion clauses during preprocessing if we ever have the need to experiment with different methodologies.
