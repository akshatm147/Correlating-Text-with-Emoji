# Correlating-Text-with-Emoji
SMAI semster course project

## Team Members
- Akshat Maheshwari (20161024)
- Kalpit Pokra (20161134)
- Kushagra Nagori (20161032)
- Kartik Garg (20161219)

## Goal
The goal of the project is: Given a tweet, predict the most likely emoji that was used along with it.

## Variants
- **Classification** : Given a tweet, predict the most likely emoji that was used along with it.

- **Emoji Embeddings** : Given a tweet without an emoji, find its embedding (averaging of word embeddings), and find the closest emoji to it. This can be extended to: given a statement, get the most relevant emoji that is suitable with that sentence using semantic analysis of that sentence using the *word2vec* and the *bag-of-words* models.
This can then be extended to paragraphs. Given a paragraph, break the paragraph to sentences and analyze each sentence and find the emoji associated with that sentence using the previous model.

## Most General Models
The most general models that we can use for the classification tasks are *LinearSVC*, *Naive Bayes*, *Logistic Regression*, *KNN Classifier*, and other classification models.

## Proposal
This is the [link](https://docs.google.com/document/d/1ESlHfxKZUEl7FuxYsGFTMZ3chAOlhFqf2GCFIrMxoTg/edit?usp=sharing) for the doc.