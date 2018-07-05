# KIXLAB Flagship AI 2018 - 3rd Project

## Goal

Classify a input sentence into one of 8 VRM categories

- Indexing 10 thousands of corpus (from TV Show, etc)
- Word Embedding
- Training with indexed test cases
- Classifying input sentence to one of category of Verbal Response Mode (with our own classifier)

> William B. Stiles, 1992, Describing Talk: A Taxonomy of Verbal Response Modes

## Weeks

1. Research for Hangul Embedding and building infrastructure for machine learning
    - Order: Embedding -> finding the best infrastructure (Tensorflow, pytorch, and etc)


## Process with Options
1. Tokenizing Data (documents -> sentences of words)
    - Spliting with whitespace
    - **Considering punctuation marks as words**
    - Removing most common words
    - Re-define words with each one's meaning
    - Morpheme analyzer (형태소 분석)
2. Word Embedding
    - **Word2Vec**
        - CBOW
        - **Skip-gram**
            - Embedding Size
            - Word Count Size
                - Uniform
                - Weighed
            - Order of training
                - **Random (epoch & batch size)**
    - fastText
    - LSA
    - GloVe