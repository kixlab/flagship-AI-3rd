# Possible Options for the Project

## Current Choices

- Dataset
  - Scenarios from kocca
- Pre-process(except tokenizing)
  - remove words in parentheses
  - consider punctuation
  - intergrate some ellipses(... , .... , ...)
- Tokenizing sentences
  - Setting maximum length of words as 3
- Word Embedding Model
  - Word2Vec
- Hyper-params
  - [Word2Vec Trials](https://docs.google.com/spreadsheets/d/1Hy2QeW-cykR5ZGM67KOepwH9vbGya2nXX_xiB1mAdNM/edit#gid=0)

## Dataset

Recommend to be spoken English in daily life

### Scenarios from kocca

- Dataset (owned by SJ, Seongo)
- [Parser (Html to JSON)](https://github.com/kixlab/TVscripts_parser)
- Possible noises
  - Wrong words
  - Multiple sentences in a line
  - Old words from historical data

### Sejong corpus

> The National Institute of the Korean Language, “Final Report on
Achievements of 21st Sejong Project: Electronic Dictionary”, 2007.
(in Korean). 

- [Homepage (needs signing up)](https://ithub.korean.go.kr/user/corpus/corpusManager.do)
- Parser (Need to develop)

## Pre-process Data (except tokenizing)

### Removing words in parentheses

To remove indicates in scenarios

### Processing punctuations & other characters

- Considering punctuations as tokens
- Integrating ellipses into single word (ex. .. , ... , ..... , … -> ..-)
- Converting numbers into Korean (ex. 23번 -> 이삼번)
- Removing non-Korean words like English, number, etc

## Tokenizing sentences

- Spliting with whitespaces
- Setting length of words maximum 3 (ex. 내일이면 -> 내일이-)
- Tokenizing with morphemes (koNLPy)
- Split long words into 1-3 character words (ex. 내일이면 -> 내일이, 면)

## Word Embeding Model

- Word2Vec (Skip-gram)
- FastText
- GloVe

## Hyper-params for the Model
> Choi et al., On Word Embedding Models and Parameters Optimized for Korean, 2016

1. #. of Sentences
    - 100,000
    - 500,000
2. Training Epoch
    - (#. of Sentences) / 10
      - Adjust with loss (stop training when loss does not change)
3. Embedding Size
    - 50
      - Better 100~, but not much improvement and takes more time to train
4. Learning Rate
    - 0.025
      - No standard
      - High Rate means faster training, but more vulnerable for divergence
5. Window Size
    - 2 (gensim default)
    - 5 (paper recommended)
6. Batch Size
    - 10000 (gensim default)
