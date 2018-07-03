# Research for Embedding

## What is Embedding?

> “Turns positive integers (indexes) into dense vectors of fixed size”

## Core Idea for Word Embedding

### VSM (Vector Space Model)
An algebraic model for representing text documents (and any objects, in general) as vectors of identifiers

### Distributional hypothesis
Words that are used and occur in the same contexts tend to purport similar meanings.


## Types of Word Embedding

### Frequency based embedding

#### Bag-of-words (BOW)
Give indexes to each word used in documents and express with counting # of words in the sentence.

#### Count Vector
Count all of words in documents and express them with D*T matrix. (D: # of documents, T: # of different words)

Problem in too big size of matrix and some solutions are following:
- use stop words (remove common words like 'a, an, this, that')
- extract some top words frequently used

#### TF-IDF Vector
Count # of words in sentence + # of words in corpus(Total words)
```
term-frequency(TF) * inverse-document-frequency(IDF)
===
TF: # of the word in the document / # of all words in the document
IDF: log(# of documents in corpus / # of documents with the word)
```
#### Co-Occurrence Matrix with a fixed context window
Words co-occurrence matrix describes how words occur together that in turn captures the relationships between words.
```
Matrix w(current) * w(next) with value (w(next)|w(current))
```

This method can preserve semantic relationship between words even after SVD. It used for text classification, sentiment analysis and etc.

### Prediction based embedding
It is hard to find similiarity between words with above methods. To solve it, **prediction based embedding** using ANN(artificial neural network) is needed.

#### One-hot representation
In vector, only one element is 1, the others are 0.

#### Continous Bag of Words (CBOW)
Guess the word with surrounding words
(multiple words -> predict one word)

#### Skip-gram
Predict surrounding words with a word
(one word -> predict surrounding words)

#### Strengths and Weaknesses
- low on memory; improve performance by using Hierarchical Softmax & Negative Sampling (V to ln(V))
- CBOW is faster with small dataset, but Skip-gram is faster with large dataset

### Etc

#### LSA
- Latent Semantic Analysis
- Diminish the number of dimension of data like Word-Document Matrix, window based o-occurrence matrix with SVD method
- Produce word-dimension & dimension-dimension & dismension-sentence matrices
- Not heavy methodology, but needs caculation again when data is changed

## Applications

### Word2Vec
CBOW or Skip-gram with optimization, co-sine similiarity

### GloVe
> While methods like LSA efficiently leverage statistical information, they do relatively poorly on the word analogy task, indicating a sub-optimal vector space structure. Methods like skip-gram may do better on the analogy ask, but they poorly utilize the statistics of the corpus since they train on separate local context windows instead of on global co-occurrence counts.

Word2vec has limitation on reflecting co-occurrence of total corpus; Word2vec uses local context window.

To cover up, GloVe uses the words' probability of co-occurrence; words appearing probability within certain window.

### fastText
Extension of Word2Vec;

FastText treats each word as composed of character ngrams and this character ngrams are the atomic entry, rather than the words in Word2Vec or GloVe.

More accurate on noisy samples, but fastText takes more long time to training


### ELMo

### Doc2Vec

## References

### Web
-  [Deep Learning #4 - Embedding Layers](https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12)
- [자연어(NLP) 처리 기초 정리](http://hero4earth.com/blog/learning/2018/01/17/NLP_Basics_01/)
- [Word embedding (Manjeet Singh, Medium)](https://medium.com/data-science-group-iitr/word-embedding-2d05d270b285)
- [word2vec 관련 이론 정리](https://shuuki4.wordpress.com/2016/01/27/word2vec-%EA%B4%80%EB%A0%A8-%EC%9D%B4%EB%A1%A0-%EC%A0%95%EB%A6%AC/)
- [빈도수 세기의 놀라운 마법, Word2Vec, Glove, Fasttext](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/11/embedding/)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [awesome-embedding-models](https://github.com/Hironsan/awesome-embedding-models)
- [Awesome Korean NLP](https://github.com/datanada/Awesome-Korean-NLP)
- [
Word Embedding의 직관적인 이해 : Count Vector에서 Word2Vec에 이르기까지](https://www.nextobe.com/single-post/2017/06/20/Word-Embedding%EC%9D%98-%EC%A7%81%
- [SVD와 PCA, 그리고 잠재의미분석(LSA)](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/04/06/pcasvdlsa/)

### Paper
- [Camacho-Collados, Pilehvar (2018),
From Word to Sense Embeddings: A Survey on Vector Representations of Meaning](https://arxiv.org/abs/1805.04032)
- [Altszyler et al. (2017), Comparative study of LSA vs Word2vec embeddings
in small corpora: a case study in dreams database](https://arxiv.org/pdf/1610.01520.pdf)
- [Landauer, T. K., Foltz, P. W., & Laham, D. (1998).
 Introduction to Latent Semantic Analysis.]
- [최상혁 (2017), 음절 기반 한국어 단어 임베팅 모델 및 학습 기법](http://s-space.snu.ac.kr/bitstream/10371/122708/1/000000142646.pdf)