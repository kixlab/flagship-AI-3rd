# Word2Vec Implementation (skip-gram)

## Process
0. Convert data into txt file (each line contains one sentence)
1. Create word_list & word_dict
2. Create skip_gram dataset
3. Train model with the dataset
4. Check accuracy of the model

## Testing the Model

### t-SNE (t-Stochastcic Neighbor Embedding)
Used for dimensionality reduction & visualization; Normally used for word2vec.

SNE + Solving Crowding Problem => t-SNE

## Optimization

### Variables

- Batch size
  - Size of the number of words in one iteration
  - Size ↑ , Performance ↑ (GPU's parellel processing), Memory Usage ↑
- Embedding size
  - Dimension of embedding vector
  - 50 or 100-200 recommended
- Training epoch
  - Depends on the size of training dataset
- Trainig rate
  - Seems 0.025 is normal
  - Slow training with low value, Disperse with too high value

### Techniques

#### Remove rare words
Set the minimum frequency of words which will be trained. Because for good training, the several presences of words are needed. Also, it imporves memory or calucation performance

#### Negative Sampling

#### Multi-core Processing (Threads)

#### Subsmapling Frequent Words

## References

### For Understanding
- [t-SNE](https://ratsgo.github.io/machine%20learning/2017/04/28/tSNE/)
- [On word embeddings - Part 3: The secret ingredients of word2vec](http://ruder.io/secret-word2vec/index.html#wordembeddingsvsdistributionalsemanticsmodels)
- [word2vec 관련 이론 정리](https://shuuki4.wordpress.com/2016/01/27/word2vec-%EA%B4%80%EB%A0%A8-%EC%9D%B4%EB%A1%A0-%EC%A0%95%EB%A6%AC/)
- [Word2Vec(skip-gram model): PART 1 - Intuition.](https://towardsdatascience.com/word2vec-skip-gram-model-part-1-intuition-78614e4d6e0b)

### For Implementing
- [TensorFlow: Save and Restore Models](http://stackabuse.com/tensorflow-save-and-restore-models/)
- [Word2Vec 모델 학습하기](https://deeplearning4j.org/kr/word2vec#%EB%AA%A8%EB%8D%B8-%ED%95%99%EC%8A%B5%ED%95%98%EA%B8%B0)