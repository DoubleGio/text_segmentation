# Papers overview

## Implementations - Supervised
* [SECTOR: A Neural Model for Coherent Topic Segmentation and Classification](https://doi.org/10.1162/tacl_a_00261)
  * **Custom labelled dataset**, **LSTM networks**, Sentence Encoding: **Bag-of-words (+ Bloom filter)** vs. **Word2Vec**, **Topic Embedding**, **Topic Segmentation**, **Topic Classification**
* [Text Segmentation as a Supervised Learning Task](https://doi.org/10.48550/arXiv.1803.09337)
  * **Custom labelled dataset**, **LSTM networks**, **Sentence Embedding**, **Text Segmentation**
* [Topic segmentation in ASR transcripts using bidirectional rnns for change detection](https://doi.org/10.1109/ASRU.2017.8268979)
  * **Concatenated news articles + videos dataset**, **LSTM networks**
* [A More Effective Sentence-Wise Text Segmentation Approach Using BERT](https://doi.org/10.1007/978-3-030-86337-1_16)
  * **BERT sentence encoder**, **Sliding Window LSTM networks**, **Dense layers**, **Attention layers**, **Data augmentation**
* [Attention-Based Neural Text Segmentation](https://doi.org/10.1007/978-3-319-76941-7_14)
  * **Fixed sentence lengths + word2vec embeddings**, **CNN transformation -> sentence representation**, **2 vertically stacked BiLSTM**, **Soft attention layer**
* 

## Interesting methods
* [Attention is All you Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
  * [Video summary](https://www.youtube.com/watch?v=iDulhoQ2pro)
  * Capture long range dependencies better (than RNNs) by using attention layers.
