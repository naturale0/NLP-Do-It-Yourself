# Natural Language Processing: DIY

Implement well-known NLP models from scratch with high-level APIs.

All implementations are in PyTorch and/or Tensorflow.

* **`NLP_with_PyTorch`**
    * `1_basics_(tutorial)`  \
      : Basic ideas and introduction to NLP. Contents from PyTorch tutorials.
    * `2_word_embedding`  \
      : Word embedding models. Contents mainly from research articles and Lee (2019)<sup>[1](#myfootnote1)</sup>.
        * [Neural Probabilistic Language Model](NLP_with_PyTorch/2_word_embedding/2-1_NPLM.ipynb)<sup>[2](#myfootnote1)</sup>.
        * Word2Vec: TBD.
    * `3_document-embedding`  \
      : Sentence/document-level embedding models. Contents mainly from research articles and Lee (2019)<sup>[1](#myfootnote1)</sup>.
    * `4_sentiment_analysis`: TBD.
* **`NLP_with_TensorFlow`**  \
: tensorflow port of `NLP_with_PyTorch`
    * `1_basics_(tutorial)`
    * `2_word_embedding`
        * [Neural Probabilistic Language Model](NLP_with_TensorFlow/2_word_embedding/2-1_NPLM.ipynb)<sup>[2](#myfootnote1)</sup>.
        * Word2Vec: TBD.

---

<a name="myfootnote1">1</a>: Lee. 2019. **한국어 임베딩 (Embedding Korean)**. 에이콘 출판사.  
<a name="myfootnote1">2</a>: Bengio et al. 2003. **A neural probabilistic language model**. The journal of machine learning research, 3, 1137-1155.