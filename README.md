# Natural Language Processing: DIY

Implement well-known NLP models from scratch with high-level APIs.

All implementations are in PyTorch and/or Tensorflow.

* **`NLP_with_PyTorch`**
    * `1_basics_(tutorial)`  \
      : Basic ideas and introduction to NLP. Contents from PyTorch tutorials.
    * `2_word_embedding`  \
      : Word embedding models. Contents mainly from research articles and Lee (2019)<sup>[1](#myfootnote1)</sup>.
        * Neural Probabilistic Language Model<sup>[2](#myfootnote1)</sup>: [[notebook]](NLP_with_PyTorch/2_word_embedding/2-1_NPLM.ipynb) [[blog]](https://naturale0.github.io/machine%20learning/natural%20language%20processing/Understanding-Neural-Probabilistic-Language-Model)
        * Word2Vec (skip gram)<sup>[3](#myfootnote3),[4](#myfootnote4)</sup>: [[notebook]](NLP_with_PyTorch/2_word_embedding/2-2_skip-gram.ipynb) [[blog]](https://naturale0.github.io/machine%20learning/natural%20language%20processing/understanding-skip-gram)
        * FastText<sup>[5](#myfootnote5)</sup>: [[notebook]](NLP_with_PyTorch/2_word_embedding/2-3_fasttext.ipynb) [[blog]](https://naturale0.github.io/machine%20learning/natural%20language%20processing/Understanding-FastText)
        * Latent Dirichlet Allocation<sup><a name="myfootnote6">6</a>,<a name="myfootnote7">7</a>,<a name="myfootnote8">8</a></sup>,: [notebook](https://github.com/naturale0/NLP-Do-It-Yourself/blob/main/NLP_with_PyTorch/3_document-embedding/3-1.%20latent%20dirichlet%20allocation.ipynb)[blog [1](https://naturale0.github.io/natural%20language%20processing/LDA-1-background-topic-modelling), [2](https://naturale0.github.io/bayesian/machine%20learning/natural%20language%20processing/LDA-2-The-Model), [3](https://naturale0.github.io/bayesian/machine%20learning/natural%20language%20processing/LDA-3-Variational-EM), [4](https://naturale0.github.io/bayesian/machine%20learning/natural%20language%20processing/LDA-4-Gibbs-Sampling), [5]()]
    * `3_document-embedding`  \
      : Sentence/document-level embedding models. Contents mainly from research articles and Lee (2019)<sup>[1](#myfootnote1)</sup>.
* `4_sentiment_analysis`: TBD.
    
* **`NLP_with_TensorFlow`**  \
: tensorflow port of `NLP_with_PyTorch`
    * `1_basics_(tutorial)`
    * `2_word_embedding`
        * Neural Probabilistic Language Model<sup>[2](#myfootnote1)</sup>: [[notebook]](NLP_with_TensorFlow/2_word_embedding/2-1_NPLM.ipynb) [[blog]](https://naturale0.github.io/machine%20learning/natural%20language%20processing/Understanding-Neural-Probabilistic-Language-Model)
        * Word2Vec: TBD.

---

<a name="myfootnote1">1</a>: Lee. 2019. **한국어 임베딩 (Embedding Korean)**. 에이콘 출판사.  
<a name="myfootnote2">2</a>: Bengio et al. 2003. **A neural probabilistic language model**. The journal of machine learning research, 3, 1137-1155.  
<a name="myfootnote3">3</a>: Mikolov et al. 2013. **Efficient Estimation of Word Representations in Vector Space**. International Conference on Learning Representations.  
<a name="myfootnote4">4</a>: Mikolov et al. 2013. **Distributed Representations of Words and Phrases and their Compositionality**. Advances in Neural Information Processing Systems, 26.  
<a name="myfootnote5">5</a>: Bojanowski et al. 2017. **Enriching Word Vectors with Subword Information**. Transactions of the Association for Computational Linguistics, 5, 135-146.  
<a name="myfootnote6">6</a>: Pritchard, Stephens, Donnelly. 2000. **Inference of Population Structure Using Multilocus Genotype Data**. Genetics. 155: 945–959.  
<a name="myfootnote7">7</a>: Blei, Ng, Jordan. 2003. **Latent Dirichlet Allocation**. Journal of Machine Learning Research. 3 (4–5): 993–1022.  
<a name="myfootnote8">8</a>: Griffiths, Steyvers. 2004. **Finding scientific topics**. Proceedings of the National Academy of Sciencess of the United States of America. 101: 5228-5235.  
