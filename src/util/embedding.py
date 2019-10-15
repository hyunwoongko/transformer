"""
@author : Hyunwoong
@when : 8/26/2019
@homepage : https://github.com/gusdnd852
"""
from os import path

import pandas as pd
from gensim.models import FastText, Word2Vec


class TextEmbedding:
    """
    Text Embedding Util using Gensim
    We should embed sentence to vector before using Transformer
    Gensim's FastText, Word2Vec Support
    """
    _embedding_vector = None
    _model = None

    def model(self, dataset, model_type='fasttext'):
        return self._model

    def embedding_vector(self, dataset, model_type='fasttext'):
        return self._embedding_vector

    def __concat(self, dataset):
        """
        concat input vector and target vector

        :param dataset: Tuple[Input, Target]
        :return: Concatenated Vector [Input, Target]
        """
        q, v = dataset
        vocab = pd.concat([q, v])
        vocab = vocab.values
        return vocab

    def __train(self, model_type, dataset):
        """
        train embedding model

        :param model_type: embedding model type
        :param dataset: data to embed
        :return: embedding model
        """
        if model_type == 'fasttext': return self.__fasttext(dataset)
        if model_type == 'word2vec': return self.__word2vec(dataset)
        return NotImplementedError('invalid embedding model type')

    def __fasttext(self, dataset):
        """
        train and return FastText model with given dataset

        :param dataset: dataset to train
        :return: FastText model
        """
        model = FastText(
            sentences=self.__concat(dataset),
            window=3,
            workers=8,
            min_count=5,
            iter=500)

        return model

    def __word2vec(self, dataset):
        """
        train and return Word2Vec model with given dataset

        :param dataset: dataset to train
        :return: Word2Vec model
        """
        model = Word2Vec(
            sentences=self.__concat(dataset),
            window=3,
            workers=8,
            min_count=5,
            iter=500)

        return model
