"""
@author : Hyunwoong
@when : 8/26/2019
@homepage : https://github.com/gusdnd852
"""
import pandas as pd


class PairDataLoader:
    """
    load sentence pair dataset
    """

    def __init__(self, train_ratio=0.9, dev_set=100):
        """
        constructor of PairDataLoader

        :param train_ratio: ratio of training data (rest is test data)
        :param dev_set: number of data for developing faster
        """
        data = pd.read_csv('data/dataset.csv')
        data = data.drop('label', axis=1)
        data = data.sample(frac=1).reset_index(drop=True)
        question, answer = data['Q'], data['A']
        split_point = int(len(question) * train_ratio)

        self._train = question[:split_point], answer[:split_point]
        self._test = question[split_point:], answer[split_point:]
        self._dev = question[:dev_set], answer[:dev_set]

    @property
    def train(self):
        """:return: training dataset"""
        return self._train

    @property
    def test(self):
        """:return: test dataset"""
        return self._test

    @property
    def dev(self):
        """:return: develop dataset"""
        return self._dev
