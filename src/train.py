"""
@author : Hyunwoong
@when : 8/20/2019
@homepage : https://github.com/gusdnd852
"""
from src.util.embedding import TextEmbedding
from src.util.loader import PairDataLoader

loader = PairDataLoader()
dev_set = loader.dev

embedding = TextEmbedding()
fasttext = embedding.model(dev_set)
print(fasttext)