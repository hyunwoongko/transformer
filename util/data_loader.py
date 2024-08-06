import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
from util.logger import get_logger

logger = get_logger(__name__)


class TranslationDataLoader:
    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        self.tokenizer_source = None
        self.tokenizer_target = None
        self.source_vocab = None
        self.target_vocab = None
        logger.info("Dataset initializing done")

    def make_dataset(self):
        if self.ext == (".de", ".en"):
            self.tokenizer_source = get_tokenizer(self.tokenize_de)
            self.tokenizer_target = get_tokenizer(self.tokenize_en)
        elif self.ext == (".en", ".de"):
            self.tokenizer_source = get_tokenizer(self.tokenize_en)
            self.tokenizer_target = get_tokenizer(self.tokenize_de)

        train_data = Multi30k(split="train", language_pair=self.ext)
        valid_data = Multi30k(split="valid", language_pair=self.ext)
        test_data = Multi30k(split="test", language_pair=self.ext)
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        def yield_tokens(data_iter, tokenizer):
            for data in data_iter:
                yield tokenizer(data[0])
                yield tokenizer(data[1])

        self.source_vocab = build_vocab_from_iterator(
            yield_tokens(train_data, self.tokenizer_source),
            specials=["<unk>", "<pad>", self.init_token, self.eos_token],
        )
        self.source_vocab.set_default_index(self.source_vocab["<unk>"])

        self.target_vocab = build_vocab_from_iterator(
            yield_tokens(train_data, self.tokenizer_target),
            specials=["<unk>", "<pad>", self.init_token, self.eos_token],
        )
        self.target_vocab.set_default_index(self.target_vocab["<unk>"])

    def collate_batch(self, batch):
        source_pipeline = lambda x: [
            self.source_vocab[token] for token in self.tokenizer_source(x)
        ]
        target_pipeline = lambda x: [
            self.target_vocab[token] for token in self.tokenizer_target(x)
        ]

        source_batch, target_batch = [], []
        for src, tgt in batch:
            source_batch.append(
                torch.tensor(
                    [self.source_vocab[self.init_token]]
                    + source_pipeline(src)
                    + [self.source_vocab[self.eos_token]],
                    dtype=torch.int64,
                )
            )
            target_batch.append(
                torch.tensor(
                    [self.target_vocab[self.init_token]]
                    + target_pipeline(tgt)
                    + [self.target_vocab[self.eos_token]],
                    dtype=torch.int64,
                )
            )

        source_batch = torch.nn.utils.rnn.pad_sequence(
            source_batch, padding_value=self.source_vocab["<pad>"]
        )
        target_batch = torch.nn.utils.rnn.pad_sequence(
            target_batch, padding_value=self.target_vocab["<pad>"]
        )
        return source_batch, target_batch

    def make_iter(self, train, validate, test, batch_size, device):
        train_loader = DataLoader(
            train, batch_size=batch_size, collate_fn=self.collate_batch
        )
        valid_loader = DataLoader(
            validate, batch_size=batch_size, collate_fn=self.collate_batch
        )
        test_loader = DataLoader(
            test, batch_size=batch_size, collate_fn=self.collate_batch
        )
        logger.info("Dataset initializing done")
        return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    ext = (".de", ".en")
    tokenize_en = "spacy"
    tokenize_de = "spacy"
    init_token = "<sos>"
    eos_token = "<eos>"
    min_freq = 2
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    translation_data_loader = TranslationDataLoader(
        ext, tokenize_en, tokenize_de, init_token, eos_token
    )
    _train_data, _valid_data, _test_data = translation_data_loader.make_dataset()
    translation_data_loader.build_vocab(_train_data, min_freq)
    _train_loader, _valid_loader, _test_loader = translation_data_loader.make_iter(
        _train_data, _valid_data, _test_data, batch_size, device
    )

    for src_batch, tgt_batch in _train_loader:
        print(src_batch)
        print(tgt_batch)
        break
