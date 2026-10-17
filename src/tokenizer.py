"""
Tokenizer: text <-> token IDs, with byte-level BPE (Byte Pair Encoding).

How BPE learns its vocabulary:
  1. Start with the 256 possible bytes. Every text can be written with them,
     so no text is ever "unknown".
  2. Count each adjacent pair of tokens in the training text.
  3. Merge the most frequent pair into one new token ("o" + "p" -> "op").
  4. Repeat until the vocabulary has vocab_size tokens.

Frequent words become one token (" Holmes"). Rare words stay as pieces.

NOTE: The tokenizer makes token IDs only. It does not make embeddings.
The model maps each token ID to a learned embedding (src/embeddings.py).
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def train_bpe(text: str, vocab_size: int, save_path: str) -> "BookTokenizer":
    """Learn BPE merges from `text` and save the tokenizer as JSON."""
    tok = Tokenizer(models.BPE())

    # Byte-level: the text is first changed to bytes. A space becomes the
    # visible symbol "Ġ" and a newline becomes "Ċ" inside the vocabulary.
    # The pre-tokenizer also splits the text at word boundaries, so a merge
    # never joins the end of one word with the start of the next word.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes
        min_frequency=2,        # merge a pair only if it occurs 2 or more times
        show_progress=False,
    )
    tok.train_from_iterator([text], trainer=trainer)
    tok.save(save_path)
    return BookTokenizer(tok)


class BookTokenizer:
    """A small wrapper with the three operations the project needs."""

    def __init__(self, tok: Tokenizer):
        self.tok = tok

    @classmethod
    def load(cls, path: str) -> "BookTokenizer":
        return cls(Tokenizer.from_file(path))

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        """Text -> token IDs."""
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Token IDs -> text."""
        return self.tok.decode(ids)

    def tokens(self, text: str) -> list[str]:
        """Text -> readable tokens. Each ID is decoded alone, so " capital"
        shows a real space and not the internal symbol "Ġ".

        NOTE: A character of more than one byte (for example the curly quote “)
        can be split across two tokens. One part alone then shows as "�".
        """
        return [self.decode([i]) for i in self.encode(text)]
