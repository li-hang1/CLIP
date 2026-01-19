import re

def load_vocab(vocab_txt_path):
    """
    return: vocab_dict {token_str: int}
    """
    vocab = {}
    with open(vocab_txt_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            token = line.strip()
            vocab[token] = idx
    return vocab

def build_tokenizer(vocab, unk_token="<unk>", bos_token="<bos>", eos_token="<eos>"):
    unk_id, bos_id, eos_id = vocab[unk_token], vocab[bos_token], vocab[eos_token]
    pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def tokenizer(text):
        tokens = pattern.findall(text)
        ids = []
        ids.append(bos_id)
        for tok in tokens:
            ids.append(vocab.get(tok, unk_id))
        ids.append(eos_id)
        return ids

    return tokenizer


if __name__ == "__main__":
    vocab = load_vocab("data/vocab.txt")
    tokenizer = build_tokenizer(vocab)
    caption = "A room with blue walls and a white sink and door."
    print(tokenizer(caption))
