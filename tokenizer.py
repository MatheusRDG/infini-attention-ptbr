from tokenizers import ByteLevelBPETokenizer


class BPETokenizer:
    def __init__(self, file_path=None):
        self.tokenizer = ByteLevelBPETokenizer()
    
    def train_tokenizer(self, file_path, vocab_size=30000, min_frequency=2):
        self.tokenizer.train(files=file_path, vocab_size=vocab_size, min_frequency=min_frequency)

    def save_tokenizer(self, path = "."):
        self.tokenizer.save_model(path)

    def load_tokenizer(self, vocab_file, merges_file):
        self.tokenizer = ByteLevelBPETokenizer().from_file(vocab_file, merges_file)
    
    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)
