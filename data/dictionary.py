import numpy as np


class EmbDict:
    def __init__(self, file_path):
        self.embs = {}
        # Load dictionary to python dict
        with open(file_path, "r") as f:
            for index, line in enumerate(f):
                if index == 0:
                    continue
                token, emb = line.split()
                self.embs[token] = float(emb)

        self.vocab_size = len(self.embs)

    def __call__(self, sequences) -> np.ndarray:
        result = []
        for tokens in sequences:
            emb = []
            for i in range(len(tokens)):
                if self.embs.get(tokens[i]) is not None:
                    emb.append(self.embs[tokens[i]])
                else:
                    emb.append(0)
            result.append(emb)

        return np.array(result)

    def get_size(self):
        return self.vocab_size



