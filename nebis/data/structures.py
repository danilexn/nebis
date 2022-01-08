class SurvivalTensor:
    def __init__(self, tensors) -> None:
        assert tensors is not None
        self.tensors = tensors
        self.idx = 0

    def size(self, idx):
        _size = self.tensors[0].shape
        if type(_size) is int:
            return _size
        else:
            return _size[idx]

    def __getitem__(self, i):
        return [t[i] for t in self.tensors]

    def __len__(self):
        return len(self.tensors)

    def __iter__(self):
        self.idx = 0
        return self[self.idx]

    def __next__(self):
        self.idx += 1
        return self[self.idx]


class SurvivalBatchTensor:
    def __init__(self, tensors) -> None:
        assert tensors is not None
        self.tensors = tensors

    def to(self, device):
        self.tensors = (t.to(device) for t in self.tensors)

    def detach(self):
        self.tensors = (t.detach() for t in self.tensors)

    def cpu(self):
        self.tensors = (t.cpu() for t in self.tensors)

    def numpy(self):
        self.tensors = (t.numpy() for t in self.tensors)
