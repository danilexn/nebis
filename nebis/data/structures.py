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
        _ret = []
        for t in self.tensors:
            try:
                _ret.append(t[i])
            except:
                _ret.append(t[0])
        return _ret

    def __len__(self):
        return len(self.tensors)

    def __iter__(self):
        self.idx = 0
        return self[self.idx]

    def __next__(self):
        self.idx += 1
        return self[self.idx]
