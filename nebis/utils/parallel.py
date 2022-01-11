from torch.nn import DataParallel


class ListDataParallel(DataParallel):
    """
    we do the scatter outside of the DataPrallel.
    input: Scattered Inputs without kwargs.
    """

    def __init__(self, module):
        super(ListDataParallel, self).__init__(module)

    def forward(self, *inputs, **kwargs):
        assert (
            len(inputs) == 0
        ), "ListDataParallel does not support positional arguments"

        _batch_size = len(kwargs[list(kwargs.keys())[0]])

        assert _batch_size <= len(
            self.device_ids
        ), "Batch size cannot be greater than the number of GPUs with ListDataParallel"

        new_inputs = [{} for _ in self.device_ids[0:_batch_size]]
        for key in kwargs:
            for i, device in enumerate(self.device_ids):
                new_inputs[i][key] = kwargs[key][i].to(device)

        nones = [[] for _ in self.device_ids[0:_batch_size]]
        replicas = self.replicate(self.module, self.device_ids[0:_batch_size])
        outputs = self.parallel_apply(replicas, nones, new_inputs)

        return self.gather(outputs, self.output_device)
