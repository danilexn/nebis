import os
import numpy as np

from sklearn.dummy import DummyClassifier

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import (
    DataLoader,
    SequentialSampler,
    TensorDataset,
)

from nebis.utils.metrics import classification_metrics, classification_roc_auc
from nebis.utils import get_survival_y_true
from nebis.data.structures import SurvivalTensor


class BaseDataset:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = None

    def _load_torch(self, *args, **kwargs):
        raise NotImplementedError()

    def load_test_train(self, path=None):
        data_dir = self.config.data_dir if path is None else path
        self.dataset = (
            self._load_torch(os.path.join(data_dir, "train.torch")),
            self._load_torch(os.path.join(data_dir, "dev.torch")),
        )

    def load(self, path=None):
        data_dir = self.config.data_dir if path is None else path
        self.dataset = self._load_torch(data_dir)

    def inference(self, batch_size=1):
        inference = self.dataset
        sampler = SequentialSampler(inference)
        dataloader = DataLoader(inference, sampler=sampler, batch_size=batch_size,)

        return dataloader

    def fitting(self, *args, **kwargs):
        raise NotImplementedError()

    def compare_dummy_classifier(self, *args, **kwargs):
        raise NotImplementedError()

    def predicting(self, batch_size=1):
        assert self.dataset != None

        _, test = self.dataset
        sampler = SequentialSampler(test)
        dataloader = DataLoader(test, sampler=sampler, batch_size=batch_size,)

        return dataloader


class DatasetForSurvival(BaseDataset):
    def fitting(self, batch_size=1, weighted=False):
        assert self.dataset != None

        train, _ = self.dataset
        normedWeights = None

        # Create sampler and dataloader
        sampler = SequentialSampler(train)
        dataloader = DataLoader(train, sampler=sampler, batch_size=batch_size,)

        return dataloader, normedWeights


class MutationDatasetForSurvival(DatasetForSurvival):
    def _load_torch(self, data_dir):

        cached_features_file = data_dir

        if os.path.exists(cached_features_file):
            _torch_data = torch.load(cached_features_file)
        else:
            raise ValueError("The file {} does not exist".format(cached_features_file))

        _input_ids = _torch_data[0].type(torch.int32)

        all_surv_E = _torch_data[6]
        all_T = _torch_data[7]
        all_surv_T, time_points = get_survival_y_true(
            all_T, all_surv_E, self.config.num_times
        )

        seq_features = _input_ids
        targets = SurvivalTensor((all_surv_E, all_T, all_surv_T, time_points))
        dataset = TensorDataset(seq_features, seq_features, targets)

        return dataset


class OmicDatasetForSurvival(DatasetForSurvival):
    def _load_torch(self, data_dir):

        cached_features_file = data_dir

        if os.path.exists(cached_features_file):
            _torch_data = torch.load(cached_features_file)
        else:
            raise ValueError("The file {} does not exist".format(cached_features_file))

        _input_ids = _torch_data[0].type(torch.int32)
        _expression = torch.tensor(
            np.digitize(
                _torch_data[3],
                bins=np.linspace(
                    self.config.digitize_min,
                    self.config.digitize_max,
                    self.config.digitize_bins,
                ),
            )
        )

        all_surv_E = _torch_data[6]
        all_T = _torch_data[7]
        all_surv_T, time_points = get_survival_y_true(
            all_T, all_surv_E, self.config.num_times
        )

        seq_features = _input_ids
        numeric_features = _expression
        targets = SurvivalTensor((all_surv_E, all_T, all_surv_T, time_points))
        dataset = TensorDataset(seq_features, numeric_features, targets)

        return dataset


class DatasetForClassification(BaseDataset):
    def fitting(self, batch_size=1, weighted=False):
        assert self.dataset != None

        train, _ = self.dataset
        target = train.tensors[2]

        class_sample_count = np.array(
            [len(np.where(target == t)[0]) for t in torch.unique(target)]
        )

        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[t] for t in target])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()

        if weighted:
            normedWeights = [
                1 - (x / np.sum(class_sample_count)) for x in class_sample_count
            ]
            normedWeights = torch.FloatTensor(normedWeights)
            normedWeights = normedWeights / normedWeights.sum()
            normedWeights = normedWeights.float().to(self.config.device)
        else:
            normedWeights = torch.FloatTensor([1 for x in class_sample_count]).to(
                self.config.device
            )

        # Create sampler and dataloader
        sampler = SequentialSampler(train)
        dataloader = DataLoader(train, sampler=sampler, batch_size=batch_size,)

        return dataloader, normedWeights

    def compare_dummy_classifier(self):
        self.dataset
        train, test = self.dataset

        Y_train = train[2]
        X_train = Y_train
        Y_test = test[2]
        X_test = Y_test

        for strategy in ["most_frequent", "prior", "uniform", "stratified"]:
            print(strategy)
            clas = DummyClassifier(strategy=strategy)
            clas.fit(X_train, Y_train)
            Y_pred = clas.predict(X_test)

            classification_metrics(Y_test, Y_pred)

    def filter_class(self, types):
        assert self.dataset is not None
        train, test = self.dataset

        for cohort in [train, test]:
            _i_cancertype = np.array(np.where(cohort == types)[0])
            cohort = (t[_i_cancertype] for t in cohort.tensors)

        return train, test


class OmicDatasetForClassification(DatasetForClassification):
    def _load_torch(self, data_dir):

        cached_features_file = data_dir

        if os.path.exists(cached_features_file):
            _torch_data = torch.load(cached_features_file)
        else:
            raise ValueError("The file {} does not exist".format(cached_features_file))

        _input_ids = _torch_data[0].type(torch.int32)
        _expression = torch.tensor(
            np.digitize(
                _torch_data[3],
                bins=np.linspace(
                    self.config.digitize_min,
                    self.config.digitize_max,
                    self.config.digitize_bins,
                ),
            )
        )
        seq_features = _input_ids
        numeric_features = _expression
        targets = _torch_data[4].type(torch.long)
        dataset = TensorDataset(seq_features, numeric_features, targets)

        return dataset


class MutationDatasetForClassification(DatasetForClassification):
    def _load_torch(self, data_dir):

        cached_features_file = data_dir

        if os.path.exists(cached_features_file):
            _torch_data = torch.load(cached_features_file)
        else:
            raise ValueError("The file {} does not exist".format(cached_features_file))

        seq_features = _torch_data[0].type(torch.int32)
        targets = _torch_data[4].type(torch.long)
        dataset = TensorDataset(seq_features, seq_features, targets)

        return dataset
