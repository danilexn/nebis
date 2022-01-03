import os
import numpy as np

from sklearn.dummy import DummyClassifier

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import (
    DataLoader,
    SequentialSampler,
    TensorDataset,
    WeightedRandomSampler,
)

from bideset.utils.metrics import classification_metrics, classification_roc_auc

class BaseDataset():
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.dataset = None
    
    def training(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()


class OmicDatasetForClassification(BaseDataset):
    def _load_torch(self, evaluate=True):
        data_dir = self.config.data_dir

        # TODO: check if in directory, download
        cached_features_file = (
            os.path.join(data_dir, "dev.torch")
            if evaluate
            else os.path.join(data_dir, "train.torch")
        )

        if os.path.exists(cached_features_file):
            features = torch.load(cached_features_file)
        else:
            raise ValueError("The file {} does not exist".format(cached_features_file))

        all_input_ids = features[0].type(torch.int32)
        all_expression = torch.tensor(
            np.digitize(
                features[3], bins=np.linspace(self.config.digitize_min, self.config.digitize_max, self.config.digitize_bins)
            )
        )
        all_labels = features[4].type(torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_expression,
            all_labels,
            names=('input_mutome', 'input_expression', 'labels')
        )

        return dataset

    def load(self):
        # In a tuple (train, test)
        self.dataset = (self._load(False), self._load(True))

        return self.dataset

    def training(self, batch_size=1, weighted=True):
        assert self.dataset != None
        
        train, _ = self.dataset
        target = train.tensors['labels']

        class_sample_count = np.array(
            [len(np.where(target == t)[0]) for t in torch.unique(target)]
        )

        normedWeights = [1 - (x / np.sum(class_sample_count)) for x in class_sample_count]
        normedWeights = torch.FloatTensor(normedWeights)
        normedWeights = normedWeights / normedWeights.sum()
        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[t] for t in target])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()

        if weighted:
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            sampler = SequentialSampler(train)

        dataloader = DataLoader(
            train,
            sampler=sampler,
            batch_size=batch_size,
        )

        return train, dataloader

    def predicting(self, batch_size=1):
        assert self.dataset != None
        
        _, test = self.dataset
        sampler = SequentialSampler(test)
        dataloader = DataLoader(
            test,
            sampler=sampler,
            batch_size=batch_size,
        )

        return test, dataloader

    def compare_dummy_classifier(self):
        self.dataset
        train, test = self.dataset

        Y_train = train['label']
        X_train = Y_train
        Y_test = test['label']
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