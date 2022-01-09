from .dataset import (
    OmicDatasetForClassification,
    MutationDatasetForClassification,
    OmicDatasetForSurvival,
    MutationDatasetForSurvival,
)


_datareader_dict = {
    "setquenceconsensus_classification": MutationDatasetForClassification,
    "setquenceconsensus_survival": MutationDatasetForSurvival,
    "setquence_classification": MutationDatasetForClassification,
    "setquence_survival": MutationDatasetForSurvival,
    "setomic_classification": OmicDatasetForClassification,
    "setomic_survival": OmicDatasetForSurvival,
    "setomicconsensus_classification": OmicDatasetForClassification,
    "setomicconsensus_survival": OmicDatasetForSurvival,
}


def list_datareaders():
    return list(_datareader_dict.keys())


def get_datareader(name):
    try:
        if type(name) is str:
            return _datareader_dict[name]
        else:
            return name
    except:
        raise ValueError("Could not retrieve downstream method '{}'".format(name))
