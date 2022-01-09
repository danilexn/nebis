from .setquence import SetQuence, SetQuenceConsensus
from .setomic import SetOmic, SetOmicConsensus
from .base import Base, parallel_fit, profiled_parallel_fit

_models = {
    "setquence": SetQuence,
    "setomic": SetOmic,
    "setquenceconsensus": SetQuenceConsensus,
    "setomicconsensus": SetOmicConsensus,
}


def get_model(name):
    try:
        return _models[name]
    except:
        raise ValueError("Could not find model {}".format(name))


def list_models():
    return list(_models.keys())
