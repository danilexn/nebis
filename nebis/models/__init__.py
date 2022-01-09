from .setquence import SetQuence, SetQuenceConsensus
from .setomic import SetOmic
from .base import Base

_models = {
    "setquence": SetQuence,
    "setomic": SetOmic,
    "setquenceconsensus": SetQuenceConsensus,
}


def get_model(name):
    try:
        return _models[name]
    except:
        raise ValueError("Could not find model {}".format(name))


def list_models():
    return list(_models.keys())
