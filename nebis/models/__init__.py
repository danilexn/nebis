from .setquence import SetQuence
from .setomic import SetOmic
from .base import Base

_models = {"setquence": SetQuence, "setomic": SetOmic}


def get_model(name):
    try:
        return _models[name]
    except:
        raise ValueError("Could not find model {}".format(name))


def list_models():
    return list(_models.keys())
