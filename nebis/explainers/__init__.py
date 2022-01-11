from .setomic import SetOmicExplainer

_explainers = {
    "setomic": SetOmicExplainer,
}


def get_explainer(name):
    try:
        return _explainers[name]
    except:
        raise ValueError("Could not find explainer model {}".format(name))


def list_explainers():
    return list(_explainers.keys())
