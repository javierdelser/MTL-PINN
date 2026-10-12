from mtl_final.problems.burgers import BurgersProblem
from mtl_final.problems.fisher_kpp import FisherKPPProblem
from mtl_final.problems.heat import HeatProblem
from mtl_final.problems.lamb_chaplygin import LambChaplyginProblem
from mtl_final.problems.poisson import PoissonProblem

_REGISTRY = {
    "poisson": PoissonProblem,
    "heat": HeatProblem,
    "burgers": BurgersProblem,
    "fisher_kpp": FisherKPPProblem,
    "lamb_chaplygin": LambChaplyginProblem,
}


def get_problem(name: str, **kwargs):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown PDE '{name}'. Choose from {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


__all__ = ["get_problem"]
