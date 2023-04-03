"""
Evaluation-s are used in cleaners.
Base is value:ValueExpression which shuuld return value compatible with
Field type.

TODO: demo Evaluation - created/updated by/at: current user, timestamp

"""

# ------------------------------------------------------------
# Evaluations - used to automatically calculate
# ------------------------------------------------------------
from __future__ import annotations

from abc import ABC
from typing import Union, Optional
from dataclasses import dataclass

from .meta          import TransMessageType
from .components    import EvaluationBase
from .expressions   import ValueExpression, VexpResult, evaluate_available_vexp


# ------------------------------------------------------------
# Evaluation - generic
# ------------------------------------------------------------
# TODO: solve this with Flag-type Enum e.g.
#       enum EvalType(enum.Enum):
#           # ALL = all 1
#           SKIP_2ND_PASS = 2

class PresaveEvaluationBase(EvaluationBase, ABC):
    pass

@dataclass
class Evaluation(PresaveEvaluationBase):
    """
    TODO: put usage - new custom evaluations could be done like this:
    """
    # TODO: check in Setup phase if type of evaluated VExpression has correct
    #       type - e.g.  EnumField should have default within enum values.
    name:           str
    label:          TransMessageType
    value:          ValueExpression
    available:      Optional[Union[bool, ValueExpression]] = True

    def __post_init__(self):
        assert isinstance(self.value, ValueExpression), self.value

    def evaluate(self, apply_session: IApplySession) -> Optional[VexpResult]:
        not_available_vexp_result = evaluate_available_vexp(self.available, apply_session=apply_session)
        if not_available_vexp_result: 
            return not_available_vexp_result
        return self.value._evaluator.evaluate(apply_session=apply_session)

# ------------------------------------------------------------
# Evaluations on object/instance initialization
# ------------------------------------------------------------

class InitEvaluationBase(EvaluationBase, ABC):
    """ called on object/instance creation, needs __init__ hook
    """
    pass


class Default(InitEvaluationBase):
    """ used for generated classes, dynamically created objects or SQL or other
        storage generated code
        for existing bound models - needs hook to ensure good value on object creation (__init__)
        in simmple cases - can be used to ensure good default value
    """
    value:          ValueExpression
    name:           Optional[str]
    label:          Optional[TransMessageType]
    available:      Optional[Union[bool, ValueExpression]] = True


    def __post_init__(self):
        assert isinstance(self.value, ValueExpression), self.value


    def evaluate(self, apply_session: IApplySession) -> Optional[VexpResult]:
        return self.value._evaluator.evaluate(apply_session=apply_session)
