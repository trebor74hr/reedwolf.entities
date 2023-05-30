"""
Evaluation-s are used in cleaners.
Base is value:DotExpression which shuuld return value compatible with
Field type.

TODO: demo Evaluation - created/updated by/at: current user, timestamp

"""

# ------------------------------------------------------------
# Evaluations - used to automatically calculate
# ------------------------------------------------------------
from __future__ import annotations

from abc import ABC
from typing import (
        Union, 
        Optional,
        ClassVar,
        )
from dataclasses import dataclass

from .meta import (
        TransMessageType, 
        LiteralType,
        )
from .base import (
        IApplySession,
        )
from .components  import (
        EvaluationBase,
        )
from .expressions import (
        DotExpression, 
        ExecResult, 
        execute_available_dexp,
        )

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
    # TODO: check in Setup phase if type of evaluated DotExpression has correct
    #       type - e.g.  for EnumField evaluated value must be within enum values.
    name:           str
    value:          DotExpression
    label:          Optional[TransMessageType] = None
    available:      Optional[Union[bool, DotExpression]] = True

    def __post_init__(self):
        assert isinstance(self.value, DotExpression), self.value

    def execute(self, apply_session: IApplySession) -> Optional[ExecResult]:
        not_available_dexp_result = execute_available_dexp(self.available, apply_session=apply_session)
        if not_available_dexp_result: 
            return not_available_dexp_result
        return self.value._evaluator.execute_dexp(apply_session=apply_session)

# ------------------------------------------------------------
# Evaluations on object/instance initialization
# ------------------------------------------------------------

class InitEvaluationBase(EvaluationBase, ABC):
    """ called on object/instance creation, needs __init__ hook
    """
    pass


@dataclass
class Default(PresaveEvaluationBase):
    # TODO: make InitEvaluationBase version too
    """ used for generated classes, dynamically created objects or SQL or other
        storage generated code
        for existing bound models - needs hook to ensure good value on object creation (__init__)
        in simmple cases - can be used to ensure good default value
        for EnumField should have default within enum values.
    """

    value:          Union[LiteralType, DotExpression]
    name:           Optional[str] = None
    label:          Optional[TransMessageType] = None
    available:      Optional[Union[bool, DotExpression]] = True

    REQUIRES_AUTOCOMPUTE: ClassVar[bool] = False

    def __post_init__(self):
        # assert isinstance(self.value, DotExpression), self.value
        pass

    def execute(self, apply_session: IApplySession) -> Optional[ExecResult]:
        if isinstance(self.value, DotExpression):
            dexp_result = self.value._evaluator.execute_dexp(apply_session=apply_session)
            value = dexp_result.value
        else:
            value = self.value
        return ExecResult.create(value)

