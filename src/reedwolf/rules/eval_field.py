"""
------------------------------------------------------------
Evaluations - used to automatically calculate
------------------------------------------------------------

Evaluation-s are used in cleaners.
Base is value:DotExpression which shuuld return value compatible with
Field type.

TODO: demo Evaluation - created/updated by/at: current user, timestamp

"""


from abc import ABC
from typing import (
        Union, 
        Optional,
        ClassVar,
        )
from dataclasses import (
        dataclass,
        field,
        )

from .exceptions import (
        RuleSetupTypeError,
        )
from .meta import (
        TransMessageType, 
        LiteralType,
        )
from .base import (
        IApplySession,
        )
from .eval_base import (
        EvaluationBase,
        )
from .expressions import (
        DotExpression, 
        ExecResult, 
        execute_available_dexp,
        )


# class InitEvaluationBase(FieldEvaluationBase, ABC):
#     """ called on object/instance creation, needs __init__ hook """
# class PresaveEvaluationBase(FieldEvaluationBase, ABC):
#     """ called after update, before save """

class FieldEvaluationBase(EvaluationBase, ABC):
    ...


# ------------------------------------------------------------
# FieldEvaluation - generic
# ------------------------------------------------------------

@dataclass
class FieldEvaluation(FieldEvaluationBase):
    """
    TODO: put usage - new custom evaluations could be done like this:
    """
    # TODO: check in Setup phase if type of evaluated DotExpression has correct
    #       type - e.g.  for EnumField evaluated value must be within enum values.
    value:          DotExpression
    name:           Optional[str] = field(default=None)
    title:          Optional[TransMessageType] = field(default=None)
    available:      Optional[Union[bool, DotExpression]] = field(default=True)

    def __post_init__(self):
        if not isinstance(self.value, DotExpression):
            raise RuleSetupTypeError(owner=self, msg=f"Argument 'value' needs to be DotExpression, got: {type(self.value)} / {self.value}")
        # self._fill_name_when_missing()
        super().__post_init__()


    def execute(self, apply_session: IApplySession) -> Optional[ExecResult]:
        not_available_dexp_result = execute_available_dexp(self.available, apply_session=apply_session)
        if not_available_dexp_result: 
            return not_available_dexp_result
        return self.value._evaluator.execute_dexp(apply_session=apply_session)

# ------------------------------------------------------------

@dataclass
class Default(FieldEvaluationBase):
    """ used for generated classes, dynamically created objects or SQL or other
        storage generated code
        for existing bound models - needs hook to ensure good value on object creation (__init__)
        in simmple cases - can be used to ensure good default value
        for EnumField should have default within enum values.
    """

    value:          Union[LiteralType, DotExpression]
    name:           Optional[str] = None
    title:          Optional[TransMessageType] = None
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

