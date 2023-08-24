"""
ChildrenEvaluations - used to automatically update some Children fields
In SubEntityItems case, this will be running against every SubEntity in list.
"""
from abc import ABC
from typing import (
        Optional,
        Union,
        )
from dataclasses import (
        dataclass,
        field,
        )

from .meta import (
        TransMessageType, 
        )
from .expressions import (
        DotExpression,
        )
from .base import (
    IApplyResult,
    ExecResult,
        )
from .eval_base import (
        EvaluationBase,
        )


class ChildrenEvaluationBase(EvaluationBase, ABC):
    ...


@dataclass
class ChildrenEvaluation(ChildrenEvaluationBase):
    ensure:         DotExpression
    name:           Optional[str] = field(default=None)
    error:          Optional[TransMessageType] = field(repr=False, default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    title:          Optional[TransMessageType] = field(repr=False, default=None)

    def execute(self, apply_result: IApplyResult) -> Optional[ExecResult]:
        raise NotImplementedError()
