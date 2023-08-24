"""
ItemsEvaluations - used to automatically create/update/delete Items used in
SubEntityItems
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


class ItemsEvaluationBase(EvaluationBase, ABC):
    ...


@dataclass
class ItemsEvaluation(ItemsEvaluationBase):
    ensure:         DotExpression
    name:           Optional[str] = field(default=None)
    error:          Optional[TransMessageType] = field(repr=False, default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    title:          Optional[TransMessageType] = field(repr=False, default=None)

    def execute(self, apply_session: IApplyResult) -> Optional[ExecResult]:
        raise NotImplementedError()
