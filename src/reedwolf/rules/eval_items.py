"""
ItemsEvaluations - used to automatically create/update/delete Items used in
SubEntityItems
"""
from __future__ import annotations

from abc import ABC
from typing import (
        Optional,
        )
from dataclasses import (
        dataclass,
        field,
        )

from .meta import (
        TransMessageType, 
        )
from .components  import (
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
    label:          Optional[TransMessageType] = field(repr=False, default=None)

    def execute(self, apply_session: IApplySession) -> Optional[ExecResult]:
        raise NotImplementedError()
