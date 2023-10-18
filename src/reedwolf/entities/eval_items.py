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

    def execute(self, apply_result: IApplyResult) -> Optional[ExecResult]:
        raise NotImplementedError()
