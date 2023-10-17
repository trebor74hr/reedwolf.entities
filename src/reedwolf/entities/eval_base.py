from abc import (
        ABC, 
        abstractmethod,
        )
from typing import (
        Optional, 
        ClassVar,
        )
from dataclasses import dataclass, field

from .base import (
    ComponentBase,
    IApplyResult,
        )
from .expressions import (
    ExecResult, ISetupSession, IThisRegistry, clean_available,
)


# ------------------------------------------------------------
# Cleaners == Validations OR Evaluations
# ------------------------------------------------------------


@dataclass
class EvaluationBase(ComponentBase, ABC):  # TODO: make it abstract
    """ Auto-compute logic - executes 'value' expression, stores into field of
        current instance/object. The execution should not fail.
    """
    REQUIRES_AUTOCOMPUTE: ClassVar[bool] = field(default=True)

    def __post_init__(self):
        if hasattr(self, "available"):
            clean_available(owner=self, attr_name="available", dexp_or_bool=self.available)


    def create_this_registry(self, setup_session: ISetupSession) -> Optional[IThisRegistry]:
        """
        return None -> inherit this_registry of the calling component.
        """
        return None

    @abstractmethod
    def execute(self, apply_result: IApplyResult) -> Optional[ExecResult]:
        """
        returns value wrapped in ExecResult which will be used to update instance.<attribute>
        if returns None, update won't be done
        """
        ...
