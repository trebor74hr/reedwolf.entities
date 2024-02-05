from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Optional,
    ClassVar, Union,
)
from dataclasses import dataclass, field

from .base import (
    IApplyResult,
    IEvaluation,
)
from .exceptions import EntitySetupValueError
from .expressions import (
    ExecResult,
    ISetupSession,
    IThisRegistry,
    clean_available,
    DotExpression,
    DEXP_VALIDATOR_NOT_MODELS,
)
from .meta import TransMessageType, STANDARD_TYPE_LIST, LiteralType
from .utils import to_repr


# ------------------------------------------------------------
# Cleaners == Validations OR Evaluations
# ------------------------------------------------------------


@dataclass
class EvaluationBase(IEvaluation, ABC):
    """ Auto-compute logic - executes 'value' expression, stores into field of
        current instance/object. The execution should not fail.
    """
    REQUIRES_AUTOCOMPUTE: ClassVar[bool] = field(default=True)

    value:          Union[LiteralType, DotExpression]
    name:           Optional[str] = field(default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    title:          Optional[TransMessageType] = field(repr=False, default=None)

    def __post_init__(self):
        if hasattr(self, "available"):
            clean_available(owner=self, attr_name="available", dexp_or_bool=self.available)

        if isinstance(self.value, DotExpression):
            self.value._SetDexpValidator(DEXP_VALIDATOR_NOT_MODELS)
        elif not isinstance(self.value, STANDARD_TYPE_LIST):
            raise EntitySetupValueError(owner=self,
                                        msg=f"Argument 'value' needs to DotExpression (e.g. F.name != '') "
                                            f"or some standard type ({STANDARD_TYPE_LIST}), got: {to_repr(self.value)}")

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
