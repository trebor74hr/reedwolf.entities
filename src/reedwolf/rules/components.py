from abc import (
        ABC, 
        abstractmethod,
        )
from typing import (
        Any,
        Union, 
        Optional, 
        ClassVar,
        )
from dataclasses import dataclass, field

from .utils import (
        UNDEFINED,
        UndefinedType,
        to_int,
        )
from .exceptions import (
        RuleSetupError,
        )
from .meta import (
        TransMessageType,
        NoneType,
        )
from .base import (
        BaseOnlyArgs,
        ComponentBase,
        IApplySession,
        ValidationFailure,
        )
from .expressions import (
        DotExpression,
        ExecResult,
        NotAvailableExecResult,
        execute_available_dexp,
        )
from .attr_nodes import (
        AttrDexpNode,
        )

# ------------------------------------------------------------
# Functions
# ------------------------------------------------------------

def _(message: str) -> TransMessageType:
    return message

# TODO: add type hint: TransMessageType -> TranslatedMessageType
# TODO: accept "{dot_node}" - can be a security issue, attr_nodes() should not make any logic
#       use .format() ... (not f"", btw. should not be possible anyway)

class msg(BaseOnlyArgs):
    pass



# ------------------------------------------------------------
# Clenaers (cleaners) == Validations OR Evaluations
# ------------------------------------------------------------


class ValidationBase(ComponentBase, ABC): # TODO: make it abstract
    """ Executes validate() method which checks all ok
    """
    def __post_init__(self):
        if hasattr(self, "ensure"):
            if not isinstance(self.ensure, DotExpression):
                raise RuleSetupError(owner=self, msg=f"ensure must be DotExpression, got: {type(self.ensure)} / {self.ensure}")

            if not self.error:
                self.error = f"Validation failed: {self.ensure}"

        if not self.error:
            title = getattr(self, "title", self.name)
            if title:
                self.error = f"Validation failed ({title})"
            else:
                self.error = "Validation failed"
        elif not self.title:
            self.title = self.error

        super().__post_init__()


    def _check_dot_expression_or_positive_int(self, attr_name:str, attr_value: Any):
        if not isinstance(attr_value, DotExpression) and not to_int(attr_value, 0) >= 0:
            raise RuleSetupError(owner=self, msg="Argument '{attr_name}' must be integer >= 0 or DotExpression, got: {attr_value}")

    def _validate_common_impl(self, apply_session: IApplySession) -> Union[NoneType, ValidationFailure]:
        not_available_dexp_result: NotAvailableExecResult  = execute_available_dexp(self.available, apply_session=apply_session)
        if not_available_dexp_result: 
            # TODO: log ...
            return None

        component = apply_session.current_frame.component
        dexp_result: ExecResult = self.ensure._evaluator.execute_dexp(apply_session)
        if not bool(dexp_result.value):
            error = self.error if self.error else "Validation failed"
            return ValidationFailure(
                            component_key_string = apply_session.get_key_string(component),
                            error=error, 
                            validation_name=self.name,
                            validation_title=self.title,
                            details=f"The validation returned '{dexp_result.value}'"
                            )
        return None


    @abstractmethod
    def validate(self, apply_session: IApplySession) -> Union[NoneType, ValidationFailure]:
        """ if all ok returns None, else returns ValidationFailure
        containing all required information about failure(s).
        """
        ...


class EvaluationBase(ComponentBase, ABC): # TODO: make it abstract
    """ Auto-compute logic - executes 'value' expression, stores into field of
        current instance/object. The execution should not fail.
    """
    REQUIRES_AUTOCOMPUTE: ClassVar[bool] = True

    @abstractmethod
    def execute(self, apply_session: IApplySession) -> Optional[ExecResult]:
        """
        returns value wrapped in ExecResult which will be used to update instance.attribute
        if returns None, update won't be done
        """
        ...


