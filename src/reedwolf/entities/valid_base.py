from abc import (
        ABC, 
        abstractmethod,
        )
from typing import (
        Any,
        Union, 
        )
from dataclasses import dataclass

from .utils import (
        to_int,
        )
from .exceptions import (
        EntitySetupError,
        )
from .meta import (
        NoneType,
        )
from .base import (
    ComponentBase,
    IApplyResult,
    ValidationFailure,
        )
from .expressions import (
        DotExpression,
        ExecResult,
        NotAvailableExecResult,
        execute_available_dexp,
        )



# ------------------------------------------------------------
# Clenaers (cleaners) == Validations OR Evaluations
# ------------------------------------------------------------


@dataclass
class ValidationBase(ComponentBase, ABC): # TODO: make it abstract
    """ Executes validate() method which checks all ok
    """

    def __post_init__(self):
        if hasattr(self, "ensure"):
            if not isinstance(self.ensure, DotExpression):
                raise EntitySetupError(owner=self, msg=f"ensure must be DotExpression, got: {type(self.ensure)} / {self.ensure}")

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


    # def __repr__(self):
    #     # due multiple inheritance - to force calling ComponentBase implementation
    #     return super().__repr__()


    def _check_dot_expression_or_positive_int(self, attr_name:str, attr_value: Any):
        if not isinstance(attr_value, DotExpression) and not to_int(attr_value, 0) >= 0:
            raise EntitySetupError(owner=self, msg="Argument '{attr_name}' must be integer >= 0 or DotExpression, got: {attr_value}")

    def _validate_common_impl(self, apply_session: IApplyResult) -> Union[NoneType, ValidationFailure]:
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
    def validate(self, apply_session: IApplyResult) -> Union[NoneType, ValidationFailure]:
        """ if all ok returns None, else returns ValidationFailure
        containing all required information about failure(s).
        """
        ...
