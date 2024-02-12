from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    Union,
    Optional,
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
    IApplyResult,
    ValidationFailure,
    IValidation,
)
from .expressions import (
    DotExpression,
    ExecResult,
    NotAvailableExecResult,
    execute_available_dexp,
    ISetupSession,
    IThisRegistry,
    clean_available,
    clean_dexp_bool_term,
)



# ------------------------------------------------------------
# Clenaers (cleaners) == Validations OR Evaluations
# ------------------------------------------------------------


@dataclass
class ValidationBase(IValidation, ABC):
    """ Executes validate() method which checks all ok
    """

    def init(self):
        if hasattr(self, "available"):
             clean_available(owner=self, attr_name="available", dexp_or_bool=self.available)

        if hasattr(self, "ensure"):
            if not isinstance(self.ensure, DotExpression):
                raise EntitySetupError(owner=self, msg=f"ensure must be DotExpression, got: {type(self.ensure)} / {self.ensure}")
            clean_dexp_bool_term(owner=self, attr_name="ensure", dexp=self.ensure)

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

        super().init()


    # def __repr__(self):
    #     # due multiple inheritance - to force calling IComponent implementation
    #     return super().__repr__()

    def create_this_registry(self, setup_session: ISetupSession) -> Optional[IThisRegistry]:
        """
        return None -> inherit this_registry of the calling component.
        """
        return None

    def _check_dot_expression_or_positive_int(self, attr_name:str, attr_value: Any):
        if not isinstance(attr_value, DotExpression) and not to_int(attr_value, 0) >= 0:
            raise EntitySetupError(owner=self, msg=f"Argument '{attr_name}' must be integer >= 0 or DotExpression, got: {attr_value}")

    def _validate_common_impl(self, apply_result: IApplyResult) -> Union[NoneType, ValidationFailure]:
        not_available_dexp_result: NotAvailableExecResult  = execute_available_dexp(self.available, apply_result=apply_result)
        if not_available_dexp_result: 
            # TODO: log ...
            return None

        component = apply_result.current_frame.component
        dexp_result: ExecResult = self.ensure._evaluator.execute_dexp(apply_result)
        if not bool(dexp_result.value):
            error = self.error if self.error else "Validation failed"
            return ValidationFailure(
                            component_key_string = apply_result.get_key_string(component),
                            error=error, 
                            validation_name=self.name,
                            validation_title=self.title,
                            details=f"The validation returned '{dexp_result.value}'"
                            )
        return None


    @abstractmethod
    def validate(self, apply_result: IApplyResult) -> Union[NoneType, ValidationFailure]:
        """ if all ok returns None, else returns ValidationFailure
        containing all required information about failure(s).
        """
        ...
