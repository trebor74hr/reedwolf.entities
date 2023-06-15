# from __future__ import annotations

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
# COMPONENTS
# ------------------------------------------------------------

# dataclass required?
@dataclass
class Component(ComponentBase, ABC):
    # NOTE: I wanted to skip saving parent reference/object within component - to
    #       preserve single and one-direction references.
    parent       : Union[ComponentBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    parent_name  : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

    # set in setup()
    dot_node:   Union[AttrDexpNode, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self.init_clean_base()

    def is_subentity_items(self):
        return False

    def is_container(self):
        return False


# ------------------------------------------------------------
# Clenaers (cleaners) == Validations OR Evaluations
# ------------------------------------------------------------


class ValidationBase(Component, ABC): # TODO: make it abstract
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


class EvaluationBase(Component, ABC): # TODO: make it abstract
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


# ------------------------------------------------------------
# OBSOLETE
# ------------------------------------------------------------
#
# NOTE: obsolete - it was used for enums only - replaced with special functiohn factory function: functions=[EnumMembers()]
# 
# @dataclass
# class StaticData(IData, Component):
#     # TODO: Data - maybe should not be component at all?
#     name:           str
#     # TODO: ModelField or Enum or ...
#     value:          Any 
#     title:          Optional[TransMessageType] = field(repr=False, default=None)
#     type_info:      TypeInfo = field(init=False)
#     # evaluate:       bool = False # TODO: describe
# 
#     def __post_init__(self):
#         if isinstance(self.value, DotExpression):
#             # TODO: should be model or enum
#             # raise RuleSetupValueError(owner=self, msg=f"{self.name} -> {type(self.value)}: {type(self.value)} - to be done a DotExpression")
#             raise NotImplementedError(f"{self.name} -> {type(self.value)}: {type(self.value)} - to be done a DotExpression")
#         self.type_info = TypeInfo.get_or_create_by_type(self.value)
#         super().__post_init__()

# # ------------------------------------------------------------
#
# NOTE: obsolete - replaced with simple functions=[Function()]
# 
# @dataclass
# class DynamicData(IData, Component):
#     # TODO: Data - maybe should not be component at all?
#     """
#     Function is invoked which returns data.
#     See functions.py for details.
#     """
#     name:           str
#     # NOTE: the name function is exposed and Function() is used, but internally
#     #       this will be function_factory instance
#     function:       CustomFunctionFactory
#     title:          Optional[TransMessageType] = field(repr=False, default=None)
#     type_info:      TypeInfo = field(init=False)
#     # evaluate:       bool = False # TODO: describe
# 
#     def __post_init__(self):
#         if not isinstance(self.function, CustomFunctionFactory):
#             raise RuleSetupValueError(owner=self, msg=f"{self.function}: {type(self.function)} - not DotExpression|Function()")
#         # self.type_info = TypeInfo.extract_function_return_type_info(self.function)
#         custon_function_factory: CustomFunctionFactory = self.function
#         self.type_info = custon_function_factory.get_type_info()
#         # TODO: check function output is Callable[..., RuleDatatype]
#         if not self.title:
#             self.title = varname_to_title(self.name)
#         super().__post_init__()

