from __future__ import annotations

from abc import (
        ABC, 
        abstractmethod,
        )
from typing import (
        Union, 
        List, 
        Optional, 
        Any,
        )
from dataclasses import dataclass, field

from .utils import (
        UNDEFINED,
        UndefinedType,
        )
from .exceptions import (
        RuleSetupValueError,
        )
from .meta import (
        TransMessageType,
        TypeInfo,
        )
from .base import (
        BaseOnlyArgs,
        ComponentBase,
        IData,
        IApplySession,
        GlobalConfig,
        ValidationFailure,
        )
from .expressions import (
        ValueExpression,
        )
from .attr_nodes import (
        AttrVexpNode,
        )
from .functions import (
        CustomFunctionFactory,
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

    # by default name should be defined
    # name: str = field(init=False, default=UNDEFINED)

    # NOTE: I wanted to skip saving owner reference/object within component - to
    #       preserve single and one-direction references.
    owner:      Union[ComponentBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    owner_name: Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

    # set in setup()
    dot_node:   Union[AttrVexpNode, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)

    def init_clean_base(self):
        # when not set then will be later defined - see set_owner()
        if self.name not in (None, "", UNDEFINED):
            if not self.name.isidentifier():
                raise RuleSetupValueError(owner=self, msg=f"{self.name} -> attribute name needs to be valid python identifier name")

    def __post_init__(self):
        self.init_clean_base()
        super().__post_init__()

    def get_key_string(self, apply_session: IApplySession):
        # TODO: is caching possible? 
        assert not self.is_container()
        container = self.get_container_owner()
        container_key_string = container.get_key_string(apply_session)
        key_string = GlobalConfig.ID_NAME_SEPARATOR.join(
                [container_key_string, self.name] 
                )
        # print("comp:", self.name, key_string)
        return key_string
        
    def is_extension(self):
        return False

    def is_container(self):
        return False


# ------------------------------------------------------------

# TODO: consider ExpressionData - to be based on ValueExpression

@dataclass
class StaticData(IData, Component):
    # TODO: Data - maybe should not be component at all?
    name:           str
    label:          TransMessageType
    # TODO: ModelField or Enum or ...
    value:          Any 
    type_info:      TypeInfo = field(init=False)
    # evaluate:       bool = False # TODO: describe

    def __post_init__(self):
        if isinstance(self.value, ValueExpression):
            print("TODO: should be model or enum")
            # raise RuleSetupValueError(owner=self, msg=f"{self.name} -> {type(self.value)}: {type(self.value)} - to be done a ValueExpression")
            raise NotImplementedError(f"{self.name} -> {type(self.value)}: {type(self.value)} - to be done a ValueExpression")
        self.type_info = TypeInfo.get_or_create_by_type(self.value)
        super().__post_init__()

# ------------------------------------------------------------

@dataclass
class DynamicData(IData, Component):
    # TODO: Data - maybe should not be component at all?
    """
    Function is invoked which returns data.
    See functions.py for details.
    """
    name:           str
    label:          TransMessageType
    # NOTE: the name function is exposed and Function() is used, but internally
    #       this will be function_factory instance
    function:       CustomFunctionFactory
    type_info:      TypeInfo = field(init=False)
    # evaluate:       bool = False # TODO: describe

    def __post_init__(self):
        if not isinstance(self.function, CustomFunctionFactory):
            raise RuleSetupValueError(owner=self, msg=f"{self.name} -> {self.function}: {type(self.function)} - not ValueExpression|Function()")
        # self.type_info = TypeInfo.extract_function_return_type_info(self.function)
        custon_function_factory: CustomFunctionFactory = self.function
        self.type_info = custon_function_factory.get_output_type_info()
        # TODO: check function output is Callable[..., RuleDatatype]
        super().__post_init__()

# ------------------------------------------------------------
# Clenaers (cleaners) == Validations OR Evaluations
# ------------------------------------------------------------


class ValidationBase(Component, ABC): # TODO: make it abstract
    """ Executes validate() method which checks all ok
    """

    def __post_init__(self):
        if not self.label:
            self.label = self.error
        super().__post_init__()


    @abstractmethod
    def validate(self, apply_session: IApplySession) -> Optional[ValidationFailure]:
        """ if all ok returns None, else returns ValidationFailure
        containing all required information about failure(s).
        """
        ...


class EvaluationBase(Component, ABC): # TODO: make it abstract
    """ Auto-compute logic - executes 'value' expression, stores into field of
        current instance/object. The execution should not fail.
    """

    @abstractmethod
    def evaluate(self, apply_session: IApplySession) -> Optional[VexpResult]:
        """
        returns value wrapped in VexpResult which will be used to update instance.attribute
        if returns None, update won't be done
        """
        ...



# ------------------------------------------------------------------------
# FieldGroups - logical groupihg and common functionality/dependency of other
#            components.
# ------------------------------------------------------------------------

@dataclass
class FieldGroup(Component):
    name:           str
    label:          str
    contains:       List[Component] = field(repr=False)
    # TODO: allow ChildrenValidation too (currently only used in Extension, e.g. Cardinality/Unique)
    cleaners:       Optional[List[Union[ValidationBase, EvaluationBase]]] = None
    available:      Union[bool, ValueExpression] = True

    @staticmethod
    def can_apply_partial() -> bool:
        return True
