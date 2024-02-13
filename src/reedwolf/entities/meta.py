"""
TypeInfo - the most interesting class
extract_* - the most interesting functions 

"""
import inspect
from collections import OrderedDict
from inspect import getmro, isclass,  getmembers, signature, Parameter
from abc import abstractmethod
from copy import copy
from collections.abc import Sequence
from functools import partial
from collections.abc import Sized
from typing import (
    Any,
    NewType,
    ClassVar,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    get_type_hints,
    TypeVar,
    Sequence as SequenceType,
    Iterable,
)
from enum import Enum
from decimal import Decimal
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from dataclasses import (
    is_dataclass,
    dataclass,
    Field as DcField,
    field,
    fields as dc_fields,
    make_dataclass,
)

MAX_RECURSIONS: int = 30

try:
    from typing import Self
except ImportError:
    # TODO: consider using typing_extensions for python version < 3.11
    Self = NewType("Self", Any)

try:
    # ----------------------------------------
    # Pydantic available
    # ----------------------------------------
    # imported and used in other modules - e.g. dropthis
    from pydantic import BaseModel as PydBaseModel
    from pydantic.main import ModelMetaclass as PydModelMetaclass
    # not used here directly, used in other modules
    from pydantic.fields import ModelField as PydModelField
    PydModelFieldType = PydModelField
except ImportError:
    PydBaseModel = None
    PydModelField = None

    class __PydanticNotPresent:
        pass

    PydModelMetaclass = __PydanticNotPresent
    # for typing
    PydModelFieldType = DcField

# ------------------------------------------------------------

from .utils import (
    UNDEFINED,
    format_arg_name_list,
    to_repr,
)
from .exceptions import (
    EntityTypeError,
    EntityInternalError,
)
from .namespaces import (
    DynamicAttrsBase,
)

# ------------------------------------------------------------
# SPECIAL DATATYPES
# ------------------------------------------------------------
NoneType                = type(None) # or None.__class__

# TODO: to use NewType or TypeVar it is the question.
#
#           - TypeVar is used when type can be different and in concrete case
#             to match used type (type variable). Example:
#
#               NumT = TypeVar("NumT", bound=Union[int, float])
#               def max_num(num_list: List[NumT]) -> NumT:
#                   ...
#               max_num([1,2,3])
#
#             used only in declaration, no constructor.
#             list of underlying types is in __constraints__ 
#             OR __bound__ when "bound=" is used.
#
#           - NewType() provides constructor (with bad repr) to mark values
#             with new type name. constructor does not add any meta-information
#             to produced instance. Example:
#               NumberType = NewType("NumberType", Union[int, float])
#               def max_num(num_list: List[NumberType]) -> NumberType:
#                   ...
#               max_num([NumberType(1), NumberType(2)])
#             can be used in declaration and in instance
#             construction. underlying type is in __supertype__
#
#           - subclass base type e.g. class Attribute(str): pass
#
#       MyPy will have the last word.

# NOTE: for dataclass there is no base type, so using Any
DataclassType = TypeVar("DataclassType", bound=Any)

if PydBaseModel:
    ModelKlassType  = Union[DataclassType, Type[PydBaseModel]]
    ModelField = Union[DcField, PydModelField]
else:
    ModelKlassType  = DataclassType
    ModelField = DcField

# instance of ModelKlassType
ModelInstanceType  = Any

STANDARD_TYPE_LIST      = (str, int, float, bool, Decimal, date, datetime, timedelta, time)
STANDARD_TYPE_W_NONE_LIST = (NoneType,) + STANDARD_TYPE_LIST 
StandardType            = Union[str, int, float, bool, Decimal, date, datetime, timedelta, time]

NUMBER_TYPES            = (int, float, Decimal)
NumberType              = Union[int, float, Decimal]

# TODO: consider maybe to replace this with AttrValue.
LiteralType             = TypeVar("LiteralType", bound=Any)

ERR_MSG_SUPPORTED = "Supporting custom and standard python types, and typing: Optional, Union[..., NoneType] and Sequence/List[ py-types | Union[py-types, NoneType]]."

# -- Following are used just to have type declaration for names
AttrName = TypeVar("AttrName", bound=str)
AttrValue = TypeVar("AttrValue", bound=Any)
AttrIndex = TypeVar("AttrIndex", bound=int)

SELF_ARG_NAME = "self"

# used in Container.container_id and Entity.containers
ContainerId = TypeVar("ContainerId", bound=str)

# ------------------------------------------------------------
# functions.py
# ------------------------------------------------------------
ValueArgValidatorPyFuncType = Callable[..., NoneType]
ValueArgValidatorPyFuncDictType = Dict[str, Union[ValueArgValidatorPyFuncType, List[ValueArgValidatorPyFuncType]]]

# ------------------------------------------------------------

# it is not dict since it must be immutable - default value for the class variable

class IFuncArgHint:
    """
    TODO: make CustomGenericAlias, check readme.txt - search
        custom type hints - IFuncArgHint - can be done:
      ref:
        https://peps.python.org/pep-0560/#specification
        https://docs.python.org/3/reference/datamodel.html#emulating-generic-types
    # if type is DotExpression -> inner_type which evaluation of DotExpression should return
    """

    def setup_check(self, setup_session: "ISetupSession", caller: Optional["IDotExpressionNode"], func_arg: "FuncArg"):
        # TODO: need to do this - currently Attrname makes problem only ...
        #       if so - then remove redundant logic (Dotexpr ...) and call super() in implementations
        #           exp_type_info = TypeInfo.get_or_create_by_type(self.type)
        #           err_msg = func_arg.type_info.check_compatible(exp_type_info)
        #           if err_msg:
        #               raise EntityTypeError(owner=self, msg=f"Function argument {func_arg} type not compatible: {err_msg}")
        ...

    @abstractmethod
    def get_type(self) -> Type:
        ...

    @abstractmethod
    def get_inner_type(self) -> Optional[Type]:
        ...

    @abstractmethod
    def __hash__(self):
        """
        must implement hash since it will be stored in TypeInfo.TYPE_INFO_REGISTRY
        """
        ...


class IExecuteFuncArgHint(IFuncArgHint):

    @abstractmethod
    def get_apply_value(self, apply_result: "IApplyResult",
                        exp_arg: "PrepArg",
                        arg_value: AttrValue,
                        prev_node_type_info: "TypeInfo",
                        ) -> AttrValue:
        ...


class IInjectFuncArgHint(IFuncArgHint):
    """
    will evaluate value and return to the caller
    """
    @abstractmethod
    def setup_check(self, setup_session: "ISetupSession", caller: Optional["IDotExpressionNode"], func_arg: "FuncArg"):
        ...

    @abstractmethod
    def get_apply_inject_value(self, apply_result: "IApplyResult", prep_arg: "PrepArg") -> AttrValue:
        ...


# e.g. list, int, dict, Person, List, Dict[str, Optional[Union[str, float]]
PyTypeHint                 = TypeVar("PyTypeHint", bound=Union[Type, IFuncArgHint])

RuleDatatype               = TypeVar("RuleDatatype", bound=Union[StandardType, List[StandardType,], Dict[str, StandardType]])

FunctionArgumentsTupleType = TypeVar("FunctionArgumentsTupleType", bound=Tuple[List[Any], Dict[str, Any]])

HookOnFinishedAllCallable = TypeVar("HookOnFinishedAllCallable", bound=Callable[[], None])

# ------------ NewType-s ------------

# A good example of NewType use-case, will require construction e.g. _("test")
TransMessageType        = NewType("TransMessageType", str)
_ = TransMessageType

KeyPairs = NewType("KeyPairs", SequenceType[Tuple[str, Any]])
Index0Type = int
KeyType = Union[KeyPairs, Index0Type]

InstanceId = NewType("InstanceId", int)

KeyString = NewType("KeyString", str)

# used for matching rows types, e.g. std_types map, filter etc.
ItemType = TypeVar("ItemType", bound=Any)

# TODO: consider using this for detection of .Items / .Children
#   ItemSetType = TypeVar("ItemSetType", bound=SequenceType[ItemType])

ComponentNameType = NewType("ComponentNameType", str)

# ------------------------------------------------------------
# Commonly/Internally used type symbols
# ------------------------------------------------------------

# Currently find no way on describing complex Dict-s
TreeNode = NewType("TreeNode", Any)

# TreeNode::
#   name: str self.name
#   component: IComponent
#   children: List[Self]
ComponentTreeType = NewType("ComponentTreeType", Dict[ComponentNameType, TreeNode])

# TreeNode::
#   name: str self.name
#   component: IComponent
#   children: List[Self]
#   attr_current_value_instance: ??
ComponentTreeWValuesType = NewType("ComponentTreeWValuesType", Dict[ComponentNameType, TreeNode])

# TreeNode::
#   name: str self.name
#   Optional[value: LiteralValue]
#   Optional[contains: List[TreeNode]]
ValuesTree = NewType("ValuesTree", Dict[ComponentNameType, TreeNode])

# TreeNode::
#   name: str self.name
#   Optional[contains: List[TreeNode]]
MetaTree = NewType("MetaTree", Dict[ComponentNameType, TreeNode])


# ------------------------------------------------------------

@dataclass
class FunctionArgumentsType:
    # ex. FunctionArgumentsType = Tuple[List[Any], Dict[str, Any]]
    args: Tuple
    kwargs: Dict[str, Any]

    def __post_init__(self):
        if not isinstance(self.args, (list, tuple)):
            raise EntityTypeError(f"Function's positional arguments must be list or tuple, got:  {type(self.args)} / {self.args}")
        if not isinstance(self.kwargs, dict):
            raise EntityTypeError(f"Function's keyword arguments must be dictionary, got:  {type(self.kwargs)} / {self.kwargs}")

    def get_args_kwargs(self) -> FunctionArgumentsTupleType:
        return self.args, self.kwargs


EmptyFunctionArguments  = FunctionArgumentsType([], {})


# NOTE: when custom type with DotExpression alias are defined, then
#       get_type_hints falls into problems producing NameError-s
#         Object <class 'type'> / '<class 'reedwolf.entities.components.BooleanField'>'
#         type hint is not possible/available: name 'OptionalBoolOrDExp' is not defined.
#
#   from .expressions import DotExpression
#   BoolOrDExpType          = Union[bool, DotExpression]
#   OptionalBoolOrDExpType  = Optional[BoolOrDExpType]
#   StandardTypeOrDExpType  = Union[StandardType, DotExpression]


# ------------------------------------------------------------

def get_underlying_types(type_: type):
    if not hasattr(type_, "__mro__"):
        return ()
    underlying_types = getmro(type_)
    idx = underlying_types.index(object)
    if idx>=0:
        underlying_types = list(underlying_types)
        del underlying_types[idx]
        underlying_types = tuple(underlying_types)
    return underlying_types


def is_pydantic(maybe_pydantic_class: Any) -> bool:
    # TODO: ALT: maybe fails for partial functions: isinstance(maybe_pydantic_class) and issubclass(maybe_pydantic_class, PydBaseModel)
    return bool(PydBaseModel) and isinstance(maybe_pydantic_class, PydModelMetaclass)

def is_model_klass(klass: Any) -> bool:
    """
    is_dataclass or is_pydantic (for now)
    in future: sqlalchemy model, django orm model, attrs
    """
    return isclass(klass) and (is_dataclass(klass) or is_pydantic(klass))

def is_model_instance(instance: ModelInstanceType) -> bool:
    """
    is_dataclass or is_pydantic (for now)
    in future: sqlalchemy model, django orm model, attrs
    """
    if isclass(instance):
        return False
    return is_model_klass(instance.__class__)

def is_enum(maybe_enum: Any) -> bool:
    return isinstance(maybe_enum, type) and issubclass(maybe_enum, Enum)

def get_enum_member_py_type(enum_kls) -> type:
    " teke first member value and return its type "
    assert is_enum(enum_kls)
    return type(list(enum_kls.__members__.values())[0].value)

def get_enum_members(enum_kls) -> List[Tuple[str, Any]]:
    " teke first member value and return its type "
    assert is_enum(enum_kls)
    return [(k, ev.value) for k, ev in enum_kls.__members__.items()]

def get_function_non_empty_arguments(py_function: Callable) -> List[AttrName]:
    # Check that function receives only single param if method(self), or no param if function()
    py_fun_signature = inspect.signature(py_function)
    # TODO: resolve properly first arg name as 'self' convention
    non_empty_params = [param.name for param in py_fun_signature.parameters.values() if
                        param.empty and param.name != SELF_ARG_NAME]
    return non_empty_params



# def is_method(obj, name):
#     return hasattr(obj, name) and ismethod(getattr(obj, name))

# def is_function(obj, name):
#     return hasattr(obj, name) and callable(getattr(obj, name))
#     # return hasattr(obj, name) and isfunction(getattr(obj, name))

def is_function(maybe_function: Any) -> bool:
    # assert not isinstance(maybe_function, DotExpression), maybe_function
    # e.g. DotExpression, Namespace
    if isinstance(maybe_function, DynamicAttrsBase):
        return False

    # NOTE: Does not recognize FunctionFactory / IFunctionDexpNode / IFunction instances 
    # if isinstance(maybe_function, "_is_custom_function_factory", None):
    #     return False

    # TODO: maybe check not is_class(maybe_function)
    # type() == type - to exclude list, dict etc.
    # TODO: type() == _GenericAlias to exclude typing.* e.g. List/Optional - any better way?
    #       py 3.10 has no _GenericAlias, so this doesn't work: and not type(maybe_function) in (type, _GenericAlias) \
    # NOTE: typing.NewType has __supertype__ - base class
    return callable(maybe_function) \
           and not type(maybe_function) in (type, ) \
           and not is_enum(maybe_function) \
           and not is_pydantic(maybe_function) \
           and not is_dataclass(maybe_function) \
           and not isclass(maybe_function) \
           and not hasattr(maybe_function, "__supertype__") \
           and not repr(maybe_function).startswith("typing.")


# def is_method(obj: Any) -> bool:
#     return is_function(getattr(owner, name))

def is_method_by_name(owner: Any, name:str) -> bool:
    return hasattr(owner, name) and is_function(getattr(owner, name))

def is_instancemethod_by_name(owner: Any, name:str) -> bool:
    """ NOTE: hm, this depends on  naming convention
              https://stackoverflow.com/questions/8408910/detecting-bound-method-in-classes-not-instances-in-python-3
        TODO: do it better: is_method and not classmethod and not staticmethod
              works for Class and Instance()
    """
    method = getattr(owner, name, None)
    if not (method and callable(method)):
        return False
    # NOTE: although signature() is preferred way to do, does not well with detecting class/static/method
    # ALT:
    #   if ismethod(method):
    #       # will return True for classmethods but not for staticmethod
    #       return True
    #   args = signature(method).parameters
    #   return bool('.' in method.__qualname__ and args and list(args.keys())[0] == SELF_ARG_NAME)
    args = inspect.getfullargspec(method).args
    return bool('.' in method.__qualname__ and args and args[0] == SELF_ARG_NAME)


def is_classmethod_by_name(owner: Any, name: str) -> bool:
    """ !! CURRENTLY NOT USED !!!
        works for Class and Instance() """
    if not isclass(owner):
        owner = owner.__class__
    if not hasattr(owner, name):
        return False
    # NOTE: must be done with vars(Class) -> returns classmethod,
    #       Instance or getattr() -> returns function/method, vars(Class)
    #       https://stackoverflow.com/questions/8408910/detecting-bound-method-in-classes-not-instances-in-python-3
    method = vars(owner).get(name, None)
    if not method and callable(method):
        return False
    return isinstance(method, classmethod)


def is_classmethod(owner: Any, name: str) -> bool:
    """ !! CURRENTLY NOT USED !!!
        works for Class and Instance() """
    return _is_method_by_name_instanceof(owner=owner, name=name, instanceof_cls=classmethod)


def is_staticmethod_by_name(owner: Any, name: str) -> bool:
    """ !! CURRENTLY NOT USED !!!
        works for Class and Instance() """
    return _is_method_by_name_instanceof(owner=owner, name=name, instanceof_cls=staticmethod)


def _is_method_by_name_instanceof(owner: Any, name: str, instanceof_cls: type) -> bool:
    """ works for Class and Instance() """
    if not isclass(owner):
        owner = owner.__class__
    if not hasattr(owner, name):
        return False
    method = vars(owner).get(name, None)
    if not method and callable(method):
        return False
    return isinstance(method, instanceof_cls)

def get_methods(owner: Any) -> Dict[str, Callable[..., Any]]:
    """ !! CURRENTLY NOT USED !!!  """ 
    # getmembers(DemoClass(), predicate=inspect.ismethod)
    methods = getmembers(owner, predicate=is_function)
    return {mname: method for (mname, method) in methods if not mname.startswith("_")}

# ------------------------------------------------------------

def get_dataclass_fields(inspect_object: Any) -> Tuple[DcField]:
    if not is_dataclass(inspect_object):
        raise ValueError(f"Expecting dataclass object, got: {inspect_object}")
    return dc_fields(inspect_object)

# ------------------------------------------------------------

def get_model_fields(inspect_object: ModelKlassType, strict: bool = True) -> Dict[AttrName, ModelField]:
    if is_dataclass(inspect_object):
        # ALT: 
        #   from dataclasses import fields
        #   field_dict = OrderedDict([f.name: f for f in fields(inspect_object)])
        field_dict = inspect_object.__dataclass_fields__
    elif is_pydantic(inspect_object):
        field_dict = inspect_object.__fields__
    else:
        if strict:
            raise EntityTypeError(item=inspect_object, msg=f"Class should be Dataclass or Pydantic, got: {inspect_object}")
        # TODO: field_dist = inspect_model.__annotations__
        field_dict = {}
    return field_dict

# ------------------------------------------------------------

def extract_model_field_meta(inspect_object: Any, attr_node_name: str) -> Tuple[Optional[ModelField], List[ModelField]]:
    """
    returns th_field, fields
    """
    fields = get_model_fields(inspect_object)
    th_field = fields.get(attr_node_name, None)
    if th_field:
        if not type(th_field) in (DcField, PydModelField):
            raise EntityTypeError(f"Invalid type of th_field: {th_field}, expected dataclass.Field or pydantic.Field")
        # assert PydModelField
    return th_field, fields

# ------------------------------------------------------------

def extract_py_type_hints(inspect_object: Any, caller_name: str = "", strict: bool = True) -> Dict[str, Union[PyTypeHint, Exception]]:
    """
        - type_hint = function.__annotations__ - will not evaluate types
        - .get("return", None) - will get return function type hint
    """
    # TODO: check if module attr_node, class, class attribute, function or method.
    try:
        # ALT: to get __annotations__ -> inspect.get_annotations
        #      https://docs.python.org/3/library/inspect.html#inspect.get_annotations
        return get_type_hints(inspect_object)
    except Exception as ex:
        if strict:
            # NOTE: sometimes there is NameError because some referenced types are not available in this place???
            #       when custom type with DotExpression alias are defined, then
            #       get_type_hints falls into problems producing NameError-s
            #         Object <class 'type'> / '<class 'reedwolf.entities.components.BooleanField'>'
            #         type hint is not possible/available: name 'OptionalBoolOrDExp' is not defined.
            raise EntityTypeError(item=inspect_object, msg=f"{caller_name}: Object type hint is not possible/available: {ex}."
                                 + " Please verify that object is properly type hinted class attribute, function or method,"
                                 + " and type hint should not include not standard type Type alias (see entity/types.py comment).")
        return {"__exception__": ex}


def extract_function_args_type_hints_dict(function: Callable[..., Any], strict: bool= True) -> Dict[str, PyTypeHint]:
    # TODO: typing
    """
    returns:
    function hinted args + return type
    strict = False:
        py_type_hint_dict will get only arguments with type hints, and when strict=False
        for missing arg or return hint will return type hint: Any
    """
    if not is_function(function):
        raise EntityTypeError(f"Invalid object, expected type, got:  {function} / {type(function)}")

    if isinstance(function, partial):
        function = function.func

    name = getattr(function, "__name__", None)
    if not name:
        name = function.__class__.__name__

    if not hasattr(function, "__annotations__"):
        raise EntityTypeError(item=function, msg=f"SetupSession: AttrDexpNode FUNCTION '{name}' is not valid, it has no __annotations__ / type hints metainfo.")

    # e.g. Optional[List[SomeCustomClass]] or SomeCustomClass or ...
    py_type_hint_dict = extract_py_type_hints(function, caller_name=f"Function {name}")

    if not strict:
        params = signature(function).parameters
        out2 = OrderedDict([(param_name, py_type_hint_dict.get(param_name, Any))
                             for param_name in params.keys()])
        out2["return"] = py_type_hint_dict.get("return", Any)
        py_type_hint_dict = out2

    return py_type_hint_dict


def extract_function_args_default_dict(py_function) -> Dict[str, Any]:
    # https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    fun_signature = signature(py_function)
    return {
            arg_name: (
                arg_info.default
                if arg_info.default is not Parameter.empty
                else UNDEFINED
                )
            for arg_name, arg_info in fun_signature.parameters.items()
        }

def type_as_str(type_: type):
    return (f"{type_.__module__}.{type_.__name__}" 
         if hasattr(type_, "__name__") and str(getattr(type_, "__module__", "")) not in ("", "builtins",) 
         else f"{getattr(type_, '__name__', str(type_))}")


# # not used any more and not currently
# def extract_bound_inner_type(inner_type_name_prefix: str, inner_type: Type) -> Tuple[bool, Type]:
#     if inner_type and getattr(inner_type, "__name__", "").startswith(inner_type_name_prefix):
#         is_matched = True
#         if hasattr(inner_type, "__bound__"):
#             # TypeVar("Name", bound=some_type)
#             inner_type = inner_type.__bound__ if inner_type.__bound__ else Any
#         elif hasattr(inner_type, "__supertype__"):
#             # NewType("Name", some_type)
#             inner_type = inner_type.__supertype__
#         else:
#             raise EntityInternalError(msg=f"inner_type '{inner_type.__name__}' is not bound TypeVar, got: {inner_type}")
#     else:
#         is_matched = False
#     return is_matched, inner_type

# ------------------------------------------------------------
# TypeInfo
# ------------------------------------------------------------

@dataclass
class TypeInfo:

    # TODO: proper type pydantic or dataclass
    py_type_hint: PyTypeHint = field(repr=True)

    # evaluated
    is_list:        bool = field(init=False, repr=False, default=UNDEFINED)
    is_dict:        bool = field(init=False, repr=False, default=UNDEFINED)
    is_optional:    bool = field(init=False, repr=False, default=UNDEFINED)
    is_enum:        bool = field(init=False, repr=False, default=UNDEFINED)
    is_union:       bool = field(init=False, repr=False, default=UNDEFINED)

    # NOTE: custom type hints - see classes that inherit IFuncArgHint.
    #       This enables custom properties and methods embedded in type-hint object instances.
    is_func_arg_hint: bool = field(init=False, repr=False, default=UNDEFINED)

    # NOTE: Special type of IInjectFuncArgHint (IFuncArgHint)
    # TODO: explain better
    is_inject_func_arg: bool = field(init=False, repr=False, default=UNDEFINED)

    # list of python type underneath - e.g. int, str, list, dict, Person, or list of accepted types
    types:          List[Type] = field(init=False, default=UNDEFINED)
    # first one - get rid of this one or leave?
    # ex. Union[type, List[type]]
    type_:          Type = field(init=False, repr=False, default=UNDEFINED)

    # ------------------------------------------------------------
    # cache - registry by class type
    TYPE_INFO_REGISTRY: ClassVar[Dict[Union[type, IFuncArgHint], Self]] = {}

    def __post_init__(self):
        if isinstance(self.py_type_hint, str):
            raise EntityTypeError(owner=self, msg=f"py_type_hint={self.py_type_hint} is still string, it should have been resolved before")

        if is_function(self.py_type_hint):
            raise EntityTypeError(owner=self, msg=f"py_type_hint={self.py_type_hint} is a function/method")

        self._extract_type_hint_details()

        if isinstance(self.type_, DcField):
            raise EntityTypeError(owner=self, msg=f"py_type_hint={self.type_} is a dataclass.Field() instance")

        # TODO: raise error: if self.is_none_type():

        # if not self.is_list and not self.is_optional:
        #     if self.py_type_hint != self.type_:
        #         raise EntityTypeError(item=self, msg=f"Unsupported type hint, got: {self.py_type_hint}. {ERR_MSG_SUPPORTED}")


    def is_none_type(self):
        return self.types == [NoneType]


    def _extract_type_hint_details(self):
        if isinstance(self.py_type_hint, IFuncArgHint):
            func_arg_hint: IFuncArgHint = self.py_type_hint
            py_type_hint = func_arg_hint.get_type()
            origin_type = func_arg_hint.get_inner_type()
            origin_type = get_underlying_type(origin_type)
        elif isclass(self.py_type_hint) and issubclass(self.py_type_hint, IFuncArgHint):
            raise EntityTypeError(owner=self, msg=f"Type hint should be instances of IFuncArgHint class, not class itself (type). Got: {self.py_type_hint}")
        else:
            func_arg_hint = None
            py_type_hint = self.py_type_hint
            origin_type = getattr(py_type_hint, "__origin__", None)

        py_type_hint = get_underlying_type(py_type_hint)

        self.is_func_arg_hint = bool(func_arg_hint)
        self.is_inject_func_arg = self.is_func_arg_hint and isinstance(func_arg_hint, IInjectFuncArgHint)

        is_optional = False
        if origin_type == Union:
            sub_types = getattr(py_type_hint, "__args__", []) 
            if sub_types:
                origin_type = sub_types

            if NoneType in sub_types:
                # Optional[SomeClass] / Union[SomeClass, OtherClass, NoneType]
                is_optional = True
                origin_type = [ty for ty in sub_types if ty != NoneType]
            else:
                # Union[SomeClass, OtherClass]
                ...

            py_type_hint = origin_type
            # get underlying types if any
            origin_type = [getattr(ty, "__origin__", ty) for ty in sub_types]

            if len(py_type_hint)==1:
                py_type_hint = py_type_hint[0]
                origin_type = origin_type[0]


        is_list = False
        is_dict = False
        inner_type = UNDEFINED

        if origin_type in (list, Sequence, Iterable):
            # List[some_type]
            is_list = True
            if len(py_type_hint.__args__) != 1:
                raise EntityTypeError(item=py_type_hint, msg=f"AttrDexpNode's annotation should have single List argument, got: {py_type_hint.__args__}.")
            inner_type = py_type_hint.__args__[0]
        elif origin_type in (dict, Dict):
            is_dict = True
            inner_type = py_type_hint.__args__[1]
        elif getattr(origin_type, "__origin__", None) in (dict,):
            is_dict = True
            # type of key is ignored
            inner_type = origin_type.__args__[1]
            origin_type = origin_type.__origin__
        elif origin_type is not None:
            inner_type = origin_type
        else:
            inner_type = py_type_hint

        if func_arg_hint:
            # assert inner_type == py_type_hint, "inner_type shouldn't be changed from original py_type_hint"
            fah_inner_type = func_arg_hint.get_inner_type()
            if fah_inner_type:
                inner_type = fah_inner_type

        # Another Union layer allowed, e.g.:  List[Union[str, int, NoneType]]
        if getattr(inner_type, "__origin__", inner_type) == Union: 
            # list of accepted types, e.g.
            #   Union[int, str, NoneType] -> int, str, NoneType
            inner_type = inner_type.__args__

        inner_type = get_underlying_type(inner_type)

        # normalize
        self.is_list = is_list
        self.is_dict = is_dict
        self.is_optional = is_optional

        # TODO: probably it will need extra attribute to hold Enum internal type (int, str, etc.)
        # if self.is_enum:
        #     subtype = type([v for v in self.type_.__members__.values()][0].value)

        self.types = [inner_type] if not isinstance(inner_type, (list, tuple)) else inner_type
        self.type_ = self.types[0]
        self.is_union = (len(self.types)!=1)

        if not self.is_union:
            # only when single type
            self.is_enum = is_enum(self.type_)

        if not is_enum:
            for type_ in self.types:
                if type_!=ItemType and type(type_)!=type:
                    raise EntityTypeError(item=self, msg=f"Unsupported type hint, got: {self.py_type_hint}. {ERR_MSG_SUPPORTED}")


    def check_compatible(self, other: Self, ignore_list_check: bool=False) -> Optional[str]:
        """
        returns error message when input type is not compatible with given type
        """
        if other is None:
            raise EntityTypeError(owner=self, msg=f"check_compatible({other}) - type_info of other not supplied")

        if self==other:
            return None

        if self.py_type_hint is Any or other.py_type_hint is Any:
            return None

        if self.is_dict!=other.is_dict:
            return "expecting a dict compatible type"

        if not ignore_list_check and self.is_list!=other.is_list:
            if self.is_list:
                err_msg = "expecting a list compatible type"
            else:
                err_msg = "did not expect a list compatible type"
            return err_msg + f"\n  expected:\n    {self.py_type_hint}\n  got:\n    {other.py_type_hint}"

        if not self.is_optional and other.is_optional:
            return "can not accept None values (Optional)"

        if other.is_union:
            return "given argument need to be single type, not Union"

        if self.is_optional and not other.is_union and other.type_==NoneType:
            # accept None
            return None

        found = False
        other_underlying_types = get_underlying_types(other.type_)

        for type_ in self.types:
            if (type_ in (ItemType, Any) 
                    or other.type_ in (ItemType, Any)
                    or (type_ == Sized and hasattr(other.type_, "__len__"))
                    or type_ in other_underlying_types):
                found = True
                break

            if is_enum(type_):
                enum_1st_member_py_type = get_enum_member_py_type(type_)
                if other.type_ == enum_1st_member_py_type:
                    found = True

        if not found:
            # TODO: can be that underlying type is not a "type"
            accepted_types = [type_as_str(type_) for type_ in self.types]
            accepted_types = format_arg_name_list(accepted_types)
            # add_msg = f" HINT: For classes that inherit Enum, inherit additional python type like `class {enum_types}(str, Enum):` " \
            #          if enum_types else ""
            return f"underlying type '{type_as_str(other.type_)}' is not compatible with {accepted_types}"

        return None

    # ------------------------------------------------------------

    def as_str(self) -> str:
        out = [f"{', '.join([type_as_str(ty) for ty in self.types])}"]

        if self.is_enum:
            out.insert(0, "Enum[")
            out.append("]")
        if self.is_list:
            out.insert(0, "List[")
            out.append("]")
        if self.is_optional:
            out.insert(0, "Optional[")
            out.append("]")
        if self.is_union:
            out.insert(0, "Union[")
            out.append("]")
        if self.is_func_arg_hint:
            out.insert(0, "FuncArgAttrname[")
            out.append("]")
        if self.is_func_arg_dot_expr:
            out.insert(0, "DotExpr[")
            out.append("]")

        return "".join(out)

    # def as_str(self) -> str:
    #     return self.py_type_hint.__name__ if type(self.py_type_hint)==type else str(self.py_type_hint)

    # ------------------------------------------------------------
    # Classmethods
    # ------------------------------------------------------------

    @classmethod
    def get_or_create_by_value(cls, value: AttrValue, caller: Optional[Any] = None) -> Self:
        """
        simple and stupid for now
        """
        if isinstance(value, (list, tuple)) and value:
            inner_type = type(value[0])
            py_type_hint = List[inner_type]
        elif isinstance(value, dict) and value:
            inner_type = type(value.values[0])
            py_type_hint = List[inner_type]
        else:
            py_type_hint = type(value)
        type_info = cls.get_or_create_by_type(py_type_hint=py_type_hint, caller=caller)
        return type_info

    @classmethod
    def get_or_create_by_type(cls, py_type_hint: PyTypeHint, caller: Optional[Any] = None) -> Self:
        """
        When 'from future import annotations + __annotations__' is used then python hints are strings. 
        Use 'typing.get_type_hints()' to resolve hints properly.

        Several ways to extract python type hints:

            1) 'extract_py_type_hints() + get_or_create_by_type()' - low level preffered

            2) 'entities.base. extract_type_info()' - high level PREFFERED - internally uses
               'extract_py_type_hints()->get_type_hints()', but requires parent object.

            2)  When you have classes/types then 'get_or_create_by_type()'

            NOTE: Do not use 'entities.meta. get_model_fields() .type' or '__annotations__'
                  since python type hints may not be resolved.
        """
        msg_prefix = f"{to_repr(caller)}:: " if caller else ""
        if isinstance(py_type_hint, str):
            raise EntityTypeError(owner=cls, msg=
                        f"{msg_prefix} Python type hint is a string, probably not resolved properly: {repr(py_type_hint)}."
                        "\nHINTS:"
                        "\n  1) if you have `from __future__ import annotations`, remove it, try to import that module, stabilize it and then try this again." 
                        "\n  2) 'get_model_fields() + .type' used instead 'extract_py_type_hints()' -> get_or_create_by_type() (internal issue, use 'extract_model_field_meta' maybe?)"
                        "Explanation: When 'from future import annotations + __annotations__' is used then python hints are strings. Use 'typing.get_type_hints()' to resolve hints properly."
                        )

        if py_type_hint not in cls.TYPE_INFO_REGISTRY:
            # py_type_hint can be IFuncArgHint
            if isinstance(py_type_hint, DcField):
                py_type_hint = py_type_hint.type
            elif PydModelField and isinstance(py_type_hint, PydModelField):
                py_type_hint = py_type_hint.type_
            cls.TYPE_INFO_REGISTRY[py_type_hint] = TypeInfo(py_type_hint=py_type_hint)

        return cls.TYPE_INFO_REGISTRY[py_type_hint]

    # ------------------------------------------------------------

    @staticmethod
    def extract_function_return_type_info(
            py_function: Callable[..., Any],
            allow_nonetype:bool=False) -> Self:

        py_type_hint_dict = extract_function_args_type_hints_dict(function=py_function)
        py_type_hint = py_type_hint_dict.get("return", None)
        name = getattr(py_function, "__name__", "?")

        msg_prefix = f"Function {py_function}::"
        if not py_type_hint:
            raise EntityTypeError(item=py_function, msg = msg_prefix + f"AttrDexpNode FUNCTION '{name}' is not valid, it has no return type hint (annotations).")
        if not allow_nonetype and py_type_hint == NoneType:
            raise EntityTypeError(item=py_function, msg = msg_prefix + f"SetupSession: AttrDexpNode FUNCTION '{name}' is not valid, returns None (from annotation).")

        output = TypeInfo.get_or_create_by_type(
                        py_type_hint=py_type_hint,
                        caller=msg_prefix,
                        )
        return output

    # ------------------------------------------------------------

    @staticmethod
    def extract_function_args_type_info_dict(
            py_function: Callable[..., Any]) -> Dict[str, Self]:
        """
        From annotations, but argument defaults could be fetched from
        inspect.getfullargspec(), see: extract_function_arguments_default_dict
        """
        py_type_hint_dict = extract_function_args_type_hints_dict(function=py_function)

        msg_prefix = f"Function {py_function}::"

        output = {}
        name = getattr(py_function, "__name__", "?")
        for arg_name, py_type_hint in py_type_hint_dict.items():
            if arg_name == "return":
                continue
            if not py_type_hint:
                raise EntityTypeError(item=py_function, msg=msg_prefix + f"AttrDexpNode FUNCTION '{name}.{arg_name}' is not valid, argument {arg_name} has no type hint (annotations).")
            if py_type_hint in (NoneType,):
                raise EntityTypeError(item=py_function, msg=msg_prefix + f"SetupSession: AttrDexpNode FUNCTION '{name}.{arg_name}' is not valid, argument {arg_name} has type hint (annotation) None.")

            output[arg_name] = TypeInfo.get_or_create_by_type(
                                    py_type_hint=py_type_hint,
                                    caller=msg_prefix,
                                    )
        return output


def get_dataclass_field_type_info(dc_model: DataclassType, field_name: str) -> Optional[TypeInfo]:
    all_fields = {field.name: field for field in dc_fields(dc_model)}
    dc_field = all_fields.get(field_name)
    return TypeInfo.get_or_create_by_type(dc_field.type) if dc_field else None


def make_dataclass_with_optional_fields(dc_model: DataclassType) -> DataclassType:
    """
    will clone the type and make all fields optinonal with None as default
    inherit the dc_model
    """
    if not is_dataclass(dc_model):
        raise EntityTypeError(f"Expected dataclass, got: {dc_model}")

    new_fields = []
    for field in dc_fields(dc_model):
        if not field.init:
            continue

        new_field = copy(field)
        type_info = TypeInfo.get_or_create_by_type(field.type)
        if not type_info.is_optional:
            new_field.type = Union[field.type, NoneType]

        # TODO: nested models - what with them? maybe recursion? 
        #       how deep? check instance_new

        if type_info.is_list:
            new_field.default_factory = list
        # NOTE: if there will be some dict() cases, cover this too
        else:
            new_field.default = None

        new_fields.append((new_field.name, new_field.type, new_field))

    temp_dataclass_model = make_dataclass(
            cls_name=f"Temp{dc_model.__name__}",
            fields = new_fields,
            # NOTE: do not inehrit - base class could have some unwanted init
            #       methods ...
            #           bases=(dc_model,),
            )
    return temp_dataclass_model

def dataclass_type_to_tuple(dataclass_klass: Type, depth: int=0) -> List[Tuple[AttrName, Type, Optional[Tuple]]]:
    """
    recursive
    used for unit tests
    """
    if depth>MAX_RECURSIONS:
        raise EntityInternalError(msg=f"Reached maximum recrusion level: {depth}")

    if not is_dataclass(dataclass_klass):
        raise TypeError(f"Expecting dataclass as inner klass, got: {dataclass_klass}")
    out = []
    for fld in dc_fields(dataclass_klass):
        fld_type = fld.type
        type_info = TypeInfo.get_or_create_by_type(fld_type)
        if is_dataclass(type_info.type_):
            # recursion:
            inner_result = dataclass_type_to_tuple(type_info.type_, depth=depth+1)
        else:
            inner_result = None
        out.append((fld.name, getattr(fld.type, "__name__", str(fld_type)), inner_result))
    return out

def dataclass_from_dict(dataclass_klass: Type, values_dict: Dict[str, Any]) -> DataclassType:
    """
    Inspired by: https://stackoverflow.com/a/54769644/565525
    Check load.py too - it has some similar logic, but for different purpose.
    """
    if not is_dataclass(dataclass_klass):
        raise TypeError(f"Expecting dataclass as inner klass, got: {dataclass_klass}")
    if not isinstance(values_dict, dict):
        raise TypeError(f"Expecting dict of values, got: {values_dict}")

    dc_field_dict: Dict[str, DcField] = {fld.name: fld for fld in dc_fields(dataclass_klass)}
    kwargs = {}
    for fld_name, value in values_dict.items():
        dc_field = dc_field_dict.get(fld_name, None)
        if not (dc_field and dc_field.init == True):
            continue
        type_info = TypeInfo.get_or_create_by_type(dc_field.type)
        if is_dataclass(type_info.type_) and value is not None:
            # ALT: type_info.is_list
            if isinstance(value, (list, tuple)):
                value = [
                    dataclass_from_dict(dataclass_klass=type_info.type_, values_dict=item)
                    for item in value
                ]
            else:
                value = dataclass_from_dict(dataclass_klass=type_info.type_, values_dict=value)
        kwargs[fld_name] = value
    return dataclass_klass(**kwargs)

def get_underlying_type(py_type_hint: Type) -> Type:
    if py_type_hint == ItemType:
        pass
    elif getattr(py_type_hint, "__supertype__", None):
        # NewType
        py_type_hint = py_type_hint.__supertype__
    elif getattr(py_type_hint, "__bound__", None):
        # TypeVar
        py_type_hint = py_type_hint.__bound__
    return py_type_hint


# ------------------------------------------------------------
# custom_attributes.py uses
# ------------------------------------------------------------

class SettingsType(str, Enum):
    SETUP_SETTINGS = "SETUP_SETTINGS"
    APPLY_SETTINGS = "APPLY_SETTINGS"


@dataclass
class SettingsSource:
    settings_type: SettingsType
    klass: ModelKlassType
    _fields: Optional[Dict[AttrName, ModelField]] = field(init=False, repr=False, default=None)

    @property
    def fields(self):
        if self._fields is None:
            self._fields = get_model_fields(self.klass)
        return self._fields

@dataclass
class IAttribute:
    # name of attribute in instance
    name: AttrName
    # name of Ctx.DotExpression attribute name
    dexp_attr_name: Optional[AttrName] = None

    # filled later
    output_type_info: TypeInfo = field(repr=False, init=False, default=None)
    settings_source: SettingsSource = field(repr=False, init=False, default=None)

    def __post_init__(self):
        if not self.dexp_attr_name:
            self.dexp_attr_name = self.name

    @abstractmethod
    def setup_dexp_attr_source(self, settings_source_list: List[SettingsSource]) -> Tuple[TypeInfo, SettingsSource]:
        ...

@dataclass
class KlassMember:
    klass: Type
    member_name: IAttribute


CustomCtxAttributeList = List[IAttribute]
FunctionNoArgs = Callable[[], Any]



# ------------------------------------------------------------
# OBSOLETE
# ------------------------------------------------------------

# Use higher level function: extract_type_info() instead
# 
# def extract_py_type_hints_for_attr(inspect_object: Any, attr_name: str, caller_name: str) -> PyTypeHint:
#     py_type_hint_dict = extract_py_type_hints(inspect_object=inspect_object, caller_name=caller_name, strict=True)
#     if attr_name not in py_type_hint_dict:
#         raise EntityTypeError(item=inspect_object, msg=f"{caller_name}: Object type hint for '{attr_name}' not available. Available are {','.join(py_type_hint_dict.keys())}.")
#     hint_type = py_type_hint_dict[attr_name]
#     if not hint_type: # check if type
#         raise EntityTypeError(item=inspect_object, msg=f"{caller_name}: Object type hint for '{attr_name}' is not valid '{hint_type}'. Expected some type.")
#     return hint_type

# ------------------------------------------------------------
# Following are used for functions to control type of function arguments
# Naming convention instead of class inheritance
# ------------------------------------------------------------

# -- Type hints for attribute name declarations e.g.
#   def Sum(fieldname: FuncArgAttrNameNumberType)

# FuncArgAttrNameType = TypeVar(f"FuncArgAttrNameType", bound=DynamicAttrsBase)
# FuncArgDotExprType = TypeVar("FuncArgDotExprType", bound=DynamicAttrsBase)

# FUNC_ARG_ATTR_NAME_TYPE_PREFIX: str = "FuncArgAttrName"
# FuncArgAttrNameType = TypeVar(f"{FUNC_ARG_ATTR_NAME_TYPE_PREFIX}Type", bound=Any)
# FuncArgAttrNameNumberType = TypeVar(f"{FUNC_ARG_ATTR_NAME_TYPE_PREFIX}NumberType", bound=Union[NumberType])
# # NOTE: *NUMBER_TYPES # self.type_.__constraints__
# #       (<class 'int'>, <class 'float'>, <class 'decimal.Decimal'>)
# FuncArgAttrNameStringType = TypeVar(f"{FUNC_ARG_ATTR_NAME_TYPE_PREFIX}StringType", bound=str)
#
# # -- Type hints for delayed DotExpression-s - dot-expression will have delayed evaluation,
# #    will be evaluated in a function for every function call e.g.
# #        def Filter(term: FuncArgDotExpression)
# #        This.Items.Filter(This.value > 10)
#
# FUNC_ARG_DOT_EXPR_TYPE_PREFIX: str = "FuncArgDotExpr"
#
# FuncArgDotExprType = TypeVar(f"{FUNC_ARG_DOT_EXPR_TYPE_PREFIX}Type", bound=DynamicAttrsBase)
# FuncArgDotExprBoolType = TypeVar(f"{FUNC_ARG_DOT_EXPR_TYPE_PREFIX}BoolType", bound=DynamicAttrsBase)
#
# FUNC_ARG_DOT_EXPR_TYPE_MAP: Dict[Type, Type] = {
#     FuncArgDotExprType: Any,
#     FuncArgDotExprBoolType: bool,
# }

