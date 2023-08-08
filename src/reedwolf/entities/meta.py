"""
TypeInfo - the most interesting class
extract_* - the most interesting functions 

"""
import inspect
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
        _GenericAlias,
        get_type_hints,
        TypeVar,
        Sequence as SequenceType,
        )
from enum import Enum
from decimal import Decimal
from datetime import date, datetime, time, timedelta
from dataclasses import (
        is_dataclass,
        dataclass,
        Field as DcField,
        field,
        fields as dc_fields,
        )

try:
    from typing import Self
except ImportError:
    Self = NewType("Self", Any)
try:
    # ----------------------------------------
    # Pydantic found
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
        )
from .exceptions import (
        EntityInternalError,
        EntitySetupError,
        EntitySetupNameError,
        EntitySetupValueError,
        )
from .namespaces import (
        DynamicAttrsBase,
        )

# ------------------------------------------------------------
# SPECIAL DATATYPES
# ------------------------------------------------------------
NoneType                = type(None) # or None.__class__

# NOTE: for dataclass there is no base type, so using Any
DataclassType = Any

if PydBaseModel:
    ModelType  = Union[DataclassType, PydBaseModel]
    ModelField = Union[DcField, PydModelField]
else:
    ModelType  = DataclassType
    ModelField = DcField

# e.g. [(1,"test"), ("C2", "test2")]
STANDARD_TYPE_LIST      = (str, int, float, bool, Decimal, date, datetime, timedelta, time)
STANDARD_TYPE_W_NONE_LIST = (NoneType,) + STANDARD_TYPE_LIST 

StandardType            = Union[str, int, float, bool, Decimal, date, datetime, timedelta, time]

LiteralType             = Any

NUMBER_TYPES            = (int, float, Decimal)

NumberType              = Union[int, float, Decimal]

ERR_MSG_SUPPORTED = "Supporting custom and standard python types, and typing: Optional, Union[..., NoneType] and Sequence/List[ py-types | Union[py-types, NoneType]]."


# e.g. list, int, dict, Person, List, Dict[str, Optional[Union[str, float]]
PyTypeHint              = Any # TODO: Union[type, Type] # or typing._GenericAlias

TransMessageType        = str

# ChoiceValueType        = Tuple[StandardType, TransMessageType]

RuleDatatype            = Union[StandardType, List[StandardType,], Dict[str, StandardType,]]

FunctionArgumentsTupleType = Tuple[List[Any], Dict[str, Any]]

HookOnFinishedAllCallable = Callable[[], None]

# ---

KeyPairs = SequenceType[Tuple[str, Any]]

InstanceId = int
KeyString = str

AttrName = str
AttrValue = Any

# ------------------------------------------------------------
# Commonly/Internally used type symbols
# ------------------------------------------------------------

# used for matching rows types, e.g. std_types map, filter etc.
ItemType = TypeVar("ItemType")

ComponentNameType = TypeVar("ComponentNameType", bound=str) 

# Naming convention instead of class inheritance
TYPE_FIELDNAME_SUFFIX = "AttrNameType"

AttrNameType = TypeVar(f"{TYPE_FIELDNAME_SUFFIX}", bound=Any) 
NumberAttrNameType = TypeVar(f"Number{TYPE_FIELDNAME_SUFFIX}", bound=Union[NumberType]) 
StringAttrNameType = TypeVar(f"String{TYPE_FIELDNAME_SUFFIX}", bound=str) 

# # https://stackoverflow.com/questions/61568462/python-typing-what-does-typevara-b-covariant-true-mean
# NumberAttrNameType = TypeVar("NumberAttrNameType") 
# # NOTE: *NUMBER_TYPES # self.type_.__constraints__
# #   (<class 'int'>, <class 'float'>, <class 'decimal.Decimal'>)

# Currently find no way on describing Dict-s
TreeNode = Any

# TreeNode :: 
#   name: str self.name
#   component: ComponentBase
#   children: List[Self]
ComponentTreeType = Dict[ComponentNameType, TreeNode]

# TreeNode :: 
#   name: str self.name
#   component: ComponentBase
#   children: List[Self]
#   attr_current_value_instance: InstanceAttrCurrentValue
ComponentTreeWValuesType = Dict[ComponentNameType, TreeNode]

# TreeNode :: 
#   name: str self.name
#   Optional[value: LiteralValue]
#   Optional[contains: List[TreeNode]]
ValuesTree = Dict[ComponentNameType, TreeNode]

# TreeNode :: 
#   name: str self.name
#   Optional[contains: List[TreeNode]]
MetaTree = Dict[ComponentNameType, TreeNode]


# ------------------------------------------------------------


@dataclass
class FunctionArgumentsType:
    # ex. FunctionArgumentsType = Tuple[List[Any], Dict[str, Any]]
    args : List[Any]
    kwargs: Dict[str, Any]

    def __post_init__(self):
        if not isinstance(self.args, (list, tuple)):
            raise EntitySetupValueError(f"Function's positional arguments must be list or tuple, got:  {type(self.args)} / {self.args}")
        if not isinstance(self.kwargs, dict):
            raise EntitySetupValueError(f"Function's keyword arguments must be dictionary, got:  {type(self.kwargs)} / {self.kwargs}")

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
    underlying_types = inspect.getmro(type_)
    idx = underlying_types.index(object)
    if idx>=0:
        underlying_types = list(underlying_types)
        del underlying_types[idx]
        underlying_types = tuple(underlying_types)
    return underlying_types


def is_pydantic(maybe_pydantic_class: Any) -> bool:
    # TODO: ALT: maybe fails for partial functions: isinstance(maybe_pydantic_class) and issubclass(maybe_pydantic_class, PydBaseModel)
    return bool(PydBaseModel) and isinstance(maybe_pydantic_class, PydModelMetaclass)

def is_model_class(klass: Any) -> bool:
    """
    is_dataclass or is_pydantic (for now)
    in future: sqlalchemy model, django orm model, attrs
    """
    return inspect.isclass(klass) and (is_dataclass(klass) or is_pydantic(klass))

def is_model_instance(instance: ModelType) -> bool:
    """
    is_dataclass or is_pydantic (for now)
    in future: sqlalchemy model, django orm model, attrs
    """
    if inspect.isclass(instance):
        return False
    return is_model_class(instance.__class__)

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


# def is_method(obj, name):
#     return hasattr(obj, name) and inspect.ismethod(getattr(obj, name))

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
    return callable(maybe_function) \
           and not type(maybe_function) in (type, _GenericAlias) \
           and not is_enum(maybe_function) \
           and not is_pydantic(maybe_function) \
           and not is_dataclass(maybe_function) \
           and not inspect.isclass(maybe_function) \
           and not repr(maybe_function).startswith("typing.")


# def is_method(obj: Any) -> bool:
#     return is_function(getattr(owner, name))

def is_method_by_name(owner: Any, name:str) -> bool:
    return hasattr(owner, name) and is_function(getattr(owner, name))

def is_instancemethod_by_name(owner: Any, name:str) -> bool:
    """ !! CURRENTLY NOT USED !!!
        NOTE: hm, this depends on  naming convention
              https://stackoverflow.com/questions/8408910/detecting-bound-method-in-classes-not-instances-in-python-3
        TODO: do it better: is_method and not classmethod and not staticmethod

        works for Class and Instance()
    """
    if not hasattr(owner, name):
        return False
    method = getattr(owner, name)
    if not callable(method):
        return False
    args = inspect.getfullargspec(method).args
    return bool('.' in method.__qualname__ and args and args[0] == 'self')


def is_classmethod_by_name(owner: Any, name: str) -> bool:
    """ !! CURRENTLY NOT USED !!!
        works for Class and Instance() """
    if not inspect.isclass(owner):
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
    if not inspect.isclass(owner):
        owner = owner.__class__
    if not hasattr(owner, name):
        return False
    method = vars(owner).get(name, None)
    if not method and callable(method):
        return False
    return isinstance(method, instanceof_cls)

def get_methods(owner: Any) -> Dict[str, Callable[..., Any]]:
    """ !! CURRENTLY NOT USED !!!  """ 
    # inspect.getmembers(DemoClass(), predicate=inspect.ismethod)
    methods = inspect.getmembers(owner, predicate=is_function)
    return {mname: method for (mname, method) in methods if not mname.startswith("_")}

# ------------------------------------------------------------

def get_dataclass_fields(inspect_object: Any) -> Tuple[DcField]:
    if not is_dataclass(inspect_object):
        raise ValueError(f"Expecting dataclass object, got: {inspect_object}")
    return dc_fields(inspect_object)

# ------------------------------------------------------------

def get_model_fields(inspect_object: ModelType, strict: bool = True) -> Dict[str, ModelField]:
    if is_dataclass(inspect_object):
        # ALT: 
        #   from dataclasses import fields
        #   field_dict = OrderedDict([f.name: f for f in fields(inspect_object)])
        field_dict = inspect_object.__dataclass_fields__
    elif is_pydantic(inspect_object):
        field_dict = inspect_object.__fields__
    else:
        if strict:
            raise EntitySetupError(item=inspect_object, msg=f"Class should be Dataclass or Pydantic ({inspect_object})")
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
            raise EntityInternalError(f"Invalid type of th_field: {th_field}, expected dataclass.Field or pydantic.Field")
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
            raise EntitySetupValueError(item=inspect_object, msg=f"{caller_name}: Object type hint is not possible/available: {ex}."
                                 + " Please verify that object is properly type hinted class attribute, function or method,"
                                 + " and type hint should not include not standard type Type alias (see entity/types.py comment).")
        return {"__exception__": ex}


def extract_function_py_type_hint_dict(function: Callable[..., Any]) -> Dict[str, PyTypeHint]:
    # TODO: typing
    """ returns: function return type + if it returns list or not """
    if not is_function(function):
        raise EntityInternalError(f"Invalid object, expected type, got:  {function} / {type(function)}")

    if isinstance(function, partial):
        function = function.func

    name = getattr(function, "__name__", None)
    if not name:
        name = function.__class__.__name__

    if not hasattr(function, "__annotations__"):
        raise EntitySetupNameError(item=function, msg=f"SetupSession: AttrDexpNode FUNCTION '{name}' is not valid, it has no __annotations__ / type hints metainfo.")

    # e.g. Optional[List[SomeCustomClass]] or SomeCustomClass or ...
    py_type_hint_dict = extract_py_type_hints(function, caller_name=f"Function {name}")
    return py_type_hint_dict


def extract_function_arguments_default_dict(py_function) -> Dict[str, Any]:
    # https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    signature = inspect.signature(py_function)
    return {
            arg_name: (
                arg_info.default
                if arg_info.default is not inspect.Parameter.empty
                else UNDEFINED
                )
            for arg_name, arg_info in signature.parameters.items()
        }

def type_as_str(type_: type):
    return (f"{type_.__module__}.{type_.__name__}" 
         if hasattr(type_, "__name__") and str(getattr(type_, "__module__", "")) not in ("", "builtins",) 
         else f"{getattr(type_, '__name__', str(type_))}")

# ------------------------------------------------------------
# TypeInfo
# ------------------------------------------------------------

@dataclass
class TypeInfo:

    # TODO: proper type pydantic or dataclass
    py_type_hint: PyTypeHint = field(repr=True)

    # evaluated
    is_list:        bool = field(init=False, repr=False, default=UNDEFINED)
    is_optional:    bool = field(init=False, repr=False, default=UNDEFINED)
    is_enum:        bool = field(init=False, repr=False, default=UNDEFINED)
    is_union:       bool = field(init=False, repr=False, default=UNDEFINED)
    is_fieldname:   bool = field(init=False, repr=False, default=UNDEFINED)

    # list of python type underneath - e.g. int, str, list, dict, Person, or list of accepted types
    types:          List[type] = field(init=False, default=UNDEFINED)
    # first one - get rid of this one or leave?
    type_:          Union[type, List[type]] = field(init=False, repr=False, default=UNDEFINED)

    # ------------------------------------------------------------

    TYPE_INFO_REGISTRY: ClassVar[Dict[type, Self]] = {}


    def __post_init__(self):
        # if self.th_field and self.th_field.name=='company_type': ...

        if isinstance(self.py_type_hint, str):
            raise EntityInternalError(owner=self, msg=f"py_type_hint={self.py_type_hint} is still string, it should have been resolved before")

        if is_function(self.py_type_hint):
            raise EntityInternalError(owner=self, msg=f"py_type_hint={self.py_type_hint} is a function/method")

        self._extract_type_hint_details()

        # TODO: raise error: if self.is_none_type():

        # if not self.is_list and not self.is_optional:
        #     if self.py_type_hint != self.type_:
        #         raise EntitySetupValueError(item=self, msg=f"Unsupported type hint, got: {self.py_type_hint}. {ERR_MSG_SUPPORTED}")

    def is_none_type(self):
        return self.types == [NoneType]

    def _extract_type_hint_details(self):
        py_type_hint = self.py_type_hint

        origin_type = getattr(py_type_hint, "__origin__", None)

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
        if origin_type in (list, Sequence):
            # List[some_type]
            is_list = True
            if len(py_type_hint.__args__) != 1:
                raise EntitySetupNameError(item=py_type_hint, msg=f"AttrDexpNode's annotation should have single List argument, got: {py_type_hint.__args__}.")
            inner_type = py_type_hint.__args__[0]

        elif origin_type is not None:
            inner_type = origin_type
        else:
            inner_type = py_type_hint


        is_fieldname = False
        if inner_type and getattr(inner_type, "__name__", "").endswith(TYPE_FIELDNAME_SUFFIX):
            is_fieldname = True
            inner_type = inner_type.__bound__ if inner_type.__bound__ else Any

        # Another Union layer allowed, e.g.:  List[Union[str, int, NoneType]]
        if getattr(inner_type, "__origin__", inner_type) == Union: 
            # list of accepted types, e.g.
            #   Union[int, str, NoneType] -> int, str, NoneType
            inner_type = inner_type.__args__ 

        # normalize
        self.is_list = is_list
        self.is_optional = is_optional
        self.is_fieldname = is_fieldname 

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
                    raise EntitySetupValueError(item=self, msg=f"Unsupported type hint, got: {self.py_type_hint}. {ERR_MSG_SUPPORTED}")


    def check_compatible(self, other: Self) -> Optional[str]:
        """
        returns error message when input type is not compatible with given type
        """
        if other is None:
            raise EntityInternalError(owner=self, msg=f"check_compatible({other}) - type_info of other not supplied")

        if self==other:
            return None

        if self.py_type_hint == Any:
            return None

        if self.is_list!=other.is_list:
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
        if self.is_fieldname:
            out.insert(0, "AttrName[")
            out.append("]")

        return "".join(out)

    # def as_str(self) -> str:
    #     return self.py_type_hint.__name__ if type(self.py_type_hint)==type else str(self.py_type_hint)

    # ------------------------------------------------------------
    # Classmethods
    # ------------------------------------------------------------

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
        # msg_prefix = f"{to_repr(caller)}:: " if caller else ""

        if isinstance(py_type_hint, str):
            raise EntitySetupValueError(owner=cls, msg=
                        "{msg_prefix}Python type hint is a string, probably not resolved properly: {repr(py_type_hint)}."
                        "\nHINTS:"
                        "\n  1) if you have `from __future__ import annotations`, remove it, try to import that module, stabilize it and then try this again." 
                        "\n  2) 'get_model_fields() + .type' used instead 'extract_py_type_hints()' -> get_or_create_by_type() (internal issue, use 'extract_model_field_meta' maybe?)"
                        "Explanation: When 'from future import annotations + __annotations__' is used then python hints are strings. Use 'typing.get_type_hints()' to resolve hints properly." 
                        )

        if py_type_hint not in cls.TYPE_INFO_REGISTRY:
            cls.TYPE_INFO_REGISTRY[py_type_hint] = TypeInfo(py_type_hint=py_type_hint)

        return cls.TYPE_INFO_REGISTRY[py_type_hint]

    # ------------------------------------------------------------

    @classmethod
    def extract_function_return_type_info(
            cls,
            py_function: Callable[..., Any], 
            allow_nonetype:bool=False) -> Self:

        py_type_hint_dict = extract_function_py_type_hint_dict(function=py_function)
        py_type_hint = py_type_hint_dict.get("return", None)
        name = getattr(py_function, "__name__", "?")

        msg_prefix = f"Function {py_function}::"
        if not py_type_hint:
            raise EntitySetupNameError(item=py_function, msg = msg_prefix + f"AttrDexpNode FUNCTION '{name}' is not valid, it has no return type hint (annotations).")
        if not allow_nonetype and py_type_hint == NoneType:
            raise EntitySetupNameError(item=py_function, msg = msg_prefix + f"SetupSession: AttrDexpNode FUNCTION '{name}' is not valid, returns None (from annotation).")

        output = TypeInfo.get_or_create_by_type(
                        py_type_hint=py_type_hint,
                        caller=msg_prefix,
                        )
        return output

    # ------------------------------------------------------------

    @classmethod
    def extract_function_arguments_type_info_dict(
            cls,
            py_function: Callable[..., Any]) -> Dict[str, Self]:
        """
        From annotations, but argument defaults could be fetched from
        inspect.getfullargspec(), see: extract_function_arguments_default_dict
        """
        py_type_hint_dict = extract_function_py_type_hint_dict(function=py_function)

        msg_prefix = f"Function {py_function}::"

        output = {}
        name = getattr(py_function, "__name__", "?")
        for arg_name, py_type_hint in py_type_hint_dict.items():
            if arg_name == "return":
                continue
            if not py_type_hint:
                raise EntitySetupNameError(item=py_function, msg=msg_prefix + f"AttrDexpNode FUNCTION '{name}.{arg_name}' is not valid, argument {arg_name} has no type hint (annotations).")
            if py_type_hint in (NoneType,):
                raise EntitySetupNameError(item=py_function, msg=msg_prefix + f"SetupSession: AttrDexpNode FUNCTION '{name}.{arg_name}' is not valid, argument {arg_name} has type hint (annotation) None.")

            output[arg_name] = TypeInfo.get_or_create_by_type(
                                    py_type_hint=py_type_hint,
                                    caller=msg_prefix,
                                    )
        return output



# ------------------------------------------------------------
# OBSOLETE
# ------------------------------------------------------------

# Use higher level function: extract_type_info() instead
# 
# def extract_py_type_hints_for_attr(inspect_object: Any, attr_name: str, caller_name: str) -> PyTypeHint:
#     py_type_hint_dict = extract_py_type_hints(inspect_object=inspect_object, caller_name=caller_name, strict=True)
#     if attr_name not in py_type_hint_dict:
#         raise EntitySetupValueError(item=inspect_object, msg=f"{caller_name}: Object type hint for '{attr_name}' not available. Available are {','.join(py_type_hint_dict.keys())}.")
#     hint_type = py_type_hint_dict[attr_name]
#     if not hint_type: # check if type
#         raise EntitySetupValueError(item=inspect_object, msg=f"{caller_name}: Object type hint for '{attr_name}' is not valid '{hint_type}'. Expected some type.")
#     return hint_type

