from enum import Enum
from collections.abc import Sized
from typing import (
    TypeVar,
    Tuple,
    Dict,
    Optional,
    List,
    Sequence,
    Any,
)

from .base import ChildField
from .exceptions import (
    EntityApplyValueError,
    EntitySetupValueError,
)
from .meta import (
    ItemType,
    NumberType,
    is_enum,
    get_enum_members,
)
from .expressions import (
    DotexprFuncArgHint,
    AttrnameFuncArgHint, JustDotexprFuncArgHint,
)
from .functions import (
    create_builtin_function_factory,
    create_builtin_items_function_factory,
    BuiltinFunctionFactory,
    CustomFunctionFactory,
    FunctionArgumentsType,
    InjectComponentTreeValuesFuncArgHint,
    DotexprExecuteOnItemFactoryFuncArgHint,
)


def get_builtin_function_factories_dict() -> Dict[str, BuiltinFunctionFactory]:
    out: Dict[str, BuiltinFunctionFactory] = {}
    for func_name, global_var in globals().copy().items():
        if not (global_var and isinstance(global_var, BuiltinFunctionFactory)):
            continue
        assert func_name
        out[func_name] = global_var
    return out


# ------------------------------------------------------------
# Builtin function factories
# ------------------------------------------------------------
def enum_members(enum: Enum) -> List[Tuple[str, Any]]:
    if not is_enum(enum):
        raise EntityApplyValueError(msg=f"Expecting enum type, got: {enum} / {type(enum)}")
    return get_enum_members(enum)


def EnumMembers(
        enum: Enum, 
        name: Optional[str] = None,
        ) -> CustomFunctionFactory:
    # TODO: consider to create/return CustomItemsFunctionFactory
    """
    used for EnumField(choices=...) - instead of obsolete data=[StaticData] and DataNs.
    it is not BuiltinFunctionFactory since it is added to functions= argument.
    """
    if not is_enum(enum):
        raise EntitySetupValueError(msg=f"Expecting enum type, got: {enum} / {type(enum)}")
    if not name:
        name = enum.__name__
    kwargs: Dict[str, Enum] = {"enum": enum}
    return CustomFunctionFactory(
                py_function=enum_members,
                name=name,
                fixed_args=FunctionArgumentsType([], kwargs),
                data=enum,
                )


# ------------------------------------------------------------
# Builtin functions
# ------------------------------------------------------------

T = TypeVar("T", bound=Any)

# def children(value: Any, inject_component_tree: ComponentTreeWValuesType) -> List[ChildField]:
#     return inject_component_tree["contains"]

def children(value: Any, component_tree: InjectComponentTreeValuesFuncArgHint()) -> List[ChildField]:
    """
    Component's list of children
    TODO: ex. was List[ItemType] - should convert "contains" to child_list
    """
    return component_tree["contains"]

Children = create_builtin_function_factory(  # noqa: E305
            children, name="Children", 
            # arg_validators={"component": ensure_is_component},
            )


def single_bool_selected(children_list: List[ItemType]) -> int:
    """ for objects from datastores - e.g. rows, iterables """
    return len(
        [child for child in children_list if child["value_instance"] is True]
    ) == 1
    # ORIG2: child["value_instance"].get_value(strict=False)...

SingleBoolSelected = create_builtin_function_factory(  # noqa: E305
            single_bool_selected, name="SingleBoolSelected", 
            )



def length(value_sized: Sized) -> int:
    """ TODO: for string and other similar """
    return len(value_sized)

Length = create_builtin_function_factory( # noqa: E305
            length, name="Length", 
            # arg_validators={"item_list": ensure_has_len},
            )

def upper(str_value: str) -> str:
    return str.upper(str_value)
Upper = create_builtin_function_factory(upper, name="Upper") # noqa: E305

def lower(str_value: str) -> str:
    return str.lower(str_value)
Lower = create_builtin_function_factory(lower, name="Lower") # noqa: E305


def substring(str_value: str, start:Optional[int], end:Optional[int]=None) -> str:
    return str_value[start:end]
Substring = create_builtin_function_factory(substring, name="Substring") # noqa: E305

def startswith(str_value: str, prefix:str) -> bool:
    return str_value.startswith(prefix)
Startswith = create_builtin_function_factory(startswith, name="Startswith") # noqa: E305


# ------------------------------------------------------------
# NOTE: In following functions there is lot of copy/paste and make small
#       changes due better performance (missing "inline" or "macro"
#       functionality).  Could be generated and eval("...").
# ------------------------------------------------------------

def count(item_list: Sequence[ItemType]) -> int:
    """ for objects from datastores - e.g. rows, iterables """
    return len(item_list)

Count = create_builtin_items_function_factory(  # noqa: E305
            items_value_arg_name="item_list",
            py_function=count, name="Count",
            # arg_validators={"item_list": ensure_is_list},
        )


def sum_(item_list: Sequence[ItemType], field_name: AttrnameFuncArgHint(NumberType)) -> NumberType:
    """
    NOTE: underlying type must be matched dynamically
    """
    # if not field_name:
    #     return sum(item_list)
    # else:
    return sum([getattr(item, field_name, 0) for item in item_list])

Sum = create_builtin_items_function_factory(
            items_value_arg_name="item_list",
            py_function=sum_, name="Sum",
            # arg_validators={"item_list": [ ensure_is_list, ensure_is_number ]},
        )


# def map_(item_list: Sequence[ItemType], callable_or_fieldname: Union[Callable[[Any], Any], str]) -> Sequence[ItemType]:
def map_(item_list: Sequence[ItemType], dot_expr: DotexprFuncArgHint(inner_type=Any)) -> Sequence[ItemType]:
    for item in item_list:
        yield dot_expr._evaluator.evaluate(item.value)
    raise NotImplementedError()
    # " returns iterator "
    # if isinstance(callable_or_fieldname, str):
    #     return (getattr(item, callable_or_fieldname, None) for item in item_list)
    # elif callable(callable_or_fieldname):
    #     return (callable_or_fieldname(item) for item in item_list)
    # raise TypeError(f"Argument expected to be callable or string (fieldname), got: {callable_or_fieldname} -> {type(callable_or_fieldname)}")

Map = create_builtin_items_function_factory(
            items_value_arg_name="item_list",
            py_function=map_, name="Map",
            # arg_validators={"item_list": ensure_is_list},
        )


# def filter_(item_list: Sequence[ItemType], bool_dot_expr: FuncArgDotExprBoolType) -> Sequence[ItemType]:
def filter_(item_list: Sequence[ItemType],
            dot_expr_execute_on_item_function: DotexprExecuteOnItemFactoryFuncArgHint(),
            term_dot_expr: JustDotexprFuncArgHint(inner_type=bool)) -> Sequence[ItemType]:
    """
    TODO: can become iterator/generator
    """
    # if not isinstance(term_dot_expr, DynamicAttrsBase):
    #     raise TypeError(f"Argument expected to be FuncArgDotExprBoolType - DotExpression, got: {term_dot_expr} -> {type(term_dot_expr)}")
    # if not callable(dot_expr_execute_on_item_function):
    #     raise TypeError(f"Argument expected to be callable, got: {term_dot_expr} -> {type(dot_expr_execute_on_item_function)}")
    output = []
    for item in item_list:
        term_value = dot_expr_execute_on_item_function(term_dot_expr, item)
        if term_value:
            output.append(item)
    return output


Filter = create_builtin_items_function_factory(
            items_value_arg_name="item_list",
            py_function=filter_,
            name="Filter",
            # arg_validators={"item_list": ensure_is_list},
        )

# ------------------------------------------------------------

# def max_(item_list: Sequence[ItemType], callable_or_fieldname: Optional[Union[Callable[[Any], Any], str]] = None) -> ItemType:
def max_(item_list: Sequence[ItemType], dot_expr: DotexprFuncArgHint(inner_type=Any)) -> Optional[Any]:
    for item in item_list:
        yield dot_expr._evaluator.evaluate(item.value)
    raise NotImplementedError
    # if not callable_or_fieldname:
    #     return max([item for item in item_list])
    # elif isinstance(callable_or_fieldname, str):
    #     return max([getattr(item, callable_or_fieldname, None) for item in item_list])
    # elif callable(callable_or_fieldname):
    #     return max([callable_or_fieldname(item) for item in item_list])
    # raise TypeError(f"Argument expected to be callable or string (fieldname), got: {callable_or_fieldname} -> {type(callable_or_fieldname)}")


Max = create_builtin_items_function_factory(
            items_value_arg_name="item_list",
            py_function=max_, name="Max",
            # arg_validators=[ensure_is_list],
        )

# ------------------------------------------------------------

# def min_(item_list: Sequence[ItemType], callable_or_fieldname: Optional[Union[Callable[[Any], Any], str]] = None) -> ItemType:
def min_(item_list: Sequence[ItemType], dot_expr: DotexprFuncArgHint(inner_type=Any)) -> Optional[Any]:
    raise NotImplementedError()
    # if not callable_or_fieldname:
    #     return min([item for item in item_list])
    # elif isinstance(callable_or_fieldname, str):
    #     return min([getattr(item, callable_or_fieldname, None) for item in item_list])
    # elif callable(callable_or_fieldname):
    #     return min([callable_or_fieldname(item) for item in item_list])
    # raise TypeError(f"Argument expected to be callable or string (fieldname), got: {callable_or_fieldname} -> {type(callable_or_fieldname)}")

Min = create_builtin_items_function_factory(
            items_value_arg_name="item_list",
            py_function=min_, name="Min",
            # arg_validators=[ensure_is_list],
        )

# def _process_item(item: ItemType, callable_or_fieldname: Optional[Union[Callable[[Any], Any], str]] = None) -> ItemType:
#     if not callable_or_fieldname:
#         return item
#     elif isinstance(callable_or_fieldname, str):
#         return getattr(item, callable_or_fieldname, UNDEFINED)
#     elif callable(callable_or_fieldname):
#         return callable_or_fieldname(item)
#     raise TypeError(f"Argument expected to be callable or string (fieldname), got: {callable_or_fieldname} -> {type(callable_or_fieldname)}")

# def first(item_list: Sequence[ItemType], callable_or_fieldname: Optional[Union[Callable[[Any], Any], str]] = None) -> ItemType:
def first(item_list: Sequence[ItemType], dot_expr: DotexprFuncArgHint(inner_type=Any)) -> Optional[Any]:
    raise NotImplementedError()
    # if not item_list:
    #     return UNDEFINED
    # return _process_item(item_list[0])

First = create_builtin_items_function_factory(
            items_value_arg_name="item_list",
            py_function=first, name="First",
            # arg_validators=[ensure_is_list],
        )

# def last(item_list: Sequence[ItemType], callable_or_fieldname: Optional[Union[Callable[[Any], Any], str]] = None) -> ItemType:
def last(item_list: Sequence[ItemType], dot_expr: DotexprFuncArgHint(inner_type=Any)) -> Optional[Any]:
    raise NotImplementedError()
    # if not item_list:
    #     return UNDEFINED
    # return _process_item(item_list[-1])

Last = create_builtin_items_function_factory(
            items_value_arg_name="item_list",
            py_function=last, name="Last",
            # arg_validators=[ensure_is_list],
        )



# ----------------------------------------------------------------------------
# ValueArgValidatioPyFuncType -> functions for value argument type validation
# ----------------------------------------------------------------------------

#    For validation error cases should:
#       * return string error message, or
#       * raise EntitySetupError based error

# def ensure_is_number(arg_name: str, value_arg_type_info: TypeInfo) -> Optional[str]:
#     if not value_arg_type_info.type_ in (int, float, Decimal):
#         raise EntitySetupTypeError(f"Expected origin type to be int/float/Decimal type, got: {value_arg_type_info.py_type_hint}")
# 
# def ensure_is_list(arg_name: str, value_arg_type_info: TypeInfo) -> Optional[str]:
#     if not value_arg_type_info.is_list:
#         raise EntitySetupTypeError(f"Expected list/tuple/sequence type, got: {value_arg_type_info.py_type_hint}")
# 
# def ensure_has_len(arg_name: str, value_arg_type_info: TypeInfo) -> Optional[str]:
#     if not hasattr(value_arg_type_info.type_, "__len__"):
#         raise EntitySetupTypeError(f"Expected type with __len__ method implemented, got: {value_arg_type_info.py_type_hint}")

# TODO: ove implementiraj - da rade na listi i da rade na jednom item-u
#       Oldest - same as First, but for date/datetime fields ?
#       Newest - same as Last, but for date/datetime fields?
#       If (CaseWhen)
#       Reduce
#       CountDistinct

# ------------------------------------------------------------
# Common validators
# ------------------------------------------------------------

# standardne funkcije za primitivne podatke, npr. upper, abs, substring, (može i [:])
# upiti prema podobjektima: kao django ORM <podobjekt>_set, npr. address_set
# obrada listi (podobjekti) - Filter, Map, Reduce + standardni agregatori Count, Min, Max i neki napredniji Newest, Oldest, Exists

#            https://docs.oracle.com/cd/E57185_01/IRWUG/ch12s04s05.html
#            str: upper, lower, substring, right, left, replace, rtrim, ltrim, trim, chr, ascii, locate, posstr
#            all: cast??, CaseWhen
#            numeric: mod, ceil, floor, round, trunc, abs, sqrt, exp, sign, sin, cos, tan
#            null: coalesce, nullif, IsNull??
#            date: current_time, current_date, current_timestamp, castDate, year, month, day, ... dateadd, datediff
#            interval: ??
#            user:
#
#          aggregatation
#            sum, min, max, count, avg, stddev, countDistinct

