# ------------------------------------------------------------
# BUILDING BLOCKS OF RULES
# ------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Union, List, Optional, Dict
from dataclasses import dataclass, field, InitVar, is_dataclass
from decimal import Decimal
from enum import Enum

# TODO: from .types         import *
# from .types         import (
#         )
from .exceptions    import RuleSetupValueError, RuleSetupError
from .namespaces    import ModelsNS, ThisNS, FieldsNS
from .utils         import (
        is_function, 
        is_enum, 
        is_pydantic, 
        get_available_vars_sample,
        UNDEFINED,
        UndefinedType,
        )
from .base import (
        BaseOnlyArgs,
        repr_obj,
        add_indent_to_strlist,
        obj_to_strlist,
        list_to_strlist,
        warn,
        ComponentBase,
        TypeHintField,
        BoundVar,
        )
from .expressions   import (ValueExpression, Operation, VExpStatusEnum)
from .variables     import Variable, VariablesHeap

# ------------------------------------------------------------
# Functions
# ------------------------------------------------------------

def _(message:str) -> TransMessageType:
    return message

# TODO: add type hint: TransMessageType -> TranslatedMessageType
# TODO: accept "{variable}" - can be a security issue, variables() should not make any logic
#       use .format() ... (not f"", btw. should not be possible anyway)

class msg(BaseOnlyArgs):
    pass


class FieldTypeEnum(str, Enum):
    INPUT  = "input"
    EMAIL  = "email"  # no special class
    NUMBER = "number" # no special class
    BOOL   = "bool"
    CHOICE = "choice" # html select
    ENUM   = "enum"
    FILE   = "file"
    PASSWORD  = "password"
    DATE  = "date"


# ------------------------------------------------------------
# COMPONENTS
# ------------------------------------------------------------

# dataclass required?
@dataclass
class Component(ComponentBase):

    # by default name should be defined
    # name: str = field(init=False, default=UNDEFINED)

    # NOTE: I wanted to skip saving owner reference/object within component - to
    #       preserve single and one-direction references. 
    owner:      Union[ComponentBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    owner_name: Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

    variable:   Union[Variable, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)

    def init_clean_base(self):
        # hopefully will be later defined - see set_owner
        # TODO: found no way to declare name 
        if self.name not in (None, "", UNDEFINED):
            if not self.name:
                raise RuleSetupValueError(owner=self, msg=f"{self} -> attribute name is not defined")

            if not self.name.isidentifier():
                raise RuleSetupValueError(owner=self, msg=f"{self.name} -> attribute name needs to be valid python identifier name")

    def __post_init__(self):
        self.init_clean_base()


    # def get_owner_variable(self, heap:VariableHeap) -> Variable:
    #     if self.owner_name:
    #         return heap.get_var(namespace=FieldsNS, var_name=self.owner_name)
    #     return None

    # ------------------------------------------------------------

# ------------------------------------------------------------


@dataclass
class DataVar(Component):
    name:           str
    label:          TransMessageType
    # TODO: the type of datatype and value should match
    value:          Union[ValueExpression, Callable[[], RuleDatatype]]
    datatype:       Optional[RuleDatatype] = None # TODO: should be calculated - field(init=False)
    evaluate:       bool = False # TODO: describe 

    def __post_init__(self):
        if not (isinstance(self.value, ValueExpression) or callable(self.value)):
            raise RuleSetupValueError(owner=self, msg=f"{self.name} -> {type(self.value)}: {type(self.value)} - not ValueExpression|Callable")
        self.init_clean_base()


@dataclass
class Validation(Component):
    """
    new custom validations could be done like this:

        @dataclass
        class ValidationHourValue(Validation):
            def __init__(self, name:str, label:TransMessageType):
                super().__init__(
                        name=name, label=label,
                        ensure=((This.value>=0) & (This.value<=23)),
                        error=_("Need valid hour value (0-23)"),
                        )

    """
    name:           str
    label:          TransMessageType
    ensure:         ValueExpression
    error:          TransMessageType
    available:      Optional[Union[bool, ValueExpression]] = True


@dataclass
class Section(Component):
    name:           str
    label:          str
    contains:       List[Component] = field(repr=False)
    # TODO: allow ChildrenValidation too (currently only used in Extension, e.g. Cardinality/Unique)
    validations:    Optional[List[Validation]] = None
    available:      Union[bool, ValueExpression] = True


@dataclass
class Field(Component):
    # to Model attribute
    bind:           ValueExpression
    label:          TransMessageType

    # optional
    type:           Optional[FieldTypeEnum] = None # TODO: Enum or implicit conversion?
    # TODO: required is also Validation() - nothing special that deserves
    #       special attribute or not? this could be shortcut, but then we will lack
    #       some special features that Validation enable.
    required:       Optional[Union[bool,ValueExpression]] = False
    # TODO: maybe UNDEFINED?
    default:        Optional[Union[StandardType, ValueExpression]] = None
    description:    Optional[TransMessageType] = None
    available:      Union[bool, ValueExpression] = True
    editable:       Union[bool, ValueExpression] = True
    validations:    Optional[List[Validation]] = None
    autocomplete:   Optional[bool] = True
    enables:        Optional[List[Component]] = field(default=None, repr=False)
    # NOTE: this has no relation to type hinting - this is used for html input placeholder attribute
    hint:           Optional[TransMessageType] = None
    # if not supplied name will be extracted from binded model variable
    name:           Optional[str] = None
    variable:       Union[Variable, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)
    bound_variable: Union[Variable, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)

    def __post_init__(self):
        self.init_clean()

    def init_clean(self):
        # TODO: check that value is simple M. value
        if not isinstance(self.bind, ValueExpression):
            raise RuleSetupValueError(owner=self, msg="'bind' needs to be ValueExpression (e.g. M.status).")
        # ModelsNs.person.surname -> surname
        if not self.name:
            self.name = self.get_name_from_bind(self.bind)

        # TODO: this is a mess - remove self.type ?
        if self.type is None:
            if self.__class__==BooleanField:
                self.type = FieldTypeEnum.BOOL
            elif self.__class__==ChoiceField:
                self.type = FieldTypeEnum.CHOICE
            elif self.__class__==EnumField:
                self.type = FieldTypeEnum.ENUM
            else:
                self.type=FieldTypeEnum.INPUT
        else:
            # type should be ommited?
            if self.__class__==BooleanField:
                raise RuleSetupValueError(owner=self, msg=f"For BooleanField type should be ommited.")
            elif self.__class__==ChoiceField:
                raise RuleSetupValueError(owner=self, msg=f"For ChoiceField type should be ommited.")
            elif self.__class__==EnumField:
                raise RuleSetupValueError(owner=self, msg=f"For EnumField type should be ommited.")

            # use classes?
            elif self.__class__!=BooleanField and self.type==FieldTypeEnum.BOOL:
                raise RuleSetupValueError(owner=self, msg=f"For type='checkbox' use BooleanField instead.")
            elif self.__class__!=ChoiceField and self.type==FieldTypeEnum.CHOICE:
                raise RuleSetupValueError(owner=self, msg=f"For type='choice' use ChoiceField instead.")
            elif self.__class__!=EnumField and self.type==FieldTypeEnum.ENUM:
                raise RuleSetupValueError(owner=self, msg=f"For type='enum' use EnumField instead.")

            # normalize to Enum value
            type_values = {ev.value:ev for ev in FieldTypeEnum}
            if self.type not in type_values:
                raise RuleSetupValueError(owner=self, msg=f"type='{self.type}' needs to be one of {type_values.keys()}.")
            self.type = type_values[self.type]

        # NOTE: for dataclass found no way to call in standard "inheritance
        #       way", super().clean_base()
        self.init_clean_base()

    def setup(self, heap:VariableHeap):
        super().setup(heap=heap)
        if self.bind:
            # within all parents catch first with namespace_only attribute
            # if such - check if namespace of all children are right. 
            # Used for Extension.
            namespace_only = ModelsNS
            owner = self.owner
            while owner!=None:
                if hasattr(owner, "namespace_only"):
                    namespace_only = owner.namespace_only
                    break
                owner = owner.owner

            if self.bind.GetNamespace()!=namespace_only:
                raise RuleSetupValueError(owner=self, msg=f"{self.bind}: 'bind' needs to be in {namespace_only} ValueExpression (e.g. M.status).")
            if len(self.bind.Path) not in (1,2,3):
                # not warning 
                warn(f"{self.bind}: 'bind' needs to be 1-3 deep ValueExpression (e.g. M.status or M.person.status).")

            self.bound_variable = heap.get_var_by_vexp(self.bind)
            if not self.bound_variable:
                # TODO: not nice :(
                owner_container = self.get_owner_container()
                owner_heap = getattr(owner_container, "owner_heap", owner_container.heap)
                if owner_heap!=heap:
                    self.bound_variable = owner_heap.get_var_by_vexp(self.bind)

            self.variable = heap.get_var(FieldsNS, self.name)
            # self.heap.Fields[self.name]
            assert self.variable
            if not self.bound_variable:
                # TODO: ignore for now ...
                pass
            else:
                # ALT: self.bound_variable.add_bound_var(BoundVar(heap.name, self.variable.namespace, self.variable.name))
                self.variable.add_bound_var(
                        BoundVar(heap.name, 
                                 self.bound_variable.namespace, 
                                 self.bound_variable.name))

    def get_variable(self, heap:VariableHeap) -> Variable: 
        assert self.variable
        return self.variable

    def get_bound_variable(self, heap:VariableHeap) -> Variable: 
        assert self.bind._all_ok, self.bind._status
        assert self.bound_variable


# ------------------------------------------------------------
# BooleanField 
# ------------------------------------------------------------

class BooleanField(Field):
    default:        Optional[Union[bool, ValueExpression]] = None

    def __post_init__(self):
        self.init_clean()
        if self.default is not None and not isinstance(self.default, (bool, ValueExpression)):
            raise RuleSetupValueError(owner=self, msg=f"'default'={self.default} needs to be bool value  (True/False).")

# ------------------------------------------------------------
# ChoiceField 
# ------------------------------------------------------------


@dataclass
class ChoiceField(Field):
    # https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
    # required is not required :) so in inherited class all attributes need to be optional
    # Note that with Python 3.10, it is now possible to do it natively with dataclasses.

    # List[Dict[str, TransMessageType]
    choices: Optional[Union[Callable[..., Any], ValueExpression, List[Union[ChoiceOption, int, str]]]] = None
    choice_value: Optional[ValueExpression] = None
    choice_label: Optional[ValueExpression] = None
    # choice_available: Optional[ValueExpression]=True # returns bool

    choice_value_th_field: TypeHintField = field(init=False, default=None, repr=False)
    choice_label_th_field: TypeHintField = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.init_clean()
        if self.choices is None:
            raise RuleSetupValueError(owner=self, msg=f"{self.name}: {self.__class__.__name__}: argument 'choices' is required.")
        if is_enum(self.choices):
            raise RuleSetupValueError(owner=self, msg=f"{self.name}: {self.__class__.__name__}: argument 'choices' is Enum, use EnumChoices instead.")

    def setup(self, heap:VariableHeap):
        super().setup(heap=heap)
        choices = self.choices
        choices_checked = False

        if isinstance(choices, ValueExpression):
            # TODO: restrict to vexp only - no operation
            if choices._status!=VExpStatusEnum.OK:
                # reported before - warn(f"TODO: There is an error with value expression {self.choices} - skip it for now.")
                choices = None
            else:
                # TODO: parent?
                # variable = choices.GetVariable(heap=heap, strict=False)
                variable = heap.get_var_by_vexp(vexp=choices)
                if not variable:
                    variable = choices.Setup(heap=heap, owner=self, parent=None)
                choices = variable.data.klass
                is_list = variable.data.is_list
                if is_list and variable.namespace==ModelsNS and is_pydantic(choices) or is_dataclass(choices):
                    # FK case - e.g. default_device_type
                    choices_checked = True
                    # TODO: deny default value - not available in this moment?

        if choices_checked:
            pass
        elif choices==None:
            # ignored
            pass
        elif is_function(choices) or is_dataclass(choices) or is_pydantic(choices):
            if is_function(choices):
                fun_return_type_hint_field = TypeHintField.extract_function_return_type_hint_field(choices)
                fun_return_type, is_list = fun_return_type_hint_field.klass, fun_return_type_hint_field.is_list
                if not is_list:
                    raise RuleSetupValueError(owner=self, msg=f"{self.name}: {self.__class__.__name__}: argument 'choices'={choices} is a function that does not return List[type]. Got: {fun_return_type}")
            else:
                fun_return_type = choices

            for aname in ("choice_value", "choice_label"):
                aval = getattr(self, aname, None)
                if not (aval and isinstance(aval, ValueExpression) and aval.GetNamespace()==ThisNS):
                    raise RuleSetupValueError(owner=self, msg=f"{self.name}: {self.__class__.__name__}: argument '{aname}' is not set or has wrong type - should be ValueExpression in This. namespace. Got: {aval} / {type(aval)}")
                if len(aval.Path)!=1:
                    raise RuleSetupValueError(owner=self, msg=f"{self.name}: {self.__class__.__name__}: argument '{aname}' should be one level deep (e.g. This.value). Got: {aval}")
                var_name = aval.Path[-1]._node
                type_hint_field = TypeHintField.extract_type_hint_field(var_name=var_name, inspect_object=fun_return_type)
                if not type_hint_field:
                    vars_avail = get_available_vars_sample(var_name, dir(fun_return_type))
                    raise RuleSetupValueError(owner=self, msg=f"{self.name}: {self.__class__.__name__}: argument '{aname}' field not found, available: {vars_avail}")
                # set choice_label_th_field and choice_value_th_field
                setattr(self, f"{aname}_th_field", type_hint_field)

            if self.choice_label_th_field.klass!=str:
                raise RuleSetupValueError(owner=self, msg=f"{self.name}: choice_label needs to be bound to string attribute, got: {choice_label_th_field.klass}")

        elif isinstance(choices, (list, tuple)):
            if len(choices)==0:
                raise RuleSetupValueError(owner=self, msg=f"{self.name}: {self.__class__.__name__}: 'choices' is an empty list, Provide list of str/int/ChoiceOption.")
            if self.choice_value or self.choice_label:
                raise RuleSetupValueError(owner=self, msg=f"{self.name}: {self.__class__.__name__}: when 'choices' is a list, choice_value and choice_label are not permitted.")
            # now supports combining - but should have the same type
            for choice in choices:
                if not isinstance(choice, (str, int, ChoiceOption)):
                    raise RuleSetupValueError(owner=self, msg=f"{self.name}: {self.__class__.__name__}: choices has invalid choice, not one of str/int/ChoiceOption: {choice} / {type(choice)}")
        else:
            raise RuleSetupValueError(owner=self, msg=f"{self.name}: {self.__class__.__name__}: choices has invalid value, not Union[Callable, ValueExpression, List[Union[ChoiceOption, int, str]]], got : {choices} / {type(choices)}")



@dataclass
class EnumField(Field):
    enum: Optional[Enum] = None

    # def __post_init__(self):
    #     self.init_clean()

    def setup(self, heap:VariableHeap):
        super().setup(heap=heap)
        # TODO: revert to: strict=True - and process exception properly
        variable = heap.get_var_by_vexp(vexp=self.bind, strict=False)
        if variable:
            # when not found -> it will be raised in other place
            if not isinstance(variable.data, TypeHintField):
                raise RuleSetupValueError(owner=self, msg=f"Data type of variable {variable} should be TypeHintField, got: {type(variable.data)}")
            enum = variable.data.py_type_hint
            if not is_enum(enum):
                if not self.enum:
                    raise RuleSetupValueError(owner=self, msg=f"Data type of variable {variable} should be Enum or supply EnumField.enum. Got: {enum}")
                # get type from first member
                enum_value_type = type(list(self.enum.__members__)[0])
                if (not issubclass(self.enum, enum) # for Enum(str, ...) and IntEnum 
                    and (type(enum)==type and enum!=enum_value_type)): # str == Enum()
                    raise RuleSetupValueError(owner=self, msg=f"Data type of variable {variable} should be the same as supplied Enum. Enum {self.enum}/{enum_value_type} is not {enum}.")
            else:
                if self.enum and self.enum!=enum:
                    raise RuleSetupValueError(owner=self, msg=f"Variable {variable} has predefined enum {self.enum} what is different from type_hint: {enum}")
                self.enum = enum

                # TODO: default=None is not good default's default, can be member of some Enum
                if self.default is not None:
                    if not isinstance(self.default, self.enum):
                        raise RuleSetupValueError(owner=self, msg=f"Default should be an Enum {self.enum} value, got: {self.default}")


@dataclass
class ChoiceOption:
    value:      ValueExpression # -> some Standard or Complex type
    label:      TransMessageType
    available:  Optional[Union[ValueExpression,bool]] = True # Vexp returns bool


