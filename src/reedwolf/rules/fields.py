"""
------------------------------------------------------------
BUILDING BLOCKS OF RULES
------------------------------------------------------------
FIELDS - can read/store data from/to storage
------------------------------------------------------------
check components.py too
"""
from __future__ import annotations

from abc import ABC
from typing import (
        Union, 
        List, 
        Optional, 
        Any, 
        ClassVar, 
        Dict,
        )
from dataclasses import dataclass, field
import inspect

from decimal import Decimal
from datetime import date, time, datetime, timedelta
# from pathlib import Path
from enum import Enum, IntEnum

from .utils import (
        UNDEFINED,
        NA_DEFAULTS_MODE,
        UndefinedType,
        message_truncate,
        varname_to_title,
        )
from .exceptions import (
        RuleSetupValueError,
        RuleInternalError,
        RuleSetupError,
        RuleSetupTypeError,
        )
from .namespaces import (
        ModelsNS,
        FieldsNS,
        ThisNS,
        )
from .meta import (
        TransMessageType,
        TypeInfo,
        is_enum,
        is_model_class,
        is_function,
        get_enum_member_py_type,
        EmptyFunctionArguments,
        STANDARD_TYPE_LIST,
        )
from .base import (
        warn,
        IFieldBase,
        IApplySession,
        ValidationFailure,
        SetupStackFrame,
        )
from .expressions   import (
        DotExpression,
        DExpStatusEnum,
        IFunctionDexpNode,
        )
from .attr_nodes import (
        AttrDexpNode
        )
from .functions import (
        CustomFunctionFactory,
        )
from .registries import (
        SetupSession,
        )
from .validations   import (
        MaxLength,
        ExactLength,
        )
from .components    import (
        Component,
        ValidationBase,
        EvaluationBase,
        )


# Items are: 
#   - when single - that must be added
#   - when tuple(validtions) then one of listed must be added
RequiredValidationsType = List[Union[ValidationBase, List[ValidationBase]]]


class AutocomputedEnum(IntEnum):
    NO       = 0  # if False; must be bool() === False
    ALLWAYS  = 1  # if True
    SOMETIMES= 2  # must be set manually

    @classmethod
    def from_value(cls, value: Union[bool, AutocomputedEnum]) -> AutocomputedEnum:
        if value is True:
            value = cls.ALLWAYS
        elif value is False:
            value = cls.NO
        return cls(value)

# ============================================================
# Fields
# ============================================================

@dataclass
class FieldBase(Component, IFieldBase, ABC):
    # abstract property:
    PYTHON_TYPE:ClassVar[type(UNDEFINED)] = UNDEFINED
    REQUIRED_VALIDATIONS:ClassVar[Optional[RequiredValidationsType]] = None

    # to Model attribute
    bind:           DotExpression
    label:          Optional[TransMessageType] = field(repr=False, default=None)

    # NOTE: required - is also Validation, i.e. Required() - just commonly used
    #       validation, nothing special that deserves special attribute.
    #       Required() adds extra possible features.
    #   required:       Optional[Union[bool,DotExpression]] = False
    #
    # NOTE: default - is initialization Evaluation, so use Default() - commonly
    #       used evaluation that is triggered on object instatation.
    #       Default() adds extra features.
    #   default:        Optional[Union[StandardType, DotExpression]] = None

    description:    Optional[TransMessageType] = field(repr=False, default=None)


    # NOTE: available that yields True(ish?) value 
    #   => this is NOT processed: 
    #       - field update by new instance, 
    #       - cleaners for the field
    #       - children tree (contains/enables)
    #   Only initial value is unchanged. See also BooleanField.enables.
    available:      Union[bool, DotExpression] = field(repr=False, default=True)
    # NOTE: replaced with Readonly(False|DotExpression...) 
    #       editable:       Union[bool, DotExpression] = field(repr=False, default=True)
    cleaners:       Optional[List[Union[ValidationBase, EvaluationBase]]] = field(repr=False, default=None)
    autocomputed:   Union[bool, AutocomputedEnum] = field(repr=False, default=False)


    # BooleanField.enables that yields False/None (value must be bool/None)
    #   => this is NOT processed:
    #       - children tree (enables) 
    #   This is equivalent to 
    #       FieldGroup(
    #           available=<this-field.value>, 
    #           contains=<this-field.enables-list>)
    #   See also "available".

    # TODO: možda složiti da radi i za Choice/Enum -> structural pattern
    #       matching like, samo nisam još našao zgodnu sintaksu.
    enables:        Optional[List[Component]] = field(repr=False, default=None)

    # NOTE: this has no relation to type hinting - this is used for html input placeholder attribute
    hint:           Optional[TransMessageType] = field(repr=False, default=None)

    # for arbitrary custom values, rules system ignores the field
    #   e.g. "autocomplete": True ...
    meta:           Optional[Dict[str, Any]] = field(repr=False, default=None)

    # if not supplied name will be extracted from binded model attr_node
    name:            Optional[str] = None
    attr_node:       Union[AttrDexpNode, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)
    bound_attr_node: Union[AttrDexpNode, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)
    python_type:     Union[type, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)
    type_info:       Optional[TypeInfo] = field(init=False, repr=False, default=UNDEFINED)

    # will be set later
    # is_key:          bool = field(init=False, repr=False, default=False)


    def __post_init__(self):
        self.init_clean()
        super().__post_init__()


    def init_clean(self):
        # TODO: check that value is simple M. value
        if not isinstance(self.bind, DotExpression):
            raise RuleSetupValueError(owner=self, msg="'bind' needs to be DotExpression (e.g. M.status).")
        # ModelsNs.person.surname -> surname
        if not self.name:
            self.name = self.get_name_from_bind(self.bind)
        if not self.label:
            self.label = varname_to_title(self.name)

        self.autocomputed = AutocomputedEnum.from_value(self.autocomputed)
        self.init_clean_base()


    def setup(self, setup_session:SetupSession):

        super().setup(setup_session=setup_session)

        if self.bind:
            # within all parents catch first with namespace_only attribute
            # if such - check if namespace of all children are right.
            # Used for Extension.
            namespace_only = ModelsNS
            owner = self.owner
            while owner is not None:
                if hasattr(owner, "namespace_only"):
                    namespace_only = owner.namespace_only
                    break
                owner = owner.owner

            if self.bind.GetNamespace()!=namespace_only:
                raise RuleSetupValueError(owner=self, msg=f"{self.bind}: 'bind' needs to be in {namespace_only} DotExpression (e.g. M.status).")
            if len(self.bind.Path) not in (1,2,3,4):
                # warn(f"{self.bind}: 'bind' needs to be 1-4 deep DotExpression (e.g. M.status, M.city.country.name ).")
                raise RuleSetupValueError(owner=self, msg=f"'bind' needs to be 1-4 deep DotExpression (e.g. M.status, M.city.country.name ), got: {self.bind}")

            self.bound_attr_node = setup_session.get_dexp_node_by_dexp(self.bind)
            if not self.bound_attr_node:
                # TODO: not nice :(
                owner_container = self.get_container_owner(consider_self=True)
                owner_setup_session = getattr(owner_container, "owner_setup_session", owner_container.setup_session)
                if owner_setup_session!=setup_session:
                    # TODO: does not goes deeper - should be done with while loop until the top
                    self.bound_attr_node = owner_setup_session.get_dexp_node_by_dexp(self.bind)

            # self.attr_node = setup_session.get_attr_node(FieldsNS, self.name, strict=True)
            self.attr_node = setup_session[FieldsNS].get(self.name)

            assert self.attr_node

            if not self.bound_attr_node:
                # warn(f"TODO: {self}.bind = {self.bind} -> bound_attr_node can not be found.")
                raise RuleSetupValueError(owner=self, msg=f"bind={self.bind}: bound_attr_node can not be found.")
            # else:
            #     # ALT: self.bound_attr_node.add_bound_attr_node(BoundVar(setup_session.name, self.attr_node.namespace, self.attr_node.name))
            #     # self.attr_node.add_bound_attr_node(
            #     #         BoundVar(setup_session.name,
            #     #                  self.bound_attr_node.namespace,
            #     #                  self.bound_attr_node.name))
            #     # if not isinstance(self.bound_attr_node.data, TypeInfo):
            #     #     raise RuleInternalError(owner=self, msg=f"Unhandled case, self.bound_attr_node.data is not TypeInfo, got: {self.bound_attr_node.data}")
            #     self._set_type_info()

        # NOTE: can have multiple Evaluation-s
        evaluations = [cleaner for cleaner in self.cleaners if isinstance(cleaner, EvaluationBase) and cleaner.REQUIRES_AUTOCOMPUTE] \
                      if self.cleaners else None

        if self.autocomputed and not evaluations:
            raise RuleSetupError(owner=self, msg=f"When 'autocomputed' is set to '{self.autocomputed.name}', you need to have at least one Evaluation cleaner defined or set 'autocomputed' to False/AutocomputedEnum.NO")
        elif not self.autocomputed and evaluations:
            raise RuleSetupError(owner=self, msg=f"'When you have at least one Evaluation cleaner, set 'autocomputed = AutocomputedEnum.ALLWAYS/SOMETIMES' (got '{self.autocomputed.name}').")

        if self.REQUIRED_VALIDATIONS:
            validations_kls_found = set([type(cleaner) for cleaner in self.cleaners if isinstance(cleaner, ValidationBase)]) if self.cleaners else set()
            missing_names = []
            for validation_kls_or_list in self.REQUIRED_VALIDATIONS:
                validation_kls_list = validation_kls_or_list if isinstance(validation_kls_or_list, (list, tuple)) else [validation_kls_or_list]
                any_found = any([validation_kls for validation_kls in validation_kls_list if validation_kls in validations_kls_found])
                if not any_found:
                    if len(validation_kls_list)>1:
                        missing = "(" + " or ".join([vk.__name__ for vk in validation_kls_list]) + ")"
                    else:
                        missing = validation_kls_list[0].__name__
                    missing_names.append(missing)
            if missing_names:
                missing_names = " and ".join(missing_names)
                raise RuleSetupError(owner=self, msg=f"'{self.__class__.__name__}' requires following Validations (cleaners attribute): {missing_names}")

            # OLD:
            #     validations_kls_required = set(self.REQUIRED_VALIDATIONS)
            #     validations_kls_found = set([type(cleaner) for cleaner in self.cleaners if isinstance(cleaner, ValidationBase)]) if self.cleaners else set()
            #     missing = (validations_kls_required - validations_kls_found)
            #     if missing:
            #         missing_names = ", ".join([validation.__name__ for validation in missing])
            #         raise RuleSetupError(owner=self, msg=f"'{self.__class__.__name__}' requires following Validations (cleaners attribute): {missing_names}")

        return self

    # ------------------------------------------------------------

    def post_setup(self):
        " to validate all internal values "
        if not self.python_type:
            if self.PYTHON_TYPE:
                self.python_type = self.PYTHON_TYPE
            else:
                raise RuleInternalError(owner=self, msg="python_type must be set in custom setup() method or define PYTHON_TYPE class constant")
        self._set_type_info()

    # ------------------------------------------------------------

    def _set_type_info(self):
        if not self.python_type:
            raise RuleInternalError(owner=self, msg="python_type not defined") 

        # TODO: explain old message "static declared types. Dynamic types can be processed later"

        self.type_info = TypeInfo.get_or_create_by_type(
                                py_type_hint=self.python_type, 
                                caller=self,
                                )

        expected_type_info = None
        if self.bound_attr_node:
            if isinstance(self.bound_attr_node.data, TypeInfo):
                expected_type_info = self.bound_attr_node.data
            elif hasattr(self.bound_attr_node.data, "type_info"):
                expected_type_info = self.bound_attr_node.data.type_info

        if not expected_type_info:
            raise RuleInternalError(owner=self, msg=f"Can't extract type_info from bound_attr_node: {self.bound_attr_node} ")

        err_msg = expected_type_info.check_compatible(self.type_info)
        if err_msg:
            expected_type_info.check_compatible(self.type_info)
            raise RuleSetupTypeError(owner=self, msg=f"Given data type '{self.type_info}' is not compatible underneath model type '{expected_type_info}: {err_msg}'")


    # ------------------------------------------------------------

    def try_adapt_value(self, value: Any) -> Any:
        " apply phase: can change value type or value itself, must return same value or changed one. SHOULD NOT raise any Validation error."
        return value


    def validate_type(self, apply_session: IApplySession, strict:bool, value: Any = UNDEFINED) -> Optional[ValidationFailure]:
        """
        returns None if all ok, otherwise ValidationFailure()
        """
        component = apply_session.current_frame.component

        if component is not self:
            raise RuleInternalError(owner=self, msg=f"Current frame component should match current objects (self), got:\n  {component}\n  !=\n  {self}") 

        if value is UNDEFINED:
            value = apply_session.get_current_value(component, strict=strict)

        # NA_DEFAULTS_MODE - no value is evaluated in defaults_mode -> no
        # validation 
        if self.PYTHON_TYPE not in (UNDEFINED, None) \
                and value is not None \
                and value is not NA_DEFAULTS_MODE \
                and not isinstance(value, self.PYTHON_TYPE):
            error = f"Value type '{type(value)}' is not compoatible with '{self.PYTHON_TYPE}' " \
                    f"(value is '{message_truncate(value)}')"
            return ValidationFailure(
                            component_key_string = apply_session.get_key_string(component),
                            error=error, 
                            validation_name=self.name,
                            validation_label="Type validation",
                            details="Provide value with correct type",
                            )
        return None


# ------------------------------------------------------------
# StringField
# ------------------------------------------------------------

@dataclass
class StringField(FieldBase):
    # django: CharField
    PYTHON_TYPE:ClassVar[type] = str
    REQUIRED_VALIDATIONS:ClassVar[List[RequiredValidationsType]] = [(MaxLength, ExactLength)]

    def try_adapt_value(self, value: Any) -> Any:
        " apply phase: if not string and is standard type, convert to string "
        if not isinstance(value, str) and isinstance(value, STANDARD_TYPE_LIST):
            value = str(value)
        return value

# ------------------------------------------------------------
# UnsizedStringField
# ------------------------------------------------------------

@dataclass
class UnsizedStringField(FieldBase):
    # NOTE: Django TextField -> MaxLength=Neki valiki, render nešto drugačiji, rows/columns
    PYTHON_TYPE:ClassVar[type] = str

# ------------------------------------------------------------
# BooleanField
# ------------------------------------------------------------

@dataclass
class BooleanField(FieldBase):
    PYTHON_TYPE:ClassVar[type] = bool
    # default:        Optional[Union[bool, DotExpression]] = None

    # def __post_init__(self):
    #     self.init_clean()
    #     if self.default is not None and not isinstance(self.default, (bool, DotExpression)):
    #         raise RuleSetupValueError(owner=self, msg=f"'default'={self.default} needs to be bool value  (True/False).")

# ------------------------------------------------------------
# ChoiceField
# ------------------------------------------------------------

@dataclass
class ChoiceOption:
    value:      DotExpression # -> some Standard or Complex type
    label:      TransMessageType
    available:  Optional[Union[DotExpression,bool]] = True # Dexp returns bool

@dataclass
class ChoiceField(FieldBase):
    # TODO: check of types needs to be done too - this indicates to skip type check, but it could be done after setup()
    PYTHON_TYPE:ClassVar[UndefinedType] = UNDEFINED

    # https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
    # required is not required :) so in inherited class all attributes need to be optional
    # Note that with Python 3.10, it is now possible to do it natively with dataclasses.

    # List[Dict[str, TransMessageType]
    choices: Optional[Union[IFunctionDexpNode, 
                            CustomFunctionFactory, 
                            DotExpression, 
                            Union[List[ChoiceOption], 
                            List[Union[int,str]]]]] = None
    choice_value: Optional[DotExpression] = None
    choice_label: Optional[DotExpression] = None
    # choice_available: Optional[DotExpression]=True # returns bool

    # computed later
    choice_value_attr_node: AttrDexpNode = field(init=False, default=None, repr=False)
    choice_label_attr_node: AttrDexpNode = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.init_clean()
        if self.choices is None:
            # {self.name}: {self.__class__.__name__}: 
            raise RuleSetupValueError(owner=self, msg="argument 'choices' is required.")
        if is_enum(self.choices):
            raise RuleSetupValueError(owner=self, msg="argument 'choices' is Enum, use EnumChoices instead.")

        if isinstance(self.choices, (IFunctionDexpNode, CustomFunctionFactory, DotExpression)):
            if not (self.choice_value and self.choice_label):
                raise RuleSetupValueError(owner=self, msg="expected 'choice_label' and 'choice_value' passed.")
        elif is_function(self.choices):
            raise RuleSetupValueError(owner=self, msg="Passing functino to 'choices={self.choices}' is not allowed. Wrap it with 'Function()'.")
        else:
            if (self.choice_value or self.choice_label):
                raise RuleSetupValueError(owner=self, msg="'choice_label' and 'choice_value' are not expected.")

    # ------------------------------------------------------------

    def setup(self, setup_session: SetupSession):

        super().setup(setup_session=setup_session)

        choices = self.choices
        choices_checked = False
        is_list = UNDEFINED
        choice_from_function = False
        dexp_node = None

        # -- Factory cases
        if is_function(choices):
            raise RuleInternalError(owner=self, msg=f"Direct functions are not allowed, wrap with Function() instead. Got: {choices}")
        elif isinstance(choices, DotExpression):
            # TODO: restrict to dexp only - no operation
            if choices._status!=DExpStatusEnum.BUILT:
                # reported before - warn(f"TODO: There is an error with value expression {self.choices} - skip it for now.")
                choices = None
            else:
                dexp_node = setup_session.get_dexp_node_by_dexp(dexp=choices)
                if not dexp_node:
                    dexp_node = choices.Setup(setup_session=setup_session, owner=self)

        elif isinstance(choices, CustomFunctionFactory):
            custom_function_factory : CustomFunctionFactory = choices
            dexp_node = custom_function_factory.create_function(
                            # NOTE: was None before
                            setup_session=setup_session,
                            caller=None,
                            func_args=EmptyFunctionArguments,
                            )

        # -- DotExpression node
        if dexp_node:
            if isinstance(dexp_node, IFunctionDexpNode):
                func_node: IFunctionDexpNode = dexp_node  # better name
                choices = func_node.get_type_info().type_
                is_list = func_node.get_type_info().is_list
                choice_from_function = True

            elif isinstance(dexp_node, AttrDexpNode):
                attr_node: AttrDexpNode = dexp_node  # better name
                if is_enum(attr_node.data.value):
                    raise RuleSetupValueError(owner=self, msg=f"Using enum {attr_node.data.value}. Use EnumField instead.")

                if not hasattr(attr_node.data, "type_"):
                    raise RuleSetupValueError(owner=self, msg=f"Wrong type for choices: {attr_node.data} / {attr_node.data.value}. You can use Function().")
                choices = attr_node.data.type_
                is_list = attr_node.data.is_list

                if is_list and attr_node.namespace==ModelsNS and is_model_class(choices):
                    # FK case - e.g. Company -> company_types : List[CompanyType]

                    # TODO: I don't like 'choices_checked' this attr_node and the logic it uses
                    choices_checked = True

                    # TODO: Explain - why [0]???
                    # complex type, another struct
                    self.python_type = choices[0]
                    # TODO: deny default value - not available in this moment?
            else:
                raise RuleInternalError(f"Unexpected: {self} -> type({type(self)})")


        # -- Final data structures 
        if choices_checked:
            warn("TODO: ChoiceField - {self} refactor and drop 'choices_checked' attr_node")
        elif choices is None:
            # ignored
            pass
        elif is_model_class(choices):
            if choice_from_function:
                if not is_list:
                    raise NotImplementedError("Not a list! Enum or what? To be done: {choices}")
                model_class = choices
                # fun_return_type_info = TypeInfo.extract_function_return_type_info(function=choices) # parent=self, 
                # model_class, is_list = fun_return_type_info.type_, fun_return_type_info.is_list
                # if not is_list:
                #     raise RuleSetupValueError(owner=self, msg=f"Argument 'choices'={choices} is a function that does not return List[type]. Got: {fun_return_type}")
            else:
                assert not is_list
                model_class = choices

            with setup_session.use_stack_frame(
                    SetupStackFrame(
                        container = self.get_container_owner(consider_self=True), 
                        component = self, 
                        local_setup_session = setup_session.create_local_setup_session(
                                                    this_ns_model_class=model_class)
                    )):
                # model_class=model_class
                self.choice_value_attr_node = self._create_attr_node(setup_session, "choice_value", dexp=self.choice_value)
                self.choice_label_attr_node = self._create_attr_node(setup_session, "choice_label", dexp=self.choice_label)

            if self.choice_label_attr_node.type_info.type_!=str:
                raise RuleSetupValueError(owner=self, msg=f"Attribute choice_label needs to be bound to string attribute, got: {self.choice_label_attr_mode.type_info.type_}")
            self.python_type = self.choice_value_attr_node.type_info.type_


        elif isinstance(choices, (list, tuple)):
            if len(choices)==0:
                raise RuleSetupValueError(owner=self, msg="Attribute 'choices' is an empty list, Provide list of str/int/ChoiceOption.")
            if self.choice_value or self.choice_label:
                raise RuleSetupValueError(owner=self, msg="When 'choices' is a list, choice_value and choice_label are not permitted.")
            # now supports combining - but should have the same type
            for choice in choices:
                if not isinstance(choice, (str, int, ChoiceOption)):
                    raise RuleSetupValueError(owner=self, msg=f"Attribute choices has invalid choice, not one of str/int/ChoiceOption: {choice} / {type(choice)}")

            if isinstance(choices[0], ChoiceOption):
                self.python_type = type(choices[0].value)
            else:
                self.python_type = type(choices[0])

        else:
            raise RuleSetupValueError(owner=self, msg=f"Attribute choices has invalid value, not Union[Function(), DotExpression, Union[List[ChoiceOption], List[int], List[str]], got : {choices} / {type(choices)}")

        if not self.python_type:
            warn(f"TODO: ChoiceField 'python_type' not set {self}")

        # TODO: on usage normalize concrete all available choices to Enum[ChoiceOption], and define:
        #       https://stackoverflow.com/questions/33690064/dynamically-create-an-enum-with-custom-values-in-python

        return self

    # ------------------------------------------------------------

    def _create_attr_node(self, 
            # local_setup_session: SetupSession,
            setup_session: SetupSession, 
            aname: str, 
            dexp: DotExpression, 
            ):
        """
        Create choice AttrDexpNode() within local ThisRegistry
        """
        if not (dexp and isinstance(dexp, DotExpression) and dexp.GetNamespace()==ThisNS):
            raise RuleSetupValueError(owner=self, msg=f"Argument '{aname}' is not set or has wrong type - should be DotExpression in This. namespace. Got: {dexp} / {type(dexp)}")

        attr_node = dexp.Setup(setup_session=setup_session, owner=self)
        if dexp._status != DExpStatusEnum.BUILT:
            raise RuleInternalError(owner=self, msg=f"Setup failed for Dexp: {dexp} -> {dexp._status}")

        return attr_node

# ------------------------------------------------------------

@dataclass
class EnumField(FieldBase):
    PYTHON_TYPE:ClassVar[type] = UNDEFINED

    enum: Optional[Enum, DotExpression] = None

    enum_value_py_type: Optional[type] = field(init=False, default=None)

    # def __post_init__(self):
    #     self.init_clean()

    def setup(self, setup_session:SetupSession):

        super().setup(setup_session=setup_session)

        # TODO: revert to: strict=True - and process exception properly
        attr_node = setup_session.get_dexp_node_by_dexp(dexp=self.bind, strict=False)
        if attr_node:

            if isinstance(self.enum, DotExpression):
                # EnumField(... enum=S.CompanyTypes)
                enum_attr_node = setup_session.get_dexp_node_by_dexp(dexp=self.enum)
                if not enum_attr_node:
                    enum_attr_node = self.enum.Setup(setup_session=setup_session, owner=self)

                # self.enum = enum_attr_node.data.value
                self.enum = enum_attr_node.data
                # EnumMembers
                if not self.enum:
                    raise RuleSetupValueError(owner=self, msg="Underlying data type of attr_node expression is not enum. You should use: functions=[... Fn.EnumMembers(enum=<EnumType>)]")
                elif not is_enum(self.enum):
                    raise RuleSetupValueError(owner=self, msg=f"Data type of attr_node expression {self.enum} should Enum, got: {type(self.enum)}. You should use: functions=[... Fn.EnumMembers(enum=<EnumType>)]")
                # attr_node.data.py_type_hint

            # when not found -> it will be raised in other place
            if not isinstance(attr_node.data, TypeInfo):
                raise RuleSetupValueError(owner=self, msg=f"Data type of attr_node {attr_node} should be TypeInfo, got: {type(attr_node.data)}")

            py_hint_type = attr_node.data.py_type_hint

            if not is_enum(py_hint_type):
                # enum
                if not self.enum:
                    raise RuleSetupValueError(owner=self, msg=f"Data type (hint) of attr_node {attr_node} should be Enum or supply EnumField.enum. Got: {py_hint_type}")

                enum_member_py_type = get_enum_member_py_type(self.enum)
                # if not issubclass(self.enum, py_hint_type)
                #     and (type(py_hint_type)==type and py_hint_type!=enum_member_py_type)):  # noqa: E129
                if not (py_hint_type in inspect.getmro(self.enum)
                        or py_hint_type==enum_member_py_type):
                    raise RuleSetupValueError(owner=self, msg=f"Data type of attr_node {attr_node} should be the same as supplied Enum. Enum {self.enum}/{enum_member_py_type} is not {py_hint_type}.")
            else:
                # EnumField(... enum=CompanyTypeEnum)
                if self.enum and self.enum!=py_hint_type:
                    raise RuleSetupValueError(owner=self, msg=f"AttrDexpNode {attr_node} has predefined enum {self.enum} what is different from type_hint: {py_hint_type}")
                self.enum = py_hint_type
                enum_member_py_type = get_enum_member_py_type(self.enum)

            self.python_type = enum_member_py_type

            # TODO: check this in Default() implementation
            # if self.default is not None:
            #     if not isinstance(self.default, self.enum):
            #         raise RuleSetupValueError(owner=self, msg=f"Default should be an Enum {self.enum} value, got: {self.default}")

        # TODO: on usage normalize concrete all available choices to Enum[ChoiceOption], and define:
        #       https://stackoverflow.com/questions/33690064/dynamically-create-an-enum-with-custom-values-in-python

        return self


# ------------------------------------------------------------

@dataclass
class IntegerField(FieldBase):
    # TODO: autoincrement = allways, never, if-not-set
    #       AutoField
    #       BigAutoField
    # TODO: size = dexp | IntegerSizeEnum(normal=0, big=1, small=2)
    #       BigIntegerField
    #       SmallAutoField
    #       SmallIntegerField
    # TODO: positive
    #       -> cleaner= [MinValue(0)]
    #       PositiveBigIntegerField
    #       PositiveIntegerField
    #       PositiveSmallIntegerField
    PYTHON_TYPE:ClassVar[type] = int

@dataclass
class FloatField(FieldBase):
    PYTHON_TYPE:ClassVar[type] = float

@dataclass
class DecimalField(FieldBase):
    PYTHON_TYPE:ClassVar[type] = Decimal

@dataclass
class DateField(FieldBase):
    PYTHON_TYPE:ClassVar[type] = date

@dataclass
class DateTimeField(FieldBase):
    PYTHON_TYPE:ClassVar[type] = datetime

@dataclass
class TimeField(FieldBase):
    PYTHON_TYPE:ClassVar[type] = time

@dataclass
class DurationField(FieldBase):
    PYTHON_TYPE:ClassVar[type] = timedelta

@dataclass
class EmailField(FieldBase):
    # TODO: Other django fields like this one:
    #       FilePathField
    #       URLField
    #       SlugField
    #       GenericIPAddressField
    #       UUIDField
    PYTHON_TYPE:ClassVar[type] = str

