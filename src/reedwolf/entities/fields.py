"""
------------------------------------------------------------
FIELDS - can read/store data from/to storage
------------------------------------------------------------
check components.py too

Notes:

    - available attribute - available yields True(ish?) value 
        => this is NOT processed: 
            - field update by new instance, 
            - cleaners for the field
            - children tree (contains/enables)
        Only initial value is unchanged. See also BooleanField.enables.

"""
from abc import ABC, abstractmethod
from typing import (
    Union,
    List,
    Sequence,
    Optional,
    Any,
    ClassVar,
    Dict, Type,
)
from dataclasses import (
    dataclass,
    field,
)
import inspect
from decimal import Decimal
from datetime import (
    date,
    time,
    datetime,
    timedelta,
)
from enum import Enum, IntEnum

from .utils import (
    to_repr,
    UNDEFINED,
    NA_DEFAULTS_MODE,
    UndefinedType,
    message_truncate,
)
from .exceptions import (
    EntitySetupValueError,
    EntityInternalError,
    EntitySetupError,
    EntitySetupTypeError,
    EntitySetupNameError, EntityInitError,
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
    is_model_klass,
    is_function,
    get_enum_member_py_type,
    EmptyFunctionArguments,
    STANDARD_TYPE_LIST,
    Self,
    ComponentStatus,
    ERR_MSG_ATTR_REQUIRED,
)
from .base import (
    get_name_from_bind,
    warn,
    IField,
    IFieldGroup,
    IApplyResult,
    ValidationFailure,
    SetupStackFrame,
    IComponent, IValidation, IEvaluation, VALIDATION_OR_EVALUATION_TYPE, )
from . import _
from .expressions import (
    DotExpression,
    IFunctionDexpNode,
    ISetupSession,
    IThisRegistry,
    DEXP_VALIDATOR_FOR_BIND,
    create_dexp_by_attr_name,
)
from .expr_attr_nodes import (
    IAttrDexpNode
)
from .functions import (
    CustomFunctionFactory,
)
from .registries import (
    ThisRegistryForComponent,
    ThisRegistryForModelKlass,
)
from .valid_field import (
    MinValue,
    MaxLength,
    ExactLength,
    FieldValidationBase
)
from .eval_field import (
    FieldEvaluationBase,
)
from .valid_children import (
    ChildrenValidationBase,
)
from .eval_children import (
    ChildrenEvaluationBase,
)
from .valid_base import (
    ValidationBase,
)
from .eval_base import (
    EvaluationBase,
)
from .value_accessors import ValueAccessorInputType

# Items are:
#   - when single - that must be added
#   - when tuple(validtions) then one of listed must be added
RequiredValidationsType = List[Union[ValidationBase, List[ValidationBase]]]


class AutocomputedEnum(IntEnum):
    NO       = 0  # if False; must be bool() === False
    ALLWAYS  = 1  # if True
    SOMETIMES= 2  # must be set manually

    @classmethod
    def from_value(cls, value: Union[bool, Self]) -> Self:
        if value is True:
            value = cls.ALLWAYS
        elif value is False:
            value = cls.NO
        return cls(value)

# ============================================================
# Fields
# ============================================================


@dataclass
class FieldBase(IField, ABC):
    # abstract property:
    PYTHON_TYPE:ClassVar[type(UNDEFINED)] = UNDEFINED
    REQUIRED_VALIDATIONS:ClassVar[Optional[RequiredValidationsType]] = None

    # to Model attribute
    bind_to:        Union[DotExpression, str] = field(default=UNDEFINED)
    title:          Optional[TransMessageType] = field(repr=False, default=None)

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

    # NOTE: see available_old_logic
    #   available:      Union[bool, DotExpression] = field(repr=False, default=True)

    # if enables is filled - then it Supports Children* cleaners too since it can have nested field hierarchy
    cleaners:       Optional[List[Union[ChildrenValidationBase, ChildrenEvaluationBase, FieldValidationBase, FieldEvaluationBase]]] = field(repr=False, default_factory=list)
    accessor:       Optional[ValueAccessorInputType] = field(repr=False, default=None)
    autocomputed:   Union[bool, AutocomputedEnum] = field(repr=False, default=False)

    # BooleanField.enables that yields False/None (value must be bool/None)
    #   => this is NOT processed:
    #       - children tree (enables) 
    #   This is equivalent to 
    #       FieldGroup(
    #           available=<this-field.value>, 
    #           contains=<this-field.enables-list>)
    #   See also "available".

    # NOTE: this has no relation to type hinting - this is used for html input placeholder attribute
    hint:           Optional[TransMessageType] = field(repr=False, default=None)

    # for arbitrary custom values, entity system ignores the field
    #   e.g. "autocomplete": True ...
    meta:           Optional[Dict[str, Any]] = field(repr=False, default=None)

    # if not supplied name will be extracted from binded model attr_node
    name:            Optional[str] = None
    attr_node:       Union[IAttrDexpNode, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)
    bound_attr_node: Union[IAttrDexpNode, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)
    python_type:     Union[type, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)
    type_info:       Optional[TypeInfo] = field(init=False, repr=False, default=UNDEFINED)

    _allowed_cleaner_base_list: List[VALIDATION_OR_EVALUATION_TYPE] = field(init=False, repr=False, default=UNDEFINED)

    # will be set later
    # is_key:          bool = field(init=False, repr=False, default=False)

    # TODO: found no way to achive this: copied from IComponent just to have
    #       these field in repr()/str() listed at the end
    #   parent:         Union[Self, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    #   parent_name:    Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

    def init_allowed_cleaners(self) -> List[VALIDATION_OR_EVALUATION_TYPE]:
        return [FieldValidationBase, FieldEvaluationBase]

    def init(self):
        self._allowed_cleaner_base_list = self.init_allowed_cleaners()

        if not self.bind_to:
            raise EntityInitError(owner=self, msg=ERR_MSG_ATTR_REQUIRED.format("bind_to"))

        if isinstance(self.bind_to, str):
            self.bind_to = create_dexp_by_attr_name(ModelsNS, self.bind_to)

        # TODO: check that value is M ns and is a simple M. value
        if not isinstance(self.bind_to, DotExpression):
            raise EntitySetupValueError(owner=self, msg=f"Argument 'bind_to' needs to be DotExpression (e.g. M.status), got: {to_repr(self.bind_to)}")

        self.bind_to._SetDexpValidator(DEXP_VALIDATOR_FOR_BIND)
        # if self.bind_to._namespace != ModelsNS:
        #     raise EntitySetupValueError(owner=self, msg=f"Argument 'bind_to' needs to be 'Models.' / 'M.' DotExpression, got: {to_repr(self.bind_to)}")

        if not self.name:
            # ModelsNs.person.surname -> surname
            self.name = get_name_from_bind(self.bind_to)

        self.autocomputed = AutocomputedEnum.from_value(self.autocomputed)

        # NOTE: see available_old_logic
        # clean_available(owner=self, attr_name="available", dexp_or_bool=self.available)

        self._check_cleaners(self._allowed_cleaner_base_list)

        super().init()


    def _set_bound_attr_node(self, setup_session: ISetupSession):
        self.bound_attr_node = setup_session.get_dexp_node_by_dexp(self.bind_to)
        if not self.bound_attr_node:
            # TODO: not nice :(
            parent_container = self.get_first_parent_container(consider_self=True)
            parent_setup_session = parent_container.parent.setup_session if parent_container.parent else parent_container.setup_session
            if parent_setup_session != setup_session:
                # TODO: does not goes deeper - should be done with while loop until the top
                self.bound_attr_node = parent_setup_session.get_dexp_node_by_dexp(self.bind_to)

        if not self.bound_attr_node:
            raise EntitySetupValueError(owner=self, msg=f"Can not fill self.bound_attr_node by bind_to: {self.bind_to}")

    # ------------------------------------------------------------

    def _setup(self, setup_session:ISetupSession):

        super()._setup(setup_session=setup_session)

        if self.bind_to:
            self.attr_node = setup_session.get_registry(FieldsNS).get(self.name)
            if not self.attr_node:
                raise EntityInternalError(owner=self, msg=f"Attribute not found: {self.name}")
            if not self.is_unbound() and not self.bound_attr_node:
                # warn(f"TODO: {self}.bind_to = {self.bind_to} -> bound_attr_node can not be found.")
                raise EntitySetupValueError(owner=self, msg=f"bind_to={self.bind_to}: bound_attr_node can not be found.")

        # NOTE: can have multiple Evaluation-s
        evaluations_w_autocomp = ", ".join([cleaner.name for cleaner in self.cleaners if isinstance(cleaner, EvaluationBase) and cleaner.REQUIRES_AUTOCOMPUTE]) \
                      if self.cleaners else ""

        if self.autocomputed and not evaluations_w_autocomp:
            raise EntitySetupError(owner=self, msg=f"When 'autocomputed' is set to '{self.autocomputed.name}', you need to have at least one Evaluation cleaner defined or set 'autocomputed' to False/AutocomputedEnum.NO")
        elif not self.autocomputed and evaluations_w_autocomp:
            raise EntitySetupError(owner=self, msg=f"'When you have at least one Evaluation cleaner (found: {evaluations_w_autocomp}), set 'autocomputed = AutocomputedEnum.ALLWAYS/SOMETIMES' (got '{self.autocomputed.name}').")

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
                raise EntitySetupError(owner=self, msg=f"'{self.__class__.__name__}' requires following Validations (cleaners attribute): {missing_names}")

    # ------------------------------------------------------------

    def _setup_phase_one_set_python_type(self, setup_session: ISetupSession):
        """
        for special cases method to setup python type and override default logic.
        """
        pass

    def _setup_phase_one_set_type_info(self, setup_session: ISetupSession):
        """
        to validate all internal values
        """
        if not self.is_unbound():
            self._set_bound_attr_node(setup_session)

        self._setup_phase_one_set_python_type(setup_session)

        if not self.python_type:
            if self.PYTHON_TYPE:
                self.python_type = self.PYTHON_TYPE
            else:
                raise EntityInternalError(owner=self, msg="python_type must be set in custom setup() method or define PYTHON_TYPE class constant")

        self._set_type_info()

    # ------------------------------------------------------------

    def create_this_registry(self, setup_session: ISetupSession) -> Optional[IThisRegistry]:
        """
        This.Value
        if BooleanField.enables -> add .Children + This.<fields>
        """
        assert getattr(self, "bind_to", None)
        if not self.bind_to.IsFinished():
            # TODO: this is 2nd place to call '.Setup()'. Explain!
            # only place where container is used ...
            container = self.get_first_parent_container(consider_self=True)
            self.bind_to.Setup(setup_session=setup_session, owner=container)

        attr_node = self.bind_to._dexp_node
        if not attr_node:
            raise EntitySetupNameError(owner=self, msg=f"{attr_node.name}.bind_to='{self.bind_to}' could not be evaluated")

        this_registry = ThisRegistryForComponent(component=self, attr_node=attr_node)

        return this_registry

    # ------------------------------------------------------------

    def get_type_info(self) -> TypeInfo:
        if not self.type_info:
            raise EntityInternalError(owner=self, msg="type_info is not yet set, probably setup was not called")
        return self.type_info

    # ------------------------------------------------------------

    def _set_type_info(self):
        if not self.python_type:
            raise EntityInternalError(owner=self, msg="python_type not defined") 

        # TODO: explain old message "static declared types. Dynamic types can be processed later"

        base_type_info = TypeInfo.get_or_create_by_type(
                                py_type_hint=self.python_type, 
                                caller=self,
                                )

        expected_type_info = None
        if self.is_unbound():
            assert not self.bound_attr_node
            expected_type_info = base_type_info
        else:
            if self.bound_attr_node:
                expected_type_info = self.bound_attr_node.type_info
                # if isinstance(self.bound_attr_node.data, TypeInfo):
                #     expected_type_info = self.bound_attr_node.data
                # elif hasattr(self.bound_attr_node.data, "type_info"):
                #     expected_type_info = self.bound_attr_node.data.type_info

            if not expected_type_info:
                raise EntityInternalError(owner=self, msg=f"Can't extract type_info from bound_attr_node: {self.bound_attr_node} ")

            err_msg = expected_type_info.check_compatible(base_type_info)
            if err_msg:
                expected_type_info.check_compatible(base_type_info)
                raise EntitySetupTypeError(owner=self, msg=f"Given data type '{base_type_info}' is not compatible underneath model type '{expected_type_info}: {err_msg}'")

        # if all ok, set to more richer type_info
        self.type_info = expected_type_info

    # ------------------------------------------------------------

    def try_adapt_value(self, value: Any) -> Any:
        " apply phase: can change value type or value itself, must return same value or changed one. SHOULD NOT raise any Validation error."
        return value


    def validate_type(self, apply_result: IApplyResult, value: Any = UNDEFINED) -> Optional[ValidationFailure]:
        """
        returns None if all ok, otherwise ValidationFailure()
        """
        component = apply_result.current_frame.component
        if component is not self:
            raise EntityInternalError(owner=self, msg=f"Current frame component should match current objects (self), got:\n  {component}\n  !=\n  {self}") 

        if value is UNDEFINED:
            value = apply_result.get_current_value(strict=True)

        # NA_DEFAULTS_MODE - no value is evaluated in defaults_mode -> no
        # validation 
        if self.PYTHON_TYPE not in (UNDEFINED, None) \
                and value is not None \
                and value is not NA_DEFAULTS_MODE \
                and not isinstance(value, self.PYTHON_TYPE):
            error = _(f"Value type '{type(value)}' is not compoatible with '{self.PYTHON_TYPE}' " \
                    f"(value is '{message_truncate(value)}')")
            return ValidationFailure(
                            component_key_string = apply_result.get_key_string(),
                            error=error, 
                            validation_name=self.name,
                            validation_title="Type validation",
                            details="Provide value with correct type",
                            )
        return None


# ------------------------------------------------------------
# SizedStringField
# ------------------------------------------------------------

class StringFieldBase(FieldBase, ABC):

    PYTHON_TYPE:ClassVar[type] = str

    def try_adapt_value(self, value: Any) -> Any:
        " apply phase: if not string and is standard type, convert to string "
        if not isinstance(value, str) and isinstance(value, STANDARD_TYPE_LIST):
            value = str(value)
        return value

@dataclass
class SizedStringField(StringFieldBase):
    """ sized / limited string field """

    REQUIRED_VALIDATIONS:ClassVar[List[RequiredValidationsType]] = [(MaxLength, ExactLength)]

# ------------------------------------------------------------
# TextField
# ------------------------------------------------------------

@dataclass
class StringField(StringFieldBase):
    """ unsized / unlimited string field """
    # ALT: name: UnsizedStringField
    ...


# ------------------------------------------------------------
# BooleanField
# ------------------------------------------------------------

@dataclass
class BooleanField(FieldBase):
    PYTHON_TYPE:ClassVar[type] = bool

    # TODO: možda složiti da radi i za Choice/Enum -> structural pattern
    #       matching like, samo nisam još našao zgodnu sintaksu.
    enables:        Optional[List[IComponent]] = field(repr=False, default=None)

    def init_allowed_cleaners(self) -> List[VALIDATION_OR_EVALUATION_TYPE]:
        out = super().init_allowed_cleaners()
        if self.enables:
            out.extend([ChildrenValidationBase, ChildrenEvaluationBase])
        return out

    # def init(self):
    #     super().init()


    @staticmethod
    def may_collect_my_children() -> bool:
        return True

    @staticmethod
    def can_have_children() -> bool:
        return True


# ------------------------------------------------------------
# ChoiceField
# ------------------------------------------------------------

@dataclass
class ChoiceOption:
    value:      DotExpression # -> some Standard or Complex type
    title:      TransMessageType
    # NOTE: see available_old_logic
    # available:  Optional[Union[DotExpression,bool]] = True # Dexp returns bool

    # def init(self):
    #     # NOTE: see available_old_logic
    #     #   clean_available(owner=self, attr_name="available", dexp_or_bool=self.available)
    #     super().init()



@dataclass
class ChoiceField(FieldBase):
    # TODO: check of types needs to be done too - this indicates to skip type check, but it could be done after setup()
    PYTHON_TYPE:ClassVar[UndefinedType] = UNDEFINED

    # https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
    # required is not required :) so in inherited class all attributes need to be optional
    # Note that with Python 3.10, it is now possible to do it natively with dataclasses.

    # List[Dict[str, TransMessageType]
    choices: Optional[
                Union[
                    IFunctionDexpNode, 
                    CustomFunctionFactory, 
                    DotExpression, 
                    Sequence[ChoiceOption], 
                    Sequence[str],
                    Sequence[int],
                    ]] = None
    choice_value: Optional[DotExpression] = None
    choice_title: Optional[DotExpression] = None
    # choice_available: Optional[DotExpression]=True # returns bool

    # computed later
    choice_value_attr_node: IAttrDexpNode = field(init=False, default=None, repr=False)
    choice_title_attr_node: IAttrDexpNode = field(init=False, default=None, repr=False)

    # TODO: možda složiti da radi i za Choice/Enum -> structural pattern
    #       matching like, samo nisam još našao zgodnu sintaksu.
    #   enables: Optional[List[IComponent]] = field(repr=False, default=None)

    def init(self):
        super().init()
        if self.choices is None:
            # {self.name}: {self.__class__.__name__}: 
            raise EntitySetupValueError(owner=self, msg="argument 'choices' is required.")
        if is_enum(self.choices):
            raise EntitySetupValueError(owner=self, msg="argument 'choices' is Enum, use EnumChoices instead.")

        if isinstance(self.choices, (IFunctionDexpNode, CustomFunctionFactory, DotExpression)):
            if not (self.choice_value and self.choice_title):
                raise EntitySetupValueError(owner=self, msg="expected 'choice_title' and 'choice_value' passed.")
        elif is_function(self.choices):
            raise EntitySetupValueError(owner=self, msg="Passing functino to 'choices={self.choices}' is not allowed. Wrap it with 'Function()'.")
        else:
            if (self.choice_value or self.choice_title):
                raise EntitySetupValueError(owner=self, msg="'choice_title' and 'choice_value' are not expected.")

    # ------------------------------------------------------------

    def _setup_phase_one_set_python_type(self, setup_session: ISetupSession):
        choices = self.choices
        choices_checked = False
        is_list = UNDEFINED
        choice_from_function = False
        dexp_node = None

        if is_function(choices):
            raise EntityInternalError(owner=self, msg=f"Direct functions are not allowed, wrap with Function() instead. Got: {choices}")

        if isinstance(choices, DotExpression):
            # if choices._status!=DExpStatusEnum.BUILT:
            dexp_node = setup_session.get_dexp_node_by_dexp(dexp=choices)
            if not dexp_node:
                dexp_node = choices.Setup(setup_session=setup_session, owner=self)
        elif isinstance(choices, CustomFunctionFactory):
            custom_function_factory: CustomFunctionFactory = choices
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

            elif isinstance(dexp_node, IAttrDexpNode):
                attr_node: IAttrDexpNode = dexp_node  # better name
                if is_enum(attr_node.data.value):
                    raise EntitySetupValueError(owner=self, msg=f"Using enum {attr_node.data.value}. Use EnumField instead.")

                type_info = attr_node.type_info
                # if not hasattr(attr_node.data, "type_"):
                #     raise EntitySetupValueError(owner=self, msg=f"Wrong type for choices: {attr_node.data} / {attr_node.data.value}. You can use Function().")

                choices = type_info.type_
                is_list = type_info.is_list

                if is_list and attr_node.namespace==ModelsNS and is_model_klass(choices):
                    # FK case - e.g. Company -> company_types: List[CompanyType]

                    # TODO: I don't like 'choices_checked' this attr_node and the logic it uses
                    choices_checked = True

                    # TODO: Explain - why [0]???
                    # complex type, another struct
                    self.python_type = choices[0]
                    # TODO: deny default value - not available in this moment?
            else:
                raise EntityInternalError(f"Unexpected: {self} -> type({type(self)})")


        # -- Final data structures 
        if choices_checked:
            warn("TODO: ChoiceField - {self} refactor and drop 'choices_checked' attr_node")
        elif choices is None:
            # ignored
            pass
        elif is_model_klass(choices):
            model_klass = choices
            if choice_from_function:
                if not is_list:
                    raise NotImplementedError("Not a list! Enum or what? To be done: {choices}")
            else:
                assert not is_list
                if is_list:
                    # TODO: explain, give example
                    raise EntityInternalError(owner=self, msg="Choices is a a list of model instances, expecting single instance.")

            #  exception to the rule - ChoiceField value and title from This.
            this_registry = ThisRegistryForModelKlass.create(
                                             setup_session=setup_session,
                                             model_klass=model_klass)

            with setup_session.use_stack_frame(
                    SetupStackFrame(
                        container = self.get_first_parent_container(consider_self=True), 
                        component = self, 
                        this_registry = this_registry,
              )):
                self.choice_value_attr_node = self._create_attr_node(setup_session, "choice_value", dexp=self.choice_value)
                self.choice_title_attr_node = self._create_attr_node(setup_session, "choice_title", dexp=self.choice_title)

            if self.choice_title_attr_node.type_info.type_!=str:
                raise EntitySetupValueError(owner=self, msg=f"Attribute choice_title needs to be bound to string attribute, got: {self.choice_title_attr_mode.type_info.type_}")
            self.python_type = self.choice_value_attr_node.type_info.type_


        elif isinstance(choices, (list, tuple)):
            if len(choices)==0:
                raise EntitySetupValueError(owner=self, msg="Attribute 'choices' is an empty list, Provide list of str/int/ChoiceOption.")
            if self.choice_value or self.choice_title:
                raise EntitySetupValueError(owner=self, msg="When 'choices' is a list, choice_value and choice_title are not permitted.")
            # now supports combining - but should have the same type
            for choice in choices:
                if not isinstance(choice, (str, int, ChoiceOption)):
                    raise EntitySetupValueError(owner=self, msg=f"Attribute choices has invalid choice, not one of str/int/ChoiceOption: {choice} / {type(choice)}")

            if isinstance(choices[0], ChoiceOption):
                self.python_type = type(choices[0].value)
            else:
                self.python_type = type(choices[0])

        else:
            raise EntitySetupValueError(owner=self, msg=f"Attribute choices has invalid value, not Union[Function(), DotExpression, Union[List[ChoiceOption], List[int], List[str]], got : {choices} / {type(choices)}")

        if not self.python_type:
            # TODO: implement if missing
            raise EntityInternalError(owner=self, msg="ChoiceField 'python_type' not set")

    # ------------------------------------------------------------

    # def _setup(self, setup_session: ISetupSession):
    #     # NOTE: must be done after to avoid calling Setup() for choice_value
    #     #       and choice_title. For these two default This namespace is not
    #     #       appliable, need special one, This.Instance + This.<model-attrs>
    #     super()._setup(setup_session=setup_session)

    #     # TODO: on usage normalize concrete all available choices to Enum[ChoiceOption], and define:
    #     #       https://stackoverflow.com/questions/33690064/dynamically-create-an-enum-with-custom-values-in-python

    #     return self

    # ------------------------------------------------------------

    def _create_attr_node(self, 
            setup_session: ISetupSession, 
            aname: str, 
            dexp: DotExpression, 
            ):
        """
        Create choice AttrDexpNodeModelKlass() within local ThisInstanceRegistry
        """
        if not (dexp and isinstance(dexp, DotExpression) and dexp._namespace==ThisNS):
            raise EntitySetupValueError(owner=self, msg=f"Argument '{aname}' is not set or has wrong type - should be DotExpression in This. namespace. Got: {dexp} / {type(dexp)}")

        attr_node = dexp.Setup(setup_session=setup_session, owner=self)
        # if dexp._status != DExpStatusEnum.BUILT:
        if dexp._status != ComponentStatus.finished:
            raise EntityInternalError(owner=self, msg=f"Setup failed for Dexp: {dexp} -> {dexp._status}")

        return attr_node

# ------------------------------------------------------------

@dataclass
class EnumField(FieldBase):
    PYTHON_TYPE:ClassVar[type] = UNDEFINED

    enum: Optional[Union[Enum, DotExpression]] = None
    enum_value_py_type: Optional[type] = field(init=False, default=None)

    def _setup_phase_one_set_python_type(self, setup_session: ISetupSession):
        # TODO: revert to: strict=True - and process exception properly
        attr_node = setup_session.get_dexp_node_by_dexp(dexp=self.bind_to, strict=False)

        if not (attr_node or self.is_unbound()):
            # TODO: should this be an error?
            #       when not found -> it will be raised in other place
            return

        if isinstance(self.enum, DotExpression):
            # EnumField(... enum=S.CompanyTypes)
            if self.is_unbound():
                raise EntitySetupValueError(owner=self, msg=f"Enum should not be bound to DotExpression, expecting Enum, got: {self.enum}")
            enum_attr_node = setup_session.get_dexp_node_by_dexp(dexp=self.enum)
            if not enum_attr_node:
                enum_attr_node = self.enum.Setup(setup_session=setup_session, owner=self)

            # self.enum = enum_attr_node.data.value
            self.enum = enum_attr_node.data
            # EnumMembers
            if not self.enum:
                raise EntitySetupValueError(owner=self, msg="Underlying data type of attr_node expression is not enum. You should use: functions=[... Ctx.EnumMembers(enum=<EnumType>)]")
            elif not is_enum(self.enum):
                raise EntitySetupValueError(owner=self, msg=f"Data type of attr_node expression {self.enum} should Enum, got: {type(self.enum)}. You should use: functions=[... Ctx.EnumMembers(enum=<EnumType>)]")
            # attr_node.data.py_type_hint

        # when not found -> it will be raised in other place
        if self.is_unbound():
            py_hint_type = self.enum
            if not is_enum(py_hint_type):
                raise EntitySetupValueError(owner=self, msg=f"In unbound mode, provide type to enum attribute, got: {py_hint_type}")
        else:
            # if not isinstance(attr_node.data, TypeInfo):
            #     raise EntitySetupValueError(owner=self, msg=f"Data type of attr_node {attr_node} should be TypeInfo, got: {type(attr_node.data)}")
            py_hint_type = attr_node.type_info.py_type_hint

        if not is_enum(py_hint_type):
            # enum
            if not self.enum:
                raise EntitySetupValueError(owner=self, msg=f"Data type (hint) of attr_node {attr_node} should be Enum or supply EnumField.enum. Got: {py_hint_type}")

            enum_member_py_type = get_enum_member_py_type(self.enum)
            # if not issubclass(self.enum, py_hint_type)
            #     and (type(py_hint_type)==type and py_hint_type!=enum_member_py_type)):  # noqa: E129
            if not (py_hint_type in inspect.getmro(self.enum)
                    or py_hint_type==enum_member_py_type):
                raise EntitySetupValueError(owner=self, msg=f"Data type of attr_node {attr_node} should be the same as supplied Enum. Enum {self.enum}/{enum_member_py_type} is not {py_hint_type}.")
        else:
            # EnumField(... enum=CompanyTypeEnum)
            if self.enum and self.enum!=py_hint_type:
                raise EntitySetupValueError(owner=self, msg=f"IAttrDexpNode {attr_node} has predefined enum {self.enum} what is different from type_hint: {py_hint_type}")
            self.enum = py_hint_type
            enum_member_py_type = get_enum_member_py_type(self.enum)

        self.python_type = enum_member_py_type

        # TODO: check this in Default() implementation
        # if self.default is not None:
        #     if not isinstance(self.default, self.enum):
        #         raise EntitySetupValueError(owner=self, msg=f"Default should be an Enum {self.enum} value, got: {self.default}")

        # TODO: on usage normalize concrete all available choices to Enum[ChoiceOption], and define:
        #       https://stackoverflow.com/questions/33690064/dynamically-create-an-enum-with-custom-values-in-python

    # ------------------------------------------------------------

    # TODO: možda složiti da radi i za Choice/Enum -> structural pattern
    #       matching like, samo nisam još našao zgodnu sintaksu.
    #   enables: Optional[List[IComponent]] = field(repr=False, default=None)
    # def _setup(self, setup_session: ISetupSession):
    #     super()._setup(setup_session=setup_session)
    #     return self


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
    #       PositiveBigIntegerField
    #       PositiveSmallIntegerField
    PYTHON_TYPE:ClassVar[type] = int

@dataclass
class PositiveIntegerField(IntegerField):
    # TODO: add unit tests 
    def init(self):
        self.cleaners.insert(0, MinValue(1))
        super().init()

@dataclass
class IdField(IntegerField):
    " can be negative "
    # TODO: add unit tests 
    ...

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


# ------------------------------------------------------------------------
# FieldGroups - logical groupihg and common functionality/dependency of other
#            components.
# ------------------------------------------------------------------------

@dataclass
class FieldGroup(IFieldGroup):
    contains:       List[IComponent] = field(repr=False)
    name:           Optional[str] = field(default=None)
    title:          Optional[TransMessageType] = field(repr=False, default=None)
    cleaners:       Optional[List[Union[ChildrenValidationBase, ChildrenEvaluationBase]]] = field(repr=False, default_factory=list)
    accessor:       Optional[ValueAccessorInputType] = field(repr=False, default=None)
    # NOTE: see available_old_logic
    #   available:      Union[bool, DotExpression] = field(repr=False, default=True)

    def init(self):
        # NOTE: see available_old_logic
        #   clean_available(owner=self, attr_name="available", dexp_or_bool=self.available)
        self._check_cleaners([ChildrenValidationBase, ChildrenEvaluationBase])
        super().init()

    @staticmethod
    def can_apply_partial() -> bool:
        return True

    @staticmethod
    def may_collect_my_children() -> bool:
        return True

    @staticmethod
    def can_have_children() -> bool:
        return True

    @staticmethod
    def is_fieldgroup() -> bool:
        return True

    def create_this_registry(self, setup_session: ISetupSession) -> Optional[IThisRegistry]:
        assert self.get_children(), self
        this_registry = ThisRegistryForComponent(component=self)
        return this_registry




