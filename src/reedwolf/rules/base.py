# TODO: Typehinting ContainerBase, Registries and AttrVexpNode could not defined
#       here, due circular imports
from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from decimal import Decimal
from enum import Enum
from typing import (
        List, 
        Any, 
        Dict, 
        Union, 
        Optional, 
        Tuple,
        Sequence,
        )
from dataclasses import (
        dataclass,
        field,
        )

from .utils import (
        UNDEFINED,
        UndefinedType,
        get_available_names_example,
        ThreadSafeCounter,
        )
from .exceptions import (
        RuleInternalError,
        RuleSetupError,
        RuleSetupNameError,
        RuleSetupValueError,
        RuleSetupNameNotFoundError,
        )
from .namespaces import (
        DynamicAttrsBase,
        ModelsNS,
        Namespace,
        )
from .meta import (
        ComponentNameType,
        NoneType,
        ModelField,
        TypeInfo,
        get_model_fields,
        is_function,
        is_method_by_name,
        is_model_class,
        ModelType,
        DataclassType,
        extract_field_meta,
        extract_py_type_hints,
        STANDARD_TYPE_LIST,
        TransMessageType,
        )
from .expressions import (
        ValueExpression,
        RawAttrValue,
        ExecResult,
        IFunctionVexpNode,
        )
from .functions import (
        CustomFunctionFactory,
        )
from .contexts import (
        IContext,
        )

# ------------------------------------------------------------

YAML_INDENT = "  "
PY_INDENT = "    "


def warn(msg):
    print(f"WARNING: {msg}")  # noqa: T001


def repr_obj(obj, limit=100):
    out = str(obj)
    if len(out) > limit-3:
        out = out[:limit-3] + "..."
    return out


def add_indent_to_strlist(out):
    return (f"\n{YAML_INDENT}".join(out)).splitlines()


def obj_to_strlist(obj, path=[]):
    return obj.to_strlist(path) if not isinstance(obj, DynamicAttrsBase) and getattr(obj, "to_strlist", None) else [str(obj)]


def list_to_strlist(args, before, after):
    out = []
    if len(args) == 0:
        out.append(f"{before}{after}")
    elif len(args) == 1:
        vstr = obj_to_strlist(args[0])
        out.append(f"{before}{vstr[0]}{after}")
    else:
        out.append(before)
        for v in args:
            out.extend(add_indent_to_strlist(obj_to_strlist(v)))
        out.append(after)
    return out


# ------------------------------------------------------------


class BaseOnlyArgs:  # noqa: SIM119

    def __init__(self, *args):
        self.args = args
        self.name = self.__class__.__name__

    def __str__(self):
        return "\n".join(self.to_strlist())

    def __repr__(self):
        return f"{self.__class__.__name__}({repr_obj(self.args)})"

    def to_strlist(self):
        return list_to_strlist(self.args, before=f"{self.__class__.__name__}(", after=")")


# ------------------------------------------------------------
# SetOwnerMixin
# ------------------------------------------------------------

class SetOwnerMixin:
    """ requires (Protocol):

        name
        owner_name
        owner
    """

    # ------------------------------------------------------------

    def set_owner(self, owner: ComponentBase):
        if not self.owner == UNDEFINED:
            raise RuleInternalError(owner=self, msg=f"Owner already defined, got: {owner}")

        assert owner is None or isinstance(owner, ComponentBase), owner
        self.owner = owner

        if not self.owner_name == UNDEFINED:
            raise RuleInternalError(owner=self, msg=f"Owner name already defined, got: {owner}")

        self.owner_name = owner.name if owner else ""
        if self.name in (None, UNDEFINED):
            suffix = self.__class__.__name__.lower()
            self.name = f"{self.owner_name}__{suffix}"


# ------------------------------------------------------------
# SubcomponentWrapper
# ------------------------------------------------------------
@dataclass
class SubcomponentWrapper:
    # TODO: strange class - check if really required or explain if yes
    name        : str # orig: vexp_node_name
    path        : str # orig: var_path
    # TODO: can be some other types too
    subcomponent: Union[ComponentBase, ValueExpression]
    th_field    : Optional[ModelField]


    def __post_init__(self):
        assert self.name and self.path and self.subcomponent and self.th_field
        # TODO: list all types available and include this check
        # if not isinstance(self.subcomponent, (ComponentBase, ValueExpression)):
        #     raise RuleInternalError(owner=self, msg=f"wrong type of subcomponent {type(self.subcomponent)} / {self.subcomponent}")


# ------------------------------------------------------------
# ComponentBase
# ------------------------------------------------------------

class ComponentBase(SetOwnerMixin, ABC):

    def __post_init__(self):
        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.register(self)
        ...

    @staticmethod
    def can_apply_partial() -> bool:
        return False

    def as_str(self):
        return "\n".join(self.to_strlist())

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self):
        return str(self)

    def to_strlist(self, path=None):
        if path is None:
            path = []
        out = []
        out.append(f"{self.__class__.__name__}(")
        # vars(self.kwargs).items():
        if len(path) > 15:
            raise RuleSetupError("Maximum object tree depth reached, not allowed depth more than 15.")
        for name, field in self.__dataclass_fields__.items():  # noqa: F402
            # if name.startswith("_") or callable(k):
            #     continue
            value = getattr(self, name)
            if type(field) in (list, tuple):
                out.extend(
                    add_indent_to_strlist(
                        list_to_strlist(
                            value,
                            before=f"{name}=[,",
                            after="],",
                        )
                    )
                )
            elif value == self:
                out.append(f"{name}=[Self...],")
            elif name in path:
                out.append(f"{name}=[...],")
            else:
                vstr = obj_to_strlist(value, path=path+[name])
                if len(vstr) <= 1:
                    out.append(f"{name}={vstr[0]},")
                else:
                    # vstr = add_indent_to_strlist(vstr)
                    out.append(f"{name}=")
                    for v2 in vstr:
                        out.append(f"{YAML_INDENT}{v2}")
        out.append(")")
        return add_indent_to_strlist(out)

    # ------------------------------------------------------------

    def get_children_dict(self) -> Dict[ComponentNameType, ComponentBase]:
        if not hasattr(self, "_children_dict"):
            self._children_dict = {comp.name : comp for comp in self.get_children()}
        return self._children_dict

    # ------------------------------------------------------------

    def get_children(self) -> List[ComponentBase]:
        """
        Get only children components. Used in apply() (+unit tests).
        to get all - components, cleaners and all other complex objects 
        - use _get_subcomponents_list()
        """
        # TODO: cached - be careful not to add new components aferwards
        if not hasattr(self, "_children"):
            children = getattr(self, "contains", None)
            if not children:
                children = getattr(self, "enables", None)
            else:
                assert not hasattr(self, "enables"), self
            self._children = children if children else []
        return self._children

    # ------------------------------------------------------------

    def _add_component(self, component: ComponentBase, components: Dict[str, ComponentBase]):
        if component.name in (None, UNDEFINED):
            raise RuleSetupValueError(owner=self, item=component, msg="Component's name is required.")
        if not (component.name and isinstance(component.name, str)):
            raise RuleSetupValueError(owner=self, item=component, msg=f"Component's name needs to be a string value, got: {component.name}': ")
        if component.name in components:
            raise RuleSetupNameError(owner=self, item=component, msg=f"Duplicate name '{component.name}': "
                        + repr(components[component.name])[:100]
                        + " --------- AND --------- "
                        + repr(component)[:100]
                        + ". Remove duplicate 'bind' or use 'name' attribute to define a distinct name.")
        # Save top container too - to preserve name and for completness (if is_top)
        components[component.name] = component

    # ------------------------------------------------------------

    def fill_components(self, components: Optional[Dict[str, ComponentBase]] = None,
                        owner: Optional[ComponentBase] = None) \
                        -> Dict[str, ComponentBase]:
        """ recursive -> flat dict
        component can be ComponentBase, Dataprovider, ...
        """

        is_top = bool(components is None)
        if is_top:
            components = {}
            # assert not owner

        # for children/contains attributes - owner is set here
        if not hasattr(self, "name"):
            raise RuleSetupError(owner=self, msg=f"Component should have 'name' attribute, got class: {self.__class__.__name__}")

        if not is_top:
            # Component
            assert owner
            assert owner != self
            self.set_owner(owner)
        else:
            if self.owner not in (None, UNDEFINED):
                # Extension()
                assert not owner
            else:
                # Rules()
                assert not owner
                self.set_owner(None)

        self._add_component(component=self, components=components)

        # includes components, cleaners and all other complex objects
        for subcomponent_wrapper in self._get_subcomponents_list():
            component = subcomponent_wrapper.subcomponent
            if isinstance(component, ValueExpression):
                pass
            elif hasattr(component, "fill_components"):
                if hasattr(component, "is_extension") and component.is_extension():
                    # for extension container don't go deeper into tree (call fill_components)
                    # it will be called later in container.setup() method
                    component.set_owner(owner=self)
                    # save only container (top) object
                    self._add_component(component=component, components=components)
                else:
                    component.fill_components(components=components, owner=self)
            elif hasattr(component, "set_owner"):
                component.set_owner(owner=self)
                self._add_component(component=component, components=components)
                # e.g. BoundModel.model - can be any custom Class(dataclass/pydantic)


        return components

    # ------------------------------------------------------------

    def _invoke_component_setup(self, 
                    subcomponent_name: str, 
                    subcomponent: Union[ComponentBase, ValueExpression], 
                    registries: 'Registries'):  # noqa: F821
        called = False

        if isinstance(subcomponent, (ValueExpression, )): # Operation
            # copy_to_registries=copy_to_registries,
            vexp: ValueExpression = subcomponent
            namespace = vexp.GetNamespace()
            if namespace._manual_setup:
                # needs manual Setup() later call with extra context - now is too
                # early (e.g. ThisNS)
                called = False
            # and vexp._status != VExpStatusEnum.INITIALIZED:
            elif namespace == ModelsNS \
               and vexp.IsFinished():
                # Setup() was called in container.setup() before
                called = False
            else:
                vexp.Setup(registries=registries, owner=self)
                called = True
        elif isinstance(subcomponent, ComponentBase):
            assert "Rules(" not in repr(subcomponent)
            # assert not isinstance(subcomponent, Rules), subcomponent
            subcomponent.setup(registries=registries)  # , owner=self)
            called = True
        elif isinstance(subcomponent, (dict, list, tuple)):
            raise RuleSetupValueError(owner=self, msg=f"Subcomponents should not be dictionaries, lists or tuples, got: {type(subcomponent)} / {subcomponent}")
        else:
            assert not hasattr(subcomponent, "Setup"), f"{self.name}.{subcomponent_name} has attribute that is not ValueExpression: {type(subcomponent)}"
            assert not hasattr(subcomponent, "setup"), f"{self.name}.{subcomponent_name} has attribute that is not Component: {type(subcomponent)}"
        return called

    # ------------------------------------------------------------

    def setup(self, registries: 'Registries'):  # noqa: F821
        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.setup_called(self)

        ret = self._setup(registries=registries)

        return ret

    # ------------------------------------------------------------
    def _get_subcomponents_list(self) -> List[SubcomponentWrapper]:
        """
        Includes components, cleaners and all other complex objects
        to get only children components, use get_children()

        TODO: document and make it better, pretty hackish.
              the logic behind is to collect all attributes (recurseively) that 
              are:
                1. component (ComponentBase)
                2. have owner set (SetOwnerMixin)
                3. ValueExpression

              all of them have some setup method:
                setup() # case 1. and 2.
                Setup() # case 3.

              so the logic should be:
                collect all attributes that have setup() and Setup()
              check if this is true and if so, implement.
        """
        # returns name, subcomponent
        fields = get_model_fields(self.__class__)
        subcomponent_list = []

        # NOTE: with vars() not the best way, other is to put metadata in field()
        for subcomponent_name, subcomponent in vars(self).items():

            if not isinstance(subcomponent, ValueExpression):
                if subcomponent in (None, (), {}, []):
                    continue

                # TODO: can be class or some simple type too - remove them from subcomponent list 
                if inspect.isclass(subcomponent):
                    continue 

                # TODO: see in meta.py there is a standard types ... use this
                if isinstance(subcomponent, STANDARD_TYPE_LIST):
                    continue

            # Skip procesing only following
            # TODO: do this better - check type hint (init=False) and then decide
            # TODO: collect names once and store it internally on class instance level?
            # type_hint = type_hints.get(subcomponent_name)
            th_field = fields.get(subcomponent_name)

            if th_field and th_field.metadata.get("skip_traverse", False):
                continue

            if is_function(subcomponent):
                # TODO: check that this is not class too
                raise RuleSetupValueError(owner=subcomponent, msg=f"Functions/callables could not be used directly - wrap with Function() and try again. Details: ({subcomponent_name}: {th_field})")

            # -------------------------------------------------------------
            # Attribute names that are not included in a subcomponents list
            # (standard attribute types like lists/dicts/int/str etc.).
            # -------------------------------------------------------------
            # "autocomplete", "evaluate",
            # TODO: not the smartest way how to do this ...
            if (subcomponent_name in ("owner", "owner_name", "owner_container", "owner_registries",
                                      "name", "label", "datatype", "components", "type", "autocomputed",
                                      "registries", "meta",
                                      # NOTE: maybe in the future will have value expressions too
                                      "error", "description", "hint", 
                                      # now is evaluated from bound_model, bound_model is processed
                                      "models", "py_type_hint", "type_info",
                                      "bound_attr_node",
                                      "read_handler",
                                      "function", "context_class", "config",
                                      "min", "max", "allow_none", "ignore_none",
                                      "keys",
                                      # "value", 
                                      # "enum",
                                      )
               or subcomponent_name[0] == "_"):
                # TODO: this should be main way how to check ...
                if subcomponent_name not in ("owner", "owner_container") and \
                    (hasattr(subcomponent, "setup") or hasattr(subcomponent, "setup")):
                    raise RuleInternalError(f"ignored attribute name '{subcomponent_name}' has setup()/Setup(). Is attribute list ok or value is not proper class (got component='{subcomponent}').")
                continue

            if not th_field or getattr(th_field, "init", True) is False:
                # warn(f"TODO: _get_subcomponents_list: {self} -> {subcomponent_name} -> {th_field}")
                raise RuleInternalError(owner=subcomponent, msg=f"Should '{subcomponent_name}' field be excluded from processing: {th_field}")

            # ----------------------------------------------------------------------
            # Classes of attribute members that are included in a subcomponents list 
            # each will be "Setup" if they have appropriate class and method
            # (see _invoke_component_setup).
            # ----------------------------------------------------------------------
            # TODO: this is not nice - do it better
            # TODO: models should not be dict()
            # dropped: "Validator" "Evaluator" "Operation"
            if subcomponent_name not in ("models", "data", "functions", "enum") \
                    and th_field \
                    and "Component" not in str(th_field.type) \
                    and "Container" not in str(th_field.type) \
                    and "ValueExpression" not in str(th_field.type) \
                    and "Validation" not in str(th_field.type) \
                    and "Evaluation" not in str(th_field.type) \
                    and "BoundModel" not in str(th_field.type) \
                    :
                # TODO: Validation should be extended to test isinstance(.., ValidationBase) ... or similar to include Required(), MaxLength etc.
                raise RuleInternalError(owner=subcomponent, msg=f"Should '{subcomponent_name}' attribute be excluded from processing." 
                        + f"\n  == {th_field})"
                        + f"\n  == owner := {self}"
                        + f"\n  == {type(subcomponent)}"
                        + f"\n  == {subcomponent}")

            if isinstance(subcomponent, (list, tuple)):
                for nr, sub_subcomponent in enumerate(subcomponent):
                    subcomponent_list.append(
                            SubcomponentWrapper(
                                name=f"{subcomponent_name}__{nr}", 
                                path=f"{subcomponent_name}[{nr}]", 
                                subcomponent=sub_subcomponent, 
                                th_field=th_field))
            elif isinstance(subcomponent, (dict,)):
                for ss_name, sub_subcomponent in subcomponent.items():
                    # NOTE: bind_to_models case - key value will be used as
                    #       attr_node name - should be registries unique
                    subcomponent_list.append(
                            SubcomponentWrapper(
                                name=ss_name, 
                                path=f"{subcomponent_name}.{ss_name}", 
                                subcomponent=sub_subcomponent, 
                                the_field=th_field))
            else:
                subcomponent_list.append(
                        SubcomponentWrapper(
                            name=subcomponent_name,
                            path=subcomponent_name, 
                            subcomponent=subcomponent, 
                            th_field=th_field))

        return subcomponent_list


    # ------------------------------------------------------------


    def _setup(self, registries: 'Registries'):  # noqa: F821
        if self.owner == UNDEFINED:
            raise RuleInternalError(owner=self, msg="Owner not set")

        if self.is_finished():
            raise RuleInternalError(owner=self, msg="Setup already called")

        for subcomponent_wrapper in self._get_subcomponents_list():
            self._invoke_component_setup(
                    subcomponent_wrapper.name, 
                    subcomponent=subcomponent_wrapper.subcomponent, 
                    registries=registries)

        # if not self.is_finished():
        self.finish()

        return self

    # ------------------------------------------------------------

    def finish(self):
        if self.is_finished():
            raise RuleSetupError(owner=self, msg="finish() should be called only once.")
        self._finished = True

    # ------------------------------------------------------------

    def is_finished(self):
        return getattr(self, "_finished", False)

    # ------------------------------------------------------------

    def get_name_from_bind(cls, bind: ValueExpression):
        # rename function to _get_name_from_bind
        if len(bind.Path) <= 2:
            # Vexpr(Person.name) -> name
            name = bind._name
        else:
            # Vexpr(Person.address.street) -> address.street
            # TODO: this is messy :( - should be one simple logic ...
            name = "__".join([bit._name for bit in bind.Path][1:])
        assert name
        return name


    def get_container_owner(self) -> 'ContainerBase':  # noqa: F821
        """ traverses up the component tree and find first container
        """
        # TODO: remove dep somehow - interface?
        from .containers import ContainerBase

        if self.owner == UNDEFINED:
            raise RuleSetupError(owner=self, msg="Owner is not set. Call .setup() method first.")

        if isinstance(self, ContainerBase):
            return self

        owner_container = self.owner
        while owner_container is not None:
            if isinstance(owner_container, ContainerBase):
                break
            owner_container = owner_container.owner
        if owner_container in (None, UNDEFINED):
            raise RuleSetupError(owner=self, msg="Did not found container in parents. Every component needs to be in some container object tree (Rules/Extension).")

        return owner_container


    # ------------------------------------------------------------

    def get_vexp_result_from_instance(self, apply_session:IApplySession, strict:bool = True) -> ExecResult:
        """ Fetch ExecResult from component.bind from INSTANCE (storage)
            by executing bind._evaluator.execute() fetch value process
            Work on stored fields only.
            A bit slower due getattr() logic and more complex structs.
            Does not work on initial values - when cache is not yet initialized
            with initial value for the component.
        """
        # TODO: put in common function
        bind_vexp = self.bind
        if not bind_vexp:
            if strict:
                # TODO: move this to Setup phase
                raise RuleApplyError(owner=self, msg=f"Component '{self.name}' has no bind")
            return None
        bind_vexp_result = bind_vexp._evaluator.execute_vexp(apply_session=apply_session)
        return bind_vexp_result

    # ------------------------------------------------------------

    # # TODO: this is rejected since vexp_result could be from non-bind vexp
    # #       and only bind vexp could provide parent_instance used for setting 
    # #       new instance attribute (see register_instance_attr_change() ).
    # def get_vexp_result_from_history(self, apply_session:IApplySession) -> ExecResult:
    #     """ Fetch ExecResult from component.bind from APPLY_SESSION.UPDATE_HISTORY
    #         last record.
    #         !!! ExecResult.value could be unadapted :( !!!
    #         Could work on non-stored fields.
    #         Probaly a bit faster, only dict queries.
    #     """
    #     # ALT: fetch from:
    #     #       bind_vexp: ValueExpression = getattr(component, "bind", None)
    #     #       bind_vexp._evaluator.execute()
    #     key_str = self.get_key_string(apply_session=apply_session)
    #     assert key_str in apply_session.update_history
    #     instance_attr_value = apply_session.update_history[key_str][-1]
    #     return instance_attr_value.vexp_result

    # ------------------------------------------------------------

    def get_current_value_from_history(self, apply_session:IApplySession) -> Any:
        """ Fetch ExecResult from component.bind from APPLY_SESSION.UPDATE_HISTORY
            last record.
            !!! ExecResult.value could be unadapted :( !!!
            Could work on non-stored fields.
            Probaly a bit faster, only dict queries.
        """
        # ALT: fetch from:
        #       bind_vexp: ValueExpression = getattr(component, "bind", None)
        #       bind_vexp._evaluator.execute()
        key_str = self.get_key_string(apply_session=apply_session)
        assert key_str in apply_session.update_history
        instance_attr_value = apply_session.update_history[key_str][-1]
        return instance_attr_value.value


# ------------------------------------------------------------

# TODO: Inheriting ABC triggers:
#   TypeError: Can't instantiate abstract class ModelsRegistry with abstract methods NAMESPACE, ...
class IRegistry:
    pass

# ------------------------------------------------------------

class IRegistries(ABC):

    def get_registry(self, namespace:Namespace):
        pass

# ------------------------------------------------------------

class IAttributeAccessorBase(ABC):
    " used in registry "

    @abstractmethod
    def get_attribute(self, apply_session: IApplySession, attr_name:str, is_last:bool) -> IAttributeAccessorBase:
        """ 
        is_last -> True - need to get final literal value from object
        (usually primitive type like int/str/date ...) 
        """
        ...


# ------------------------------------------------------------
# IData
# ------------------------------------------------------------
class IData(ABC):
    pass

# ------------------------------------------------------------
# IFieldBase
# ------------------------------------------------------------

class IFieldBase(ABC):

    @abstractmethod
    def get_attr_node(self, registries:"Registries") -> "AttrVexpNode":  # noqa: F821
        ...

    @abstractmethod
    def get_bound_attr_node(self, registries:"Registries") -> "AttrVexpNode":  # noqa: F821
        ...

# ------------------------------------------------------------
# ContainerBase
# ------------------------------------------------------------

class IContainerBase(ABC):

    @abstractmethod
    def add_fieldgroup(self, fieldgroup:"FieldGroup"):  # noqa: F821
        ...

    @abstractmethod
    def is_extension(self):
        ...

    @abstractmethod
    def __getitem__(self, name):
        ...

    @abstractmethod
    def setup(self) -> IContainerBase:
        ...

    @abstractmethod
    def get_bound_model_attr_node(self) -> "AttrVexpNode":  # noqa: F821
        ...

    @abstractmethod
    def get_component(self, name:str) -> ComponentBase:
        ...

    @abstractmethod
    def pp(self):
        """ pretty print - prints to stdout all components """
        ...


# ------------------------------------------------------------
# BoundModelBase
# ------------------------------------------------------------

class BoundModelBase(ComponentBase, ABC):


    def fill_models(self, models=None):
        """
        Recursion
        """
        if models is None:
            models = {}
        if self.name in models:
            raise RuleSetupNameError(owner=self, msg=f"Currently model names in tree dependency should be unique. Model name {self.name} is not, found: {models[self.name]}")
        models[self.name] = self
        if hasattr(self, "contains"):
            for dep_bound_model in self.contains:
                # recursion
                dep_bound_model.fill_models(models=models)
        return models

    def get_attr_node(self, registries: 'Registries') -> Union["AttrVexpNode", UndefinedType]:  # noqa: F821
        return registries.models_registry.get_attr_node_by_bound_model(bound_model=self)

# ------------------------------------------------------------

# TODO: put in Config and use only in ApplySession.config ...
class GlobalConfig:
    ID_NAME_SEPARATOR: ClassVar[str] = "::"


    # ID_PREFIX_WHEN_INTERNAL: ClassVar[str] = "_IID_"

    # ID_KEY_COUNTER: ClassVar[int] = 0 # field(repr=False, init=False, default=0)
    ID_KEY_COUNTER: ClassVar[ThreadSafeCounter] = ThreadSafeCounter()
    ID_KEY_PREFIX_FOR_MISSING: ClassVar[str] = "_MK_"


def get_instance_key_string_attrname_pair(key_string: str) -> Tuple[str, str]:
    idx = key_string.rfind(GlobalConfig.ID_NAME_SEPARATOR)
    if idx < 0:
        raise RuleInternalError(f"Invalid key_string='{key_string}'")
    instance_key_string = key_string[:idx]
    attrname = key_string[idx + len(GlobalConfig.ID_NAME_SEPARATOR):]
    return instance_key_string, attrname, 


# --------------------------------------------------
# Key types aliases
# --------------------------------------------------
KeyPairs = Sequence[Tuple[str, Any]]

InstanceId = int
KeyString = str

AttrName = str
AttrValue = Any

class ChangeOpEnum(str, Enum):
    CREATE = "CREATE"
    DELETE = "DELETE"
    UPDATE = "UPDATE"

@dataclass
class InstanceChange:
    key_string: KeyString
    # model_class: Any 
    key_pairs: KeyPairs = field()
    operation: ChangeOpEnum
    instance: ModelType = field(repr=True)
    updated_values: Optional[ Dict[AttrName, Tuple[AttrValue, AttrValue]] ] = field(default=None)

@dataclass
class InstanceAttrValue:
    # NOTE: value could be adapted version of vexp_result.value (can be different)
    value: Any

    # * first / initial record in owner list is from bind, first
    #   this .value could be unadapted value version (field.try_adapt_value())
    #   therefore value is in a special field.
    # * second+ - are from evaluation results
    vexp_result: ExecResult = field(repr=False, compare=False)

    # TODO: this is not filled good - allways component.name
    # just to have track who read/set this value
    value_owner_name: str = field(repr=False, compare=False)

    # is from bind
    is_from_bind: bool = field(repr=False, compare=False, default=False)

# ------------------------------------------------------------

@dataclass
class StackFrame:
    # current container data instance which will be procesed: changed/validated/evaluated
    instance: DataclassType
    container: 'ContainerBase' = field(repr=False)
    component: ComponentBase

    # UPDATE by this instance - can be RULES_LIKE or MODELS_LIKE (same dataclass) struct
    # required but is None when not update
    instance_new: Union[ModelType, NoneType, UndefinedType] = field(repr=False)

    # TODO: this is ugly 
    # set only in single case: partial mode, in_component_only_tree, compoent=component_only, instance_new
    on_component_only: Optional[ComponentBase] = field(repr=False, default=None)

    # this is currently created on 1st level and then copied to each next level
    # instance_new_struct_type: Union[StructEnum, NoneType, UndefinedType] = field(repr=False)

    # filled only when container list member
    index0: Optional[int] = None

    # internal - filled in __post_init__
    key_string: Optional[str] = field(init=False, repr=False, default=None)

    # used to check root value in models registry 
    bound_model_root : Optional[BoundModelBase] = field(repr=False, init=False, default=None)

    def __post_init__(self):
        from .containers import ContainerBase
        assert isinstance(self.container, ContainerBase)
        assert isinstance(self.component, ComponentBase)

        if self.index0 is not None:
            assert self.index0 >= 0

        if self.on_component_only:
            self.bound_model_root = (self.on_component_only 
                                     if   self.on_component_only.is_extension()
                                     else self.on_component_only.get_container_owner()\
                                    ).bound_model
            # can be list in this case
            # TODO: check if list only: if self.bound_model_root.type_info.is_list:
            instance_to_test = self.instance[0] \
                               if isinstance(self.instance, (list, tuple)) \
                               else self.instance
            assert is_model_class(instance_to_test.__class__), self.instance
        else:
            self.bound_model_root = self.container.bound_model
            assert is_model_class(self.instance.__class__), self.instance


        assert self.bound_model_root


@dataclass
class ValidationFailure:
    component_key_string: str
    error: TransMessageType
    validation_name: str
    validation_label: str
    details: Optional[str]


# ------------------------------------------------------------

class StructEnum(str, Enum):
    # models like - follows flat, storage/db like structure
    MODELS_LIKE = "MODELS"
    # rules like - follows hierachical structure like defined in rules
    RULES_LIKE  = "RULES"

# TODO: consider using classes instead, e.g.
#       class InputStructBase:
#           pass
#       class ModelInputStruct(InputStructBase):
#           # models like - follows flat, storage/db like structure
#           pass
#       class RulesInputStruct(InputStructBase):
#           # rules like - follows hierachical structure like defined in rules
#           pass

# ------------------------------------------------------------

@dataclass
class IApplySession:
    # TODO: možda bi ovo trebalo izbaciti ... - link na IRegistry u vexp node-ovima 
    registries: IRegistries = field(repr=False)
    rules: "Rules" = field(repr=False) # ex. ComponentBase 
    instance: Any = field(repr=False)
    # TODO: consider: instance_new: Union[ModelType, UndefinedType] = UNDEFINED,
    instance_new: Optional[ModelType] = field(repr=False)

    context: Optional[IContext] = field(repr=False)
    # apply_partial
    component_name_only: Optional[str] = field(repr=False, default=None)

    # automatically computed
    # extracted from component
    bound_model : BoundModelBase = field(repr=False, init=False)
    finished: bool = field(repr=False, init=False, default=False)

    # ---- internal structs ----

    # stack of frames - first frame is current. On the end of the process the
    # stack must be empty
    frames_stack: List[StackFrame] = field(repr=False, init=False, default_factory=list)

    # I do not prefer attaching key_string to instance i.e. setattr(instance, key_string)
    # but having instance not expanded, but to store in a special structure
    key_string_container_cache : Dict[InstanceId, KeyString] = \
            field(repr=False, init=False, default_factory=dict)

    # used when when collecting list of instances/attributes which are updated
    instance_by_key_string_cache : Dict[KeyString, ModelType] = \
            field(repr=False, init=False, default_factory=dict)

    # list of attribute values - from initial to final
    update_history: Dict[KeyString, List[InstanceAttrValue]] = \
            field(repr=False, init=False, default_factory=dict)

    # final list of channged instances / sub-instances (operation + orig values)
    changes: List[InstanceChange] = field(repr=False, init=False, default_factory=list)

    # When first argument is <type> and second is <type> then call function Callable
    # which will adapt second type to first one. Example: <string> + <int> -> <string> + str(<int>)
    binary_operations_type_adapters : Dict[Tuple[type, type], Callable[[Any], Any]] = field(repr=False, init=False, default_factory=dict)

    # on validation errors this will be filled
    errors: Dict[KeyString, ValidationFailure] = field(repr=False, init=False, default_factory=dict)

    # for instance_new (root struct), is set, in normal mode set in init, in partial mode set on component detection
    instance_new_struct_type: Optional[StructEnum] = field(repr=False, init=False, default=None)

    # computed from component_name_only
    component_only: Optional[ComponentBase] = field(repr=False, init=False, default=None)


    @abstractmethod
    def apply(self) -> IApplySession:
        ...


    # def get_current_component_bind_value(self):
    #     component = self.current_frame.component
    #     bind_vexp_result = component.get_vexp_result_from_instance(apply_session=self)
    #     return bind_vexp_result.value

    def validate_type(self, component: ComponentBase, value: Any = UNDEFINED):
        validation_failure = None
        if isinstance(component, IFieldBase):
            validation_failure = component.validate_type(apply_session=self, value=value)
            if validation_failure:
                self.register_instance_validation_failed(component, validation_failure)
        return validation_failure


    def register_instance_attr_change(self, 
            component: ComponentBase, 
            vexp_result: ExecResult,
            new_value: Any,
            is_from_init_bind:bool=False) -> InstanceAttrValue:

        # NOTE: new_value is required - since vexp_result.value
        #       could be unadapted (see field.try_adapt_value()

        assert component == self.current_frame.component

        if new_value is UNDEFINED:
            raise RuleInternalError(owner=self, msg=f"new value should not be UNDEFINED, fix the caller (comp={component})")

        key_str = component.get_key_string(apply_session=self)
        # if "can_be_accessed" in key_str: import pdb;pdb.set_trace() 
        if key_str not in self.update_history:
            if not is_from_init_bind:
                raise RuleInternalError(owner=self, msg="key_str not found in update_history and this is not initialization")
            self.update_history[key_str] = []

            # NOTE: initial value from instance is not checked - only
            #       intermediate and the last value
            #   self.validate_type(component, new_value)
        else:
            if is_from_init_bind:
                raise RuleInternalError(owner=self, msg="key_str found in update_history and this is initialization")

            if not self.update_history[key_str]:
                raise RuleInternalError(owner=self, msg="change history is empty")

            # -- check if current value is different from new one
            value_current = self.update_history[key_str][-1].value
            if value_current == new_value:
                raise RuleApplyError(owner=self, msg=f"register change failed, the value is the same: {value_current}")

            # is this really necessary
            self.validate_type(component, new_value)

            # -- parent instance
            # parent_raw_attr_value = vexp_result.value_history[-2]
            # parent_instance = parent_raw_attr_value.value

            parent_instance = self.current_frame.instance
            # TODO: not sure if this validation is ok
            assert isinstance(parent_instance, 
                        self.current_frame.container.bound_model.get_type_info().type_)

            # -- attr_name - fetch from initial bind vexp (very first)
            init_instance_attr_value = self.update_history[key_str][0]
            assert init_instance_attr_value.is_from_bind
            init_bind_vexp_result = init_instance_attr_value.vexp_result
            # attribute name is in the last item
            init_raw_attr_value = init_bind_vexp_result.value_history[-1]
            attr_name = init_raw_attr_value.attr_name

            assert hasattr(parent_instance, attr_name), f"{parent_instance}.{attr_name}"
            # ----------------------------------------
            # Finally change instance value
            # ----------------------------------------
            setattr(parent_instance, attr_name, new_value)

            # NOTE: bind_vexp_result last value is not changed
            #       maybe should be changed but ... 

        # TODO: pass input arg value_owner_name - component.name does not have
        #       any purpose
        instance_attr_value = InstanceAttrValue(
                                value_owner_name=component.name, 
                                value=new_value,
                                vexp_result=vexp_result,
                                is_from_bind = is_from_init_bind,
                                # TODO: source of change ...
                                )

        self.update_history[key_str].append(instance_attr_value)

        return instance_attr_value


    def push_frame_to_stack(self, frame: StackFrame):
        self.frames_stack.insert(0, frame)

    def pop_frame_from_stack(self) -> StackFrame:
        assert self.frames_stack
        return self.frames_stack.pop(0)

    @property
    def current_frame(self) -> StaciFrame:
        assert self.frames_stack
        return self.frames_stack[0]

    def set_finished(self):
        assert not self.finished
        assert len(self.frames_stack) == 0
        self.finished = True

    def register_instance_validation_failed(self, component: ComponentBase, failure: ValidationFailure):
        if failure.component_key_string not in self.errors:
            self.errors[failure.component_key_string] = []
        self.errors[failure.component_key_string].append(failure)
        # TODO: mark invalid all children and this component

# ------------------------------------------------------------

def extract_type_info(
        attr_node_name: str,
        inspect_object: Any,
        ) -> Tuple[
                TypeInfo, 
                Optional[ModelField], 
                Optional[IFunctionVexpNode]  # noqa: F821
                ]:
    """
    TODO: move to meta.py?

    main logic for extraction from parent object (dataclass, py class, vexp instances) 
    member by name 'attr_node_name' -> data (struct, plain value), or IFunctionVexpNode instances . E.g.

    """
    if isinstance(inspect_object, BoundModelBase):
        inspect_object = inspect_object.model

    # function - callable and not class and not pydantic?
    parent_type_info = None
    if isinstance(inspect_object, CustomFunctionFactory):
        parent_type_info = inspect_object.get_output_type_info()
        parent_object = parent_type_info.type_
    elif isinstance(inspect_object, IFunctionVexpNode):
        parent_type_info = inspect_object.output_type_info
        parent_object = parent_type_info.type_
    elif is_function(inspect_object):
        raise Exception("should not happen, use Function/Factory-ies")
    else:
        # normal object - hopefully with with type hinting
        parent_object = inspect_object

    if isinstance(parent_object, TypeInfo):
        # go one level deeper
        parent_type_info = parent_object
        if not isinstance(parent_object.type_, type):
            raise RuleSetupValueError(item=inspect_object, msg=f"Inspected object's type hint is not a class object/type: {parent_object.type_} : {parent_object.type_}, got: {type(parent_object.type_)} ('.{attr_node_name}' process)")
        # Can be method call, so not pydantic / dataclass are allowed too
        # if not is_model_class(parent_object.type_):
        #     raise RuleSetupValueError(item=inspect_object, msg=f"Inspected object's type hint type is not 'dataclass'/'Pydantic.BaseModel' type: {parent_object.type_}, got: {parent_object.type_} ('.{attr_node_name}' process)")
        parent_object = parent_object.type_


    fields = []

    type_info = None
    th_field = None
    # func_node = None

    is_vexp = isinstance(parent_object, ValueExpression)
    if is_vexp:
        raise RuleSetupNameNotFoundError(item=inspect_object, msg=f"Attribute '{attr_node_name}' is not member of expression '{parent_object}'.")

    # when parent virtual expression is not assigned to data-type/data-attr_node, 
    # then it is sent directly what will later raise error "attribute not found" 
    if not is_vexp:
        if is_method_by_name(parent_object, attr_node_name):
            # parent_object, attr_node_name
            raise RuleSetupNameError(item=inspect_object,
                        msg=f"Inspected object's is a method: {parent_object}.{attr_node_name}. "
                        "Calling methods on instances are not allowed.")

        if is_model_class(parent_object):
            # raise RuleInternalError(item=inspect_object, msg=f"'Parent is not DC/PYD class -> {parent_object} : {type(parent_object)}'.")

            if not hasattr(parent_object, "__annotations__"):
                # TODO: what about pydantic?
                raise RuleInternalError(item=inspect_object, msg=f"'DC/PYD class {parent_object}' has no metadata (__annotations__ / type hints), can't read '{attr_node_name}'. Add type-hints or check names.")

            model_class: ModelType = parent_object

            # === Dataclass / pydantic field metadata
            # TODO: can it be a List[]?
            th_field, fields = extract_field_meta(inspect_object=model_class, attr_node_name=attr_node_name)

            # === parent type hint
            parent_py_type_hints = extract_py_type_hints(model_class, f"registries->{attr_node_name}:DC/PYD")

            py_type_hint = parent_py_type_hints.get(attr_node_name, None)
            if py_type_hint:
                type_info = TypeInfo.get_or_create_by_type(
                                py_type_hint=py_type_hint,
                                )
    if not type_info:
        # fallback -> will match fields and is_vexp mode
        if not fields:
            fields = get_model_fields(parent_object, strict=False)
        field_names = list(fields.keys())
        valid_names = f"Valid attributes: {get_available_names_example(attr_node_name, field_names)}" if field_names else "Type has no attributes at all."
        raise RuleSetupNameNotFoundError(msg=f"Type object {parent_object} has no attribute '{attr_node_name}'. {valid_names}")

    assert isinstance(type_info, TypeInfo)
    return type_info, th_field # , func_node



