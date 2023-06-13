# https://stackoverflow.com/questions/58986031/type-hinting-child-class-returning-self/74545764#74545764
from __future__ import annotations
import inspect
from abc import (
        ABC, 
        abstractmethod,
        )
from typing import (
        Tuple,
        Any,
        Dict,
        List,
        Optional,
        Union,
        Type,
        )
from inspect import isclass, isabstract

from dataclasses import dataclass, field

from .utils import (
        get_available_names_example,
        UNDEFINED,
        NA_DEFAULTS_MODE,
        UndefinedType,
        varname_to_title,
        )
from .exceptions import (
        RuleSetupError,
        RuleSetupValueError,
        RuleSetupNameError,
        RuleInternalError,
        RuleNameNotFoundError,
        RuleSetupNameNotFoundError,
        RuleApplyNameNotFoundError,
        RuleValidationError,
        )
from .namespaces import (
        ModelsNS,
        FieldsNS,
        )
from .meta import (
        STANDARD_TYPE_LIST,
        TransMessageType,
        TypeInfo,
        is_model_class,
        get_model_fields,
        ModelType,
        DataclassType,
        )
from .base import (
        ComponentBase,
        IContainerBase,
        BoundModelBase,
        GlobalConfig,
        KeyPairs,
        IApplySession,
        SetupStackFrame,
        )
from .expressions import (
        DotExpression,
        )
from .bound_models import (
        BoundModel,
        BoundModelWithHandlers,
        )
from .attr_nodes import AttrDexpNode
from .functions import CustomFunctionFactory
from .registries import (
        SetupSession,
        ModelsRegistry,
        FieldsRegistry,
        FunctionsRegistry,
        OperationsRegistry,
        ContextRegistry,
        ConfigRegistry,
        )

from .valid_children import (
        ICardinalityValidation
        )
from .components import (
        Component,
        FieldGroup,
        ValidationBase,
        EvaluationBase,
        )
from .contexts import (
        IContext,
        )
from .config import (
        Config,
        )

from ..rules import components, fields, validations, evaluations

# ------------------------------------------------------------

def create_setup_session(
        parent: ContainerBase,
        config: Config,
        functions: Optional[List[CustomFunctionFactory]] = None, 
        context_class: Optional[IContext] = None,
        ) -> SetupSession:
    setup_session = SetupSession(
                    parent=parent, 
                    functions=functions,
                    parent_setup_session=parent.setup_session if parent.parent else None,
                    include_builtin_functions=parent.is_top_parent())

    setup_session.add_registry(ModelsRegistry())
    setup_session.add_registry(FieldsRegistry())
    setup_session.add_registry(FunctionsRegistry())
    setup_session.add_registry(OperationsRegistry())
    setup_session.add_registry(ContextRegistry(context_class=context_class))
    setup_session.add_registry(ConfigRegistry(config=config))

    return setup_session

# ------------------------------------------------------------
# Rules
# ------------------------------------------------------------

class ContainerBase(IContainerBase, ComponentBase, ABC):

    def _get_function(self, name: str, strict:bool=True):
        if not self.functions:
            raise KeyError(f"{self.name}: Function '{name}' not found, no functions available.")
        return self.setup_session.functions_factory_registry.get(name, strict=strict)

    def add_fieldgroup(self, fieldgroup:FieldGroup):
        if self.is_finished():
            raise RuleSetupError(owner=self, msg="FieldGroup can not be added after setup() is called.")
        found = [sec for sec in self.contains if sec.name==fieldgroup.name]
        if found:
            raise RuleSetupError(owner=self, msg=f"FieldGroup {fieldgroup.name} is already added.")
        self.contains.append(fieldgroup)

    def is_top_parent(self):
        return not bool(self.parent)

    def is_extension(self):
        # TODO: if self.parent is not None could be used as the term, put validation somewhere
        " if start model is value expression - that mean that the the Rules is Extension "
        return isinstance(self.bound_model.model, DotExpression)

    def is_container(self):
        return True

    @staticmethod
    def can_apply_partial() -> bool:
        return True

    def __getitem__(self, name):
        if name not in self.components:
            vars_avail = get_available_names_example(name, self.components.keys())
            raise KeyError(f"{self.name}: Component name '{name}' not found, available: {vars_avail}")
        return self.components[name]

    # ------------------------------------------------------------

    def _register_model_attr_nodes(self):
        # ----------------------------------------
        # A. 1st level attr_nodes
        # ------------------------------------------------------------
        # A.1. MODELS - collect attr_nodes from managed models
        # ------------------------------------------------------------
        self.models = self.bound_model.fill_models()

        if not self.models:
            raise RuleSetupError(owner=self, msg="Rules(models=List[models]) is required.")

        for bound_model_name, bound_model in self.models.items():
            assert bound_model_name.split(".")[-1] == bound_model.name
            self._register_bound_model(bound_model=bound_model)

    # ------------------------------------------------------------

    def _register_bound_model(self, bound_model:BoundModelBase):
        # ex. type_info.metadata.get("bind_to_parent_setup_session")
        is_main_model = (bound_model==self.bound_model)
        is_extension_main_model = (self.is_extension() and is_main_model)

        # is_list = False
        if not isinstance(bound_model, BoundModelBase):
            raise RuleSetupError(owner=self, msg=f"{bound_model.name}: Needs to be Boundbound_model* instance, got: {bound_model}")

        model = bound_model.model

        attr_node = None
        if is_extension_main_model:
            if not isinstance(model, DotExpression):
                raise RuleSetupError(owner=self, msg=f"{bound_model.name}: For Extension main bound_model needs to be DotExpression: {bound_model.model}")

        # alias_saved = False
        is_list = False


        if isinstance(model, DotExpression):
            # TODO: for functions value expressions need to be stored
            #       with all parameters (func_args)
            if model.GetNamespace()!=ModelsNS:
                raise RuleSetupError(owner=self, msg=f"{bound_model.name}: DotExpression should be in ModelsNS namespace, got: {model.GetNamespace()}")

            if is_extension_main_model:
                # TODO: DRY this - the only difference is setup_session - extract common logic outside / 
                # bound attr_node
                assert hasattr(self, "parent_setup_session")
                setup_session_from = self.parent_setup_session
            else:
                # Rules - top parent container / normal case
                setup_session_from = self.setup_session

            attr_node = setup_session_from.get_dexp_node_by_dexp(dexp=model)
            if attr_node:
                raise RuleInternalError(owner=self, msg=f"AttrDexpNode data already in setup_session: {model} -> {attr_node}")

            attr_node = model.Setup(setup_session=setup_session_from, parent=bound_model)
            if not attr_node:
                raise RuleInternalError(owner=self, msg=f"AttrDexpNode not recognized: {model}")

            if not isinstance(attr_node.data, TypeInfo):
                raise RuleInternalError(owner=self, msg=f"AttrDexpNode data is not TypeInfo, got: {type(attr_node.data)} / {attr_node.data}")

            model   = attr_node.data.type_
            is_list = attr_node.data.is_list

        if not is_model_class(model) and not (is_list and model in STANDARD_TYPE_LIST):
            raise RuleSetupError(owner=self, msg=f"Managed model {bound_model.name} needs to be a @dataclass, pydantic.BaseModel or List[{STANDARD_TYPE_LIST}], got: {type(model)}")


        # == M.name version
        self.setup_session[ModelsNS].register_all_nodes(root_attr_node=attr_node, bound_model=bound_model, model=model)

        # if not isinstance(model, DotExpression) and isinstance(bound_model, BoundModel):
        #     bound_model._register_nested_models(self.setup_session)


        # == M.company.name version
        # if not attr_node:
        #     attr_node = self.setup_session[ModelsNS].create_root_attr_node(bound_model=bound_model) -> _create_root_attr_node()
        # # self.setup_session.register(attr_node, alt_attr_node_name=bound_model.name if attr_node.name!=bound_model.name else None)
        # self.setup_session[ModelsNS].register_attr_node(
        #                             attr_node, 
        #                             alt_attr_node_name=(
        #                                 bound_model.name 
        #                                 if attr_node.name!=bound_model.name 
        #                                 else None))


        return attr_node

    # ------------------------------------------------------------

    # def _register_data_attr_nodes(self):
    #     # ------------------------------------------------------------
    #     # A.2. DATAPROVIDERS - Collect all attr_nodes from dataproviders fieldgroup
    #     # ------------------------------------------------------------
    #     for data_var in self.data:
    #         self.setup_session[DataNS].register(data_var)

    # ------------------------------------------------------------

    def _register_fields_components_attr_nodes(self):
        """
        Traverse the whole tree (recursion) and collect all components into
        simple flat list. It will set parent for each child component.
        """
        self.components = self.fill_components()

        # A.3. COMPONENTS - collect attr_nodes - previously flattened (recursive function fill_components)
        for component_name, component in self.components.items():
            self.setup_session[FieldsNS].register(component)

    # ------------------------------------------------------------

    def setup(self) -> ContainerBase:
        # components are flat list, no recursion/hierarchy browsing needed
        if self.bound_model is None:
            raise RuleSetupError(owner=self, msg="bound_model not set. Initialize in constructor or call bind_to() first.")

        if not self.contains:
            raise RuleSetupError(owner=self, msg="'contains' attribute is required with list of components")

        if self.is_finished():
            raise RuleSetupError(owner=self, msg="setup() should be called only once")

        if self.setup_session is not None:
            raise RuleSetupError(owner=self, msg="SetupSession.setup() should be called only once")

        self.setup_session = create_setup_session(
                                parent=self,
                                functions = self.functions,
                                config = self.config,
                                context_class = self.context_class,
                                )


        # ----------------------------------------
        # A. 1st level attr_nodes
        # ----------------------------------------
        # ModelsNS
        self._register_model_attr_nodes()

        # # DataNS
        # self._register_data_attr_nodes()

        # FieldsNS
        self._register_fields_components_attr_nodes()

        # NOTE: ThisNS and FunctionsNS - are initialized differently and later

        # ----------------------------------------
        # B. other level attr_nodes - recursive
        # ----------------------------------------
        # now when all attr_nodes are set, now setup() can be called for all
        # components recursively:
        #   it will validate every component attribute
        #   if attribute is another component - it will call recursively component.setup()
        #   if component attribute is DotExpression -> will call dexp.Setup()
        #   if component is another container i.e. is_extension() - it will
        #       process only that component and will not go deeper. later
        #       extension.setup() will do this within own tree dep (own .components / .setup_session)

        # iterate all subcomponents and call _setup() for each
        with self.setup_session.use_stack_frame(
                SetupStackFrame(
                    container = self, 
                    component = self, 
                )):
            self._setup(setup_session=self.setup_session)

        # check all ok?
        for component_name, component in self.components.items():
            # TODO: maybe bound Field.bind -> Model attr_node?
            if not component.is_finished():
                raise RuleInternalError(owner=self, msg=f"{component} not finished. Is in overriden setup()/Setup() parent method super().setup()/Setup() been called (which sets parent and marks finished)?")

        self.setup_session.finish()

        if self.keys:
            # Inner BoundModel can have self.bound_model.model = DotExpression
            self.keys.validate(self.bound_model.get_type_info().type_)

        if self.is_top_parent():
            self.setup_session.call_hooks_on_finished_all()

        return self

    # ------------------------------------------------------------

    def get_bound_model_attr_node(self) -> AttrDexpNode:
        return self.setup_session[ModelsNS].get_attr_node_by_bound_model(bound_model=self.bound_model)

    # ------------------------------------------------------------

    def get_component(self, name:str) -> ComponentBase:
        # TODO: currently components are retrieved only from contains - but should include validations + cardinality
        if name not in self.components:
            vars_avail = get_available_names_example(name, self.components.keys())
            raise RuleNameNotFoundError(owner=self, msg=f"Component '{name}' not found, some valid_are: {vars_avail}")
        return self.components[name]

    # ------------------------------------------------------------

    def pprint(self):
        if not hasattr(self, "components"):
            raise RuleSetupError(owner=self, msg="Call .setup() first")
        print(f"{self.name}: {self.__class__.__name__} ::")
        for nr, (name, component) in enumerate(self.components.items(),1):
            print(f"  {nr:02}. {name}: {component.__class__.__name__} ::")
            print(f"      {repr(component)[:100]}") # noqa: T001
            # TODO: if pp() exists -> recursion with depth+1 (indent)

    # ------------------------------------------------------------

    def get_key_pairs_or_index0(self, 
                                instance: ModelType, 
                                index0: int, 
                                ) -> Union[List[(str, Any)], int]:
        " index0 is 0 based index of item in the list"
        # TODO: move to ApplyResult:IApplySession?
        if self.keys:
            ret = self.get_key_pairs(instance)
        else:
            ret = index0
        return ret


    def get_key_pairs(self, instance: ModelType) -> List[(str, Any)]:
        if not self.keys:
            raise RuleInternalError(msg="get_key_pairs() should be called only when 'keys' are defined")
        key_pairs = self.keys.get_key_pairs(instance)
        return key_pairs


# ------------------------------------------------------------


class KeysBase(ABC):

    @abstractmethod
    def validate(self, model_or_children_model: ModelType):
        ...

    @abstractmethod
    def get_key_pairs(self, instance: ModelType) -> List[Tuple[str, Any]]:
        """ returns list of (key-name, key-value) """
        ...

    # def get_keys(self, apply_session:IApplySession) -> List[Any]:
    #     return [key for name,key in self.get_keys_tuple(apply_session)]


def get_new_unique_id() -> int:
    return GlobalConfig.ID_KEY_COUNTER.get_new()


class MissingKey:
    def __init__(self, id=None):
        self.id = id if id else get_new_unique_id()

    def __str__(self):
        return f"{GlobalConfig.ID_KEY_PREFIX_FOR_MISSING}{self.id}"
    def __repr__(self):
        return f"{self.__class__.__name__}('{str(self)}')"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(repr(self.id))

@dataclass
class KeyFields(KeysBase):

    # list of field names
    field_names : List[str]

    def __post_init__(self):
        if not self.field_names:
            raise RuleSetupError("field_names are required")
        for field_name in self.field_names:
            if not isinstance(field_name, str):
                raise RuleSetupNameError(f"Fields needs to be list of strings, got '{field_name}'")
        if len(set(self.field_names)) != len(self.field_names):
            raise RuleSetupNameError(f"Fields needs to be unique list of names, found duplicates: '{self.field_names}'")

    def validate(self, model: ModelType):
        model_fields = get_model_fields(model)
        for field_name in self.field_names:
            if field_name not in model_fields:
                available_names = get_available_names_example(field_name, model_fields.keys())
                raise RuleSetupNameNotFoundError(f"Field name '{field_name}' not found in list of attributes of '{model}'. Available names: {available_names}")


    def get_key_pairs(self, instance: ModelType) -> KeyPairs:
        # apply_session:IApplySession
        # frame = apply_session.current_frame
        # instance = frame.instance

        keys = []

        for field_name in self.field_names:
            if not hasattr(instance, field_name):
                raise RuleApplyNameNotFoundError(f"Field name '{field_name}' not found in list of attributes of '{type(instance)}'!?")
            key = getattr(instance, field_name)
            if key is None:
                # if missing then setup temp unique id
                key = MissingKey()
            else:
                assert not str(key).startswith(GlobalConfig.ID_KEY_PREFIX_FOR_MISSING), key
            keys.append((field_name, key))
        return tuple(keys)



# class ListIndexKey(KeysBase):
#     # Can be applied only for children - e.g. Extension with multiple items
#     # parent assignes key as index of item in the list of "children").
# 
#     def validate(self, children_model: ModelType):
#         # children_model - check if get_type_info().is_list
#         raise NotImplementedError()
# 
#     def get_key_pairs(instance: ModelType) -> List[Tuple[str, Any]]:
#         raise NotImplementedError()

# ------------------------------------------------------------

@dataclass
class Rules(ContainerBase):
    name            : str
    contains        : List[Component]      = field(repr=False)

    # --- optional - following can be bound later with .bind_to()
    label           : Optional[TransMessageType] = field(repr=False, default=None)

    # binding interface - not dumped/exported
    bound_model     : Optional[BoundModel] = field(repr=False, default=None, metadata={"skip_dump": True})
    # will be filled automatically with Config() if not supplied
    config          : Optional[Type[Config]] = field(repr=False, default=None, metadata={"skip_dump": True})
    context_class   : Optional[Type[IContext]] = field(repr=False, default=None, metadata={"skip_dump": True})
    functions       : Optional[List[CustomFunctionFactory]] = field(repr=False, default_factory=list, metadata={"skip_dump": True})

    # --- only list of model names allowed
    keys            : Optional[KeysBase] = field(repr=False, default=None)

    # --- validators and evaluators
    cleaners        : Optional[List[Union[ValidationBase, EvaluationBase]]] = field(repr=False, default_factory=list)

    # --- Evaluated later
    setup_session      : Optional[SetupSession]    = field(init=False, repr=False, default=None)
    components      : Optional[Dict[str, Component]]  = field(init=False, repr=False, default=None)
    models          : Dict[str, Union[type, DotExpression]] = field(repr=False, init=False, default_factory=dict)
    # in Rules (top object) this case allway None - since it is top object
    parent           : Union[None, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    parent_name      : Union[str, UndefinedType]  = field(init=False, default=UNDEFINED)

    def __post_init__(self):
        # TODO: check that BoundModel.model is_model_class() and not DotExpression

        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.register(self)
        if not self.config:
            # default setup
            self.config = Config()

        self.init_clean()
        if not self.label:
            self.label = varname_to_title(self.name)
        super().__post_init__()

    def init_clean(self):
        if not isinstance(self.config, Config):
            raise RuleSetupValueError(owner=self, msg=f"config needs Config instance, got: {type(self.config)} / {self.config}")

        if self.context_class and not (
                inspect.isclass(self.context_class)
                and IContext in inspect.getmro(self.context_class)):
            raise RuleSetupValueError(owner=self, msg=f"context_class needs to be class that inherits IContext, got: {self.context_class}")

        # if self.functions:
        #     for function in self.functions:
        #         assert isinstance(function, IFunctionFactory), function


    # def setup(self):
    #     ret = super().setup()
    #     self.setup_session.call_hooks_on_finished_all()
    #     return ret

    def bind_to(self, 
                bound_model:Optional[BoundModel]=None, 
                config: Optional[Config]=None,
                context_class: Optional[IContext]=None,
                functions: Optional[List[CustomFunctionFactory]]=None,
                do_setup:bool = True,
                ):
        """
        late binding, will call .setup()
        """
        # TODO: DRY this - for loop
        if bound_model:
            if self.bound_model is not None:
                raise RuleSetupError(owner=self, msg="bound_model already already set, late binding not allowed.")
            self.bound_model = bound_model

        # if data:
        #     if self.data:
        #         raise RuleSetupError(owner=self, msg="data already set, late binding not allowed.")
        #     self.data = data

        if functions:
            if self.functions:
                raise RuleSetupError(owner=self, msg="functions already set, late binding not allowed.")
            self.functinos = functions

        if context_class:
            if self.context_class:
                raise RuleSetupError(owner=self, msg="context already set, late binding not allowed.")
            self.context_class = context_class

        if config:
            # overwrite
            self.config = config

        self.init_clean()

        if do_setup:
            self.setup()

    # ------------------------------------------------------------
    # apply - API entries
    # ------------------------------------------------------------

    def apply(self, 
              instance: DataclassType, 
              instance_new: Optional[ModelType] = None,
              context: Optional[IContext] = None, 
              raise_if_failed:bool = True) -> IApplySession:
        return self._apply(
                  instance=instance,
                  instance_new=instance_new,
                  context=context,
                  raise_if_failed=raise_if_failed)

    def apply_partial(self, 
              component_name_only:str,
              instance: DataclassType, 
              instance_new: Optional[ModelType] = None,
              context: Optional[IContext] = None, 
              raise_if_failed:bool = True) -> IApplySession:
        return self._apply(
                  instance=instance,
                  instance_new=instance_new,
                  component_name_only=component_name_only,
                  context=context,
                  raise_if_failed=raise_if_failed)


    def _apply(self, 
              instance: DataclassType, 
              instance_new: Optional[ModelType] = None,
              component_name_only:Optional[str] = None,
              context: Optional[IContext] = None, 
              raise_if_failed:bool = True) -> IApplySession:
        """
        create and config ApplyResult() and call apply_session.apply()
        """
        from .apply import ApplyResult
        container = self.get_container_parent(consider_self=True)

        apply_session = \
                ApplyResult(setup_session=container.setup_session, 
                      rules=self, 
                      component_name_only=component_name_only,
                      context=context, 
                      instance=instance,
                      instance_new=instance_new,
                      )\
                  .apply()

        if not apply_session.finished:
            raise RuleInternalError(owner=self, msg="Apply process is not finished")

        if raise_if_failed:
            apply_session.raise_if_failed()

        return apply_session

    # ------------------------------------------------------------

    def dump_defaults(self, 
              context: Optional[IContext] = None, 
              ) -> IApplySession:
        """
        In defaults mode:
            - context should be applied if Rules have (same as in apply())
            - validations are not called
        """
        from .apply import ApplyResult
        container = self.get_container_parent(consider_self=True)

        apply_session = \
                ApplyResult(
                    defaults_mode=True,
                    setup_session=container.setup_session, 
                    rules=self, 
                    component_name_only=None,
                    context=context, 
                    instance=NA_DEFAULTS_MODE,
                    instance_new=None,
                    )\
                  .apply()

        if not apply_session.finished:
            raise RuleInternalError(owner=self, msg="Apply process is not finished")

        if apply_session.errors:
            validation_error = RuleValidationError(owner=apply_session.rules, errors=apply_session.errors)
            raise RuleInternalError(owner=self, msg=f"Internal issue, apply process should not yield validation error(s), got: {validation_error}")

        output = apply_session._dump_defaults()

        return output


# ------------------------------------------------------------

@dataclass
class Extension(ContainerBase):
    """ can not be used individually - must be directly embedded into Other
        Extension or top Rules """

    # required since if it inherit name from BoundModel then the name will not
    # be unique in self.components (Extension and BoundModel will share the same name)
    name            : str
    bound_model     : Union[BoundModel, BoundModelWithHandlers] = field(repr=False)
    # metadata={"bind_to_parent_setup_session" : True})

    cardinality     : ICardinalityValidation
    contains        : List[Component] = field(repr=False)

    label           : Optional[TransMessageType] = field(repr=False, default=None)
    functions       : Optional[List[CustomFunctionFactory]] = field(repr=False, default_factory=list)
    # --- can be index based or standard key-fields names
    keys            : Optional[KeysBase] = field(repr=False, default=None)
    # --- validators and evaluators
    cleaners        : Optional[List[Union[ValidationBase, EvaluationBase]]] = field(repr=False, default_factory=list)

    # --- Evaluated later
    setup_session      : Optional[SetupSession] = field(init=False, repr=False, default=None)
    components      : Optional[Dict[str, Component]]  = field(init=False, repr=False, default=None)
    models          : Dict[str, Union[type, DotExpression]] = field(init=False, repr=False, default_factory=dict)
    parent           : Union[ComponentBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    parent_name      : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

    # extension specific - is this top parent or what? what is the difference to self.parent

    # in parents' chain (including self) -> first container
    parent_container : Union[ContainerBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)

    # in parents' chain (not including self) -> first container's setup_session
    parent_setup_session: Optional[SetupSession] = field(init=False, repr=False, default=None)

    # copy from first non-self container parent
    context_class   : Optional[Type[IContext]] = field(repr=False, init=False, default=None)
    config          : Optional[Type[Config]] = field(repr=False, init=False, default=None)

    # bound_attr_node  : Union[AttrDexpNode, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)

    # Class attributes
    # namespace_only  : ClassVar[Namespace] = ThisNS
    def __post_init__(self):
        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.register(self)
        if not self.label:
            self.label = varname_to_title(self.name)
        super().__post_init__()

    def set_parent(self, parent:ContainerBase):
        super().set_parent(parent=parent)

        # can be self
        self.parent_container     = self.get_container_parent(consider_self=True)

        # take from real first container parent
        non_self_parent_container = self.get_container_parent(consider_self=False)
        self.context_class = non_self_parent_container.context_class
        self.config = non_self_parent_container.config
        if not self.config:
            raise RuleInternalError(owner=self, msg=f"Config not set from parent: {self.parent_container}") 

    def setup(self, setup_session:SetupSession):
        # NOTE: setup_session is not used, can be reached with parent.setup_session(). left param
        #       for same function signature as for components.
        self.parent_setup_session = setup_session
        super().setup()
        self.cardinality.validate_setup()
        return self



def collect_classes(componnents_registry: Dict, module: Any, klass_match: type) -> Dict:
    if module:
        names = [(n, getattr(module, n)) 
                 for n in dir(module) 
                 if not n.startswith("_")]
    else:
        names = [(n,c) 
                 for n,c in globals().items() 
                 if not n.startswith("_")]

    for name, comp_klass in names:
        if isclass(comp_klass) \
          and issubclass(comp_klass, klass_match)\
          and not isabstract(comp_klass) \
          and not name.endswith("Base") \
          and name not in ("Component",) \
          :
            # and not hasattr(comp_klass, "__abstractmethods__"):
            componnents_registry[name] = comp_klass
    return componnents_registry


# TODO: not the best way + move function to some utils
COMPONNENTS_REGISTRY = {}
collect_classes(COMPONNENTS_REGISTRY, components, ComponentBase)
collect_classes(COMPONNENTS_REGISTRY, fields, ComponentBase)
collect_classes(COMPONNENTS_REGISTRY, validations, ComponentBase)
collect_classes(COMPONNENTS_REGISTRY, evaluations, ComponentBase)
collect_classes(COMPONNENTS_REGISTRY, None, ComponentBase)
