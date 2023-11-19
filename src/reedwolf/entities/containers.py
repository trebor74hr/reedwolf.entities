import inspect
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Type,
)
from dataclasses import (
    dataclass,
    field,
)

from .utils import (
    get_available_names_example,
    UNDEFINED,
    NA_DEFAULTS_MODE,
    UndefinedType,
    camel_case_to_snake,
    )
from .exceptions import (
    EntitySetupError,
    EntitySetupValueError,
    EntitySetupNameError,
    EntitySetupTypeError,
    EntityInternalError,
    EntityNameNotFoundError,
    EntitySetupNameNotFoundError,
    EntityApplyNameNotFoundError,
    EntityValidationError, EntityTypeError,
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
    Self,
    NoneType,
    Index0Type,
    KeyType,
    KeyPairs,
)
from .base import (
    get_name_from_bind,
    IComponent,
    IContainer,
    IBoundModel,
    GlobalConfig,
    IApplyResult,
    SetupStackFrame,
    IEntity,
    IUnboundModel,
)
from .expressions import (
    DotExpression,
)
from .bound_models import (
    BoundModel,
    BoundModelWithHandlers, UnboundModel,
)
from .attr_nodes import (
    AttrDexpNode,
    )
from .functions import (
    CustomFunctionFactory,
    IFunction,
    )
from .registries import (
    SetupSession,
    ModelsRegistry,
    FieldsRegistry,
    FunctionsRegistry,
    OperationsRegistry,
    ContextRegistry,
    ConfigRegistry, UnboundModelsRegistry,
)
from .valid_children import (
    ChildrenValidationBase,
    )
from .eval_children import (
    ChildrenEvaluationBase,
    )
from .valid_items import (
    ItemsValidationBase,
    SingleValidation,
    )
from .eval_items import (
    ItemsEvaluationBase,
    )
from .valid_base import (
    ValidationBase,
    )
from .eval_base import (
    EvaluationBase,
    )
from .fields import (
    FieldGroup, DEXP_VALIDATOR_FOR_BIND,
)
from .contexts import (
    IContext,
)
from .config import (
    Config,
)
from .struct_converters import (
    StructConverterRunner,
)
# import modules that hold any kind of component - will be used for COMPONENTS_REGISTRY
from . import (
    fields,
    valid_field,
    valid_items,
    valid_children,
    eval_field,
    eval_items,
    eval_children,
)
from .values_accessor import (
    IValueAccessor,
    STANDARD_VALUE_ACCESSOR_CLASS_REGISTRY,
    DEFAULT_VALUE_ACCESSOR_CLASS,
)


# ------------------------------------------------------------
# Entity
# ------------------------------------------------------------

class ContainerBase(IContainer, ABC):

    @staticmethod
    def is_container() -> bool:
        return True

    @staticmethod
    def can_have_children() -> bool:
        return True

    # def is_subentity(self):
    #     # TODO: not sure if this is good way to do it. Maybe: isinstance(ISubEntity) would be safer?
    #     # TODO: if self.parent is not None could be used as the term, put validation somewhere
    #     " if start model is value expression - that mean that the the Entity is SubEntityItems "
    #     return isinstance(self.bound_model.model, DotExpression)

    def get_type_info(self) -> TypeInfo:
        """
        Currently used only for UnboundModel case
        """
        # _component_fields_dataclass must be created before, thus is setup_session None
        _component_fields_dataclass, _ = self.get_component_fields_dataclass(setup_session=None)
        if self.is_subentity_items():
            type_hint = List[_component_fields_dataclass]
        else:
            type_hint = _component_fields_dataclass
        type_info = TypeInfo.get_or_create_by_type(type_hint)
        return type_info

    def _get_function(self, name: str, strict:bool=True) -> Optional[IFunction]:
        if not self.functions:
            raise KeyError(f"{self.name}: Function '{name}' not found, no functions available.")
        return self.setup_session.functions_factory_registry.get(name, strict=strict)

    def add_fieldgroup(self, fieldgroup:FieldGroup):
        if self.is_finished():
            raise EntitySetupError(owner=self, msg="FieldGroup can not be added after setup() is called.")
        found = [sec for sec in self.contains if sec.name==fieldgroup.name]
        if found:
            raise EntitySetupError(owner=self, msg=f"FieldGroup {fieldgroup.name} is already added.")
        self.contains.append(fieldgroup)


    @staticmethod
    def can_apply_partial() -> bool:
        return True

    def __getitem__(self, name) -> IComponent:
        if name not in self.components:
            vars_avail = get_available_names_example(name, list(self.components.keys()))
            raise KeyError(f"{self.name}: Component name '{name}' not found, available: {vars_avail}")
        return self.components[name]

    # ------------------------------------------------------------

    def create_setup_session(self, config: Optional[Config] = None) -> SetupSession:
        """
        config param - only for unit testing
        """
        #
        # functions: Optional[List[CustomFunctionFactory]]
        # context_class: Optional[IContext]
        assert self.setup_session is None

        functions = self.functions
        config = config if config else self.config
        context_class = self.context_class

        setup_session = SetupSession(
            container=self,
            functions=functions,
            parent_setup_session=self.setup_session if self.parent else None,
            include_builtin_functions=self.is_entity())

        if self.is_unbound():
            setup_session.add_registry(UnboundModelsRegistry())
        else:
            setup_session.add_registry(ModelsRegistry())
        setup_session.add_registry(FieldsRegistry())
        setup_session.add_registry(FunctionsRegistry())
        setup_session.add_registry(OperationsRegistry())
        setup_session.add_registry(ContextRegistry(context_class=context_class))
        setup_session.add_registry(ConfigRegistry(config=config))

        self.setup_session = setup_session

        return setup_session

    # ------------------------------------------------------------

    def _register_model_attr_nodes(self):
        # ----------------------------------------
        # A. 1st level attr_nodes
        # ------------------------------------------------------------
        # A.1. MODELS - collect attr_nodes from managed models
        # ------------------------------------------------------------
        models = self.bound_model.fill_models()
        if not models:
            raise EntitySetupError(owner=self, msg="Entity(models=List[models]) is required.")

        # if not isinstance(self.models, dict):
        #     # TODO: this never happens - define new test case, implement (or drop this logic)
        #     self.bound_model.fill_models()

        assert isinstance(models, dict), models

        # can have several, first is main model, other are submodels e.g. BoundModelWithHandlers
        for nr, (bound_model_name, bound_model) in enumerate(models.items()):
            assert bound_model_name.split(".")[-1] == bound_model.name
            self._register_bound_model(bound_model=bound_model)

        # NOTE: self.models was used only here
        #   self.models = models

    # ------------------------------------------------------------

    def _setup_bound_model_dot_expression(self, bound_model:IBoundModel, setup_session: Optional[SetupSession] = None) -> AttrDexpNode:
        model = bound_model.model
        if not isinstance(model, DotExpression):
            raise EntityInternalError(owner=self, msg=f"Expecting model is DotExpression instance, got: {model}") 

        is_main_model = (bound_model==self.bound_model)
        is_subentity_main_model = (self.is_subentity() and is_main_model)
        if is_subentity_main_model:
            if not isinstance(model, DotExpression):
                raise EntitySetupError(owner=self, msg=f"{bound_model.name}: For SubEntityItems/SubEntitySingle main bound_model needs to be DotExpression: {bound_model.model}")

        # TODO: for functions value expressions need to be stored
        #       with all parameters (func_args)
        if not (self.is_subentity() or not is_main_model):
            raise EntitySetupTypeError(owner=self, msg=f"{bound_model.name}: DotExpression should be used only in SubEntity containers and nested BoundModels")

        model._SetDexpValidator(DEXP_VALIDATOR_FOR_BIND)
        # if model._namespace!=ModelsNS:
        #     raise EntitySetupTypeError(owner=self, msg=f"{bound_model.name}: DotExpression should be in ModelsNS namespace, got: {model._namespace}")

        if setup_session:
            # TODO: not sure if this is the best solution
            setup_session_from = setup_session
        else:
            if is_subentity_main_model:
                # TODO: DRY this - the only difference is setup_session - extract common logic outside / 
                # bound attr_node
                setup_session_from = self.parent.setup_session
            else:
                # Entity - top parent container / normal case
                setup_session_from = self.setup_session

        attr_node = setup_session_from.get_dexp_node_by_dexp(dexp=model)
        if self.is_unbound():
            if not attr_node:
                raise EntityInternalError(owner=self, msg=f"AttrDexpNode data should already be in setup_session: {model} -> {attr_node}")
        else:
            if attr_node:
                # NOTE: it seems this does not happen, so I added this exception. Remove it if it will be required in the future
                raise EntityInternalError(owner=self, msg=f"TODO: AttrDexpNode data already in setup_session: {model} -> {attr_node}")
            attr_node = model.Setup(setup_session=setup_session_from, owner=bound_model)

        if not attr_node:
            raise EntityInternalError(owner=self, msg=f"AttrDexpNode not recognized: {model}")

        if not isinstance(attr_node.data, TypeInfo):
            raise EntityInternalError(owner=self, msg=f"AttrDexpNode data is not TypeInfo, got: {type(attr_node.data)} / {attr_node.data}")

        return attr_node

    # ------------------------------------------------------------

    def _register_bound_model(self, bound_model:IBoundModel, unbound_mode: bool = False):
        # Entity can have one main bound_model and optionally some dependent
        # models nested in tree structure

        if not isinstance(bound_model, IBoundModel):
            raise EntitySetupError(owner=self, msg=f"{bound_model.name}: Needs to be BoundModel* instance, got: {bound_model}")

        attr_node = None
        # alias_saved = False
        is_list = False
        model = bound_model.model

        if isinstance(model, DotExpression):
            attr_node = self._setup_bound_model_dot_expression(bound_model)

            model   = attr_node.data.type_
            is_list = attr_node.data.is_list
            py_type_hint = attr_node.data.py_type_hint

            if self.is_subentity_single() and is_list:
                raise EntitySetupTypeError(owner=self, msg=f"{bound_model.name}: For SubEntitySingle did not expect List model type, got: {py_type_hint}")
            elif self.is_subentity_items() and not is_list:
                raise EntitySetupTypeError(owner=self, msg=f"{bound_model.name}: For SubEntityItems expected List model type, got: {py_type_hint}")

            # TODO: check bound_model cases - list, not list, etc.
            # elif self.is_entity_model() and ...
        else:
            if self.is_subentity():
                raise EntitySetupTypeError(owner=self, msg=f"{bound_model.name}: For SubEntity use DotExpression as model, got: {model}")
            # TODO: maybe check model type_info is_list ...

        if not is_model_class(model) and not (is_list and model in STANDARD_TYPE_LIST):
            raise EntitySetupError(owner=self, msg=f"Managed model {bound_model.name} needs to be a @dataclass, pydantic.BaseModel or List[{STANDARD_TYPE_LIST}], got: {type(model)}")

        # == M.name version
        models_registry = self.setup_session.get_registry(ModelsNS)
        models_registry.register_all_nodes(root_attr_node=attr_node,
                                           bound_model=bound_model,
                                           model=model,
                                           unbound_mode=unbound_mode)

        return attr_node

    # ------------------------------------------------------------

    def _register_fields_components_attr_nodes(self):
        """
        Traverse the whole tree (recursion) and collect all components into
        simple flat list. It will set parent for each child component.
        """
        # A.3. COMPONENTS - collect attr_nodes - previously flattened (recursive function fill_components)
        for component_name, component in self.components.items():
            self.setup_session.get_registry(FieldsNS).register(component)

    # ------------------------------------------------------------
    def _replace_modelsns_registry(self, setup_session: SetupSession):
        # subentity case
        assert (self.is_unbound())
        self.bound_model.set_parent(self)

        if isinstance(self.bound_model, IUnboundModel):
            assert self.is_entity()
            # setup_session=None - dataclass should be already created
            fields_dataclass, _ = self.get_component_fields_dataclass(setup_session=None)
            self.bound_model = BoundModel(model=fields_dataclass)
        else:
            assert not self.is_entity()
            # model_dexp_node: IDotExpressionNode
            if not isinstance(self.bound_model.model, DotExpression):
                raise EntityInternalError(owner=self, msg=f"In unbound mode bound_model is expected to be DotExpression, got: {self.bound_model}")
            # Setup must be called before - in order to set type_info
            self.bound_model.model.Setup(setup_session=setup_session, owner=self)

        models_registry = self.setup_session.get_registry(ModelsNS)
        assert models_registry.is_unbound_models_registry()
        # TODO: any need to save old models_registry?
        models_registry = ModelsRegistry()
        self.setup_session.add_registry(registry=models_registry, replace=True)
        self._register_bound_model(bound_model=self.bound_model, unbound_mode=True)

    # ------------------------------------------------------------

    def setup(self) -> Self:
        # components are flat list, no recursion/hierarchy browsing needed
        if self.is_finished():
            raise EntitySetupError(owner=self, msg="setup() should be called only once")

        if self.bound_model is None:
            self.bound_model = UnboundModel()
            # raise EntitySetupError(owner=self, msg="bound_model not set. Initialize in constructor or call bind_to() first.")

        if not self.contains:
            raise EntitySetupError(owner=self, msg="'contains' attribute is required with list of components")

        if self.is_entity():
            # ----------------------------------------
            # SETUP PHASE one (recursive)
            # ----------------------------------------
            # Traverse all subcomponents and call the same method for each (recursion)
            # NOTE: Will setup all bound_model and bind and ModelsNS.
            #       In phase two will setup all other components and FieldsNS, ThisNS and FunctionsNS.
            self._setup_phase_one()

        with self.setup_session.use_stack_frame(
                SetupStackFrame(
                    container = self, 
                    component = self, 
                    this_registry = None,
                )):

            # TODO: FieldsNS - in session - for future improvement - to allow
            #       use of SubentityItems() (now in denied)
            self._register_fields_components_attr_nodes()

            # ----------------------------------------
            # SETUP PHASE TWO (recursive)
            # ----------------------------------------
            # iterate all subcomponents and call _setup() for each
            # now when all attr_nodes are set, now setup() can be called for all
            # components recursively:
            #   it will validate every component attribute
            #   if attribute is another component - it will call recursively component.setup()
            #   if component attribute is DotExpression -> will call dexp.Setup()
            #   if component is another container i.e. is_subentity_items() - it will
            #       process only that component and will not go deeper. later
            #       subentity_items.setup() will do this within own tree dep (own .components / .setup_session)

            # NOTE: setup this_registry objects must be inside of stack_frame due
            #       premature component.bind setup in some ThisRegistryFor* classes.
            # this_registry = self.create_this_registry(component=self, setup_session=self.setup_session)
            this_registry = self.get_or_create_this_registry(setup_session=self.setup_session)
            self.setup_session.current_frame.set_this_registry(this_registry)

            # ------------------------------------------------------------
            # setup me and all my children and children's children and ...
            # ------------------------------------------------------------
            # RECURSION - goes through all tree nodes
            self._setup_phase_two(setup_session=self.setup_session)
            # ------------------------------------------------------------


        # check all ok?
        for component_name, component in self.components.items():
            # TODO: maybe bound Field.bind -> Model attr_node?
            if not isinstance(component, UnboundModel) and not component.is_finished():
                raise EntityInternalError(owner=self, msg=f"{component} not finished. Is in overriden setup()/Setup() parent method super().setup()/Setup() been called (which sets parent and marks finished)?")

        self.setup_session.finish()

        if self.keys:
            # Inner BoundModel can have self.bound_model.model = DotExpression
            self.keys.validate(self.bound_model.get_type_info().type_)

        if self.is_top_parent():
            self.setup_session.call_hooks_on_finished_all()

        return self

    # ------------------------------------------------------------

    def get_bound_model_attr_node(self) -> AttrDexpNode:
        return self.setup_session.get_registry(ModelsNS).get_attr_node_by_bound_model(bound_model=self.bound_model)

    # ------------------------------------------------------------

    def get_component(self, name:str) -> IComponent:
        # TODO: currently components are retrieved only from contains - but should include validations + cardinality
        if name not in self.components:
            vars_avail = get_available_names_example(name, self.components.keys())
            raise EntityNameNotFoundError(owner=self, msg=f"Component '{name}' not found, some valid_are: {vars_avail}")
        return self.components[name]

    # ------------------------------------------------------------

    def pprint(self):
        if not hasattr(self, "components"):
            raise EntitySetupError(owner=self, msg="Call .setup() first")
        print(f"{self.name}: {self.__class__.__name__} ::")
        for nr, (name, component) in enumerate(self.components.items(),1):
            print(f"  {nr:02}. {name}: {component.__class__.__name__} ::")
            print(f"      {repr(component)[:100]}") # noqa: T001
            # TODO: if pp() exists -> recursion with depth+1 (indent)

    # ------------------------------------------------------------

    def get_key_pairs_or_index0(self,
                                apply_result: IApplyResult,
                                instance: ModelType,
                                index0: Index0Type,
                                ) -> KeyType:
        """
        index0 is 0 based index of item in the list
        """
        # TODO: move to ApplyResult:IApplyResult?
        if self.keys:
            ret = self.get_key_pairs(instance, apply_result=apply_result)
        else:
            ret = index0
        return ret


    def get_key_pairs(self, instance: ModelType, apply_result: IApplyResult) \
            -> KeyPairs:
        if not self.keys:
            raise EntityInternalError(msg="get_key_pairs() should be called only when 'keys' are defined")
        key_pairs = self.keys.get_key_pairs(instance, apply_result=apply_result)
        return key_pairs


    # @staticmethod
    # def create_this_registry_for_model_class(
    #         setup_session: ISetupSession,
    #         model_class: ModelType,
    #         ) -> IThisRegistry:
    #     # NOTE: must be here since:
    #     #   - expressions.py don't see ThisRegistry
    #     #   - setup.py does not see registries
    #     #   - registries - needed by setup.py which can not see registries
    #     # TODO: try to resolve this and put
    #     """
    #     - ThisRegistry is unavailable to low-level modules -
    #       e.g. func_args -> setup.
    #     - .Instance + <attr-names> is used only in manual setup cases,
    #       e.g. ChoiceField()
    #     """
    #     this_registry = ThisRegistry(model_class=model_class)
    #     this_registry.setup(setup_session=setup_session)
    #     this_registry.finish()

    #     return this_registry

# ------------------------------------------------------------



# ------------------------------------------------------------


class KeysBase(ABC):

    @abstractmethod
    def validate(self, model_or_children_model: ModelType):
        ...

    @abstractmethod
    def get_key_pairs(self, instance: ModelType, container: IContainer) -> KeyPairs:
        """ returns list of (key-name, key-value) """
        ...

    # def get_keys(self, apply_result:IApplyResult) -> List[Any]:
    #     return [key for name,key in self.get_keys_tuple(apply_result)]


# def get_new_unique_id() -> int:
#     return GlobalConfig.ID_KEY_COUNTER.get_new()


class MissingKey:
    def __init__(self, id):
        # old: self.id = id if id else get_new_unique_id()
        assert id
        self.id = id

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
            raise EntitySetupError("field_names are required")
        for field_name in self.field_names:
            if not isinstance(field_name, str):
                raise EntitySetupNameError(f"Fields needs to be list of strings, got '{field_name}'")
        if len(set(self.field_names)) != len(self.field_names):
            raise EntitySetupNameError(f"Fields needs to be unique list of names, found duplicates: '{self.field_names}'")

    def validate(self, model: ModelType):
        model_fields = get_model_fields(model)
        for field_name in self.field_names:
            if field_name not in model_fields:
                available_names = get_available_names_example(field_name, model_fields.keys())
                raise EntitySetupNameNotFoundError(f"Field name '{field_name}' not found in list of attributes of '{model}'. Available names: {available_names}")


    def get_key_pairs(self, instance: ModelType, apply_result: IApplyResult) -> KeyPairs:
        # container: IContainer
        # apply_result:IApplyResult
        # frame = apply_result.current_frame
        # instance = frame.instance

        keys = []

        for field_name in self.field_names:
            if not hasattr(instance, field_name):
                raise EntityApplyNameNotFoundError(f"Field name '{field_name}' not found in list of attributes of '{type(instance)}'!?")
            key = getattr(instance, field_name)
            if key is None:
                # if missing then setup temp unique id
                missing_key_id = apply_result._get_new_id()
                # container._get_new_id_by_parent_name(GlobalConfig.ID_KEY_PREFIX_FOR_MISSING)
                key = MissingKey(id=missing_key_id)
            else:
                assert not str(key).startswith(GlobalConfig.ID_KEY_PREFIX_FOR_MISSING), key
            keys.append((field_name, key))
        return tuple(keys)


# ------------------------------------------------------------


@dataclass
class Entity(IEntity, ContainerBase):

    contains        : List[IComponent] = field(repr=False)

    # --- optional - following can be bound later with .bind_to()
    name            : Optional[str] = field(default=None)
    title           : Optional[TransMessageType] = field(repr=False, default=None)

    # binding interface - not dumped/exported
    bound_model     : Optional[BoundModel] = field(repr=False, default=None, metadata={"skip_dump": True})
    # will be filled automatically with Config() if not supplied
    config          : Optional[Config] = field(repr=False, default=None, metadata={"skip_dump": True})
    context_class   : Optional[Type[IContext]] = field(repr=False, default=None, metadata={"skip_dump": True})
    functions       : Optional[List[CustomFunctionFactory]] = field(repr=False, default_factory=list, metadata={"skip_dump": True})

    # --- only list of model names allowed
    keys            : Optional[KeysBase] = field(repr=False, default=None)

    # --- validators and evaluators
    cleaners        : Optional[List[Union[ChildrenValidationBase, ChildrenEvaluationBase]]] = field(repr=False, default_factory=list)

    # --- Evaluated later
    setup_session      : Optional[SetupSession]    = field(init=False, repr=False, default=None)
    components      : Optional[Dict[str, IComponent]]  = field(init=False, repr=False, default=None)
    # NOTE: used only internally:
    # models          : Dict[str, Union[type, DotExpression]] = field(repr=False, init=False, default_factory=dict)

    # in Entity (top object) this case allways None - since it is top object
    # NOTE: not DRY: Entity, SubentityBase and ComponentBase
    parent           : Union[NoneType, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    parent_name      : Union[str, UndefinedType]  = field(init=False, default=UNDEFINED)
    entity           : Union[ContainerBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    value_accessor   : IValueAccessor = field(init=False, repr=False)

    # used for automatic component's naming, <parent_name/class_name>__<counter>
    name_counter_by_parent_name: Dict[str, int] = field(init=False, repr=False, default_factory=dict)
    value_accessor_class_registry: Dict[str, Type[IValueAccessor]] = field(init=False, repr=False)
    value_accessor_default: IValueAccessor = field(init=False, repr=False)

    # is_unbound case - must be cached since bound_model could be dynamically changed in setup phase with normal
    _is_unbound: bool = field(init=False, repr=False)

    def __post_init__(self):
        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.register(self)
        if not self.config:
            # default setup
            self.config = Config()

        self.init_clean()
        super().__post_init__()

    def is_unbound(self) -> bool:
        return self._is_unbound

    def init_clean(self):
        if self.bound_model:
            if not (isinstance(self.bound_model, BoundModel) and is_model_class(self.bound_model.model)):
                raise EntitySetupTypeError(owner=self, msg=f"Attribute 'bound_model' needs to be BoundModel with model DC/PYD, got: {self.bound_model}") 

            if not self.name:
                self.name = "__".join([
                    camel_case_to_snake(self.__class__.__name__),
                    camel_case_to_snake(self.bound_model.model.__name__),
                    ])

        self._is_unbound = self.bound_model is None

        self._check_cleaners([ChildrenValidationBase, ChildrenEvaluationBase])

        if not isinstance(self.config, Config):
            raise EntitySetupValueError(owner=self, msg=f"config needs Config instance, got: {type(self.config)} / {self.config}")

        # TODO: it is copied to enable user to extend with new ones (should be added in Config/Resource/Repository)
        self.value_accessor_class_registry = STANDARD_VALUE_ACCESSOR_CLASS_REGISTRY.copy()
        self.value_accessor_default = self.config.value_accessor if self.config.value_accessor \
                                      else DEFAULT_VALUE_ACCESSOR_CLASS()
        assert isinstance(self.value_accessor_default, IValueAccessor)

        if self.context_class and not (
                inspect.isclass(self.context_class)
                and IContext in inspect.getmro(self.context_class)):
            raise EntitySetupValueError(owner=self, msg=f"context_class needs to be class that inherits IContext, got: {self.context_class}")

        # if self.functions:
        #     for function in self.functions:
        #         assert isinstance(function, IFunctionFactory), function


    # def setup(self):
    #     ret = super().setup()
    #     self.setup_session.call_hooks_on_finished_all()
    #     return ret

    @staticmethod
    def is_entity() -> bool:
        return True

    def bind_to(self, 
                bound_model:Union[UndefinedType, BoundModel]=UNDEFINED,
                config: Optional[Config]=None,
                context_class: Optional[IContext]=None,
                functions: Optional[List[CustomFunctionFactory]]=None,
                do_setup:bool = True,
                ):
        """
        late binding, will call .setup()
        """
        # TODO: DRY this - for loop
        if self.is_finished():
            raise EntityInternalError(owner=self, msg="Entity already marked as finished.")

        if bound_model != UNDEFINED:
            # NOTE: allowed to change to None (not yet tested)
            # if self.bound_model is not None:
            #     raise EntitySetupError(owner=self, msg="bound_model already already set, late binding not allowed.")
            self.bound_model = bound_model

        # if data:
        #     if self.data:
        #         raise EntitySetupError(owner=self, msg="data already set, late binding not allowed.")
        #     self.data = data

        if functions:
            if self.functions:
                raise EntitySetupError(owner=self, msg="functions already set, late binding not allowed.")
            self.functions = functions

        if context_class:
            if self.context_class:
                raise EntitySetupError(owner=self, msg="context already set, late binding not allowed.")
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
              raise_if_failed:bool = True) -> IApplyResult:
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
              raise_if_failed:bool = True) -> IApplyResult:
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
              raise_if_failed:bool = True) -> IApplyResult:
        """
        create and config ApplyResult() and call apply_result.apply()
        """
        from .apply import ApplyResult

        apply_result = \
                ApplyResult(
                      entity=self,
                      component_name_only=component_name_only,
                      context=context, 
                      instance=instance,
                      instance_new=instance_new,
                      )\
                  .apply()

        if not apply_result.finished:
            raise EntityInternalError(owner=self, msg="Apply process is not finished")

        if raise_if_failed:
            apply_result.raise_if_failed()

        return apply_result

    # ------------------------------------------------------------

    def dump_defaults(self, 
              context: Optional[IContext] = None, 
              ) -> IApplyResult:
        """
        In defaults mode:
            - context should be applied if Entity have (same as in apply())
            - validations are not called
        """
        from .apply import ApplyResult

        # container = self.get_first_parent_container(consider_self=True)
        apply_result = \
                ApplyResult(
                    defaults_mode=True,
                    entity=self,
                    component_name_only=None,
                    context=context,
                    instance=NA_DEFAULTS_MODE,
                    instance_new=None,
                    )\
                  .apply()

        if not apply_result.finished:
            raise EntityInternalError(owner=self, msg="Apply process is not finished")

        if apply_result.errors:
            validation_error = EntityValidationError(owner=apply_result.entity, errors=apply_result.errors)
            raise EntityInternalError(owner=self, msg=f"Internal issue, apply process should not yield validation error(s), got: {validation_error}")

        output = apply_result._dump_defaults()

        return output

    def create_dto_instance_from_model_instance(self, instance: ModelType, dto_class: Type[ModelType]) -> ModelType:
        if not isinstance(instance, self.bound_model.model):
            raise EntityTypeError(owner=self, msg=f"Expected  {self.bound_model.model} instance, got: {instance} : {type(instance)}")

        struct_converter_runner = StructConverterRunner()
        dto_instance = struct_converter_runner.create_dto_instance_from_model_instance(
                            component=self,
                            instance=instance,
                            dto_class=dto_class)

        return dto_instance


# ------------------------------------------------------------


@dataclass
class SubEntityBase(ContainerBase, ABC):
    """
    can not be used individually - must be directly embedded into Other
    SubEntityItems or top Entity
    """
    # DotExpression based model -> can be dumped
    bound_model     : Union[BoundModel, BoundModelWithHandlers] = field(repr=False)

    # cardinality     : ICardinalityValidation
    contains        : List[IComponent] = field(repr=False)

    # required since if it inherit name from BoundModel then the name will not
    # be unique in self.components (SubEntityItems and BoundModel will share the same name)
    name            : Optional[str] = field(default=None)
    title           : Optional[TransMessageType] = field(repr=False, default=None)
    functions       : Optional[List[CustomFunctionFactory]] = field(repr=False, default_factory=list, metadata={"skip_dump": True})
    # --- can be index based or standard key-fields names
    keys            : Optional[KeysBase] = field(repr=False, default=None)
    # --- validators and evaluators
    cleaners        : Optional[List[Union[ValidationBase, EvaluationBase]]] = field(repr=False, default_factory=list)

    # --- Evaluated later
    setup_session    : Optional[SetupSession] = field(init=False, repr=False, default=None)
    components       : Optional[Dict[str, IComponent]]  = field(init=False, repr=False, default=None)
    # NOTE: used only internally:
    # models           : Dict[str, Union[type, DotExpression]] = field(init=False, repr=False, default_factory=dict)

    # --- IComponent common attrs
    # NOTE: not DRY: Entity, SubentityBase and ComponentBase
    parent           : Union[IComponent, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    parent_name      : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)
    entity           : Union[ContainerBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    value_accessor   : IValueAccessor = field(init=False, repr=False)

    # subentity_items specific - is this top parent or what? what is the difference to self.parent

    # in parents' chain (including self) -> first container
    parent_container : Union[ContainerBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)

    # NOTE: this is dropped and replaced with parent.setup_session
    #   in parents' chain (not including self) -> first container's setup_session
    # parent_setup_session: Optional[SetupSession] = field(init=False, repr=False, default=None)

    # copy from first non-self container parent
    context_class   : Optional[Type[IContext]] = field(repr=False, init=False, default=None)
    config          : Optional[Type[Config]] = field(repr=False, init=False, default=None)

    # used for automatic component's naming, <class_name>__<counter>
    name_counter_by_parent_name: Dict[str, int] = field(init=False, repr=False, default_factory=dict)

    # bound_attr_node  : Union[AttrDexpNode, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)

    # Class attributes
    # namespace_only  : ClassVar[Namespace] = ThisNS

    def __post_init__(self):
        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.register(self)
        if not (isinstance(self.bound_model, IBoundModel) and isinstance(self.bound_model.model, DotExpression)):
            raise EntityInternalError(owner=self, msg=f"Attribute bound_model='{self.bound_model}' is not BoundModel instance or '.model' is not a DotExpression") 
        if not self.name:
            self.name = get_name_from_bind(self.bound_model.model)

        super().__post_init__()

    def set_parent(self, parent:ContainerBase):
        super().set_parent(parent=parent)

        # can be self
        self.parent_container     = self.get_first_parent_container(consider_self=True)

        # take from real first container parent
        non_self_parent_container = self.get_first_parent_container(consider_self=False)
        self.context_class = non_self_parent_container.context_class
        self.config = non_self_parent_container.config
        if not self.config:
            raise EntityInternalError(owner=self, msg=f"Config not set from parent: {self.parent_container}") 

    def setup(self, setup_session:SetupSession):
        # NOTE: setup_session is not used, can be reached with parent.setup_session(). left param
        #       for same function signature as for components.
        super().setup()
        # self.cardinality.validate_setup()
        return self


# ------------------------------------------------------------

@dataclass
class SubEntityItems(SubEntityBase):
    """ one to many relations - e.g. Person -> PersonAddresses """

    def __post_init__(self):
        # TODO: ChildrenValidationBase, ChildrenEvaluationBase
        self._check_cleaners([ItemsValidationBase, ItemsEvaluationBase])
        # for cleaner in self.cleaners:
        #     if not isinstance(cleaner, (ItemsValidationBase, ChildrenValidationBase, ItemsEvaluationBase, ChildrenEvaluationBase)):
        #         raise EntitySetupTypeError(owner=self, msg=f"Cleaners should be instances of ItemsValidationBase, ChildrenValidationBase, ItemsEvaluationBase or ChildrenEvaluationBase, got: {type(cleaner)} / {cleaner}") 
        super().__post_init__()

    @staticmethod
    def is_subentity_items() -> bool:
        return True


# ------------------------------------------------------------

@dataclass
class SubEntitySingle(SubEntityBase):
    """ one to one relations - e.g. Person -> PersonAccess """

    cleaners        : Optional[List[Union[SingleValidation, ChildrenValidationBase, ChildrenEvaluationBase]]] = field(repr=False, default_factory=list)

    def __post_init__(self):
        self._check_cleaners([SingleValidation, ChildrenValidationBase, ChildrenEvaluationBase])
        # for cleaner in self.cleaners:
        #     if not isinstance(cleaner, (SingleValidation, ChildrenValidationBase, ChildrenEvaluationBase)):
        #         raise EntitySetupTypeError(owner=self, msg=f"Cleaners should be instances of SingleValidation, ChildrenValidationBase or ChildrenEvaluationBase, got: {type(cleaner)} / {cleaner}") 
        super().__post_init__()

    @staticmethod
    def is_subentity_single() -> bool:
        return True

    # @staticmethod
    # def may_collect_my_children() -> bool:
    #     # currently not possible - sometimes child.bind need to be setup and
    #     # setup can be done only fields which are inside the same container
    #     # share the same bound_model 
    #     return True

# ------------------------------------------------------------

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
        if inspect.isclass(comp_klass) \
          and issubclass(comp_klass, klass_match)\
          and not inspect.isabstract(comp_klass) \
          and not name.endswith("Base") \
          and name not in ("Component",) \
          :
            # and not hasattr(comp_klass, "__abstractmethods__"):
            componnents_registry[name] = comp_klass
    return componnents_registry


# TODO: not the best way + move function to some utils
COMPONENTS_REGISTRY = {}
collect_classes(COMPONENTS_REGISTRY, fields, IComponent)
collect_classes(COMPONENTS_REGISTRY, valid_field, IComponent)
collect_classes(COMPONENTS_REGISTRY, valid_items, IComponent)
collect_classes(COMPONENTS_REGISTRY, valid_children, IComponent)
collect_classes(COMPONENTS_REGISTRY, eval_field, IComponent)
collect_classes(COMPONENTS_REGISTRY, eval_items, IComponent)
collect_classes(COMPONENTS_REGISTRY, eval_children, IComponent)
collect_classes(COMPONENTS_REGISTRY, None, IComponent)

# ------------------------------------------------------------
# OBSOLETE - DELETE THIS
# ------------------------------------------------------------
# with setup_session_from.use_changed_current_stack_frame(
# with setup_session_from.use_stack_frame(
#         SetupStackFrame(
#             container=self,
#             component=self,
#             dexp_validator=DexpValidator(
#                 allow_functions=False,
#                 allow_operations=False,
#                 allow_namespaces=[ModelsNS],
#                 max_path_depth=self.DEXP_MODELSNS_MAX_PATH_DEPTH)
#         )):

