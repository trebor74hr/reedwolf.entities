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
    EntityValidationError,
    EntityTypeError,
    EntityValueError,
    EntityInitError,
)
from .namespaces import (
    ModelsNS,
    FieldsNS,
    )
from .meta_dataclass import (
    ComponentStatus,
    Self,
)
from .meta import (
    STANDARD_TYPE_LIST,
    TransMessageType,
    TypeInfo,
    is_model_klass,
    get_model_fields,
    ModelKlassType,
    NoneType,
    Index0Type,
    KeyType,
    KeyPairs,
    ModelInstanceType,
    ERR_MSG_ATTR_REQUIRED, ComponentName, MessageType,
)
from .base import (
    get_name_from_bind,
    IComponent,
    IContainer,
    IDataModel,
    GlobalConfig,
    IApplyResult,
    SetupStackFrame,
    IEntity,
    IUnboundDataModel,
)
from .expressions import (
    DotExpression,
    create_dexp_by_attr_name,
    IThisRegistry,
)
from .data_models import (
    DataModel,
    DataModelWithHandlers,
    UnboundModel,
)
from .expr_attr_nodes import (
    IAttrDexpNode,
    AttrDexpNodeForTypeInfo,
)
from .functions import (
    IFunction,
    FunctionsFactoryRegistry,
)
from .setup import (
    SetupSession,
    TopSetupSession,
)
from .registries import (
    ModelsRegistry,
    LocalFieldsRegistry,
    OperationsRegistry,
    ContextRegistry,
    UnboundModelsRegistry,
    TopFieldsRegistry,
    ThisRegistryForComponent,
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
    DEXP_VALIDATOR_FOR_BIND,
)
from .settings import (
    Settings, ApplySettings,
)
from .struct_converters import (
    StructConverterRunner,
)
# import MODULES that hold any kind of component - will be used for COMPONENTS_REGISTRY
from . import (
    fields,
    valid_field,
    valid_items,
    valid_children,
    eval_field,
    eval_items,
    eval_children,
)
from .value_accessors import (
    ValueAccessorInputType,
)


# ------------------------------------------------------------
# Entity
# ------------------------------------------------------------

@dataclass
class ContainerBase(IContainer, ABC):

    # ------ later evaluated ----
    _type_info: TypeInfo = field(init=False, repr=False, default=UNDEFINED)

    @staticmethod
    def is_container() -> bool:
        return True

    @staticmethod
    def can_have_children() -> bool:
        return True

    def get_this_registry_for_item(self) -> IThisRegistry:
        raise EntityInternalError(owner=self, msg=f"Function should be called only for SubEntityItems.")

    # def is_subentity(self):
    #     # TODO: not sure if this is good way to do it. Maybe: isinstance(ISubEntity) would be safer?
    #     # TODO: if self.parent is not None could be used as the term, put validation somewhere
    #     " if start model is value expression - that mean that the the Entity is SubEntityItems "
    #     return isinstance(self.data_model.model, DotExpression)

    def get_type_info(self) -> TypeInfo:
        """
        Currently used only for UnboundModel case
        """
        # _component_fields_dataclass must be created before, thus is setup_session None
        if self._type_info is UNDEFINED:
            _component_fields_dataclass, _ = self.get_component_fields_dataclass(setup_session=None)
            if self.is_subentity_items():
                type_hint = List[_component_fields_dataclass]
            else:
                type_hint = _component_fields_dataclass
            self._type_info = TypeInfo.get_or_create_by_type(type_hint)
        return self._type_info

    def _get_function(self, name: str, strict:bool=True) -> Optional[IFunction]:
        # TODO: used only in unit tests, remove it and make better test maybe?
        if not self.settings.custom_functions:
            raise KeyError(f"{self.name}: Function '{name}' not found, no functions available.")
        return self.setup_session.functions_factory_registry.get(name, strict=strict)

    # def add_fieldgroup(self, fieldgroup:FieldGroup):
    #     if self.is_finished:
    #         raise EntitySetupError(owner=self, msg="FieldGroup can not be added after setup() is called.")
    #     found = [sec for sec in self.contains if sec.name==fieldgroup.name]
    #     if found:
    #         raise EntitySetupError(owner=self, msg=f"FieldGroup {fieldgroup.name} is already added.")
    #     self.contains.append(fieldgroup)


    @staticmethod
    def can_apply_partial() -> bool:
        return True

    def __getitem__(self, name) -> IComponent:
        if self._status != ComponentStatus.finished:
            raise EntitySetupError(owner=self, msg=f"Not allowed, call .setup() first (current status={self._status})")
        if name not in self.components:
            vars_avail = get_available_names_example(name, list(self.components.keys()))
            raise KeyError(f"{self.name}: Component name '{name}' not found, available: {vars_avail}")
        return self.components[name]

    # ------------------------------------------------------------

    def create_setup_session(self, settings: Optional[Settings] = None) -> SetupSession:
        """
        settings param - only for unit testing
        """
        assert self.setup_session is None

        # TODO: remove exception case: entity can be UNDEFINED in unit test cases only
        apply_settings_class = self.entity.apply_settings_class if self.entity else None
        functions = self.entity.settings.get_all_custom_functions() if self.entity else []
        settings = settings if settings else self.settings
        builtin_functions_dict = self.entity.settings.get_all_builtin_functions_dict() \
                                 if self.entity and self.is_entity() else {}

        functions_factory_registry: FunctionsFactoryRegistry = \
            FunctionsFactoryRegistry(functions=functions,
                                     builtin_functions_dict=builtin_functions_dict)

        if not self.parent:
            top_fields_registry = TopFieldsRegistry(entity=self)
            setup_session = TopSetupSession(
                container=self,
                functions_factory_registry = functions_factory_registry,
                top_fields_registry = top_fields_registry,
            )
        else:
            # for FieldGroup/BooleanField + children cases needs to use parent_container instead of parent
            setup_session = SetupSession(
                container=self,
                functions_factory_registry = functions_factory_registry,
                parent_setup_session=self.parent_container.setup_session
            )
            top_fields_registry = setup_session.top_setup_session.top_fields_registry

        if self.is_unbound():
            setup_session.add_registry(UnboundModelsRegistry())
        else:
            setup_session.add_registry(ModelsRegistry())
        setup_session.add_registry(LocalFieldsRegistry(
                                        container=self,
                                        top_fields_registry=top_fields_registry,
                                        ))
        # setup_session.add_registry(FunctionsRegistry())
        setup_session.add_registry(OperationsRegistry())
        setup_session.add_registry(ContextRegistry(setup_settings=settings,
                                                   apply_settings_class=apply_settings_class))

        self.setup_session = setup_session

        return setup_session

    # ------------------------------------------------------------

    def _register_model_attr_nodes(self) -> Dict[str, IDataModel]:
        # ----------------------------------------
        # A. 1st level attr_nodes
        # ------------------------------------------------------------
        # A.1. MODELS - collect attr_nodes from managed models
        # ------------------------------------------------------------
        models = self.data_model.fill_models()
        if not models:
            raise EntitySetupError(owner=self, msg="Entity(models=List[models]) is required.")

        # if not isinstance(self.models, dict):
        #     # TODO: this never happens - define new test case, implement (or drop this logic)
        #     self.data_model.fill_models()

        assert isinstance(models, dict), models

        # can have several, first is main model, other are submodels e.g. DataModelWithHandlers
        for nr, (data_model_name, data_model) in enumerate(models.items()):
            assert data_model_name.split(".")[-1] == data_model.name
            self._register_data_model(data_model=data_model)

        # NOTE: self.models was used only here
        #   self.models = models
        return models

    # ------------------------------------------------------------

    def _setup_data_model_dot_expression(self, data_model:IDataModel, setup_session: Optional[SetupSession] = None) -> IAttrDexpNode:
        model_klass = data_model.model_klass
        if not isinstance(model_klass, DotExpression):
            raise EntityInternalError(owner=self, msg=f"Expecting model is DotExpression instance, got: {model_klass}")

        is_main_model = (data_model==self.data_model)
        is_subentity_main_model = (self.is_subentity_any() and is_main_model)
        if is_subentity_main_model:
            if not isinstance(model_klass, DotExpression):
                raise EntitySetupError(owner=self, msg=f"{data_model.name}: For SubEntityItems/SubEntity main data_model needs to be DotExpression: {data_model.model_klass}")

        # TODO: for functions value expressions need to be stored
        #       with all parameters (func_args)
        if not (self.is_subentity_any() or not is_main_model):
            raise EntitySetupTypeError(owner=self, msg=f"{data_model.name}: DotExpression should be used only in SubEntity containers and nested /DataModelDataModels")

        model_klass._SetDexpValidator(DEXP_VALIDATOR_FOR_BIND)
        # if model._namespace!=ModelsNS:
        #     raise EntitySetupTypeError(owner=self, msg=f"{data_model.name}: DotExpression should be in ModelsNS namespace, got: {model._namespace}")

        if setup_session:
            # TODO: not sure if this is the best solution
            setup_session_from = setup_session
        else:
            if is_subentity_main_model:
                # TODO: DRY this - the only difference is setup_session - extract common logic outside / 
                # bound attr_node
                setup_session_from = self.parent_container.setup_session
            else:
                # Entity - top parent container / normal case
                setup_session_from = self.setup_session

        attr_node = setup_session_from.get_dexp_node_by_dexp(dexp=model_klass)
        if self.is_unbound():
            if not attr_node:
                raise EntityInternalError(owner=self, msg=f"AttrDexpNode data should already be in setup_session: {model_klass} -> {attr_node}")
        else:
            if attr_node:
                # NOTE: it seems this does not happen, so I added this exception. Remove it if it will be required in the future
                raise EntityInternalError(owner=self, msg=f"TODO: AttrDexpNode data already in setup_session: {model_klass} -> {attr_node}")
            attr_node = model_klass.Setup(setup_session=setup_session_from, owner=data_model)

        if not attr_node:
            raise EntityInternalError(owner=self, msg=f"AttrDexpNode not recognized: {model_klass}")

        if not isinstance(attr_node, AttrDexpNodeForTypeInfo):
            raise EntityInternalError(owner=self, msg=f"AttrDexpNode data is not TypeInfo, got: {attr_node}")

        return attr_node

    # ------------------------------------------------------------

    def _register_data_model(self, data_model:IDataModel, unbound_mode: bool = False):
        # Entity can have one main data_model and optionally some dependent
        # models nested in tree structure

        if not isinstance(data_model, IDataModel):
            raise EntitySetupError(owner=self, msg=f"{data_model.name}: Needs to be DataModel* instance, got: {data_model}")

        attr_node = None
        # alias_saved = False
        is_list = False
        if not hasattr(data_model, "model_klass"):
            raise EntityInternalError(owner=self, msg=f"data_model.model_klass not set")
        model_klass = data_model.model_klass

        if isinstance(model_klass, DotExpression):
            attr_node = self._setup_data_model_dot_expression(data_model)

            # model_klass   = attr_node.data.type_
            type_info = attr_node.type_info
            model_klass = type_info.type_
            is_list = type_info.is_list
            py_type_hint = type_info.py_type_hint

            if self.is_subentity() and is_list:
                raise EntitySetupTypeError(owner=self, msg=f"{data_model.name}: For SubEntity did not expect List model type, got: {py_type_hint}")
            elif self.is_subentity_items() and not is_list:
                raise EntitySetupTypeError(owner=self, msg=f"{data_model.name}: For SubEntityItems expected List model type, got: {py_type_hint}")

            # TODO: check data_model cases - list, not list, etc.
            # elif self.is_entity_model() and ...
        else:
            if self.is_subentity_any():
                raise EntitySetupTypeError(owner=self, msg=f"{data_model.name}: For SubEntity use DotExpression as model, got: {model_klass}")
            # TODO: maybe check model type_info is_list ...

        if not is_model_klass(model_klass) and not (is_list and model_klass in STANDARD_TYPE_LIST):
            raise EntitySetupError(owner=self, msg=f"Managed model {data_model.name} needs to be a @dataclass, pydantic.BaseModel or List[{STANDARD_TYPE_LIST}], got: {type(model_klass)}")

        # == M.name version
        models_registry = self.setup_session.get_registry(ModelsNS)
        models_registry.register_all_nodes(root_attr_node=attr_node,
                                           data_model=data_model,
                                           model_klass=model_klass,
                                           unbound_mode=unbound_mode)

        return attr_node

    # ------------------------------------------------------------
    def _replace_modelsns_registry(self, setup_session: SetupSession):
        # subentity case?
        assert self.is_unbound()

        is_unbound_data_model = isinstance(self.data_model, IUnboundDataModel)
        if is_unbound_data_model:
            # top level - concrete class
            assert self.is_entity()
            # setup_session=None - dataclass should be already created
            fields_dataclass, _ = self.get_component_fields_dataclass(setup_session=None)
            self.data_model = DataModel(model_klass=fields_dataclass)
        else:
            if self.data_model is None:
                raise EntityInternalError(owner=self, msg=f"data_model not set")

        self.data_model.set_parent(self)

        if not is_unbound_data_model:
            # not-top level - dot-expression
            assert not self.is_entity()
            # model_dexp_node: IDotExpressionNode
            if not isinstance(self.data_model.model_klass, DotExpression):
                raise EntityInternalError(owner=self, msg=f"In unbound mode data_model is expected to be DotExpression, got: {self.data_model}")
            # Setup must be called before - in order to set type_info
            self.data_model.model_klass.Setup(setup_session=setup_session, owner=self)

        models_registry = self.setup_session.get_registry(ModelsNS)
        assert models_registry.is_unbound_models_registry()
        # TODO: any need to save old models_registry?
        models_registry = ModelsRegistry()
        self.setup_session.add_registry(registry=models_registry, replace=True)
        self._register_data_model(data_model=self.data_model, unbound_mode=True)

    # ------------------------------------------------------------

    def init(self):
        super().init()
        if self.data_model is None:
            self.data_model = UnboundModel()
            # raise EntitySetupError(owner=self, msg="data_model not set. Initialize in constructor or call bind_to() first.")

    # ------------------------------------------------------------

    def _setup(self, setup_session: SetupSession):

        # components are flat list, no recursion/hierarchy browsing needed
        if self.is_finished:
            raise EntitySetupError(owner=self, msg="setup() should be called only once")

        # if not self.contains:
        #     raise EntitySetupError(owner=self, msg="'contains' attribute is required with list of components")

        with self.setup_session.use_stack_frame(
                SetupStackFrame(
                    container = self, 
                    component = self, 
                    this_registry = None,
                )):


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
            #       premature component.bind_to setup in some ThisRegistryFor* classes.
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
            # NOTE: normally .finish() is called within _setup_phase_two(). This call covers rare cases
            #       when finish() is not called. This complex term is consequence of having
            #       subentity* components in two containers: their owner and themselves.
            if not isinstance(component, UnboundModel) \
              and not component.is_finished \
              and (component is not self or component.is_entity()):
                component.finish()
                # raise EntityInternalError(owner=self, msg=f"{component} not finished. Is in overriden setup()/Setup() parent method super().setup()/Setup() been called (which sets parent and marks finished)?")

        self.setup_session.finish()

        if self.keys:
            # Inner DataModel can have self.data_model.model_klass = DotExpression
            self.keys.validate(self.data_model.get_type_info().type_)

    # ------------------------------------------------------------

    def create_this_registry(self, setup_session: SetupSession) -> Optional[IThisRegistry]:
        """
        This.<field>
        This.Children
        """
        if self.is_data_model():
            raise EntityInternalError(owner=self, msg=f"For DataModel create_this_registry() needs to be overridden.")

        if not self.has_children():
            raise EntityInternalError(owner=self, msg="Non-fields should have children")

        this_registry = ThisRegistryForComponent(component=self)
        return this_registry

    # ------------------------------------------------------------

    def get_data_model_attr_node(self) -> IAttrDexpNode:
        return self.setup_session.get_registry(ModelsNS).get_attr_node_by_data_model(data_model=self.data_model)

    # ------------------------------------------------------------

    def get_component(self, name:str) -> IComponent:
        if name not in self.components:
            vars_avail = get_available_names_example(name, self.components.keys())
            raise EntityNameNotFoundError(owner=self, msg=f"Component '{name}' not found, some valid_are: {vars_avail}")
        return self.components[name]

    # ------------------------------------------------------------

    def pprint(self):
        if not hasattr(self, "components"):
            raise EntitySetupError(owner=self, msg="Call .setup() first")
        print(f"{self.name}: {self.__class__.__name__}::")
        for nr, (name, component) in enumerate(self.components.items(),1):
            print(f"  {nr:02}. {name}: {component.__class__.__name__}::")
            print(f"      {repr(component)[:100]}") # noqa: T001
            # TODO: if pp() exists -> recursion with depth+1 (indent)

    # ------------------------------------------------------------

    def get_key_pairs_or_index0(self,
                                apply_result: IApplyResult,
                                instance: ModelInstanceType,
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


    def get_key_pairs(self, instance: ModelInstanceType, apply_result: IApplyResult) \
            -> KeyPairs:
        if not self.keys:
            raise EntityInternalError(msg="get_key_pairs() should be called only when 'keys' are defined")
        key_pairs = self.keys.get_key_pairs(instance, apply_result=apply_result)
        return key_pairs


# ------------------------------------------------------------


# TODO: inherit ReedwolfDataclassBase to have copy() or not?
class KeysBase(ABC):

    @abstractmethod
    def validate(self, model_or_children_model: ModelKlassType):
        ...

    @abstractmethod
    def get_key_pairs(self, instance: ModelInstanceType, container: IContainer) -> KeyPairs:
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
    field_names: List[str]

    def __post_init__(self):
        if not self.field_names:
            raise EntitySetupError("field_names are required")
        for field_name in self.field_names:
            if not isinstance(field_name, str):
                raise EntitySetupNameError(f"Fields needs to be list of strings, got '{field_name}'")
        if len(set(self.field_names)) != len(self.field_names):
            raise EntitySetupNameError(f"Fields needs to be unique list of names, found duplicates: '{self.field_names}'")

    def validate(self, model_klass: ModelKlassType):
        model_fields = get_model_fields(model_klass)
        for field_name in self.field_names:
            if field_name not in model_fields:
                available_names = get_available_names_example(field_name, model_fields.keys())
                raise EntitySetupNameNotFoundError(f"Field name '{field_name}' not found in list of attributes of '{model_klass}'. Available names: {available_names}")


    def get_key_pairs(self, instance: ModelInstanceType, apply_result: IApplyResult) -> KeyPairs:
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

    # NOTE: only this one is obligatory
    contains:           List[IComponent] = field(repr=False, default_factory=list)

    # --- optional - following can be bound later with .bind_to()
    name:               Optional[ComponentName] = field(default=None)
    title:              Union[MessageType, TransMessageType, NoneType] = field(repr=False, default=None)

    # binding interface - not dumped/exported
    bind_to:            Optional[Union[DataModel, ModelKlassType]] = field(repr=False, default=None,
                                                                           metadata={"skip_dump": True, "skip_setup": True})
    # list of names from model / contains element names
    keys:               Optional[KeysBase] = field(repr=False, default=None)
    # will be filled automatically with Settings() if not supplied
    settings:           Optional[Settings] = field(repr=False, default=None, metadata={"skip_dump": True})
    apply_settings_class: Optional[Type[ApplySettings]] = field(repr=False, default=None, metadata={"skip_dump": True})
    accessor:           Optional[ValueAccessorInputType] = field(repr=False, default=None)

    # --- validators and evaluators
    cleaners:           Optional[List[Union[ChildrenValidationBase, ChildrenEvaluationBase]]] = field(repr=False, default_factory=list)

    # custom metadata
    meta:               Optional[Dict[str, Any]] = field(repr=False, default=None)

    # ------------------------------------------------------------
    # --- Evaluated later
    # ------------------------------------------------------------
    data_model:        Optional[DataModel] = field(init=False, repr=False, metadata={"skip_dump": True})

    setup_session:      Optional[SetupSession] = field(init=False, repr=False, default=None)
    # see comment in IContainer
    components:         Optional[Dict[str, IComponent]] = field(init=False, repr=False, default=None)

    # in Entity (top object) this case allways None - since it is top object
    # NOTE: not DRY: Entity, SubentityBase and ComponentBase
    parent:             Union[NoneType, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    parent_name:        Union[str, UndefinedType]  = field(init=False, default=UNDEFINED)
    entity:             Union[Self, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)

    # is_unbound to model case - must be cached since data_model could be dynamically changed in setup phase with normal
    _is_unbound: bool = field(init=False, repr=False)


    def init(self):
        if not (self.contains and isinstance(self.contains, (tuple, list))):
            raise EntityInitError(owner=self, msg=f"Attribute 'contains' is required and needs to be a list of components (got: {self.contains})")

        if not self.settings:
            # default setup
            self.settings = Settings()

        if self.bind_to:
            if is_model_klass(self.bind_to):
                self.data_model = DataModel(model_klass=self.bind_to)
            else:
                self.data_model = self.bind_to

            if not (isinstance(self.data_model, DataModel) and is_model_klass(self.data_model.model_klass)):
                raise EntitySetupTypeError(owner=self, msg=f"Attribute 'data_model' needs to be model DC/PYD OR DataModel with model DC/PYD, got: {self.data_model}")

            if not self.name:
                self.name = "__".join([
                    camel_case_to_snake(self.__class__.__name__),
                    camel_case_to_snake(self.data_model.model_klass.__name__),
                    ])
        else:
            # unbound mode
            self.data_model = None

        self._is_unbound = self.data_model is None

        self._check_cleaners([ChildrenValidationBase, ChildrenEvaluationBase])

        if not isinstance(self.settings, Settings):
            raise EntitySetupValueError(owner=self, msg=f"settings needs to be Settings instance, got: {type(self.settings)} / {self.settings}")

        # ------------------------------------------------------------
        # Settings check / setup not done in Settings.__post_init__ since the method is not guaranteed to be called
        # for custom implementations.
        # ------------------------------------------------------------
        self.settings._init()

        if self.settings.apply_settings_class:
            if self.apply_settings_class:
                raise EntityValueError(owner=self,
                                       msg=f"Argument apply_settings_class set in settings too. Skip this argument or remove from settings instance. Got: {self.apply_settings_class}")
            self.apply_settings_class = self.settings.apply_settings_class

        if self.apply_settings_class and not (
                inspect.isclass(self.apply_settings_class)
                and ApplySettings in inspect.getmro(self.apply_settings_class)):
            raise EntityTypeError(owner=self, msg=f"apply_settings_class needs to be class that inherits ApplySettings, got: {self.apply_settings_class}")

        super().init()

    def is_unbound(self) -> bool:
        return self._is_unbound

    def set_parent(self, parent: NoneType):
        assert parent is None
        self.containers_dict = {}
        self.container_id_counter = 1
        self.container_id = self.name
        self.containers_dict[self.container_id] = self
        self.containers_id_path = [self.container_id]
        super().set_parent(parent=parent)

    @staticmethod
    def is_entity() -> bool:
        return True

    # NOTE: replaced with generic change() method - see meta::ReedwolfMetaclass.change()
    # def change(self,
    #            bind_to:Union[UndefinedType, DataModel]=UNDEFINED,
    #            settings: Optional[Settings]=None,
    #            apply_settings_class: Optional[Type[ApplySettings]]=None,
    #            # do_setup:bool = True,
    #            ) -> Self:
    #     """
    #     Handy function that returns entity back -> allows dot-chaining.
    #     Just sets/updates some attributes. No checks until setup() is called.
    #     """
    #     if self.is_finished:
    #         raise EntityInternalError(owner=self, msg="Entity already marked as finished.")

    #     if bind_to != UNDEFINED:
    #         # NOTE: allowed to change to None (not yet tested)
    #         self.bind_to = bind_to

    #     if apply_settings_class is not None:
    #         # overwrite
    #         # if self.apply_settings_class:
    #         #     raise EntitySetupError(owner=self, msg="settings already set, late binding not allowed.")
    #         self.apply_settings_class = apply_settings_class

    #     if settings is not None:
    #         # overwrite
    #         self.settings = settings

    #     return self

    # ------------------------------------------------------------
    # setup
    # ------------------------------------------------------------
    def setup(self) -> Self:
        # ----------------------------------------
        # SETUP PHASE one (recursive)
        # ----------------------------------------
        # Traverse all subcomponents and call the same method for each (recursion)
        # NOTE: Will setup all data_model and bind_to and ModelsNS.
        #       In phase two will setup all other components and FieldsNS, ThisNS and FunctionsNS.
        self._setup_phase_one()

        super()._setup(setup_session=self.setup_session)
        # if self.is_top_parent():
        assert self.is_entity()
        self.setup_session.call_hooks_on_finished_all()

        return self


    def _setup_phase_one(self, components: Optional[Dict[str, Self]] = None) -> NoneType:
        super()._setup_phase_one(components=components)

        # A.3. COMPONENTS - collect attr_nodes - previously flattened (recursive function fill_components)
        #      each container will have own Store and then associated LocalFieldsRegistry
        #      (with assigned TopFieldsRegistry)
        fields_registry: LocalFieldsRegistry = self.setup_session.get_registry(FieldsNS)

        # Only Entity can do setup - it will walk through all components and fill the stores which
        # are later used in LocalFieldsRegistry-s.
        fields_registry.register_all()

    # ------------------------------------------------------------
    # apply - API entries
    # ------------------------------------------------------------

    def apply(self,
              instance: ModelInstanceType,
              instance_new: Optional[ModelInstanceType] = None,
              settings: Optional[ApplySettings] = None,
              accessor: Optional[ValueAccessorInputType] = None,
              raise_if_failed:bool = True) -> IApplyResult:
        return self._apply(
                    instance=instance,
                    instance_new=instance_new,
                    settings=settings,
                    accessor=accessor,
                    raise_if_failed=raise_if_failed)

    def apply_partial(self,
                      component_name_only:str,
                      instance: ModelInstanceType,
                      instance_new: Optional[ModelInstanceType] = None,
                      settings: Optional[Settings] = None,
                      accessor: Optional[ValueAccessorInputType] = None,
                      raise_if_failed:bool = True) -> IApplyResult:
        return self._apply(
                    instance=instance,
                    instance_new=instance_new,
                    component_name_only=component_name_only,
                    settings=settings,
                    accessor=accessor,
                    raise_if_failed=raise_if_failed)


    def _apply(self,
               instance: ModelInstanceType,
               instance_new: Optional[ModelInstanceType] = None,
               component_name_only:Optional[str] = None,
               settings: Optional[Settings] = None,
               accessor: Optional[ValueAccessorInputType] = None,
               raise_if_failed:bool = True) -> IApplyResult:
        """
        create and settings ApplyResult() and call apply_result.apply()
        """
        from .apply import ApplyResult

        apply_result = \
                ApplyResult(
                      entity=self,
                      component_name_only=component_name_only,
                      settings=settings,
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
                      settings: Optional[Settings] = None,
                      ) -> IApplyResult:
        """
        In defaults mode:
            - settings should be applied if Entity have (same as in apply())
            - validations are not called
        """
        from .apply import ApplyResult

        # container = self.get_first_parent_container(consider_self=True)
        apply_result = \
                ApplyResult(
                    defaults_mode=True,
                    entity=self,
                    component_name_only=None,
                    settings=settings,
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

    def create_dto_instance_from_model_instance(self, instance: ModelInstanceType, dto_class: ModelKlassType) -> ModelInstanceType:
        if not isinstance(instance, self.data_model.model_klass):
            raise EntityTypeError(owner=self, msg=f"Expected  {self.data_model.model_klass} instance, got: {instance}: {type(instance)}")

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
    Can not be used individually - must be directly embedded into Other
    SubEntityItems or top Entity
    """
    # DotExpression based model -> can be dumped, obligatory
    bind_to:        Union[DataModel, DataModelWithHandlers, DotExpression, str] \
                        = field(repr=True, metadata = {"skip_setup": True}, default=UNDEFINED)

    # cardinality:  ICardinalityValidation
    contains:       List[IComponent] = field(repr=False, default_factory=list)

    accessor:       Optional[ValueAccessorInputType] = field(repr=False, default=None)

    # required since if it inherit name from DataModel then the name will not
    # be unique in self.components (SubEntityItems and DataModel will share the same name)
    name:           Optional[ComponentName] = field(default=None)
    title:          Union[MessageType, TransMessageType, NoneType] = field(repr=False, default=None)

    # --- can be index based or standard key-fields names
    keys:           Optional[KeysBase] = field(repr=False, default=None)
    # --- validators and evaluators
    cleaners:       Optional[List[Union[ValidationBase, EvaluationBase]]] = field(repr=False, default_factory=list)

    # custom metadata
    meta:               Optional[Dict[str, Any]] = field(repr=False, default=None)

    # ------------------------------------------------------------
    # --- Evaluated later
    # ------------------------------------------------------------

    # DotExpression based model -> can be dumped
    data_model:    Union[DataModel, DataModelWithHandlers] = field(init=False, repr=False)

    setup_session:  Optional[SetupSession] = field(init=False, repr=False, default=None)
    # see comment in IContainer
    components:     Optional[Dict[str, IComponent]]  = field(init=False, repr=False, default=None)

    # --- IComponent common attrs
    # NOTE: not DRY: Entity, SubentityBase and ComponentBase
    parent:         Union[IComponent, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    parent_name:    Union[str, UndefinedType] = field(init=False, default=UNDEFINED)
    entity:         Union[Entity, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)

    # subentity_items specific - is this top parent or what? what is the difference to self.parent

    # NOTE: in parents' chain (not including self) -> first container.
    parent_container: Union[ContainerBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)

    # NOTE: this is dropped and replaced with parent.setup_session
    #   in parents' chain (not including self) -> first container's setup_session
    # parent_setup_session: Optional[SetupSession] = field(init=False, repr=False, default=None)

    # copy from first non-self container parent
    #   NTOE: for this use self.entity.apply_settings_class instad
    #       apply_settings_class: Optional[Type[ApplySettings]] = field(repr=False, init=False, default=None)
    settings: Optional[Settings] = field(repr=False, init=False, default=None)

    # bound_attr_node: Union[IAttrDexpNode, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)

    # Class attributes
    # namespace_only: ClassVar[Namespace] = ThisNS

    def init(self):
        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.register(self)

        if not self.bind_to:
            raise EntityInitError(owner=self, msg=ERR_MSG_ATTR_REQUIRED.format("bind_to"))

        if not (self.contains and isinstance(self.contains, (tuple, list))):
            raise EntityInitError(owner=self, msg=f"Attribute 'contains' is required and needs to be a list of components (got: {self.contains})")

        if isinstance(self.bind_to, str):
            self.bind_to = create_dexp_by_attr_name(ModelsNS, self.bind_to)

        if isinstance(self.bind_to, DotExpression):
            # Clone it to be sure that is not used - if problem, then clone only if not built
            model_klass_dexp = self.bind_to.Clone()
            self.data_model = DataModel(model_klass=model_klass_dexp)
        else:
            self.data_model = self.bind_to

        if not (isinstance(self.data_model, IDataModel) and isinstance(self.data_model.model_klass, DotExpression)):
            raise EntityInternalError(owner=self, msg=f"Attribute data_model needs to be DotExpression "
                                                      f"OR DataModel with DotExpression '.model', "
                                                      f"got: '{self.data_model}' : {type(self.data_model)}")

        if not self.name:
            self.name = get_name_from_bind(self.data_model.model_klass)

        super().init()

    def set_parent(self, parent: ContainerBase):
        super().set_parent(parent=parent)

        # first parent which is container (not including self)
        self.parent_container = self.get_first_parent_container(consider_self=False)
        assert self.parent_container and self.parent_container != self

        # --- setup container_id and register container in entity.containers_dict ---
        container_id = self.name
        if container_id in self.entity.containers_dict:
            # using - to avoid another variable clash. Component's name must be identifiers, no - is allowed.
            container_id = f"{container_id}--{self.entity.container_id_counter}"
            self.entity.container_id_counter += 1
        if container_id in self.entity.containers_dict:
            raise EntityInternalError(owner=self, msg=f"Failed to get container unique name (id_name={container_id}.")
        self.container_id = container_id
        self.entity.containers_dict[self.container_id] = self
        self.containers_id_path = self.parent_container.containers_id_path + [self.container_id]

        # take from real first container parent
        # non_self_parent_container = self.get_first_parent_container(consider_self=False)
        # # self.apply_settings_class = non_self_parent_container.apply_settings_class
        # self.settings = non_self_parent_container.settings

        # --- Setup settings ---
        self.settings = self.parent_container.settings
        if not self.settings:
            raise EntityInternalError(owner=self, msg=f"settings not set from parent: {self.parent_container}")

        # self._accessor = self.settings.get_accessor(self._accessor) if self._accessor is not None else non_self_parent_container._accessor


    # def setup(self, setup_session:SetupSession):
    #     # NOTE: setup_session is not used, can be reached with parent.setup_session(). left param
    #     #       for same function signature as for components.
    #     super().setup()
    #     # self.cardinality.validate_setup()
    #     return self


# ------------------------------------------------------------

@dataclass
class SubEntityItems(SubEntityBase):
    """
    one to many relations - e.g. Person -> PersonAddresses
    """

    # _this_registry          - will be used from List[Item] perspective, i.e. for This.Items
    # _this_registry_for_item - will be used from Item perspective (insde of an Item from List), i.e. This.<field-name>
    _this_registry_for_item: Union[IThisRegistry, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)

    def init(self):
        # TODO: ChildrenValidationBase, ChildrenEvaluationBase
        self._check_cleaners([ItemsValidationBase, ItemsEvaluationBase])
        # for cleaner in self.cleaners:
        #     if not isinstance(cleaner, (ItemsValidationBase, ChildrenValidationBase, ItemsEvaluationBase, ChildrenEvaluationBase)):
        #         raise EntitySetupTypeError(owner=self, msg=f"Cleaners should be instances of ItemsValidationBase, ChildrenValidationBase, ItemsEvaluationBase or ChildrenEvaluationBase, got: {type(cleaner)} / {cleaner}") 
        super().init()

    @staticmethod
    def is_subentity_items() -> bool:
        return True

    def get_this_registry_for_item(self) -> IThisRegistry:
        if not self._this_registry_for_item:
            raise EntityInternalError(owner=self, msg="self._this_registry_for_item not initialized")
        return self._this_registry_for_item

    def create_this_registry(self, setup_session: SetupSession) -> Optional[IThisRegistry]:
        this_registry = super().create_this_registry(setup_session)
        assert self._this_registry_for_item is UNDEFINED
        # model_klass = type_info.py_type_hint
        this_registry_for_item = ThisRegistryForComponent(component=self, is_items_for_each_mode=True)
        # do setup() immediately. For self._this_registry it is done in self.get_or_create_this_registry()
        this_registry_for_item.setup(setup_session=setup_session)
        this_registry_for_item.finish()
        self._this_registry_for_item = this_registry_for_item

        return this_registry


# ------------------------------------------------------------

@dataclass
class SubEntity(SubEntityBase):
    """ one to one relations - e.g. Person -> PersonAccess """

    cleaners: Optional[List[Union[SingleValidation, ChildrenValidationBase, ChildrenEvaluationBase]]] = field(repr=False, default_factory=list)

    def init(self):
        self._check_cleaners([SingleValidation, ChildrenValidationBase, ChildrenEvaluationBase])
        # for cleaner in self.cleaners:
        #     if not isinstance(cleaner, (SingleValidation, ChildrenValidationBase, ChildrenEvaluationBase)):
        #         raise EntitySetupTypeError(owner=self, msg=f"Cleaners should be instances of SingleValidation, ChildrenValidationBase or ChildrenEvaluationBase, got: {type(cleaner)} / {cleaner}") 
        super().init()

    @staticmethod
    def is_subentity() -> bool:
        return True

    # @staticmethod
    # def may_collect_my_children() -> bool:
    #     # currently not possible - sometimes child.bind_to need to be setup and
    #     # setup can be done only fields which are inside the same container
    #     # share the same data_model
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
          and name not in ("Component",):
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

