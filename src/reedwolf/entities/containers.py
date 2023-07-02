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
        )
from .base import (
        get_name_from_bind,
        ComponentBase,
        IContainerBase,
        BoundModelBase,
        GlobalConfig,
        KeyPairs,
        IApplySession,
        SetupStackFrame,
        ISetupSession,
        IFieldBase,
        )
from .expressions import (
        DotExpression,
        IThisRegistry,
        )
from .bound_models import (
        BoundModel,
        BoundModelWithHandlers,
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
        ConfigRegistry,
        ThisRegistryForValue,
        ThisRegistryForChildren,
        ThisRegistryForValueAndChildren,
        ThisRegistryForItemsAndChildren,
        ThisRegistryForInstance,
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
        FieldGroup,
        )
from .contexts import (
        IContext,
        )
from .config import (
        Config,
        )

from . import (
        fields, 
        valid_field, 
        valid_items, 
        valid_children, 
        eval_field,
        eval_items,
        eval_children,
        )

# ------------------------------------------------------------
# Entity
# ------------------------------------------------------------

class ContainerBase(IContainerBase, ComponentBase, ABC):

    def is_container(self) -> bool:
        return True

    # def is_subentity(self):
    #     # TODO: not sure if this is good way to do it. Maybe: isinstance(ISubEntity) would be safer?
    #     # TODO: if self.parent is not None could be used as the term, put validation somewhere
    #     " if start model is value expression - that mean that the the Entity is SubEntityItems "
    #     return isinstance(self.bound_model.model, DotExpression)

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

    def is_top_parent(self):
        return not bool(self.parent)


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
            raise EntitySetupError(owner=self, msg="Entity(models=List[models]) is required.")

        if not isinstance(self.models, dict):
            # TODO: this never happens - define new test case, implenet (or drop this logic)
            self.bound_model.fill_models()

        assert isinstance(self.models, dict), self.models

        for bound_model_name, bound_model in self.models.items():
            assert bound_model_name.split(".")[-1] == bound_model.name
            self._register_bound_model(bound_model=bound_model)

    # ------------------------------------------------------------

    def _register_bound_model(self, bound_model:BoundModelBase):
        # ex. type_info.metadata.get("bind_to_parent_setup_session")

        # Entity can have one main bound_model and optionally some dependent
        # models nested in tree structure
        is_main_model = (bound_model==self.bound_model)

        is_subentity_main_model = (self.is_subentity() and is_main_model)

        # is_list = False
        if not isinstance(bound_model, BoundModelBase):
            raise EntitySetupError(owner=self, msg=f"{bound_model.name}: Needs to be Boundbound_model* instance, got: {bound_model}")

        model = bound_model.model

        attr_node = None
        if is_subentity_main_model:
            if not isinstance(model, DotExpression):
                raise EntitySetupError(owner=self, msg=f"{bound_model.name}: For SubEntityItems/SubEntitySingle main bound_model needs to be DotExpression: {bound_model.model}")

        # alias_saved = False
        is_list = False

        if isinstance(model, DotExpression):
            # TODO: for functions value expressions need to be stored
            #       with all parameters (func_args)
            if not (self.is_subentity() or not is_main_model):
                raise EntitySetupTypeError(owner=self, msg=f"{bound_model.name}: DotExpression should be used only in SubEntity containers and nested BoundModels")

            if model._namespace!=ModelsNS:
                raise EntitySetupTypeError(owner=self, msg=f"{bound_model.name}: DotExpression should be in ModelsNS namespace, got: {model._namespace}")

            if is_subentity_main_model:
                # TODO: DRY this - the only difference is setup_session - extract common logic outside / 
                # bound attr_node
                assert hasattr(self, "parent_setup_session")
                setup_session_from = self.parent_setup_session
            else:
                # Entity - top parent container / normal case
                setup_session_from = self.setup_session

            attr_node = setup_session_from.get_dexp_node_by_dexp(dexp=model)
            if attr_node:
                raise EntityInternalError(owner=self, msg=f"AttrDexpNode data already in setup_session: {model} -> {attr_node}")

            attr_node = model.Setup(setup_session=setup_session_from, owner=bound_model)
            if not attr_node:
                raise EntityInternalError(owner=self, msg=f"AttrDexpNode not recognized: {model}")

            if not isinstance(attr_node.data, TypeInfo):
                raise EntityInternalError(owner=self, msg=f"AttrDexpNode data is not TypeInfo, got: {type(attr_node.data)} / {attr_node.data}")

            model   = attr_node.data.type_
            is_list = attr_node.data.is_list
            py_type_hint = attr_node.data.py_type_hint

            if self.is_subentity_single() and is_list :
                raise EntitySetupTypeError(owner=self, msg=f"{bound_model.name}: For SubEntitySingle did not expect List model type, got: {py_type_hint}")
            elif self.is_subentity_items() and not is_list:
                raise EntitySetupTypeError(owner=self, msg=f"{bound_model.name}: For SubEntityItems expected List model type, got: {py_type_hint}")

            # TODO: check bound_model cases - list, not list, etc.
            # elif self.is_bound_model() and ...

        else:
            if self.is_subentity():
                raise EntitySetupTypeError(owner=self, msg=f"{bound_model.name}: For SubEntity use DotExpression as model, got: {model}")
            # TODO: maybe check model type_info is_list ...

        if not is_model_class(model) and not (is_list and model in STANDARD_TYPE_LIST):
            raise EntitySetupError(owner=self, msg=f"Managed model {bound_model.name} needs to be a @dataclass, pydantic.BaseModel or List[{STANDARD_TYPE_LIST}], got: {type(model)}")

        # == M.name version
        self.setup_session[ModelsNS].register_all_nodes(root_attr_node=attr_node, bound_model=bound_model, model=model)

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

    def setup(self) -> Self:
        # components are flat list, no recursion/hierarchy browsing needed
        if self.bound_model is None:
            raise EntitySetupError(owner=self, msg="bound_model not set. Initialize in constructor or call bind_to() first.")

        if not self.contains:
            raise EntitySetupError(owner=self, msg="'contains' attribute is required with list of components")

        if self.is_finished():
            raise EntitySetupError(owner=self, msg="setup() should be called only once")

        if self.setup_session is not None:
            raise EntitySetupError(owner=self, msg="SetupSession.setup() should be called only once")

        self.setup_session = create_setup_session(
                                container=self,
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
        #   if component is another container i.e. is_subentity_items() - it will
        #       process only that component and will not go deeper. later
        #       subentity_items.setup() will do this within own tree dep (own .components / .setup_session)

        # for containers
        # iterate all subcomponents and call _setup() for each
        with self.setup_session.use_stack_frame(
                SetupStackFrame(
                    container = self, 
                    component = self, 
                    local_setup_session = None,
                )):
            # setup this_registry objects must be inside of stack_frame due
            # premature component.bind setup in some ThisRegistryFor* classes.
            this_registry = self.try_create_this_registry(component=self, setup_session=self.setup_session)
            assert this_registry
            local_setup_session = self.setup_session.create_local_setup_session(this_registry)
            self.setup_session.current_frame.set_local_setup_session(local_setup_session)

            # setup me
            self._setup(setup_session=self.setup_session)

        # check all ok?
        for component_name, component in self.components.items():
            # TODO: maybe bound Field.bind -> Model attr_node?
            if not component.is_finished():
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
        return self.setup_session[ModelsNS].get_attr_node_by_bound_model(bound_model=self.bound_model)

    # ------------------------------------------------------------

    def get_component(self, name:str) -> ComponentBase:
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
                                instance: ModelType, 
                                index0: int, 
                                ) -> Union[Tuple[(str, Any)], int]:
        " index0 is 0 based index of item in the list"
        # TODO: move to ApplyResult:IApplySession?
        if self.keys:
            ret = self.get_key_pairs(instance)
        else:
            ret = index0
        return ret


    def get_key_pairs(self, instance: ModelType) -> Tuple[(str, Any)]:
        if not self.keys:
            raise EntityInternalError(msg="get_key_pairs() should be called only when 'keys' are defined")
        key_pairs = self.keys.get_key_pairs(instance, container=self)
        return key_pairs

    # ------------------------------------------------------------

    @staticmethod
    def create_this_registry_for_instance(
            model_class: ModelType, 
            owner: Optional[ComponentBase], 
            children: Optional[List[ComponentBase]], 
            setup_session: ISetupSession
            ) -> IThisRegistry:
        """ 
        - ThisRegistryForInstance it is unavailable to low-level modules -
          e.g. func_args -> setup.
        - .Instance + <attr-names> is used only in manual setup cases, 
          e.g. ChoiceField()
        """
        return ThisRegistryForInstance(model_class=model_class, owner=owner, children=children, setup_session=setup_session)


    def try_create_this_registry(self, component: ComponentBase, setup_session: ISetupSession) -> Optional[IThisRegistry]:

        children = component.get_children()

        if isinstance(component, IFieldBase):
            # ==== similar logic in apply.py :: _apply() ====
            # TODO: this is 2nd place to call '.Setup()'. Explain!
            assert not component.is_container()
            assert getattr(component, "bind", None)

            # Field -> This.Value == This.<all-attributes>
            if not component.bind.IsFinished():
                attr_node = component.bind.Setup(setup_session=setup_session, owner=self)
            else:
                attr_node = component.bind._dexp_node

            if not attr_node:
                raise EntitySetupNameError(owner=component, msg=f"{attr_node.name}.bind='{component.bind}' could not be evaluated")

            if children:
                # Field with children (e.g. BooleanField.enables)
                # Children -> This..Children + This.<all-attributes>
                # model_class = component.get_type_info().type_
                this_registry = ThisRegistryForValueAndChildren(attr_node=attr_node, owner=component, children=children, setup_session=setup_session)
            else:
                this_registry = ThisRegistryForValue(attr_node)


        elif children:
            if component.is_subentity_items():
                # Items -> This.Items 
                this_registry = ThisRegistryForItemsAndChildren(owner=component, children=children)
            else:
                # Children -> This.Children + This.<all-attributes>
                this_registry = ThisRegistryForChildren(owner=component, children=children, setup_session=setup_session)
        else:
            if component.is_container():
                raise EntityInternalError(owner=self, msg="Container should have local_session created") 
            # no This for you!
            this_registry = None

        return this_registry


# ------------------------------------------------------------


def create_setup_session(
        container: ContainerBase,
        config: Config,
        functions: Optional[List[CustomFunctionFactory]] = None, 
        context_class: Optional[IContext] = None,
        ) -> SetupSession:
    setup_session = SetupSession(
                    container=container, 
                    functions=functions,
                    parent_setup_session=container.setup_session if container.parent else None,
                    include_builtin_functions=container.is_top_parent())

    setup_session.add_registry(ModelsRegistry())
    setup_session.add_registry(FieldsRegistry())
    setup_session.add_registry(FunctionsRegistry())
    setup_session.add_registry(OperationsRegistry())
    setup_session.add_registry(ContextRegistry(context_class=context_class))
    setup_session.add_registry(ConfigRegistry(config=config))

    return setup_session


# ------------------------------------------------------------


class KeysBase(ABC):

    @abstractmethod
    def validate(self, model_or_children_model: ModelType):
        ...

    @abstractmethod
    def get_key_pairs(self, instance: ModelType, container: IContainerBase) -> List[Tuple[str, Any]]:
        """ returns list of (key-name, key-value) """
        ...

    # def get_keys(self, apply_session:IApplySession) -> List[Any]:
    #     return [key for name,key in self.get_keys_tuple(apply_session)]


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


    def get_key_pairs(self, instance: ModelType, container: IContainerBase) -> KeyPairs:
        # apply_session:IApplySession
        # frame = apply_session.current_frame
        # instance = frame.instance

        keys = []

        for field_name in self.field_names:
            if not hasattr(instance, field_name):
                raise EntityApplyNameNotFoundError(f"Field name '{field_name}' not found in list of attributes of '{type(instance)}'!?")
            key = getattr(instance, field_name)
            if key is None:
                # if missing then setup temp unique id
                missing_key_id = container._get_new_id_by_parent_name(GlobalConfig.ID_KEY_PREFIX_FOR_MISSING)
                key = MissingKey(id=missing_key_id)
            else:
                assert not str(key).startswith(GlobalConfig.ID_KEY_PREFIX_FOR_MISSING), key
            keys.append((field_name, key))
        return tuple(keys)


# ------------------------------------------------------------


@dataclass
class Entity(ContainerBase):

    contains        : List[ComponentBase] = field(repr=False)

    # --- optional - following can be bound later with .bind_to()
    name            : Optional[str] = field(default=None)
    title           : Optional[TransMessageType] = field(repr=False, default=None)

    # binding interface - not dumped/exported
    bound_model     : Optional[BoundModel] = field(repr=False, default=None, metadata={"skip_dump": True})
    # will be filled automatically with Config() if not supplied
    config          : Optional[Type[Config]] = field(repr=False, default=None, metadata={"skip_dump": True})
    context_class   : Optional[Type[IContext]] = field(repr=False, default=None, metadata={"skip_dump": True})
    functions       : Optional[List[CustomFunctionFactory]] = field(repr=False, default_factory=list, metadata={"skip_dump": True})

    # --- only list of model names allowed
    keys            : Optional[KeysBase] = field(repr=False, default=None)

    # --- validators and evaluators
    cleaners        : Optional[List[Union[ChildrenValidationBase, ChildrenEvaluationBase]]] = field(repr=False, default_factory=list)

    # --- Evaluated later
    setup_session      : Optional[SetupSession]    = field(init=False, repr=False, default=None)
    components      : Optional[Dict[str, ComponentBase]]  = field(init=False, repr=False, default=None)
    models          : Dict[str, Union[type, DotExpression]] = field(repr=False, init=False, default_factory=dict)
    # in Entity (top object) this case allway None - since it is top object
    parent           : Union[None, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    parent_name      : Union[str, UndefinedType]  = field(init=False, default=UNDEFINED)

    # used for automatic component's naming, <parent_name/class_name>__<counter>
    name_counter_by_parent_name: Dict[str, int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):

        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.register(self)
        if not self.config:
            # default setup
            self.config = Config()

        self.init_clean()
        super().__post_init__()

    def init_clean(self):
        if self.bound_model:
            if not (isinstance(self.bound_model, BoundModel) and is_model_class(self.bound_model.model)):
                raise EntitySetupTypeError(owner=self, msg=f"Attribute 'bound_model' needs to be BoundModel with model DC/PYD, got: {self.bound_model}") 

            if not self.name:
                self.name = "__".join([
                    camel_case_to_snake(self.__class__.__name__),
                    camel_case_to_snake(self.bound_model.model.__name__),
                    ])

        self._check_cleaners([ChildrenValidationBase, ChildrenEvaluationBase])

        if not isinstance(self.config, Config):
            raise EntitySetupValueError(owner=self, msg=f"config needs Config instance, got: {type(self.config)} / {self.config}")

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
                raise EntitySetupError(owner=self, msg="bound_model already already set, late binding not allowed.")
            self.bound_model = bound_model

        # if data:
        #     if self.data:
        #         raise EntitySetupError(owner=self, msg="data already set, late binding not allowed.")
        #     self.data = data

        if functions:
            if self.functions:
                raise EntitySetupError(owner=self, msg="functions already set, late binding not allowed.")
            self.functinos = functions

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
        container = self.get_first_parent_container(consider_self=True)

        apply_session = \
                ApplyResult(setup_session=container.setup_session, 
                      entity=self, 
                      component_name_only=component_name_only,
                      context=context, 
                      instance=instance,
                      instance_new=instance_new,
                      )\
                  .apply()

        if not apply_session.finished:
            raise EntityInternalError(owner=self, msg="Apply process is not finished")

        if raise_if_failed:
            apply_session.raise_if_failed()

        return apply_session

    # ------------------------------------------------------------

    def dump_defaults(self, 
              context: Optional[IContext] = None, 
              ) -> IApplySession:
        """
        In defaults mode:
            - context should be applied if Entity have (same as in apply())
            - validations are not called
        """
        from .apply import ApplyResult
        container = self.get_first_parent_container(consider_self=True)

        apply_session = \
                ApplyResult(
                    defaults_mode=True,
                    setup_session=container.setup_session, 
                    entity=self, 
                    component_name_only=None,
                    context=context, 
                    instance=NA_DEFAULTS_MODE,
                    instance_new=None,
                    )\
                  .apply()

        if not apply_session.finished:
            raise EntityInternalError(owner=self, msg="Apply process is not finished")

        if apply_session.errors:
            validation_error = EntityValidationError(owner=apply_session.entity, errors=apply_session.errors)
            raise EntityInternalError(owner=self, msg=f"Internal issue, apply process should not yield validation error(s), got: {validation_error}")

        output = apply_session._dump_defaults()

        return output


# ------------------------------------------------------------


@dataclass
class SubEntityBase(ContainerBase, ABC):
    """ can not be used individually - must be directly embedded into Other
        SubEntityItems or top Entity """

    # DotExpression based model -> can be dumped
    bound_model     : Union[BoundModel, BoundModelWithHandlers] = field(repr=False)
    # metadata={"bind_to_parent_setup_session" : True})

    # cardinality     : ICardinalityValidation
    contains        : List[ComponentBase] = field(repr=False)

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
    components       : Optional[Dict[str, ComponentBase]]  = field(init=False, repr=False, default=None)
    models           : Dict[str, Union[type, DotExpression]] = field(init=False, repr=False, default_factory=dict)

    # --- ComponentBase common attrs
    parent           : Union[ComponentBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    parent_name      : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

    # subentity_items specific - is this top parent or what? what is the difference to self.parent

    # in parents' chain (including self) -> first container
    parent_container : Union[ContainerBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)

    # in parents' chain (not including self) -> first container's setup_session
    parent_setup_session: Optional[SetupSession] = field(init=False, repr=False, default=None)

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
        if not (isinstance(self.bound_model, BoundModelBase) and isinstance(self.bound_model.model, DotExpression)):
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
        self.parent_setup_session = setup_session
        super().setup()
        # self.cardinality.validate_setup()
        return self

# ------------------------------------------------------------

@dataclass
class SubEntityItems(SubEntityBase):
    """ one to many relations - e.g. Person -> PersonAddresses """

    def __post_init__(self):
        self._check_cleaners((ItemsValidationBase, ChildrenValidationBase, ItemsEvaluationBase, ChildrenEvaluationBase))
        # for cleaner in self.cleaners:
        #     if not isinstance(cleaner, (ItemsValidationBase, ChildrenValidationBase, ItemsEvaluationBase, ChildrenEvaluationBase)):
        #         raise EntitySetupTypeError(owner=self, msg=f"Cleaners should be instances of ItemsValidationBase, ChildrenValidationBase, ItemsEvaluationBase or ChildrenEvaluationBase, got: {type(cleaner)} / {cleaner}") 
        super().__post_init__()

    def is_subentity_items(self):
        return True

# ------------------------------------------------------------

@dataclass
class SubEntitySingle(SubEntityBase):
    """ one to one relations - e.g. Person -> PersonAccess """

    cleaners        : Optional[List[Union[SingleValidation, ChildrenValidationBase, ChildrenEvaluationBase]]] = field(repr=False, default_factory=list)

    def __post_init__(self):
        self._check_cleaners((SingleValidation, ChildrenValidationBase, ChildrenEvaluationBase))
        # for cleaner in self.cleaners:
        #     if not isinstance(cleaner, (SingleValidation, ChildrenValidationBase, ChildrenEvaluationBase)):
        #         raise EntitySetupTypeError(owner=self, msg=f"Cleaners should be instances of SingleValidation, ChildrenValidationBase or ChildrenEvaluationBase, got: {type(cleaner)} / {cleaner}") 
        super().__post_init__()

    def is_subentity_single(self):
        return True

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
COMPONENTS_REGISTRY = {}
collect_classes(COMPONENTS_REGISTRY, fields, ComponentBase)
collect_classes(COMPONENTS_REGISTRY, valid_field, ComponentBase)
collect_classes(COMPONENTS_REGISTRY, valid_items, ComponentBase)
collect_classes(COMPONENTS_REGISTRY, valid_children, ComponentBase)
collect_classes(COMPONENTS_REGISTRY, eval_field, ComponentBase)
collect_classes(COMPONENTS_REGISTRY, eval_items, ComponentBase)
collect_classes(COMPONENTS_REGISTRY, eval_children, ComponentBase)
collect_classes(COMPONENTS_REGISTRY, None, ComponentBase)
