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
# https://peps.python.org/pep-0673/
# python 3.11+ from typing import Self

from dataclasses import dataclass, field

from .utils import (
        get_available_names_example,
        UNDEFINED,
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
        )
from .expressions import (
        ValueExpression,
        )
from .models import (
        BoundModel,
        BoundModelWithHandlers,
        )
from .attr_nodes import AttrVexpNode
from .functions import CustomFunctionFactory
from .registries import (
        Registries,
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

# ------------------------------------------------------------

def create_registries(
        owner: ContainerBase,
        config: Config,
        functions: Optional[List[CustomFunctionFactory]] = None, 
        context_class: Optional[IContext] = None,
        ) -> Registries:
    registries = Registries(
                    owner=owner, 
                    functions=functions,
                    owner_registries=owner.registries if owner.owner else None,
                    include_builtin_functions=owner.is_top_owner())
    registries.add_registry(ModelsRegistry())
    registries.add_registry(FieldsRegistry())
    registries.add_registry(FunctionsRegistry())
    registries.add_registry(OperationsRegistry())
    registries.add_registry(ContextRegistry(context_class=context_class))
    registries.add_registry(ConfigRegistry(config=config))

    return registries

# ------------------------------------------------------------
# Rules
# ------------------------------------------------------------

class ContainerBase(IContainerBase, ComponentBase, ABC):

    def _get_function(self, name: str, strict:bool=True):
        if not self.functions:
            raise KeyError(f"{self.name}: Function '{name}' not found, no functions available.")
        return self.registries.functions_factory_registry.get(name, strict=strict)

    def add_fieldgroup(self, fieldgroup:FieldGroup):
        if self.is_finished():
            raise RuleSetupError(owner=self, msg="FieldGroup can not be added after setup() is called.")
        found = [sec for sec in self.contains if sec.name==fieldgroup.name]
        if found:
            raise RuleSetupError(owner=self, msg=f"FieldGroup {fieldgroup.name} is already added.")
        self.contains.append(fieldgroup)

    def is_top_owner(self):
        return not bool(self.owner)

    def is_extension(self):
        # TODO: if self.owner is not None could be used as the term, put validation somewhere
        " if start model is value expression - that mean that the the Rules is Extension "
        return isinstance(self.bound_model.model, ValueExpression)

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
        # ex. type_info.metadata.get("bind_to_owner_registries")
        is_main_model = (bound_model==self.bound_model)
        is_extension_main_model = (self.is_extension() and is_main_model)

        # is_list = False
        if not isinstance(bound_model, BoundModelBase):
            raise RuleSetupError(owner=self, msg=f"{bound_model.name}: Needs to be Boundbound_model* instance, got: {bound_model}")

        model = bound_model.model

        attr_node = None
        if is_extension_main_model:
            if not isinstance(model, ValueExpression):
                raise RuleSetupError(owner=self, msg=f"{bound_model.name}: For Extension main bound_model needs to be ValueExpression: {bound_model.model}")

        # alias_saved = False
        is_list = False


        if isinstance(model, ValueExpression):
            # TODO: for functions value expressions need to be stored
            #       with all parameters (func_args)
            if model.GetNamespace()!=ModelsNS:
                raise RuleSetupError(owner=self, msg=f"{bound_model.name}: ValueExpression should be in ModelsNS namespace, got: {model.GetNamespace()}")

            if is_extension_main_model:
                # TODO: DRY this - the only difference is registries - extract common logic outside / 
                # bound attr_node
                assert hasattr(self, "owner_registries")
                registries_from = self.owner_registries
            else:
                # Rules - top owner container / normal case
                registries_from = self.registries

            attr_node = registries_from.get_vexp_node_by_vexp(vexp=model)
            if attr_node:
                raise RuleInternalError(owner=self, msg=f"AttrVexpNode data already in registries: {model} -> {attr_node}")

            attr_node = model.Setup(registries=registries_from, owner=bound_model)
            if not attr_node:
                raise RuleInternalError(owner=self, msg=f"AttrVexpNode not recognized: {model}")

            if not isinstance(attr_node.data, TypeInfo):
                raise RuleInternalError(owner=self, msg=f"AttrVexpNode data is not TypeInfo, got: {type(attr_node.data)} / {attr_node.data}")

            model   = attr_node.data.type_
            is_list = attr_node.data.is_list

        if not is_model_class(model) and not (is_list and model in STANDARD_TYPE_LIST):
            raise RuleSetupError(owner=self, msg=f"Managed model {bound_model.name} needs to be a @dataclass, pydantic.BaseModel or List[{STANDARD_TYPE_LIST}], got: {type(model)}")


        # == M.name version
        self.registries[ModelsNS].register_all_nodes(root_attr_node=attr_node, bound_model=bound_model, model=model)

        # if not isinstance(model, ValueExpression) and isinstance(bound_model, BoundModel):
        #     bound_model._register_nested_models(self.registries)


        # == M.company.name version
        # if not attr_node:
        #     attr_node = self.registries[ModelsNS].create_root_attr_node(bound_model=bound_model) -> _create_root_attr_node()
        # # self.registries.register(attr_node, alt_attr_node_name=bound_model.name if attr_node.name!=bound_model.name else None)
        # self.registries[ModelsNS].register_attr_node(
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
    #         self.registries[DataNS].register(data_var)

    # ------------------------------------------------------------

    def _register_fields_components_attr_nodes(self):
        """
        Traverse the whole tree (recursion) and collect all components into
        simple flat list. It will set owner for each child component.
        """
        self.components = self.fill_components()

        # A.3. COMPONENTS - collect attr_nodes - previously flattened (recursive function fill_components)
        for component_name, component in self.components.items():
            self.registries[FieldsNS].register(component)

    # ------------------------------------------------------------

    def setup(self) -> ContainerBase:
        # components are flat list, no recursion/hierarchy browsing needed
        if self.bound_model is None:
            raise RuleSetupError(owner=self, msg="bound_model not set. Initialize in constructor or call bind_to() first.")

        if not self.contains:
            raise RuleSetupError(owner=self, msg="'contains' attribute is required with list of components")

        if self.is_finished():
            raise RuleSetupError(owner=self, msg="setup() should be called only once")

        if self.registries is not None:
            raise RuleSetupError(owner=self, msg="Registries.setup() should be called only once")

        self.registries = create_registries(
                                owner=self,
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
        #   if component attribute is ValueExpression -> will call vexp.Setup()
        #   if component is another container i.e. is_extension() - it will
        #       process only that component and will not go deeper. later
        #       extension.setup() will do this within own tree dep (own .components / .registries)

        # iterate all subcomponents and call _setup() for each
        self._setup(registries=self.registries)

        # check all ok?
        for component_name, component in self.components.items():
            # TODO: maybe bound Field.bind -> Model attr_node?
            if not component.is_finished():
                raise RuleInternalError(owner=self, msg=f"{component} not finished. Is in overriden setup()/Setup() parent method super().setup()/Setup() been called (which sets parent and marks finished)?")

        self.registries.finish()

        if self.keys:
            # Inner BoundModel can have self.bound_model.model = ValueExpression
            self.keys.validate(self.bound_model.get_type_info().type_)

        if self.is_top_owner():
            self.registries.call_hooks_on_finished_all()

        return self

    # ------------------------------------------------------------

    def get_bound_model_attr_node(self) -> AttrVexpNode:
        return self.registries[ModelsNS].get_attr_node_by_bound_model(bound_model=self.bound_model)

    # ------------------------------------------------------------

    def get_component(self, name:str) -> ComponentBase:
        # TODO: currently components are retrieved only from contains - but should include validations + cardinality
        if name not in self.components:
            vars_avail = get_available_names_example(name, self.components.keys())
            raise RuleNameNotFoundError(owner=self, msg=f"Component '{name}' not found, some valid_are: {vars_avail}")
        return self.components[name]

    # ------------------------------------------------------------

    def pp(self):
        if not hasattr(self, "components"):
            raise RuleSetupError(owner=self, msg="Call .setup() first")
        print(f"{self.name}: {self.__class__.__name__} ::")
        for nr, (name, component) in enumerate(self.components.items(),1):
            print(f"  {nr:02}. {name}: {component.__class__.__name__} ::")
            print(f"      {repr(component)[:100]}") # noqa: T001
            # TODO: if pp() exists -> recursion with depth+1 (indent)

    # ------------------------------------------------------------
    # Apply phase
    # ------------------------------------------------------------

    def get_key_string_by_instance(self, apply_session:IApplySession, instance: ModelType, index0: Optional[int]) -> str:
        # NOTE: it is good enough key to have current name only without
        #       owner_container.get_key_string() attached
        instance_id = id(instance)
        key_string = apply_session.key_string_container_cache.get(instance_id, None)
        if key_string is None:
            if self.keys:
                key_pairs = self.get_key_pairs(instance)
                assert key_pairs
                key_string = "{}[{}]".format(
                                self.name, 
                                GlobalConfig.ID_NAME_SEPARATOR.join(
                                    [f"{name}={value}" for name, value in key_pairs]
                                ))
            elif index0 is not None:
                key_string = f"{self.name}[{index0}]"
            else:
                key_string = self.name
            apply_session.key_string_container_cache[instance_id] = key_string
            apply_session.instance_by_key_string_cache[key_string] = instance
        #     from_cache = "new"
        # else:
        #     from_cache = "cache"

        # TODO: apply_session.config.logger.debug("cont:", self.name, key_string, f"[{from_cache}]")

        return key_string


    def get_key_string(self, apply_session:IApplySession) -> str:
        " uses cache, when not found then gets intances and index0 from current frame "
        return self.get_key_string_by_instance(
                apply_session = apply_session,
                instance = apply_session.current_frame.instance, 
                index0 = apply_session.current_frame.index0)


    def get_key_pairs_or_index0(self, instance: ModelType, index0: int) -> Union[List[(str, Any)], int]:
        " index0 is 0 based index of item in the list"
        return self.get_key_pairs(instance) if self.keys else index0


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
    bound_model     : Optional[BoundModel] = field(repr=False, default=None)
    # will be filled automatically with Config() if not supplied
    config          : Optional[Type[Config]] = field(repr=False, default=None)
    context_class   : Optional[Type[IContext]] = field(repr=False, default=None)
    functions       : Optional[List[CustomFunctionFactory]] = field(repr=False, default_factory=list)
    # --- only list of model names allowed
    keys            : Optional[KeysBase] = field(repr=False, default=None)
    # --- validators and evaluators
    cleaners        : Optional[List[Union[ValidationBase, EvaluationBase]]] = field(repr=False, default_factory=list)

    # --- Evaluated later
    registries      : Optional[Registries]    = field(init=False, repr=False, default=None)
    components      : Optional[Dict[str, Component]]  = field(repr=False, default=None)
    models          : Dict[str, Union[type, ValueExpression]] = field(repr=False, init=False, default_factory=dict)
    # in Rules (top object) this case allway None - since it is top object
    owner           : Union[None, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    owner_name      : Union[str, UndefinedType]  = field(init=False, default=UNDEFINED)

    def __post_init__(self):
        # TODO: check that BoundModel.model is_model_class() and not ValueExpression

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
    #     self.registries.call_hooks_on_finished_all()
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
        proxy to Result().apply()
        TODO: check that this is container / extension / fieldgroup
        """
        from .apply import ApplyResult
        container = self.get_container_owner(include_self=True)

        apply_result = \
                ApplyResult(registries=container.registries, 
                      rules=self, 
                      component_name_only=component_name_only,
                      context=context, 
                      instance=instance,
                      instance_new=instance_new,
                      )\
                  .apply()

        if raise_if_failed:
            apply_result.raise_if_failed()

        return apply_result


# ------------------------------------------------------------

@dataclass
class Extension(ContainerBase):
    """ can not be used individually - must be directly embedded into Other
        Extension or top Rules """

    # required since if it inherit name from BoundModel then the name will not
    # be unique in self.components (Extension and BoundModel will share the same name)
    name            : str
    bound_model     : Union[BoundModel, BoundModelWithHandlers] = field(repr=False)
    # metadata={"bind_to_owner_registries" : True})

    cardinality     : ICardinalityValidation
    contains        : List[Component] = field(repr=False)

    label           : Optional[TransMessageType] = field(repr=False, default=None)
    functions       : Optional[List[CustomFunctionFactory]] = field(repr=False, default_factory=list)
    # --- can be index based or standard key-fields names
    keys            : Optional[KeysBase] = field(repr=False, default=None)
    # --- validators and evaluators
    cleaners        : Optional[List[Union[ValidationBase, EvaluationBase]]] = field(repr=False, default_factory=list)

    # --- Evaluated later
    registries      : Optional[Registries] = field(init=False, repr=False, default=None)
    components      : Optional[Dict[str, Component]]  = field(repr=False, default=None)
    models          : Dict[str, Union[type, ValueExpression]] = field(repr=False, init=False, default_factory=dict)
    owner           : Union[ComponentBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    owner_name      : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

    # extension specific - is this top owner or what? what is the difference to self.owner

    # in owners' chain (including self) -> first container
    owner_container : Union[ContainerBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)

    # in owners' chain (not including self) -> first container's registries
    owner_registries: Optional[Registries] = field(init=False, repr=False, default=None)

    # copy from first non-self container owner
    context_class   : Optional[Type[IContext]] = field(repr=False, init=False, default=None)
    config          : Optional[Type[Config]] = field(repr=False, init=False, default=None)

    # bound_attr_node  : Union[AttrVexpNode, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)

    # Class attributes
    # namespace_only  : ClassVar[Namespace] = ThisNS
    def __post_init__(self):
        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.register(self)
        if not self.label:
            self.label = varname_to_title(self.name)
        super().__post_init__()

    def set_owner(self, owner:ContainerBase):
        super().set_owner(owner=owner)

        # can be self
        self.owner_container     = self.get_container_owner(include_self=True)

        # take from real first container owner
        non_self_owner_container = self.get_container_owner(include_self=False)
        self.context_class = non_self_owner_container.context_class
        self.config = non_self_owner_container.config
        if not self.config:
            raise RuleInternalError(owner=self, msg=f"Config not set from owner: {self.owner_container}") 

    def setup(self, registries:Registries):
        # NOTE: registries is not used, can be reached with owner.registries(). left param
        #       for same function signature as for components.
        self.owner_registries = registries
        super().setup()
        self.cardinality.validate_setup()
        return self


