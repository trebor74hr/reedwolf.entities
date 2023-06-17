import inspect

from typing import (
        Any,
        List,
        Union,
        Optional,
        ClassVar,
        Dict,
        Type,
        Tuple,
        )
from .utils import (
        UNDEFINED,
        UndefinedType,
        )
from .exceptions import (
        RuleSetupError,
        RuleSetupNameError,
        RuleSetupValueError,
        RuleInternalError,
        RuleApplyTypeError,
        RuleApplyNameError,
        )
from .namespaces import (
        ModelsNS,
        FieldsNS,
        ThisNS,
        FunctionsNS,
        ContextNS,
        ConfigNS,
        OperationsNS,
        )
from .expressions import (
        DotExpression,
        IDotExpressionNode,
        )
from .meta import (
        Self,
        is_model_class,
        ModelType,
        get_model_fields,
        TypeInfo,
        AttrName,
        )
from .base import (
        ReservedAttributeNames,
        ComponentBase,
        IFieldBase, 
        IContainerBase,
        IApplySession,
        BoundModelBase,
        IFieldGroup,
        )
from .attr_nodes import (
        AttrDexpNode,
        )
from .valid_base import (
        ValidationBase,
        )
from .eval_base import (
        EvaluationBase,
        )
from .contexts import (
        IContext,
        )
from .config import (
        Config,
        )
from .setup import (
        RegistryBase,
        RegistryUseDenied,
        ComponentAttributeAccessor,
        SetupSessionBase,
        )

# ------------------------------------------------------------

class FunctionsRegistry(RegistryUseDenied):
    NAMESPACE = FunctionsNS

class OperationsRegistry(RegistryUseDenied):
    NAMESPACE = OperationsNS

# ------------------------------------------------------------


class ModelsRegistry(RegistryBase):
    """
    All models have full path from top container (for better readibility)
    e.g. M.address_set.city.street so reading begins from root
    instance, instead of apply_session.current_frame.instance.
    See get_root_value() implementation.
    """
    NAMESPACE = ModelsNS

    # == M.company.name case
    # # Caller should not fetch attribute by name from return value of
    # # get_root_value()
    # ROOT_VALUE_NEEDS_FETCH_BY_NAME: ClassVar[bool] = False

    def __init__(self):
        super().__init__()
        self.root_attr_nodes : Optional[Dict[str, AttrDexpNode]] = {}

    # NOTE: no register() method due complex logic - see
    #       ContainerBase._register_bound_model()

    def _create_root_attr_node(self, bound_model:BoundModelBase) -> AttrDexpNode:
        " models specific method "
        # standard DTO class attr_node
        # if not bound_model.type_info:
        #     bound_model.set_type_info()
        # assert bound_model.type_info.type_==model
        attr_node = AttrDexpNode(
                        name=bound_model.name,
                        data=bound_model,
                        namespace=self.NAMESPACE,
                        type_info=bound_model.get_type_info())
        return attr_node


    # ------------------------------------------------------------

    def register_all_nodes(self, root_attr_node: Optional[AttrDexpNode],  bound_model: BoundModelBase, model: ModelType):
        " models specific method "
        if not root_attr_node:
            root_attr_node = self._create_root_attr_node(bound_model=bound_model)

        name = bound_model.get_full_name()
        is_root = "." not in name # TODO: hack
        if name in self.root_attr_nodes:
            raise RuleInternalError(owner=self, msg=f"Duplicate {name} -> {self.root_attr_nodes[name]}, already set, failed to set: {root_attr_node}")

        self.root_attr_nodes[name] = root_attr_node

        # company.business_type.name --> company__business_types__name
        name_for_reg = name.replace(".", "__")

        for attr_name in get_model_fields(model):
            attr_node = self._create_attr_node_for_model_attr(model, attr_name)
            alt_attr_node_name = None if is_root else f"{name_for_reg}__{attr_name}"
            self.register_attr_node(attr_node, alt_attr_node_name=alt_attr_node_name)

        # M.Instance
        self.register_instance_attr_node(model_class=model,
                attr_name_prefix = None if is_root else f"{name_for_reg}__",
                )


    # ------------------------------------------------------------

    def get_attr_node_by_bound_model(self,
                               bound_model:BoundModelBase,
                               # default:[None, UndefinedType]=UNDEFINED,
                               # strict:bool=False
                               ) -> Union[AttrDexpNode, None, UndefinedType]:
        " models specific method "
        # attr_node_name = bound_model.name
        # == M.name mode
        name = bound_model.get_full_name() 

        # company.business_type.name --> company__business_types__name
        name_for_reg = name.replace(".", "__")

        if name_for_reg not in self.root_attr_nodes:
            raise RuleInternalError(owner=self, msg=f"Name not found {name_for_reg} in {self.root_attr_nodes.keys()}")
        return self.root_attr_nodes[name_for_reg] 

        # == M.company mode
        # allways in models
        # attr_node_name = bound_model.name
        # assert attr_node_name
        # # return self.store.get(attr_node_name, default)
        # return self.store[attr_node_name]

    # ------------------------------------------------------------

    def get_root_value(self, apply_session: IApplySession, attr_name: AttrName) -> Tuple[Any, Optional[AttrName]]:
        # ROOT_VALUE_NEEDS_FETCH_BY_NAME = False
        # component = apply_session.current_frame.component
        instance = apply_session.current_frame.instance

        # bound_model = apply_session.current_frame.container.bound_model
        bound_model_root = apply_session.current_frame.bound_model_root

        expected_type = bound_model_root.type_info.type_ \
                        if isinstance(bound_model_root.model, DotExpression) \
                        else bound_model_root.model

        if instance is None:
            if not bound_model_root.type_info.is_optional:
                raise RuleInternalError(owner=bound_model_root, msg="Got None and type is not 'Optional'")
        else:
            if bound_model_root.type_info.is_list and isinstance(instance, (list, tuple)):
                # raise RuleApplyTypeError(owner=self, msg=f"Wrong type, expected list/tuple, got '{instance}'")
                # check only first
                instance_to_test = instance[0] if instance else None
            else:
                instance_to_test = instance

            # == M.name case
            if instance_to_test and not isinstance(instance_to_test, expected_type):
                raise RuleApplyTypeError(owner=self, msg=f"Wrong type, expected '{expected_type}', got '{instance}'")

        return instance, None


# ------------------------------------------------------------


class FieldsRegistry(RegistryBase):

    NAMESPACE = FieldsNS

    ALLOWED_BASE_TYPES: ClassVar[List[type]] = (IFieldBase, )

    DENIED_BASE_TYPES: ClassVar[List[type]] = (BoundModelBase, ValidationBase, EvaluationBase, IFieldGroup, IContainerBase, )

    def create_attr_node(self, component:ComponentBase):
        # TODO: put class in container and remove these local imports
        # ------------------------------------------------------------
        # A.3. COMPONENTS - collect attr_nodes - previously flattened (recursive function fill_components)
        # ------------------------------------------------------------
        if not isinstance(component, (ComponentBase, )):
            raise RuleSetupError(owner=self, msg=f"Register expexted ComponentBase, got {component} / {type(component)}.")

        component_name = component.name

        # TODO: to have standard types in some global list in fields.py
        #           containers, validations, evaluations, 
        if isinstance(component, (IFieldBase, )):
            denied = False
            deny_reason = ""
            type_info = component.type_info if component.type_info else None
        # TODO: to have standard types in some global list in fields.py
        #           containers, validations, evaluations,
        elif isinstance(component, self.DENIED_BASE_TYPES): # 
            # stored - but should not be used
            denied = True
            deny_reason = f"Component of type {component.__class__.__name__} can not be referenced in DotExpressions"
            if hasattr(component, "type_info"):
                type_info=component.type_info
            else:
                type_info=None
        else:
            # TODO: this should be automatic, a new registry for field types
            valid_types = ', '.join([t.__name__ for t in self.ALLOWED_BASE_TYPES])
            raise RuleSetupError(owner=self, msg=f"RuleSetup does not support type {type(component)}: {repr(component)[:100]}. Valid type of objects or objects inherited from: {valid_types}")

        attr_node = AttrDexpNode(
                        name=component_name,
                        data=component,
                        namespace=FieldsNS,
                        type_info=type_info,
                        denied=denied,
                        deny_reason=deny_reason)
        return attr_node


    def register(self, component:ComponentBase):
        attr_node = self.create_attr_node(component)
        self.register_attr_node(attr_node) # , is_list=False))
        return attr_node


    def get_root_value(self, apply_session: IApplySession, attr_name: AttrName) -> Tuple[Any, Optional[AttrName]]:
        # container = apply_session.current_frame.component.get_first_parent_container(consider_self=True)
        component = apply_session.current_frame.component
        instance  = apply_session.current_frame.instance
        top_attr_accessor = ComponentAttributeAccessor(component, instance)
        return top_attr_accessor, None


# ------------------------------------------------------------


class ContextRegistry(RegistryBase):

    NAMESPACE = ContextNS

    def __init__(self, 
                 context_class: Optional[Type[IContext]], 
                 ):
        super().__init__() # functions_factory_registry=functions_factory_registry)

        self.context_class = context_class
        if self.context_class:
            self.register_all_nodes()


    def create_node(self, 
                    dexp_node_name: str, 
                    owner_dexp_node: IDotExpressionNode, 
                    owner: ComponentBase,
                    ) -> IDotExpressionNode:

        if not isinstance(owner, ComponentBase):
            raise RuleInternalError(owner=self, msg=f"Owner needs to be Component, got: {type(owner)} / {owner}")  

        if not owner.get_first_parent_container(consider_self=True).context_class:
            raise RuleSetupNameError(owner=owner, msg=f"Namespace '{self.NAMESPACE}' (referenced by '{self.NAMESPACE}.{dexp_node_name}') should not be used since 'Entity.context_class' is not set. Define 'context_class' to 'Entity()' constructor and try again.")

        return super().create_node(
                dexp_node_name=dexp_node_name,
                owner_dexp_node=owner_dexp_node,
                owner=owner,
                )


    def register_all_nodes(self):
        if IContext not in inspect.getmro(self.context_class):
            raise RuleSetupValueError(owner=self, msg=f"Context should inherit IContext, got: {self.context_class}")

        for attr_name in get_model_fields(self.context_class):
            attr_node = self._create_attr_node_for_model_attr(self.context_class, attr_name)
            self.register_attr_node(attr_node)

        # map User, Session, Now and similar Attribute -> function calls
        for attr_name, py_function in self.context_class.get_dexp_attrname_dict().items():
            type_info = TypeInfo.extract_function_return_type_info(
                            py_function,
                            allow_nonetype=True)
            if attr_name in self.store:
                raise RuleSetupNameError(f"Attribute name '{attr_name}' is reserved. Rename class attribute in '{self.context_class}'")
            attr_node = AttrDexpNode(
                            name=attr_name,
                            data=type_info,
                            namespace=self.NAMESPACE,
                            type_info=type_info, 
                            # TODO: the type or name of th_field is not ok
                            th_field=py_function, 
                            )
            self.register_attr_node(attr_node, attr_name)


    def get_root_value(self, apply_session: IApplySession, attr_name: AttrName) -> Tuple[Any, Optional[AttrName]]:
        context = apply_session.context
        if context in (UNDEFINED, None):
            component = apply_session.current_frame.component
            raise RuleApplyNameError(owner=self, msg=f"ContextNS attribute '{component.name}' can not be fetched since context is not set ({type(context)}).")
        # if attr_name in self.context_class.get_dexp_attrname_dict():
        return context, None


# ------------------------------------------------------------


class ConfigRegistry(RegistryBase):

    NAMESPACE = ConfigNS

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        if not self.config:
            raise RuleInternalError(owner=self, msg="Config is required")

        self.register_all_nodes()

    def register_all_nodes(self):
        if not isinstance(self.config, Config):
            raise RuleInternalError(owner=self, msg=f".config is not Config instance, got: {type(self.config)} / {self.config}")

        config_class = self.config.__class__
        for attr_name in get_model_fields(config_class):
            attr_node = self._create_attr_node_for_model_attr(config_class, attr_name)
            self.register_attr_node(attr_node)

    def get_root_value(self, apply_session: IApplySession, attr_name: AttrName) -> Tuple[Any, Optional[AttrName]]:
        # ALT: config = apply_session.entity.config
        config = self.config
        if config in (UNDEFINED, None):
            component = apply_session.current_frame.component
            raise RuleInternalError(owner=self, 
                msg=f"ConfigNS attribute '{component.name}' can not be fetched since config is not set ({type(config)}).")
        return config, None


# ------------------------------------------------------------


class ThisInstanceRegistry(RegistryBase):
    """ 
    Applies to model instances, e.g. 
        company <= Company(name="Cisco", city="London")

        This.Instance <= company
        This.name <= "Cisco"
        This.city <= "London"

    Uses ReservedAttributeNames.INSTANCE_ATTR_NAME == "Instance" 
    """

    NAMESPACE = ThisNS

    def __init__(self, model_class: ModelType):
        super().__init__() # functions_factory_registry=functions_factory_registry)
        self.model_class = model_class
        if not is_model_class(self.model_class):
            raise RuleSetupValueError(owner=self, msg=f"Expected model class (DC/PYD), got: {type(self.model_class)} / {self.model_class} ")

        for attr_name in get_model_fields(self.model_class):
            # th_field: ModelField in .values()
            attr_node = self._create_attr_node_for_model_attr(self.model_class, attr_name)
            self.register_attr_node(attr_node)

        # This.Instance == ReservedAttributeNames.INSTANCE_ATTR_NAME 
        self.register_instance_attr_node(model_class=model_class)


    def get_root_value(self, apply_session: IApplySession, attr_name: AttrName) -> Tuple[Any, Optional[AttrName]]:
        if not isinstance(apply_session.current_frame.instance, self.model_class):
            raise RuleInternalError(owner=self, msg=f"Type of apply session's instance expected to be '{self.model_class}, got: {apply_session.current_frame.instance}") 
        return apply_session.current_frame.instance, None


# ------------------------------------------------------------

class ThisValueRegistry(RegistryBase):
    """ 
    Applies to model instance's attribute, e.g. 
        company <= Company(name="Cisco", city="London")
        company_name = company.name

        This.Value <= company_name

    Uses ReservedAttributeNames.VALUE_ATTR_NAME == "Value" 
    """

    NAMESPACE = ThisNS

    def __init__(self, attr_node: AttrDexpNode):
        # TODO: model_class: ModelType
        super().__init__()

        self.attr_node = attr_node
        if not isinstance(self.attr_node, AttrDexpNode):
            raise RuleSetupValueError(owner=self, msg=f"Expected AttrDexpNode, got: {type(self.attr_node)} / {self.attr_node}")
        self.attr_name = self.attr_node.name

        # TODO: consider adding this too? how to fetch Parent?
        #   self.model_class = model_class
        #   if not is_model_class(self.model_class):
        #       raise RuleSetupValueError(owner=self, msg=f"Expected model class (DC/PYD), got: {type(self.model_class)} / {self.model_class}")
        #
        #   model_fields = get_model_fields(self.model_class)
        #   if self.attr_name not in model_fields:
        #       raise RuleSetupValueError(owner=self, msg=f"Attribute '{attr_name}' not found in {type(self.model_class)} / {self.model_class}")

        # This.Value == ReservedAttributeNames.VALUE_ATTR_NAME.value
        self.register_value_attr_node(attr_node=attr_node)


    def get_root_value(self, apply_session: IApplySession, attr_name: AttrName) -> Tuple[Any, Optional[AttrName]]:
        # TODO: 
        # if not isinstance(apply_session.current_frame.instance, self.model_class):
        #     raise RuleInternalError(owner=self, msg=f"Type of apply session's instance expected to be '{self.model_class}, got: {apply_session.current_frame.instance}") 

        if attr_name != ReservedAttributeNames.VALUE_ATTR_NAME.value:
            raise RuleApplyNameError(owner=self, msg=f"Only '.Value' attribute is expected, got: {attr_name}") 

        # instead of fetching .Value, instruct caller to fetch component's bound attribute
        return apply_session.current_frame.instance, self.attr_name

# ------------------------------------------------------------


class SetupSession(SetupSessionBase):

    def create_local_setup_session(self, 
            this_ns_instance_model_class: Optional[ModelType],
            this_ns_value_attr_node: Optional[AttrDexpNode] = None,
            ) -> Self:
        """ Currently creates only local ThisNS registry, which is used for
        some local context, e.g. Component This. dexps 
        Args:

            a) this_ns_instance_model_class -> .Instance + all-attributes logic

            b) this_ns_value_attr_node -> .Value logic, no other attributes for now

        TODO:??
            this_ns_value_attr_name:
                when None -> .Instance + all-attributes logic
                when not None -> .Value logic, no other attributes for now

        """
        # if not this_ns_instance_model_class:
        #     raise RuleInternalError(owner=self, msg="Expected this_ns_instance_model_class.") 

        if this_ns_instance_model_class:
            assert this_ns_value_attr_node is None
            this_registry = ThisInstanceRegistry(
                    model_class=this_ns_instance_model_class,
                    )
        elif this_ns_value_attr_node:
            this_registry = ThisValueRegistry(
                    # model_class=this_ns_instance_model_class,
                    attr_node=this_ns_value_attr_node,
                    )
        else:
            raise RuleInternalError(owner=self, msg="Expected this_ns_instance_model_class or this_ns_value_attr_node") 

        local_setup_session = SetupSession(
                                container=self.container,
                                parent_setup_session=None,
                                functions_factory_registry=self.functions_factory_registry,
                                )

        local_setup_session.add_registry(this_registry)

        return local_setup_session

# ------------------------------------------------------------
# OBSOLETE
# ------------------------------------------------------------
