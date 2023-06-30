import inspect
from dataclasses import (
        dataclass, 
        field,
        )
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
        to_repr,
        )
from .exceptions import (
        EntitySetupError,
        EntitySetupNameError,
        EntitySetupValueError,
        EntityInternalError,
        EntityApplyTypeError,
        EntityApplyNameError,
        )
from .namespaces import (
        Namespace,
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

@dataclass
class FunctionsRegistry(RegistryUseDenied):
    NAMESPACE: ClassVar[Namespace] = FunctionsNS

@dataclass
class OperationsRegistry(RegistryUseDenied):
    NAMESPACE: ClassVar[Namespace] = OperationsNS

# ------------------------------------------------------------


@dataclass
class ModelsRegistry(RegistryBase):
    """
    All models have full path from top container (for better readibility)
    e.g. M.address_set.city.street so reading begins from root
    instance, instead of apply_session.current_frame.instance.
    See get_root_value() implementation.
    """
    # == M.company.name case
    # # Caller should not fetch attribute by name from return value of
    # # get_root_value()
    # ROOT_VALUE_NEEDS_FETCH_BY_NAME: ClassVar[bool] = False

    root_attr_nodes : Optional[Dict[str, AttrDexpNode]] = field(repr=False, init=False, default_factory=dict)

    NAMESPACE: ClassVar[Namespace] = ModelsNS

    # def __post_init__(self):
    #     super().__post_init__()

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
            raise EntityInternalError(owner=self, msg=f"Duplicate {name} -> {self.root_attr_nodes[name]}, already set, failed to set: {root_attr_node}")

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
            raise EntityInternalError(owner=self, msg=f"Name not found {name_for_reg} in {self.root_attr_nodes.keys()}")
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
                raise EntityInternalError(owner=bound_model_root, msg="Got None and type is not 'Optional'")
        else:
            if bound_model_root.type_info.is_list and isinstance(instance, (list, tuple)):
                # raise EntityApplyTypeError(owner=self, msg=f"Wrong type, expected list/tuple, got '{instance}'")
                # check only first
                instance_to_test = instance[0] if instance else None
            else:
                instance_to_test = instance

            # == M.name case
            if instance_to_test and not isinstance(instance_to_test, expected_type):
                raise EntityApplyTypeError(owner=self, msg=f"Wrong type, expected '{expected_type}', got '{instance}'")

        return instance, None


# ------------------------------------------------------------


@dataclass
class FieldsRegistry(RegistryBase):

    NAMESPACE: ClassVar[Namespace] = FieldsNS

    ALLOWED_BASE_TYPES: ClassVar[List[type]] = (IFieldBase, )

    DENIED_BASE_TYPES: ClassVar[List[type]] = (BoundModelBase, ValidationBase, EvaluationBase, IFieldGroup, IContainerBase, )

    def create_attr_node(self, component:ComponentBase):
        # TODO: put class in container and remove these local imports
        # ------------------------------------------------------------
        # A.3. COMPONENTS - collect attr_nodes - previously flattened (recursive function fill_components)
        # ------------------------------------------------------------
        if not isinstance(component, (ComponentBase, )):
            raise EntitySetupError(owner=self, msg=f"Register expexted ComponentBase, got {component} / {type(component)}.")

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
            raise EntitySetupError(owner=self, msg=f"Valid type of objects or objects inherited from: {valid_types}. Got: {type(component)} / {to_repr(component)}. ")

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


@dataclass
class ContextRegistry(RegistryBase):

    context_class: Optional[IContext] = field(repr=False)

    NAMESPACE: ClassVar[Namespace] = ContextNS

    def __post_init__(self):
        super().__post_init__() 
        if self.context_class:
            self.register_all_nodes()


    def create_node(self, 
                    dexp_node_name: str, 
                    owner_dexp_node: IDotExpressionNode, 
                    owner: ComponentBase,
                    ) -> IDotExpressionNode:

        if not isinstance(owner, ComponentBase):
            raise EntityInternalError(owner=self, msg=f"Owner needs to be Component, got: {type(owner)} / {owner}")  

        if not owner.get_first_parent_container(consider_self=True).context_class:
            raise EntitySetupNameError(owner=owner, msg=f"Namespace '{self.NAMESPACE}' (referenced by '{self.NAMESPACE}.{dexp_node_name}') should not be used since 'Entity.context_class' is not set. Define 'context_class' to 'Entity()' constructor and try again.")

        return super().create_node(
                dexp_node_name=dexp_node_name,
                owner_dexp_node=owner_dexp_node,
                owner=owner,
                )


    def register_all_nodes(self):
        if IContext not in inspect.getmro(self.context_class):
            raise EntitySetupValueError(owner=self, msg=f"Context should inherit IContext, got: {self.context_class}")

        for attr_name in get_model_fields(self.context_class):
            attr_node = self._create_attr_node_for_model_attr(self.context_class, attr_name)
            self.register_attr_node(attr_node)

        # map User, Session, Now and similar Attribute -> function calls
        for attr_name, py_function in self.context_class.get_dexp_attrname_dict().items():
            type_info = TypeInfo.extract_function_return_type_info(
                            py_function,
                            allow_nonetype=True)
            if attr_name in self.store:
                raise EntitySetupNameError(f"Attribute name '{attr_name}' is reserved. Rename class attribute in '{self.context_class}'")
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
            raise EntityApplyNameError(owner=self, msg=f"ContextNS attribute '{component.name}' can not be fetched since context is not set ({type(context)}).")
        # if attr_name in self.context_class.get_dexp_attrname_dict():
        return context, None


# ------------------------------------------------------------


@dataclass
class ConfigRegistry(RegistryBase):

    config: Config = field(repr=False)

    NAMESPACE: ClassVar[Namespace] = ConfigNS

    def __post_init__(self):
        super().__post_init__()

        if not self.config:
            raise EntityInternalError(owner=self, msg="Config is required")

        self.register_all_nodes()

    def register_all_nodes(self):
        if not isinstance(self.config, Config):
            raise EntityInternalError(owner=self, msg=f".config is not Config instance, got: {type(self.config)} / {self.config}")

        config_class = self.config.__class__
        for attr_name in get_model_fields(config_class):
            attr_node = self._create_attr_node_for_model_attr(config_class, attr_name)
            self.register_attr_node(attr_node)

    def get_root_value(self, apply_session: IApplySession, attr_name: AttrName) -> Tuple[Any, Optional[AttrName]]:
        # ALT: config = apply_session.entity.config
        config = self.config
        if config in (UNDEFINED, None):
            component = apply_session.current_frame.component
            raise EntityInternalError(owner=self, 
                msg=f"ConfigNS attribute '{component.name}' can not be fetched since config is not set ({type(config)}).")
        return config, None


# ------------------------------------------------------------


@dataclass
class ThisRegistry(RegistryBase):
    """ 
    Applies to model instances, e.g. 
        company <= Company(name="Cisco", city="London")

        This.Instance <= company
        This.name <= "Cisco"
        This.city <= "London"

    Uses ReservedAttributeNames.INSTANCE_ATTR_NAME == "Instance" 
    """

    children_mode: bool = False
    model_class: Optional[ModelType] = None
    attr_node: Optional[AttrDexpNode] = None

    NAMESPACE: ClassVar[Namespace] = ThisNS

    # autocomputed
    attr_name: Optional[str] = field(init=False, repr=False, default=None)


    def __post_init__(self):

        super().__post_init__()

        if not (self.model_class or self.attr_node):
            raise EntityInternalError(owner=self, msg="Pass model_class or attr_node.") 

        if self.attr_node:
            if not isinstance(self.attr_node, AttrDexpNode):
                raise EntitySetupValueError(owner=self, msg=f"Expected AttrDexpNode, got: {type(self.attr_node)} / {self.attr_node}")
            self.attr_name = self.attr_node.name

            # This.Value == ReservedAttributeNames.VALUE_ATTR_NAME.value
            self.register_value_attr_node(attr_node=self.attr_node)

        if self.model_class:
            if not is_model_class(self.model_class):
                raise EntitySetupValueError(owner=self, msg=f"Expected model class (DC/PYD), got: {type(self.model_class)} / {self.model_class} ")

            for attr_name in get_model_fields(self.model_class):
                # th_field: ModelField in .values()
                attr_node = self._create_attr_node_for_model_attr(self.model_class, attr_name)
                self.register_attr_node(attr_node)

            # This.Instance == ReservedAttributeNames.INSTANCE_ATTR_NAME 
            self.register_instance_attr_node(model_class=self.model_class)

        else:
            if self.children_mode:
                raise EntityInternalError(owner=self, msg="Children mode allowed only when model_class is passed") 



    def get_root_value(self, 
            apply_session: IApplySession, 
            attr_name: AttrName
            ) -> Tuple[Any, Optional[AttrName]]:
        " TODO: explain: when 2nd param is not None, then ... "
        # TODO: 
        # if not isinstance(apply_session.current_frame.instance, self.model_class):
        #     raise EntityInternalError(owner=self, msg=f"Type of apply session's instance expected to be '{self.model_class}, got: {apply_session.current_frame.instance}") 

        out = UNDEFINED

        if self.attr_name and attr_name == ReservedAttributeNames.VALUE_ATTR_NAME.value:
            # instead of fetching .Value, instruct caller to fetch component's
            # bound attribute
            out = apply_session.current_frame.instance, self.attr_name

        elif self.model_class:
            if not isinstance(apply_session.current_frame.instance, self.model_class):
                raise EntityInternalError(owner=self, msg=f"Type of apply session's instance expected to be '{self.model_class}, got: {apply_session.current_frame.instance}") 
            out = apply_session.current_frame.instance, None

        if not out:
            raise EntityApplyNameError(owner=self, msg=f"Unknown attribute '.{attr_name}'.") 

        return out


# ------------------------------------------------------------


class SetupSession(SetupSessionBase):

    def create_local_setup_session(self, 
            this_ns_instance_model_class: Optional[ModelType],
            this_ns_value_attr_node: Optional[AttrDexpNode] = None,
            ) -> Self:
        """ Currently creates only local ThisNS registry, which is used for
        some local context, e.g. Component This. dexps 
        Args:

            a) this_ns_instance_model_class -> This.Instance + This.<all-attribute-names> logic

            b) this_ns_value_attr_node -> .Value logic, no other attributes for now

        TODO:??
            this_ns_value_attr_name:
                when None -> .Instance + all-attributes logic
                when not None -> .Value logic, no other attributes for now

        """
        # if not this_ns_instance_model_class:
        #     raise EntityInternalError(owner=self, msg="Expected this_ns_instance_model_class.") 

        kwargs = {}
        if this_ns_instance_model_class:
            kwargs["model_class"]=this_ns_instance_model_class

        if this_ns_value_attr_node:
            kwargs["attr_node"]=this_ns_value_attr_node

        if not kwargs:
            raise EntityInternalError(owner=self, msg="Expected this_ns_instance_model_class and/or this_ns_value_attr_node") 

        this_registry = ThisRegistry(**kwargs)

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

# # ------------------------------------------------------------
# 
# class ThisValueRegistry(RegistryBase):
#     """ 
#     Applies to model instance's attribute, e.g. 
#         company <= Company(name="Cisco", city="London")
#         company_name = company.name
# 
#         This.Value <= company_name
# 
#     Uses ReservedAttributeNames.VALUE_ATTR_NAME == "Value" 
#     """
# 
#     NAMESPACE = ThisNS
# 
#     def __init__(self, attr_node: AttrDexpNode):
#         # TODO: model_class: ModelType
#         super().__init__()
# 
#         self.attr_node = attr_node
#         if not isinstance(self.attr_node, AttrDexpNode):
#             raise EntitySetupValueError(owner=self, msg=f"Expected AttrDexpNode, got: {type(self.attr_node)} / {self.attr_node}")
#         self.attr_name = self.attr_node.name
# 
#         # TODO: consider adding this too? how to fetch Parent?
#         #   self.model_class = model_class
#         #   if not is_model_class(self.model_class):
#         #       raise EntitySetupValueError(owner=self, msg=f"Expected model class (DC/PYD), got: {type(self.model_class)} / {self.model_class}")
#         #
#         #   model_fields = get_model_fields(self.model_class)
#         #   if self.attr_name not in model_fields:
#         #       raise EntitySetupValueError(owner=self, msg=f"Attribute '{attr_name}' not found in {type(self.model_class)} / {self.model_class}")
# 
#         # This.Value == ReservedAttributeNames.VALUE_ATTR_NAME.value
#         self.register_value_attr_node(attr_node=attr_node)
# 
# 
#     def get_root_value(self, apply_session: IApplySession, attr_name: AttrName) -> Tuple[Any, Optional[AttrName]]:
#         # TODO: 
#         # if not isinstance(apply_session.current_frame.instance, self.model_class):
#         #     raise EntityInternalError(owner=self, msg=f"Type of apply session's instance expected to be '{self.model_class}, got: {apply_session.current_frame.instance}") 
# 
#         if attr_name != ReservedAttributeNames.VALUE_ATTR_NAME.value:
#             raise EntityApplyNameError(owner=self, msg=f"Only '.Value' attribute is expected, got: {attr_name}") 
# 
#         # instead of fetching .Value, instruct caller to fetch component's bound attribute
#         return apply_session.current_frame.instance, self.attr_name
