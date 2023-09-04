import inspect
from dataclasses import (
        dataclass, 
        field,
        InitVar,
        )
from typing import (
        List,
        Union,
        Optional,
        ClassVar,
        Dict,
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
        IThisRegistry,
        )
from .meta import (
        Self,
        ModelType,
        get_model_fields,
        TypeInfo,
        AttrName,
        AttrValue,
        )
from .base import (
        ReservedAttributeNames,
        ComponentBase,
        IFieldBase,
        IContainerBase,
        IApplyResult,
        BoundModelBase,
        IFieldGroup,
        ISetupSession,
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


@dataclass
class RootValue:
    value_root: AttrValue
    attr_name_new : Optional[AttrName]
    do_fetch_by_name: Union[bool, UndefinedType] = UNDEFINED

# ------------------------------------------------------------


@dataclass
class ModelsRegistry(RegistryBase):
    """
    All models have full path from top container (for better readibility)
    e.g. M.address_set.city.street so reading begins from root
    instance, instead of apply_result.current_frame.instance.
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

        # register
        self._register_children_attr_node(
                        model_class=model,
                        attr_name = ReservedAttributeNames.INSTANCE_ATTR_NAME.value,
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

    def get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RootValue:
        # ROOT_VALUE_NEEDS_FETCH_BY_NAME = False
        # component = apply_result.current_frame.component
        instance = apply_result.current_frame.instance

        # bound_model = apply_result.current_frame.container.bound_model
        bound_model_root = apply_result.current_frame.bound_model_root

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
            if not apply_result.instance_none_mode and instance_to_test \
              and not isinstance(instance_to_test, expected_type):
                raise EntityApplyTypeError(owner=self, msg=f"Wrong type, expected '{expected_type}', got '{instance}'")

        return RootValue(instance, None)


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


    def get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RootValue:
        # container = apply_result.current_frame.component.get_first_parent_container(consider_self=True)
        component = apply_result.current_frame.component
        instance  = apply_result.current_frame.instance
        top_attr_accessor = ComponentAttributeAccessor(component, instance)
        return RootValue(top_attr_accessor, None)


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


    def get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RootValue:
        context = apply_result.context
        if context in (UNDEFINED, None):
            component = apply_result.current_frame.component
            raise EntityApplyNameError(owner=self, msg=f"ContextNS attribute '{component.name}' can not be fetched since context is not set ({type(context)}).")
        # if attr_name in self.context_class.get_dexp_attrname_dict():
        return RootValue(context, None)


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

    def get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RootValue:
        # ALT: config = apply_result.entity.config
        config = self.config
        if config in (UNDEFINED, None):
            component = apply_result.current_frame.component
            raise EntityInternalError(owner=self, 
                msg=f"ConfigNS attribute '{component.name}' can not be fetched since config is not set ({type(config)}).")
        return RootValue(config, None)


# ------------------------------------------------------------


@dataclass
class ThisRegistryForValue(IThisRegistry, RegistryBase):
    """
    This.Value - returns current Component's value
    """
    attr_node: AttrDexpNode
    # autocomputed
    attr_name: Optional[str] = field(init=False, repr=False, default=None)

    NAMESPACE: ClassVar[Namespace] = ThisNS

    def __post_init__(self):
        if not isinstance(self.attr_node, AttrDexpNode):
            raise EntitySetupValueError(owner=self, msg=f"Expected AttrDexpNode, got: {type(self.attr_node)} / {self.attr_node}")
        self.attr_name = self.attr_node.name
        # This.Value == ReservedAttributeNames.VALUE_ATTR_NAME
        self.register_value_attr_node(attr_node=self.attr_node)

    def get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RootValue:
        if attr_name != ReservedAttributeNames.VALUE_ATTR_NAME.value:
            raise EntityInternalError(owner=self, msg=f"Expected attribute name: {ReservedAttributeNames.VALUE_ATTR_NAME.value}, got: {attr_name}") 

        # with 2nd param -> instead of fetching .Value, instruct caller to
        #                   fetch component's bound attribute
        return RootValue(apply_result.current_frame.instance, self.attr_name)

# --------------------

@dataclass
class ThisRegistryForChildren(IThisRegistry, RegistryBase):
    """
    This.Children + This.<component's-fields>
    """
    owner: ComponentBase
    # extendd list of arguments to constructor, but not stored
    setup_session: InitVar[Optional[ISetupSession]] = field(repr=False)

    NAMESPACE: ClassVar[Namespace] = ThisNS

    def __post_init__(self, setup_session: Optional[ISetupSession]):
        self._register_children(
                setup_session=setup_session,
                attr_name=ReservedAttributeNames.CHILDREN_ATTR_NAME,
                owner=self.owner, 
                )

    def get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RootValue:
        if not isinstance(apply_result.current_frame.instance, self.model_class):
            raise EntityInternalError(owner=self, msg=f"Type of apply session's instance expected to be '{self.model_class}, got: {apply_result.current_frame.instance}") 

        if attr_name == ReservedAttributeNames.CHILDREN_ATTR_NAME:
            # with 2nd param == None -> do not fetch further
            atrr_name_to_fetch = None
            raise NotImplementedError()
        else:
            # with 2nd param like this -> fetch further by attr_name
            atrr_name_to_fetch = attr_name

        return RootValue(apply_result.current_frame.instance, atrr_name_to_fetch)

# ------------------------------------------------------------

@dataclass
class ThisRegistryForValueAndChildren(ThisRegistryForChildren):
    """
    Inherits ThisRegistryForChildren + adds .Value", resulting:
        This.Children + This.<component's-fields> + This.Value
    Example: BooleanField with enables.
             .Children -> see ThisRegistryForItemsAndChildren
    """
    attr_node: AttrDexpNode
    owner: ComponentBase = field(repr=False)
    setup_session: InitVar[Optional[ISetupSession]] = field(repr=False)

    # autocomputed
    attr_name: Optional[str] = field(init=False, repr=False, default=None)

    # TODO: not good!!!
    def __post_init__(self, setup_session: Optional[ISetupSession]):
        super().__post_init__(setup_session=setup_session)
        # TODO: DRY this
        if not isinstance(self.attr_node, AttrDexpNode):
            raise EntitySetupValueError(owner=self, msg=f"Expected AttrDexpNode, got: {type(self.attr_node)} / {self.attr_node}")
        self.attr_name = self.attr_node.name
        # This.Value == ReservedAttributeNames.VALUE_ATTR_NAME
        self.register_value_attr_node(attr_node=self.attr_node)
        # TODO: .Children?

    def get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RootValue:
        instance, atrr_name_to_fetch = super().get_root_value(apply_result=apply_result, attr_name=attr_name)

        if atrr_name_to_fetch and attr_name == ReservedAttributeNames.VALUE_ATTR_NAME:
            # instance = apply_result.current_frame.instance
            atrr_name_to_fetch = self.attr_name

        return RootValue(apply_result.current_frame.instance, atrr_name_to_fetch)

# ------------------------------------------------------------

# TODO: for SubEntitySingle
# @dataclass
# class ThisRegistryForInstanceAndChildren(ThisRegistryForChildren):
#     " inherits ThisRegistryForChildren + adds .Instance"


# --------------------


@dataclass
class ThisRegistryForInstance(IThisRegistry, RegistryBase):
    """
    This.Instance + This.<component's-fields>
    Example: model instances:
        company := Company(name="Cisco", city="London")
        This.Instance := company
        This.name := "Cisco"
        This.city := "London"
    """

    model_class: ModelType

    NAMESPACE: ClassVar[Namespace] = ThisNS

    def __post_init__(self):
        # M.Instance / Models.Instance + # M.<all-attributes>
        self._register_model_nodes(model_class=self.model_class)
        self._register_children_attr_node(
                        model_class=self.model_class,
                        attr_name=ReservedAttributeNames.INSTANCE_ATTR_NAME,
                        attr_name_prefix=None)


    def get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RootValue:
        " TODO: explain: when 2nd param is not None, then ... "
        if not isinstance(apply_result.current_frame.instance, self.model_class):
            raise EntityInternalError(owner=self, msg=f"Type of apply session's instance expected to be '{self.model_class}, got: {apply_result.current_frame.instance}") 

        if attr_name == ReservedAttributeNames.INSTANCE_ATTR_NAME:
            # with 2nd param == None -> do not fetch further
            atrr_name_to_fetch = None
        else:
            # with 2nd param like this -> fetch further by attr_name
            atrr_name_to_fetch = attr_name
        return RootValue(apply_result.current_frame.instance, atrr_name_to_fetch)

# --------------------

@dataclass
class ThisRegistryForItemsAndChildren(IThisRegistry, RegistryBase):
    """
    .Items -> input is list of items, which can be filtered, mapped, counter, selected single ...
    .Children -> list of ChildField instances.
    Applies to SubentityItemss.
    Company(name== "Cisco", address_set = [Address(street="First", city="London"), Address(street="Second", city="Paris")])
    This. on address_set:
        This.Items := [Address(street="First", city="London"),
                       Address(street="Second", city="Paris")]
        This.Children := for each item (Address) list of fields with values (ChildField). Example - for first:
                := [ChildField(name="street", Value="First"),
                    ChildField(name="city", value="London")]
    """
    # TODO: consider to include Children + attributes too :
    #       -> validation will be runned againts all items

    owner: ComponentBase
    setup_session: InitVar[ISetupSession] = field(repr=False)

    NAMESPACE: ClassVar[Namespace] = ThisNS

    def __post_init__(self, setup_session: ISetupSession):
        # This.Items == ReservedAttributeNames.ITEMS_ATTR_NAME.value
        self.register_items_attr_node(owner=self.owner)

        # TODO: Children + <attributes> - explain case when this will be used
        #       This.Children.age >= 18 ==> every item should be at least 18 years old
        self._register_children(
                setup_session=setup_session,
                attr_name=ReservedAttributeNames.CHILDREN_ATTR_NAME,
                owner=self.owner, 
                )

    def get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RootValue:
        if attr_name == ReservedAttributeNames.ITEMS_ATTR_NAME.value:
            assert isinstance(apply_result.current_frame.instance, (list, tuple))
            root_value = RootValue(
                            value_root=apply_result.current_frame.instance,
                            attr_name_new=None, 
                            do_fetch_by_name=False)
        elif attr_name == ReservedAttributeNames.CHILDREN_ATTR_NAME.value:
            assert not isinstance(apply_result.current_frame.instance, (list, tuple))
            if not isinstance(apply_result.current_frame.component.child_field_list, (list, tuple)):
                raise EntityInternalError(owner=apply_result.current_frame.component, msg=f"_child_field_list not a list, got: {apply_result.current_frame.component.child_field_list}")

            # TODO: .Children?
            root_value = RootValue(
                            value_root=apply_result.current_frame.component.child_field_list,
                            attr_name_new=None,
                            do_fetch_by_name=False,
                            )
        else:
            raise EntityInternalError(owner=self, msg=f"Expected attribute name: {ReservedAttributeNames.ITEMS_ATTR_NAME.value} or {ReservedAttributeNames.CHILDREN_ATTR_NAME.value} , got: {attr_name}") 
        return root_value



# ------------------------------------------------------------


class SetupSession(SetupSessionBase):

    def create_local_setup_session(self, this_registry: IThisRegistry) -> Self:
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
        if not (this_registry and isinstance(this_registry, IThisRegistry)):
            raise EntityInternalError(owner=self, msg=f"Expected IThisRegistry instance, got: {this_registry}") 

        local_setup_session = SetupSession(
                                container=self.container,
                                parent_setup_session=None,
                                functions_factory_registry=self.functions_factory_registry,
                                )
        local_setup_session.add_registry(this_registry)

        return local_setup_session


