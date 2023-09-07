import inspect
from collections import OrderedDict
from dataclasses import (
    dataclass,
    field,
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
    to_repr, get_available_names_example,
)
from .exceptions import (
    EntitySetupError,
    EntitySetupNameError,
    EntitySetupValueError,
    EntityInternalError,
    EntityApplyTypeError,
    EntityApplyNameError,
    EntityApplyValueError,
    EntitySetupTypeError,
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
    RegistryRootValue,
)
from .meta import (
    ModelType,
    get_model_fields,
    TypeInfo,
    AttrName,
    Self,
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

    # Just to check not duplicate. Can have several, first is main model, other are submodels
    models_dict : Dict[str, ModelType] = field(repr=True, init=False, default_factory=OrderedDict)

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

        if bound_model.name in self.models_dict:
            raise EntityInternalError(owner=self, msg=f"Model {bound_model.name} already set {self.models_dict[bound_model.name]}, got: {model}")
        self.models_dict[bound_model.name] = model

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
        self._register_special_attr_node(
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

    def _apply_to_get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RegistryRootValue:
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
            # TODO: caller should check expected type ...
            if not apply_result.instance_none_mode and instance_to_test \
              and not isinstance(instance_to_test, expected_type):
                raise EntityApplyTypeError(owner=self, msg=f"Wrong type, expected '{expected_type}', got '{instance}'")

        return RegistryRootValue(instance, None)\


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


    def _apply_to_get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RegistryRootValue:
        # container = apply_result.current_frame.component.get_first_parent_container(consider_self=True)
        component = apply_result.current_frame.component
        instance  = apply_result.current_frame.instance
        top_attr_accessor = ComponentAttributeAccessor(component, instance)
        return RegistryRootValue(top_attr_accessor, None)


# ------------------------------------------------------------


@dataclass
class ContextRegistry(RegistryBase):

    context_class: Optional[IContext] = field(repr=False)

    NAMESPACE: ClassVar[Namespace] = ContextNS

    def __post_init__(self):
        # super().__post_init__()
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


    def _apply_to_get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RegistryRootValue:
        context = apply_result.context
        if context in (UNDEFINED, None):
            component = apply_result.current_frame.component
            raise EntityApplyNameError(owner=self, msg=f"ContextNS attribute '{component.name}' can not be fetched since context is not set ({type(context)}).")
        # if attr_name in self.context_class.get_dexp_attrname_dict():
        return RegistryRootValue(context, None)


# ------------------------------------------------------------


@dataclass
class ConfigRegistry(RegistryBase):

    config: Config = field(repr=False)

    NAMESPACE: ClassVar[Namespace] = ConfigNS

    def __post_init__(self):
        # super().__post_init__()
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

    def _apply_to_get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RegistryRootValue:
        # ALT: config = apply_result.entity.config
        config = self.config
        if config in (UNDEFINED, None):
            component = apply_result.current_frame.component
            raise EntityInternalError(owner=self, 
                msg=f"ConfigNS attribute '{component.name}' can not be fetched since config is not set ({type(config)}).")
        return RegistryRootValue(config, None)


# ------------------------------------------------------------
@dataclass
class ThisRegistry(IThisRegistry, RegistryBase):
    """
    General idea/overview:
                                                Field   bool-ena field-gr subent-sin subent-its FuncArg: Filter(T.) FuncArg: T.attr
                                                ------- -------- -------- ---------- ---------- ------------------- ---------------
    a) .<field-name>     - direktno             -       yes      yes      yes        yes (*1)   - (vidi f)          yes- if Item
    b) This.Value                               yes     yes      -        -          -          -                   yes- if std.
    c) This.Instance     - jel mi treba???      -       -        -        yes        -- (*2)    yes(Self)           yes- if Item?
    d) This.Items: List[Item]                   -       -        -        -          yes        - (vidi g)          -
    e) This.Children: List[ChildField]          -       yes      yes      yes        yes (*1)   -                   -

    f) This.Item.<field-name>                   -       -        -        -          -          yes(*3)             -
    g) This.Item.Children                       -       -        -        -          -          yes(*3)             -

    (*1) - in which case to put this? - if this is ok, then put in first too
    (*2) - maybe InstanceList
    (*3) - the question: how to make This.Item work? maybe simple as done in a) i e)
    """
    NAMESPACE: ClassVar[Namespace] = ThisNS

    attr_node: Optional[AttrDexpNode] = field(default=None)
    component: Optional[ComponentBase] = field(default=None)
    model_class: Optional[ModelType] = field(default=None)
    # Used for ItemsFunctions when Item attribute will be available, e.g. This.name -> AttrValue
    is_items_for_each_mode: bool = field(default=False)

    # autocomputed
    # This.Items -> List[Item]
    is_items_mode: bool = field(init=False, default=False)
    attr_name: Optional[str] = field(init=False, repr=False, default=None)
    model_class_type_info: Optional[TypeInfo] = field(default=None)

    @classmethod
    def create_for_model_class(cls,
            setup_session: ISetupSession,
            model_class: ModelType,
            is_items_for_each_mode: bool = False,
    ) -> Self:
        # NOTE: must be here since:
        #   - expressions.py don't see ThisRegistry
        #   - setup.py does not see registries
        #   - registries - needed by setup.py which can not see registries
        # TODO: try to resolve this and put
        """
        - ThisRegistry is unavailable to low-level modules -
          e.g. func_args -> setup.
        - .Instance + <attr-names> is used only in manual setup cases,
          e.g. ChoiceField()
        """
        this_registry = ThisRegistry(model_class=model_class, is_items_for_each_mode=is_items_for_each_mode)
        this_registry.setup(setup_session=setup_session)
        this_registry.finish()

        return this_registry

    def __post_init__(self):
        if self.attr_node:
            if not isinstance(self.attr_node, AttrDexpNode):
                raise EntitySetupValueError(owner=self, msg=f"Expected AttrDexpNode, got: {type(self.attr_node)} / {self.attr_node}")
            self.attr_name = self.attr_node.name

        if self.model_class:
            if self.attr_node or self.component:
                raise EntityInternalError(owner=self, msg="model_class can not be combined with attr_node nor component cases.")
            self.model_class_type_info = TypeInfo.get_or_create_by_type(self.model_class)

        if self.is_items_for_each_mode:
            if not ((self.component and self.component.is_subentity_items())   \
                    or (self.model_class and self.model_class_type_info.is_list)):
                raise EntitySetupTypeError(owner=self, msg=f"is_items_for_each_mode needs to operate on List[Any], got: {self.model_class} / {self.component} ")
        else:
            if self.component and self.component.is_subentity_items():
                self.is_items_mode = True
            elif self.model_class and self.model_class_type_info.is_list:
                self.is_items_mode = True

    def setup(self, setup_session: ISetupSession) -> None:
        super().setup(setup_session)

        if self.attr_node:
            # This.Value == ReservedAttributeNames.VALUE_ATTR_NAME
            type_info = self.attr_node.get_type_info()
            self._register_special_attr_node(
                model_class=type_info.type_,
                attr_name=ReservedAttributeNames.VALUE_ATTR_NAME.value,
                )

        if self.model_class:
            if not self.is_items_for_each_mode and self.is_items_mode:
                # This.Items
                self._register_special_attr_node(
                    model_class=self.model_class_type_info.type_,
                    attr_name=ReservedAttributeNames.ITEMS_ATTR_NAME.value,
                )
            else:
                # NOTE: Includes self.is_items_for_each_mode too
                # This.<all-attributes>
                self._register_model_nodes(model_class=self.model_class_type_info.type_)
                # This.Instance
                self._register_special_attr_node(
                    model_class=self.model_class_type_info.type_,
                    attr_name=ReservedAttributeNames.INSTANCE_ATTR_NAME,
                    attr_name_prefix=None)

        if self.component:
            if not self.is_items_for_each_mode and self.is_items_mode:
                # This.Items : List[component_fields_dataclass]
                component_fields_dataclass, _ = self.component.get_component_fields_dataclass(
                                                    setup_session=setup_session)
                self._register_special_attr_node(
                    model_class=component_fields_dataclass,
                    attr_name=ReservedAttributeNames.ITEMS_ATTR_NAME.value)
            else:
                # NOTE: Includes self.is_items_for_each_mode too
                # This.<all-attribute> + This.Children: List[ChildField]
                self._register_all_children(
                    setup_session=setup_session,
                    attr_name=ReservedAttributeNames.CHILDREN_ATTR_NAME,
                    owner=self.component,
                )

    def _apply_to_get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RegistryRootValue:
        if not self.finished:
            raise EntityInternalError(owner=self, msg="Setup not called")

        root_value: Optional[RegistryRootValue] = None

        if self.attr_node and attr_name == ReservedAttributeNames.VALUE_ATTR_NAME.value:
            # raise EntityInternalError(owner=self, msg=f"Expected attribute name: {ReservedAttributeNames.VALUE_ATTR_NAME.value}, got: {attr_name}")
            # with 2nd return value -> instead of fetching .Value, instruct caller to
            #                   fetch component's bound attribute
            root_value = RegistryRootValue(value_root=apply_result.current_frame.instance,
                                           attr_name_new=self.attr_name)
        elif self.model_class:
            if not isinstance(apply_result.current_frame.instance, self.model_class):
                raise EntityInternalError(owner=self,
                                          msg=f"Type of apply session's instance expected to be '{self.model_class}, got: {apply_result.current_frame.instance}")

            if attr_name == ReservedAttributeNames.INSTANCE_ATTR_NAME:
                # with 2nd param == None -> do not fetch further
                atrr_name_to_fetch = None
            else:
                # with 2nd param like this -> fetch further by attr_name
                atrr_name_to_fetch = attr_name

            root_value = RegistryRootValue(apply_result.current_frame.instance, atrr_name_to_fetch)

        if self.component:
            if self.is_items_mode:
                # multiple items case
                if attr_name == ReservedAttributeNames.ITEMS_ATTR_NAME.value:
                    if not isinstance(apply_result.current_frame.instance, (list, tuple)):
                        raise EntityInternalError(f"Items expected, got: {apply_result.current_frame.instance}")

                    root_value = RegistryRootValue(
                        value_root=apply_result.current_frame.instance,
                        attr_name_new=None,
                        do_fetch_by_name=False)
                elif attr_name == ReservedAttributeNames.CHILDREN_ATTR_NAME.value:
                    if  isinstance(apply_result.current_frame.instance, (list, tuple)):
                        raise EntityInternalError(f"Single item expected, got: {apply_result.current_frame.instance}")

                    if not isinstance(apply_result.current_frame.component.child_field_list, (list, tuple)):
                        raise EntityInternalError(owner=apply_result.current_frame.component,
                                                  msg=f"_child_field_list not a list, got: {apply_result.current_frame.component.child_field_list}")
                    # TODO: .Children?
                    # raise NotImplementedError()
                    root_value = RegistryRootValue(
                        value_root=apply_result.current_frame.component.child_field_list,
                        attr_name_new=None,
                        do_fetch_by_name=False,
                    )
                else:
                    raise EntityInternalError(owner=self, msg=f"Expected attribute name: {ReservedAttributeNames.ITEMS_ATTR_NAME.value} or {ReservedAttributeNames.CHILDREN_ATTR_NAME.value} , got: {attr_name}")
            else:
                # single item component
                if not isinstance(apply_result.current_frame.instance, self.model_class):
                    raise EntityInternalError(owner=self,
                                              msg=f"Type of apply session's instance expected to be '{self.model_class}, got: {apply_result.current_frame.instance}")

                if attr_name == ReservedAttributeNames.CHILDREN_ATTR_NAME:
                    # with 2nd param == None -> do not fetch further
                    atrr_name_to_fetch = None
                    raise NotImplementedError()
                else:
                    # with 2nd param like this -> fetch further by attr_name
                    atrr_name_to_fetch = attr_name
                root_value = RegistryRootValue(apply_result.current_frame.instance, atrr_name_to_fetch)

        if not root_value:
            raise EntityInternalError(owner=self, msg="Invalid case")

        return root_value

    def __repr__(self):
        out: List[str] = []
        if self.attr_node:
            out.append(f"attr={self.attr_node.name}")
        if self.component:
            out.append(
                f"{'items' if self.is_items_mode else 'component'}=" 
                f"{self.component.__class__.__name__}({self.component.name})")
        if self.model_class:
            out.append(f"model={self.model_class}")
        # if not out:
        #     raise EntityInternalError(owner=self, msg="__repr__ failed -> no mode selected")
        return f'{self.__class__.__name__}({", ".join(out)})'

    __str__ = __repr__


# ------------------------------------------------------------

class SetupSession(SetupSessionBase):
    ...

    # def create_local_setup_session(self, this_registry: IThisRegistry) -> Self:
    #     """ Currently creates only local ThisNS registry, which is used for
    #     some local context, e.g. Component This. dexps
    #     Args:
    #         a) this_ns_instance_model_class -> This.Instance + This.<all-attribute-names> logic
    #         b) this_ns_value_attr_node -> .Value logic, no other attributes for now
    #     """
    #     if not (this_registry and isinstance(this_registry, IThisRegistry)):
    #         raise EntityInternalError(owner=self, msg=f"Expected IThisRegistry instance, got: {this_registry}")

    #     local_setup_session = SetupSession(
    #                             container=self.container,
    #                             parent_setup_session=None,
    #                             functions_factory_registry=self.functions_factory_registry,
    #                             )
    #     local_setup_session.add_registry(this_registry)

    #     return local_setup_session


