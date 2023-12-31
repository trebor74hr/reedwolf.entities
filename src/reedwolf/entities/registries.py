import inspect
from collections import OrderedDict
from dataclasses import (
    dataclass,
    field,
)
from enum import Enum
from typing import (
    List,
    Union,
    Optional,
    ClassVar,
    Dict,
    Type,
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
    EntitySetupTypeError,
    EntityApplyNameError,
)
from .namespaces import (
    Namespace,
    ModelsNS,
    FieldsNS,
    ThisNS,
    FunctionsNS,
    ContextNS,
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
    ExpressionsAttributesMap,
    MethodName,
    FunctionNoArgs,
    is_instancemethod_by_name,
    FieldName,
    ModelField,
    KlassMember,
)
from .base import (
    ReservedAttributeNames,
    IComponent,
    IField,
    IApplyResult,
    IBoundModel,
    IFieldGroup,
    ISetupSession,
    IContainer,
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
from .settings import (
    Settings, ApplySettings,
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
class UnboundModelsRegistry(IThisRegistry, RegistryBase):
    """
    This is temporary registry used only in unbound mode.
    It has not predefined registry entries, rather will acceept all new
    fiedls and register them in a store.
    Later this registry is replaced and this instance and store is rejected.
    """
    NAMESPACE: ClassVar[Namespace] = ModelsNS

    @staticmethod
    def is_unbound_models_registry() -> bool:
        return True

    def register_unbound_attr_node(self, component: Union[IField, IContainer], full_dexp_node_name: AttrName) -> AttrDexpNode:
        type_info = component.get_type_info()
        # python_type = field.python_type
        # if not field.PYTHON_TYPE:
        #     raise EntityInternalError(owner=self, msg=f"Python type not yet set.")
        # type_info = TypeInfo.get_or_create_by_type(python_type)
        attr_node = AttrDexpNode(
            name=full_dexp_node_name,
            data=type_info, # must be like this
            namespace=self.NAMESPACE,
            type_info=type_info,
            type_object=type_info.type_,
        )
        self.register_attr_node(attr_node, alt_attr_node_name=None)
        return attr_node

    # ------------------------------------------------------------

    def _apply_to_get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RegistryRootValue:
        raise NotImplementedError()

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

    root_attr_nodes: Optional[Dict[str, AttrDexpNode]] = field(repr=False, init=False, default_factory=dict)

    # Just to check not duplicate. Can have several, first is main model, other are submodels
    models_dict: Dict[str, ModelType] = field(repr=True, init=False, default_factory=OrderedDict)

    NAMESPACE: ClassVar[Namespace] = ModelsNS

    # def __post_init__(self):
    #     super().__post_init__()

    # NOTE: no register() method due complex logic - see
    #       ContainerBase._register_bound_model()

    def _create_root_attr_node(self, bound_model:IBoundModel) -> AttrDexpNode:
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

    def register_all_nodes(self, root_attr_node: Optional[AttrDexpNode],
                           bound_model: IBoundModel,
                           model: ModelType,
                           unbound_mode: bool = False):
        " models specific method "
        if not root_attr_node:
            root_attr_node = self._create_root_attr_node(bound_model=bound_model)

        if bound_model.name in self.models_dict:
            raise EntityInternalError(owner=self, msg=f"Model {bound_model.name} already set {self.models_dict[bound_model.name]}, got: {model}")
        self.models_dict[bound_model.name] = model

        name = bound_model.get_full_name(init=unbound_mode)
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
        type_info = bound_model.get_type_info()
        type_info_from_model = TypeInfo.get_or_create_by_type(model)
        if not type_info_from_model.type_ == type_info.type_:
            raise EntityInternalError(owner=self, msg=f"Model type info inner type <> bound_model's: {type_info} <> {type_info_from_model}")

        self._register_special_attr_node(
                        type_info = type_info,
                        attr_name = ReservedAttributeNames.INSTANCE_ATTR_NAME.value,
                        attr_name_prefix = None if is_root else f"{name_for_reg}__",
                        # model_class=model,
                        )

    # ------------------------------------------------------------

    def get_attr_node_by_bound_model(self,
                                     bound_model:IBoundModel,
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
            if not apply_result.entity.is_unbound() \
              and not apply_result.instance_none_mode \
              and instance_to_test:
                apply_result.current_frame.component.value_accessor.validate_instance_type(
                        owner_name=f"{apply_result.current_frame.component.name} -> {self}.{attr_name}",
                        instance=instance_to_test,
                        model_type=expected_type,
                )
                # if not isinstance(instance_to_test, expected_type):
                #     raise EntityApplyTypeError(owner=self, msg=f"Wrong type, expected '{expected_type}', got '{instance}'")

        return RegistryRootValue(instance, None)


# ------------------------------------------------------------


@dataclass
class FieldsRegistry(RegistryBase):

    NAMESPACE: ClassVar[Namespace] = FieldsNS

    ALLOWED_BASE_TYPES: ClassVar[List[type]] = (IField,)

    # TODO: zamijeni IContainer s; IFieldGroup, IEntityBase, a dodaj u Allowed: ISubentityBase
    DENIED_BASE_TYPES: ClassVar[List[type]] = (IBoundModel, ValidationBase, EvaluationBase, IContainer, IFieldGroup,)

    def create_attr_node(self, component:IComponent):
        # TODO: put class in container and remove these local imports
        # ------------------------------------------------------------
        # A.3. COMPONENTS - collect attr_nodes - previously flattened (recursive function fill_components)
        # ------------------------------------------------------------
        if not isinstance(component, (IComponent,)):
            raise EntitySetupError(owner=self, msg=f"Register expexted ComponentBase, got {component} / {type(component)}.")

        component_name = component.name

        # TODO: to have standard types in some global list in fields.py
        #           containers, validations, evaluations, 
        if isinstance(component, (IField,)):
            denied = False
            deny_reason = ""
            type_info = component.type_info if component.type_info else None
        # elif isinstance(component, (ISubentityBase, )):
        #     denied = False
        #     deny_reason = ""
        #     d1, d2 = component.get_component_fields_dataclass(setup_session=self.setup_session)
        #     type_info = None # component.get_type_info()

        # TODO: to have standard types in some global list in fields.py
        #           containers, validations, evaluations,
        elif isinstance(component, self.DENIED_BASE_TYPES): # 
            # stored - but should not be used
            assert not isinstance(component, self.ALLOWED_BASE_TYPES), component
            denied = True
            deny_reason = f"Component of type {component.__class__.__name__} can not be referenced in DotExpressions"
            if hasattr(component, "type_info"):
                type_info=component.type_info
            else:
                type_info=None
        else:
            if isinstance(component, self.ALLOWED_BASE_TYPES):
                raise EntityInternalError(owner=self, msg=f"Component is in ALLOWED_BASE_TYPES, and is not processed: {type(component)}. Add new if isinstance() here.")
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


    def register(self, component:IComponent):
        attr_node = self.create_attr_node(component)
        self.register_attr_node(attr_node) # , is_list=False))
        return attr_node


    def _apply_to_get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RegistryRootValue:
        # container = apply_result.current_frame.component.get_first_parent_container(consider_self=True)
        component = apply_result.current_frame.component
        instance  = apply_result.current_frame.instance
        # value_node = apply_result.current_frame.value_node
        top_attr_accessor = ComponentAttributeAccessor(component, instance)
        return RegistryRootValue(top_attr_accessor, None)


# -------------------------------------------------------------

class SettingsType(str, Enum):
    SETUP_SETTINGS = "SETUP_SETTINGS"
    APPLY_SETTINGS = "APPLY_SETTINGS"


@dataclass
class SettingsKlassMember(KlassMember):
    settings_type: SettingsType


@dataclass
class SettingsSource:
    settings_type: SettingsType
    klass: ModelType
    fields: Dict[AttrName, ModelField] = field(init=False, repr=False)

    def __post_init__(self):
        self.fields = get_model_fields(self.klass)

# ------------------------------------------------------------

@dataclass
class ContextRegistry(RegistryBase):

    setup_settings: Settings = field(repr=False)
    apply_settings_class: Optional[Type[ApplySettings]] = field(repr=False)

    NAMESPACE: ClassVar[Namespace] = ContextNS

    attributes_dict: ExpressionsAttributesMap = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.setup_settings, Settings):
            raise EntitySetupValueError(owner=self, msg=f"setup_settings must be instance of Settings, got: {self.apply_settings_class}")

        if self.apply_settings_class is not None:
            if Settings not in inspect.getmro(self.apply_settings_class):
                raise EntitySetupValueError(owner=self, msg=f"apply_settings_class should inherit Settings, got: {self.apply_settings_class}")
        self.register_all_nodes()

    def register_all_nodes(self):
        """
        For the same attribute name - this is order of preferences - which will win:
            1. custom attributes in apply settings
            2. custom attributes in setup settings
            3. common attributes in apply settings (usually not overridden)
            4. common attributes in setup settings (usually not overridden)
        """
        setup_settings_source = SettingsSource(SettingsType.SETUP_SETTINGS, self.setup_settings.__class__)
        common_dict = self.setup_settings._get_common_contextns_attributes()
        setup_custom_dict = self.setup_settings.get_custom_contextns_attributes()

        if self.apply_settings_class:
            apply_settings_source = SettingsSource(SettingsType.APPLY_SETTINGS, self.apply_settings_class)
            apply_custom_dict = self.apply_settings_class.get_custom_contextns_attributes()
            settings_source_list_pairs = [
                (common_dict, [setup_settings_source, apply_settings_source]),
                (setup_custom_dict, [setup_settings_source]),
                (apply_custom_dict, [apply_settings_source]),
            ]
        else:
            settings_source_list_pairs = [
                (common_dict, [setup_settings_source]),
                (setup_custom_dict, [setup_settings_source]),
            ]

        for attributes_dict, settings_source_list in settings_source_list_pairs:

            for attr_name, attr_getter in attributes_dict.items():

                if isinstance(attr_getter, MethodName):
                    attr_getter: MethodName = attr_getter
                    py_function = settings_source = UNDEFINED
                    for settings_source in settings_source_list:
                        py_function: FunctionNoArgs = getattr(settings_source.klass, attr_getter, UNDEFINED)
                        if py_function is not UNDEFINED:
                            break

                    if py_function is UNDEFINED:
                        # TODO: could I get all methods with no args?
                        models = [settings_source.klass for settings_source in settings_source_list]
                        raise EntitySetupNameError(owner=self,
                                                   msg=f"Attribute {attr_name} must be name of method with no arguments from class(es) '{models}', got: {attr_getter}")

                    function_name = py_function.__name__
                    if not is_instancemethod_by_name(settings_source.klass,  function_name):
                        raise EntitySetupNameError(owner=self,
                                                   msg=f"Attribute {attr_name} must be name of method with no arguments of class '{settings_source.klass}', function {attr_getter} is not instance method of this class.")

                    # Check that function receives only single param if method(self), or no param if function()
                    py_fun_signature = inspect.signature(py_function)
                    # TODO: resolve properly first arg name as 'self' convention
                    non_empty_params = [param.name for param in py_fun_signature.parameters.values() if param.empty and param.name != 'self']
                    if len(non_empty_params)!=0:
                        raise EntitySetupNameError(owner=self,
                                                   msg=f"{attr_name}: Method '{settings_source.klass.__name__}.{attr_getter}()' must not have arguments without defaults. Found: {', '.join(non_empty_params)} ")

                    # NOTE: py_function is not used later
                    type_info = TypeInfo.extract_function_return_type_info(py_function, allow_nonetype=True)
                    data = attr_getter
                    type_object = SettingsKlassMember(settings_type=settings_source.settings_type,
                                                      klass=settings_source.klass,
                                                      name=attr_getter)

                elif isinstance(attr_getter, FieldName):
                    attr_getter: FieldName = attr_getter
                    attr_field = settings_source = UNDEFINED
                    for settings_source in settings_source_list:
                        attr_field = settings_source.fields.get(attr_getter, UNDEFINED)
                        if attr_field is not UNDEFINED:
                            break

                    if attr_field is UNDEFINED:
                        all_keys = set()
                        for settings_source in settings_source_list:
                            all_keys.union(set(settings_source.fields.keys()))
                        models = [settings_source.klass for settings_source in settings_source_list]
                        aval_names = get_available_names_example(attr_field, list(all_keys))
                        raise EntitySetupNameError(owner=self, msg=f"Attribute {attr_name} must be field name of class(es) '{models}', got: {attr_getter}, available: {aval_names}")

                    # NOTE: attr_field is not used later
                    type_info = TypeInfo.get_or_create_by_type(py_type_hint=attr_field,
                                                               caller=settings_source.klass)
                    data = attr_getter
                    type_object = SettingsKlassMember(settings_type=settings_source.settings_type,
                                                      klass=settings_source.klass,
                                                      name=attr_getter)
                else:
                    raise EntitySetupValueError(owner=self, msg=f"Attribute {attr_name} expected FieldName or MethodName instance, got: {attr_getter} / {type(attr_getter)}")

                # NOTE: No problem with override any more!
                #           if attr_name in self.store:
                #               raise EntitySetupNameError(f"Attribute name '{attr_name}' is reserved. Rename class attribute in '{self.apply_settings_class}'")

                attr_node = AttrDexpNode(
                    name=attr_name,
                    namespace=self.NAMESPACE,
                    type_info=type_info,
                    data=data,
                    type_object=type_object,
                )

                self.register_attr_node(attr_node, attr_name, replace_when_duplicate=True)

        return

    def create_node(self,
                    dexp_node_name: str,
                    owner_dexp_node: IDotExpressionNode,
                    owner: IComponent,
                    ) -> IDotExpressionNode:

        if not isinstance(owner, IComponent):
            raise EntityInternalError(owner=self, msg=f"Owner needs to be Component, got: {type(owner)} / {owner}")

        return super().create_node(
            dexp_node_name=dexp_node_name,
            owner_dexp_node=owner_dexp_node,
            owner=owner,
        )


    def _apply_to_get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RegistryRootValue:
        if attr_name not in self.store:
            avail_names = get_available_names_example(attr_name, list(self.store.keys()))
            raise EntityApplyNameError(owner=self, msg=f"Invalid attribute name '{attr_name}', available: {avail_names}.")

        attr_dexp_node = self.store[attr_name]
        klass_member = attr_dexp_node.type_object

        if not isinstance(klass_member, SettingsKlassMember):
            raise EntityInternalError(owner=self, msg=f"Expected SettingsClassMember instance, got: {klass_member}")

        # can be method name or field name
        attr_name_new = klass_member.name
        if klass_member.settings_type == SettingsType.APPLY_SETTINGS:
            value_root = apply_result.settings
        elif klass_member.settings_type == SettingsType.SETUP_SETTINGS:
            value_root = apply_result.entity.settings
        else:
            raise EntityInternalError(owner=self, msg=f"Invalid settingss type: got: {klass_member.settings_type}")

        return RegistryRootValue(value_root, attr_name_new)

# ------------------------------------------------------------

@dataclass
class ThisRegistry(IThisRegistry, RegistryBase):
    """
    General idea/overview:
                                                Field   bool-ena field-gr subent-sin subent-lst FuncArg: Filter(T.) FuncArg: T.attr
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
    component: Optional[IComponent] = field(default=None)
    model_class: Optional[ModelType] = field(default=None)
    # Used for ItemsFunctions when Item attribute will be available, e.g. This.name -> AttrValue
    # TODO: do I really need this parameter - is it overlapping with is_items_mode?
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
                type_info=type_info,
                attr_name=ReservedAttributeNames.VALUE_ATTR_NAME.value,
                # model_class=type_info.type_,
                )

        if self.model_class:
            if not self.is_items_for_each_mode and self.is_items_mode:
                # This.Items
                self._register_special_attr_node(
                    type_info=self.model_class_type_info,  # already a List
                    attr_name=ReservedAttributeNames.ITEMS_ATTR_NAME.value,
                )
            else:
                # NOTE: Includes self.is_items_for_each_mode too
                # This.<all-attributes>
                self._register_model_nodes(model_class=self.model_class_type_info.type_)
                # This.Instance
                self._register_special_attr_node(
                    type_info=self.model_class_type_info,
                    attr_name=ReservedAttributeNames.INSTANCE_ATTR_NAME,
                    attr_name_prefix=None,
                    # model_class=self.model_class_type_info.type_,
                )

        if self.component:
            if not self.is_items_for_each_mode and self.is_items_mode:
                # This.Items: List[component_fields_dataclass]
                component_fields_dataclass, _ = self.component.get_component_fields_dataclass(
                                                    setup_session=setup_session)
                py_type_hint = List[component_fields_dataclass]
                type_info = TypeInfo.get_or_create_by_type(py_type_hint)
                self._register_special_attr_node(
                    type_info=type_info,
                    attr_name=ReservedAttributeNames.ITEMS_ATTR_NAME.value,
                    # model_class=component_fields_dataclass,
                )
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

            root_value = RegistryRootValue(value_root=apply_result.current_frame.instance,
                                           attr_name_new=atrr_name_to_fetch)

        if self.component:
            if root_value:
                raise EntityInternalError(owner=self, msg=f"component mode and some other mode clash, root_value already set: {root_value}")
            # TODO: root_value could be already defined, should I put "elif ..." instead?
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
                root_value = RegistryRootValue(value_root=apply_result.current_frame.instance,
                                               attr_name_new=atrr_name_to_fetch)

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
    #     some local settings, e.g. Component This. dexps
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


