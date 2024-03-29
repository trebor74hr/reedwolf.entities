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
    Type,
    Tuple,
)

from .utils import (
    UndefinedType,
    get_available_names_example,
    UNDEFINED, to_repr,
)
from .exceptions import (
    EntitySetupError,
    EntitySetupValueError,
    EntityInternalError,
    EntitySetupTypeError,
    EntityApplyNameError,
    EntitySetupNameError,
    EntityApplyTypeError,
)
from .namespaces import (
    Namespace,
    ModelsNS,
    FieldsNS,
    ThisNS,
    ContextNS,
    OperationsNS,
)
from .meta_dataclass import Self
from .meta import (
    ModelKlassType,
    get_model_fields,
    TypeInfo,
    AttrName,
    KlassMember,
    SettingsType,
    IAttribute,
    CustomCtxAttributeList,
    ContainerId, is_model_klass, ComponentName, ListChildField,
)
from .expressions import (
    DotExpression,
    IDotExpressionNode,
    IThisRegistry,
    RegistryRootValue, )
from .base import (
    ReservedAttributeNames,
    IComponent,
    IField,
    IApplyResult,
    IDataModel,
    IFieldGroup,
    ISetupSession,
    IContainer,
    IEntity,
    IValueNode, AttrDexpNodeStore, AttrDexpNodeStoreForComponent, )
from .expr_attr_nodes import (
    IAttrDexpNode,
    AttrValueContainerPath,
    AttrDexpNodeWithValuePath,
    AttrDexpNodeForTypeInfo,
    AttrDexpNodeForDataModel,
    AttrDexpNodeForComponent,
    AttrDexpNodeForKlassMember,
)
from .settings import (
    Settings,
    ApplySettings,
)
from .setup import (
    RegistryBase,
    RegistryUseDenied,
    RegistryBaseWithStoreBase, )
from .value_nodes import (
    SubentityItemsValueNode,
)


# ------------------------------------------------------------

# @dataclass
# class FunctionsRegistry(RegistryUseDenied):
#     NAMESPACE: ClassVar[Namespace] = FunctionsNS

@dataclass
class OperationsRegistry(RegistryUseDenied):
    NAMESPACE: ClassVar[Namespace] = OperationsNS

    def _apply_to_get_root_value(self, apply_result: "IApplyResult", attr_name: AttrName) -> RegistryRootValue:
        raise EntityInternalError(owner=self, msg="Method should not be called")


@dataclass
class UnboundModelsRegistry(IThisRegistry, RegistryBaseWithStoreBase):
    """
    This is temporary registry used only in unbound mode.
    It has not predefined registry entries, rather will accept all new
    fields and register them in a store.
    Later this registry is replaced and this instance and store is rejected.
    """
    NAMESPACE: ClassVar[Namespace] = ModelsNS

    @staticmethod
    def is_unbound_models_registry() -> bool:
        return True

    def register_unbound_attr_node(self,
                                   component: Union[IField, IContainer],
                                   full_dexp_node_name: AttrName
                                   ) -> AttrDexpNodeForTypeInfo:
        type_info = component.get_type_info()
        # python_type = field.python_type
        # if not field.PYTHON_TYPE:
        #     raise EntityInternalError(owner=self, msg=f"Python type not yet set.")
        # type_info = TypeInfo.get_or_create_by_type(python_type)
        attr_node = AttrDexpNodeForTypeInfo(
            name=full_dexp_node_name,
            type_info=type_info,  # must be like this
            namespace=self.NAMESPACE,
            # type_object=type_info.type_,
        )
        store = self.get_store()
        store.register_attr_node(attr_node, alt_attr_node_name=None)
        return attr_node

    # ------------------------------------------------------------

    def _apply_to_get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RegistryRootValue:
        raise NotImplementedError()


@dataclass
class ModelsRegistry(RegistryBaseWithStoreBase):
    """
    All models have full path from top container (for better readability)
    e.g. M.address_set.city.street so reading begins from root
    instance, instead of apply_result.current_frame.instance.
    See get_root_value() implementation.
    """
    # == M.company.name case
    # # Caller should not fetch attribute by name from return value of
    # # get_root_value()
    # ROOT_VALUE_NEEDS_FETCH_BY_NAME: ClassVar[bool] = False

    root_attr_nodes: Dict[str, AttrDexpNodeForDataModel] = field(repr=False, init=False, default_factory=dict)

    # Just to check not duplicate. Can have several, first is main model, other are sub-models
    models_dict: Dict[str, ModelKlassType] = field(repr=True, init=False, default_factory=OrderedDict)

    NAMESPACE: ClassVar[Namespace] = ModelsNS

    # def __post_init__(self):
    #     super().__post_init__()

    # NOTE: no register() method due complex logic - see
    #       ContainerBase._register_data_model()

    def _create_root_attr_node(self, data_model: IDataModel) -> AttrDexpNodeForDataModel:
        """
        Models specific method
        """
        # standard DTO class attr_node
        # if not data_model.type_info:
        #     data_model.set_type_info()
        # assert data_model.type_info.type_==model
        attr_node = AttrDexpNodeForDataModel(
                        name=data_model.name,
                        data_model=data_model,
                        namespace=self.NAMESPACE,
                        # type_info=data_model.get_type_info()
                    )
        return attr_node

    def register_all_nodes(self, root_attr_node: Optional[AttrDexpNodeForDataModel],
                           data_model: IDataModel,
                           model_klass: ModelKlassType,
                           unbound_mode: bool = False):
        """
        Models specific method
        """
        if not root_attr_node:
            root_attr_node = self._create_root_attr_node(data_model=data_model)

        if data_model.name in self.models_dict:
            raise EntityInternalError(owner=self,
                    msg=f"Model {data_model.name} already set {self.models_dict[data_model.name]}, got: {model_klass}")
        self.models_dict[data_model.name] = model_klass

        name = data_model.get_full_name(init=unbound_mode)
        is_root = "." not in name  # TODO: hack
        if name in self.root_attr_nodes:
            raise EntityInternalError(owner=self,
                    msg=f"Duplicate {name} -> {self.root_attr_nodes[name]}, already set, failed to set: {root_attr_node}")

        self.root_attr_nodes[name] = root_attr_node

        # Fetch and check type_info
        type_info = data_model.get_type_info()
        type_info_from_model = TypeInfo.get_or_create_by_type(model_klass)
        if not type_info_from_model.type_ == type_info.type_:
            raise EntityInternalError(owner=self, msg=f"Model type info inner type <> data_model's: {type_info} <> {type_info_from_model}")

        # company.business_type.name --> company__business_types__name
        name_for_reg = name.replace(".", "__")

        # --------------------------------------------------
        # Register attributes
        # --------------------------------------------------
        # Register M.<all-fields>
        store = self.get_store()
        for attr_name in get_model_fields(model_klass):
            attr_node = self._create_attr_node_for_model_attr(model_klass, attr_name)
            alt_attr_node_name = None if is_root else f"{name_for_reg}__{attr_name}"
            store.register_attr_node(attr_node, alt_attr_node_name=alt_attr_node_name)

        # Register M.Instance - can be used for Functions(arguments)
        # Exception: allowed to access: .Instance -> apply_session.current_frame.instance
        store = self.get_store()
        store.register_special_attr_node(attr_name=ReservedAttributeNames.INSTANCE_ATTR_NAME,
                                         component=None,
                                         attr_name_prefix=None if is_root else f"{name_for_reg}__",
                                         type_info=type_info)

    # ------------------------------------------------------------

    def get_attr_node_by_data_model(self,
                                    data_model: IDataModel,
                                    # default:[None, UndefinedType]=UNDEFINED,
                                    # strict:bool=False
                                    ) -> Union[IAttrDexpNode, None, UndefinedType]:
        """
        Models specific method
        """
        # attr_node_name = data_model.name
        # == M.name mode
        name = data_model.get_full_name()

        # company.business_type.name --> company__business_types__name
        name_for_reg = name.replace(".", "__")

        if name_for_reg not in self.root_attr_nodes:
            raise EntityInternalError(owner=self, msg=f"Name not found {name_for_reg} in {self.root_attr_nodes.keys()}")
        return self.root_attr_nodes[name_for_reg] 

        # == M.company mode
        # allways in models
        # attr_node_name = data_model.name
        # assert attr_node_name
        # # return self.get_store().get(attr_node_name, default)
        # return self.get_store()[attr_node_name]

    # ------------------------------------------------------------

    def _apply_to_get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RegistryRootValue:
        # ROOT_VALUE_NEEDS_FETCH_BY_NAME = False

        # NOTE: instance can be different from apply_result.current_frame.value_node.instance
        #       in update from instance_new mode (it is set in use_stack() call).
        instance = apply_result.current_frame.instance

        # data_model = apply_result.current_frame.container.data_model
        data_model_root = apply_result.current_frame.data_model_root

        expected_type = data_model_root.type_info.type_ \
                        if isinstance(data_model_root.model_klass, DotExpression) \
                        else data_model_root.model_klass

        if instance is None:
            if not data_model_root.type_info.is_optional:
                raise EntityInternalError(owner=data_model_root, msg="Got None and type is not 'Optional'")
        else:
            if data_model_root.type_info.is_list and isinstance(instance, (list, tuple)):
                # NOTE: can be list and only a single item
                # TODO: only first is checked - maybe should test all
                instance_to_test = instance[0] if instance else None
            else:
                instance_to_test = instance

            # == M.name case
            # TODO: caller should check expected type ...
            if not apply_result.entity.is_unbound() \
              and not apply_result.instance_none_mode \
              and instance_to_test:
                component = apply_result.current_frame.component
                component._accessor.validate_instance_type(
                        owner_name=f"{component.name} -> {self}.{attr_name}",
                        instance=instance_to_test,
                        model_klass=expected_type,
                )

        return RegistryRootValue(instance, None)

# ------------------------------------------------------------

@dataclass
class LocalFieldsRegistry(RegistryBase):
    """
    Created on a per container base.
    Sees only children of a container.
    Uses container._attr_dexp_node_store for
    Attribute DotExpression node store (setup phase),
    and in apply phase gets  from ValueNode.
    Stores attr dexp node in component too, along with special arguments.
    """
    NAMESPACE: ClassVar[Namespace] = FieldsNS
    # fallback to check in TopFieldRegistry if attribute name is available
    CALL_DEXP_NOT_FOUND_FALLBACK: ClassVar[bool] = True
    ALLOWED_BASE_TYPES: ClassVar = (IField, IContainer, IFieldGroup)

    container: IContainer = field(repr=False)
    top_fields_registry: "TopFieldsRegistry" = field(repr=False)

    # ------------------------------------------------------------

    def get_store(self) -> Union[AttrDexpNodeStoreForComponent, UndefinedType]:
        """
        Use component/container store
        """
        return self.container.get_attr_dexp_node_store()

    def register_all(self):
        """
        container.components - flattened list of descendants - traversed tree of components (recursion)
            but does not go deeper for SubEntity* children.
        # TODO: same attr_dexp_node will be created here and in TopFieldsRegistry.register_fields_of_container()
        #       cache in TopFieldsRegistry and reuse here.
        """
        # A.3. COMPONENTS - collect attr_nodes - previously flattened (recursive function fill_components)
        # container_id = self.container.container_id
        # parent_dict: Dict[ComponentName, IComponent] = {}
        # if self.container.name == "D": print("here33")
        if not self.container.is_entity():
            raise EntityInternalError(owner=self, msg=f"This method sets up for all Component's dexp-stores and therefore for all LocalFieldsRegistry-s")

        store = self.get_store()
        store.setup_all_nodes(setup_session=self.setup_session)


    def dexp_not_found_fallback(self, owner: IComponent, full_dexp_node_name: AttrName) -> Union[IAttrDexpNode, UndefinedType]:
        """
        when field name is not found in self.get_store() (local store)
        tries to find if attribute is in TopFieldsRegistry store and is visible.
        returns:
            - UNDEFINED if does not find or is not visible
            - otherwise return IAttrDexpNode.
        """

        # NOTE: this is now allowed, e.g. F.access (on .access.some-field) - will use fallback to target node's container.
        # if full_dexp_node_name == self.container.container_id:
        #     raise EntitySetupNameError(owner=owner,
        #             msg=f"Namespace '{self.NAMESPACE}': Attribute name '{full_dexp_node_name}' matches current container name. Use F.{full_dexp_node_name} instead.")

        container_attr_dexp_node_pair_list: Optional[List[ContainerAttrDexpNodePair]] = \
            self.top_fields_registry.cont_attr_dexp_pair_store.get(full_dexp_node_name, default=None)
        if container_attr_dexp_node_pair_list is None:
            return UNDEFINED

        containers_dict = self.container.entity.containers_dict
        my_containers_id_path = self.container.containers_id_path
        assert my_containers_id_path
        my_containers_id = my_containers_id_path[-1]
        candidates: List[Tuple[IAttrDexpNode, AttrValueContainerPath]] = []

        for container_attr_dexp_node_pair in container_attr_dexp_node_pair_list:
            ho_container_id = container_attr_dexp_node_pair.container_id
            holder_container: IContainer = containers_dict.get(ho_container_id, UNDEFINED)
            assert holder_container

            # TODO: if this block will become problematic - put in plain utils function and make unit tests on it
            #       find path from one container to another
            path_up: Optional[List[ContainerId]] = None
            path_down: Optional[List[ContainerId]] = None

            # orig: for idx, ho_container_id_bit in enumerate(holder_container.containers_id_path, 0):
            # orig:     if not (idx < len(my_containers_id_path) and ho_container_id_bit == my_containers_id_path[idx]):
            len_my = len(my_containers_id_path)
            len_ho = len(holder_container.containers_id_path)
            max_len = max([len_my, len_ho])
            for idx in range(max_len):
                ho_container_id_bit = holder_container.containers_id_path[idx] if idx < len_ho else UNDEFINED
                my_container_id_bit = my_containers_id_path[idx] if idx < len_my else UNDEFINED
                assert not (ho_container_id_bit is UNDEFINED and my_container_id_bit is UNDEFINED), "should not happen"
                if ho_container_id_bit != my_container_id_bit:
                    # break on first different node - this is junction of two paths
                    if idx == 0:
                        raise EntityInternalError(owner=self,
                                msg=f"First node must match (entity) - Entity references itself? LocalFieldsRegistry should have done this")
                    path_up = list(reversed(my_containers_id_path[idx-1:])) if my_container_id_bit is not UNDEFINED else [my_containers_id]
                    path_down = holder_container.containers_id_path[idx:] if ho_container_id_bit is not UNDEFINED else []
                    break

            if not path_up:
                # The owner's container
                # NOTE: this is now allowed, e.g. F.access (on .access.some-field) - will use fallback to target node's container.
                #   raise EntityInternalError(owner=self, msg=f"Path up empty for: {owner}::{full_dexp_node_name} -> {ho_container_id} (down is: {path_down})")
                assert not path_down
                path_up = None
                path_down = None

            if path_down:
                # in path down - skip this if you find any SubEntityItems in the path
                subentity_items_in_path_down = [
                    container_id_in_path for container_id_in_path in path_down
                    if containers_dict[container_id_in_path].is_subentity_items()]

                # but if Container is to be found and there is only single sub-items container
                # and there is no other after -> it is visible since it is unique.
                # first operator after must operate with list.
                if len(subentity_items_in_path_down) == 1 \
                  and path_down[-1] == subentity_items_in_path_down[-1] \
                  and full_dexp_node_name == subentity_items_in_path_down[-1]:
                    subentity_items_in_path_down = None
            else:
                subentity_items_in_path_down = None

            ok = (not subentity_items_in_path_down)
            if ok:
                # Store path up and down
                attr_value_container_path = AttrValueContainerPath(
                    attr_name=full_dexp_node_name,
                    container_id_from=self.container.container_id,
                    container_id_to=ho_container_id,
                    container_node_mode=(ho_container_id == full_dexp_node_name),
                    path_up=path_up,
                    path_down=path_down,
                )
                candidates.append((container_attr_dexp_node_pair.attr_dexp_node,
                                   attr_value_container_path))

            # TODO: if settings.debug:
            # print(f"Container {self.container.container_id} refs {full_dexp_node_name} -> testing {ho_container_id} - found: up={path_up} + down={path_down}, items={subentity_items_in_path_down}, ok={ok}")

        if len(candidates) > 1:
            custom_msg_bits = ["{}{}{}".format("^".join(cand[1].path_up),
                                               "->" if cand[1].path_up and cand[1].path_down else "",
                                               ".".join(cand[1].path_down))
                               for cand in candidates]
            # TODO: when more candidates, try with Container.<field> ...
            custom_msg = f"Field name '{full_dexp_node_name}' is ambiguous, can not use it directly. " \
                         f"Found {len(custom_msg_bits)} candidates: {', '.join(custom_msg_bits)}."
            raise self.create_exception_name_not_found_error(owner=owner, full_dexp_node_name=full_dexp_node_name, custom_msg=custom_msg)

        if len(candidates) == 1:
            # found one unique field in some other visible container -> use it.
            # convert this node to a new one AttrDexpNodeWithValuePath:
            #   will be created with same input arguments + one extra (path info)
            attr_dexp_node_orig, attr_value_container_path = candidates[0]
            attr_dexp_node_w_value_path = attr_dexp_node_orig.copy(
                traverse=False,  # NOTE: hope this is not bad idea
                as_class=AttrDexpNodeWithValuePath,
                change=dict(attr_value_container_path=attr_value_container_path),
            )
            store = self.get_store()
            existing_attr_dexp_node = store.get(full_dexp_node_name, default=UNDEFINED)
            if existing_attr_dexp_node is not UNDEFINED:
                raise EntityInternalError(owner=self, msg=f"Value '{full_dexp_node_name}' already in the store: {existing_attr_dexp_node}?")
            store.set(full_dexp_node_name, attr_dexp_node_w_value_path, replace_when_duplicate=False)
            return attr_dexp_node_w_value_path

        return UNDEFINED

    def _apply_to_get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RegistryRootValue:
        value_node = apply_result.current_frame.value_node
        store = self.get_store()
        attr_dexp_node = store.get(attr_name, default=UNDEFINED)

        if attr_dexp_node is UNDEFINED:
            raise EntityInternalError(owner=self, msg=f"{attr_name} not found in store: {store}")

        if isinstance(attr_dexp_node, AttrDexpNodeWithValuePath):
            # complex case - field is not in current value_node container
            # - the setup phase has found it in some other container
            # - follow the path, first go up and then go down to locate the container
            # - in down path no SubentityItemsValueNode container should be found
            # TODO: logger.log() # print(f"{attr_name} -> up: {attr_dexp_node.attr_value_container_path.path_up}, down: {attr_dexp_node.attr_value_container_path.path_down}")
            # ------------------------------------------------------------
            # GO UP
            # ------------------------------------------------------------
            path_up = attr_dexp_node.attr_value_container_path.path_up[:]
            container_value_node_curr: Optional[IValueNode] = None
            while True:
                if not path_up:
                    break  # travel up ended
                container_id = path_up.pop(0)
                container_value_node_curr = (container_value_node_curr if container_value_node_curr else value_node).parent_container_node
                if isinstance(container_value_node_curr, SubentityItemsValueNode):
                    # when items, need to jump over - i.e. one level up
                    # SubentityItemsValueNode is intermediate node object that holds all items
                    container_value_node_curr = container_value_node_curr.parent_container_node

                if not (container_value_node_curr and container_value_node_curr.name == container_id):
                    raise EntityInternalError(owner=self, msg=f"Expected container_id={container_id}, got: {container_value_node_curr}")

            assert container_value_node_curr

            # ------------------------------------------------------------
            # GO DOWN
            # ------------------------------------------------------------
            path_down = list(attr_dexp_node.attr_value_container_path.path_down)

            # Fetching attribute from container node will be done later
            while True:
                if not path_down:
                    break  # travel up ended
                value_node_name = path_down.pop(0)
                if not hasattr(container_value_node_curr, "container_children"):
                    # SubentityItemsValueNode does not have this property
                    raise EntityInternalError(owner=self, msg=f"IValueNode is not ValueNode container {container_value_node_curr}")
                value_node_temp = container_value_node_curr.container_children.get(value_node_name, None)
                if not value_node_temp:
                    raise EntityInternalError(owner=self, msg=f"{value_node_name} not found, available for container {container_value_node_curr.name} are: {', '.join(container_value_node_curr.children.keys())}")
                container_value_node_curr = value_node_temp
            # fetch by name only if not container
            do_fetch_by_name = not attr_dexp_node.attr_value_container_path.container_node_mode
        else:
            # simple case - from current value_node -> the searched attribute should be in the same container
            assert isinstance(attr_dexp_node, IAttrDexpNode), attr_dexp_node
            container_value_node_curr = value_node.parent_container_node
            do_fetch_by_name = True

        return RegistryRootValue(value_root=container_value_node_curr, attr_name_new=None, do_fetch_by_name=do_fetch_by_name)

# -------------------------------------------------------------


@dataclass
class ContainerAttrDexpNodePair:
    container_id: ContainerId
    attr_dexp_node: IAttrDexpNode


@dataclass
class TopFieldsRegistry(RegistryBase):
    """
    Created for all fields, used in 2nd try to reach not-local fields.
    Checks upward and downward (stops on SubEntityItems).
    Not standard registry - stores only referenced fields.
    Stored within Entity.
    """

    NAMESPACE: ClassVar[Namespace] = FieldsNS

    entity: IEntity = field(repr=False)

    # TODO: typing is not good, AttrDexpNodeStore value's type does not support List[ContainerAttrDexpNodePair]
    #       orig. Dict[AttrName, List[ContainerAttrDexpNodePair]]
    cont_attr_dexp_pair_store: AttrDexpNodeStore = field(repr=False, init=False)

    def __post_init__(self):
        # super().__post_init__()
        self.cont_attr_dexp_pair_store = AttrDexpNodeStore(namespace=self.NAMESPACE)

    def get_store(self) -> Union[AttrDexpNodeStore, UndefinedType]:
        """
        For each attribute dict value is a list of container ID-s i stored (instead of standard AttrDexpNode-s).
        """
        return self.cont_attr_dexp_pair_store

    def _apply_to_get_root_value(self, apply_result: "IApplyResult", attr_name: AttrName) -> RegistryRootValue:
        raise EntityInternalError(owner=self, msg=f"Should no be called")

    def register_fields_of_container(self, container: IContainer):
        """
        container.components - flattened list of descendants - traversed tree of components (recursion)
            but does not go deeper for SubEntity* children.
        """
        # A.3. COMPONENTS - collect attr_nodes - previously flattened (recursive function fill_components)
        container_id = container.container_id
        for component_name, component in container.components.items():
            # TODO: same content attr_dexp_node will be created here and in LocalFieldsRegistry.register_all()
            #       Cache this one and reuse.
            if ((isinstance(component, IContainer) and component is not container)
              or not component.has_data()):
                # 1) container is in list of parent's components and in its own list. only last will be added.
                # 2) skip cleaners, data_models and similar - that holds no data
                continue

            attr_dexp_node = AttrDexpNodeStoreForComponent.create_attr_node(component)
            if not attr_dexp_node.denied:
                attr_dexp_pair = self.cont_attr_dexp_pair_store.get(attr_dexp_node.name, default=UNDEFINED)
                if attr_dexp_pair is UNDEFINED:
                    attr_dexp_pair = []
                    self.cont_attr_dexp_pair_store.set(attr_name=attr_dexp_node.name,
                                                       attr_dexp_node=attr_dexp_pair,
                                                       replace_when_duplicate=False)
                attr_dexp_pair.append(ContainerAttrDexpNodePair(container_id, attr_dexp_node))
        return


@dataclass
class SettingsKlassMember(KlassMember):
    settings_type: SettingsType


@dataclass
class ContextRegistry(RegistryBaseWithStoreBase):

    setup_settings: Settings = field(repr=False)
    apply_settings_class: Optional[Type[ApplySettings]] = field(repr=False)

    NAMESPACE: ClassVar[Namespace] = ContextNS

    attributes_dict: CustomCtxAttributeList = field(init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.setup_settings, Settings):
            raise EntitySetupValueError(owner=self, msg=f"setup_settings must be instance of Settings, got: {self.apply_settings_class}")

        if self.apply_settings_class is not None:
            if ApplySettings not in inspect.getmro(self.apply_settings_class):
                raise EntitySetupValueError(owner=self, msg=f"apply_settings_class should inherit ApplySettings, got: {self.apply_settings_class}")
        self.register_all_nodes()

    def register_all_nodes(self):
        """
        when there are several sources of attribute names,
        check in self.setup_settings._get_attribute_settings_source_list_pairs()
        by which order / priority which attribute source wins.
        """
        settings_source_list_pairs = self.setup_settings._get_attribute_settings_source_list_pairs(self.apply_settings_class)

        for attributes_dict, settings_source_list in settings_source_list_pairs:

            for attr_name, attr_getter in attributes_dict.items():
                if not isinstance(attr_getter, IAttribute):
                    raise EntitySetupValueError(owner=self,
                                                msg=f"Attribute {attr_name} expected FieldName or MethodName instance, got: {attr_getter} / {type(attr_getter)}")
                settings_source = attr_getter.setup_dexp_attr_source(settings_source_list)
                klass_member = SettingsKlassMember(settings_type=settings_source.settings_type,
                                                   klass=settings_source.model_klass,
                                                   attribute=attr_getter)
                # NOTE: No problem with override any more!
                #           if attr_name in self.get_store():
                #               raise EntitySetupNameError(f"Attribute name '{attr_name}' is reserved. Rename class attribute in '{self.apply_settings_class}'")

                attr_node = AttrDexpNodeForKlassMember(
                    name=attr_name,
                    namespace=self.NAMESPACE,
                    # attribute=attr_getter,
                    # type_info=type_info,
                    klass_member=klass_member,
                )
                store = self.get_store()
                store.register_attr_node(attr_node=attr_node, alt_attr_node_name=attr_name, replace_when_duplicate=True)

    def create_node(self,
                    dexp_node_name: str,
                    owner_dexp_node: Optional[IDotExpressionNode],
                    owner: IComponent,
                    # is_1st_node: bool,
                    ) -> IDotExpressionNode:

        if not isinstance(owner, IComponent):
            raise EntityInternalError(owner=self, msg=f"Owner needs to be Component, got: {type(owner)} / {owner}")

        return super().create_node(
            dexp_node_name=dexp_node_name,
            owner_dexp_node=owner_dexp_node,
            # is_1st_node=is_1st_node,
            owner=owner,
        )

    def _apply_to_get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RegistryRootValue:
        store = self.get_store()
        attr_dexp_node = store.get(attr_name, default=UNDEFINED)
        if attr_dexp_node is UNDEFINED:
            avail_names = get_available_names_example(attr_name, list(store.keys()))
            raise EntityApplyNameError(owner=self, msg=f"Invalid attribute name '{attr_name}', available: {avail_names}.")

        # attr_dexp_node = store[attr_name]
        assert isinstance(attr_dexp_node, AttrDexpNodeForKlassMember)
        klass_member = attr_dexp_node.klass_member

        if not isinstance(klass_member, SettingsKlassMember):
            raise EntityInternalError(owner=self, msg=f"Expected SettingsClassMember instance, got: {klass_member}")

        # can be method name or field name
        attr_name_new = klass_member.attribute
        if klass_member.settings_type == SettingsType.APPLY_SETTINGS:
            value_root = apply_result.settings
        elif klass_member.settings_type == SettingsType.SETUP_SETTINGS:
            value_root = apply_result.entity.settings
        else:
            raise EntityInternalError(owner=self, msg=f"Invalid settings type: got: {klass_member.settings_type}")

        return RegistryRootValue(value_root, attr_name_new)

# ------------------------------------------------------------


@dataclass
class ThisRegistryBase(IThisRegistry, RegistryBaseWithStoreBase):
    """
    General idea/overview:
                                                Field   bool-ena field-gr subent-sin subent-lst FuncArg: Filter(T.) FuncArg: T.attr
                                                ------- -------- -------- ---------- ---------- ------------------- ---------------
    a) .<field-name>     - direct               -       yes      yes      yes        yes (*1)   - (see f)           yes- if Item
    b) This.Value                               yes     yes      -        -          -          -                   yes- if std.
    c) This.Instance     - do I need it???      -       -        -        yes        -- (*2)    yes(Self)           yes- if Item?
    d) This.Items: List[Item]                   -       -        -        -          yes        - (see g)           -
    e) This.Children: ListChildField            -       yes      yes      yes        yes (*1)   -                   -

    f) This.Item.<field-name>                   -       -        -        -          -          yes(*3)             -
    g) This.Item.Children                       -       -        -        -          -          yes(*3)             -

    (*1) - in which case to put this? - if this is ok, then put in first too
    (*2) - maybe InstanceList
    (*3) - the question: how to make This.Item work? maybe simple as done in a) i e)
    """
    # attr_node: Optional[IAttrDexpNode] = field(default=None)
    # component: Optional[IComponent] = field(default=None)
    # model_klass: Optional[ModelKlassType] = field(default=None)

    # it is not overlapping with is_items_mode:
    #   - is_items_for_each_mode - applies to each Item from .Items
    #     (used for ItemsFunctions when Item attribute will be
    #      available, e.g. This.name -> AttrValue)
    #   - is_items - apply to .Items : List[Item]
    is_items_for_each_mode: bool = field(default=False)

    # --- autocomputed ---
    # This.Items -> List[Item]
    is_items_mode: bool = field(init=False, default=False)

    def _apply_to_get_root_value(self, apply_result: "IApplyResult", attr_name: AttrName) -> RegistryRootValue:
        raise EntityInternalError(owner=self, msg="Should not be called.")


@dataclass
class ThisRegistryForModelKlass(ThisRegistryBase):
    """
    Used for:
        M.    - used for bind_to
        This. - DataModel function arguments in handler functions
        This. - function arguments in general when ModelKlass is current node
        This. - choices / Enum
    """
    NAMESPACE: ClassVar[Namespace] = ThisNS

    model_klass: ModelKlassType = field(default=None)

    # --- autocomputed:
    # attr_name: Optional[str] = field(init=False, repr=False, default=None)
    model_klass_type_info: Optional[TypeInfo] = field(default=None)

    @classmethod
    def create(cls,
               setup_session: ISetupSession,
               model_klass: ModelKlassType,
               # is_items_for_each_mode: bool = False,
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
        this_registry = ThisRegistryForModelKlass(model_klass=model_klass)  # is_items_for_each_mode=is_items_for_each_mode)
        this_registry.setup(setup_session=setup_session)
        this_registry.finish()

        return this_registry

    def __post_init__(self):
        super().__post_init__()

        if not self.model_klass:
            raise EntitySetupTypeError(owner=self, msg=f"Argument 'model_klass' is obligatory.")

        self.model_klass_type_info = TypeInfo.get_or_create_by_type(self.model_klass)
        if self.is_items_for_each_mode:
            if not (self.model_klass and self.model_klass_type_info.is_list):
                raise EntitySetupTypeError(owner=self, msg=f"is_items_for_each_mode needs to operate on List[Any], got: {self.model_klass}")
        elif self.model_klass_type_info.is_list:
            self.is_items_mode = True

    def setup(self, setup_session: ISetupSession) -> None:
        super().setup(setup_session)

        store = self.get_store()
        if inspect.isclass(self.model_klass) and issubclass(self.model_klass, IValueNode):
            raise EntityInternalError(owner=self, msg=f"Did not expected ValueNode class, got: {self.model_klass}. Use ThisRegistryForComponent().")
        if not self.is_items_for_each_mode and self.is_items_mode:
            # This.Items
            store.register_special_attr_node(attr_name=ReservedAttributeNames.ITEMS_ATTR_NAME,
                                             component=None,
                                             type_info=self.model_klass_type_info)  # already a List
        else:
            # NOTE: Covers self.is_items_for_each_mode too

            # This.<all-attributes>
            self._register_model_nodes(model_klass=self.model_klass_type_info.type_)

            # This.Instance
            # Exception: allowed to access: .Instance -> apply_session.current_frame.instance
            # NOTE: not allowed for ValueNode-s, since .instance holds initial values only, only ValueNode-s
            #       hold current values. Additionally, .Instance would break ValueNode tree logic.
            store.register_special_attr_node(attr_name=ReservedAttributeNames.INSTANCE_ATTR_NAME,
                                             component=None,
                                             type_info=self.model_klass_type_info,
                                             attr_name_prefix=None)

    def _register_model_nodes(self, model_klass: ModelKlassType):
        if not is_model_klass(model_klass):
            raise EntitySetupValueError(owner=self, msg=f"Expected model class (DC/PYD), got: {type(model_klass)} / {to_repr(model_klass)} ")

        store = self.get_store()
        for attr_name in get_model_fields(model_klass):
            attr_node = self._create_attr_node_for_model_attr(model_klass, attr_name)
            store.register_attr_node(attr_node)


    def _apply_to_get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RegistryRootValue:
        if not self.finished:
            raise EntityInternalError(owner=self, msg="Setup not called")

        instance = apply_result.current_frame.instance
        # NOTE: can not use apply_result.current_frame.value_node, it is not set (nor value_node.instance)
        if not isinstance(instance, self.model_klass):
            raise EntityInternalError(owner=self, msg=f"Type of apply session's instance expected to be '{self.model_klass}', got: {instance}")

        if attr_name == ReservedAttributeNames.INSTANCE_ATTR_NAME:
            # with 2nd param == None -> do not fetch further
            attr_name_to_fetch = None
        else:
            # with 2nd param like this -> fetch further by attr_name
            attr_name_to_fetch = attr_name

        root_value = RegistryRootValue(value_root=instance, attr_name_new=attr_name_to_fetch)

        return root_value

    def __repr__(self):
        out: List[str] = []
        if self.model_klass:
            out.append(f"model={self.model_klass}")
        return f'{self.__class__.__name__}({", ".join(out)})'

    __str__ = __repr__


@dataclass
class ThisRegistryForComponent(ThisRegistryBase):
    """
    component - for Containers or FieldGroup or Fields
                with children (BooleanField+enables).
                Supports SubEntityItems -> items
    attr_node - for Fields (have final value)

    all combinations are supported except when both are empty

    TODO: extract ThisRegistryForItems if SubEntityItems.
    """
    NAMESPACE: ClassVar[Namespace] = ThisNS

    component: Optional[IComponent] = field(default=None)
    attr_node: Optional[IAttrDexpNode] = field(default=None)

    # autocomputed:
    has_children: bool = field(init=False, repr=False)
    attr_name: Optional[str] = field(init=False, repr=False, default=None)
    # model_klass_type_info: Optional[TypeInfo] = field(default=None)

    @classmethod
    def create(cls,
               setup_session: ISetupSession,
               component: IComponent,
               attr_node: Optional[IAttrDexpNode],
               ) -> Self:
        this_registry = ThisRegistryForComponent(attr_node=attr_node, component=component)
        this_registry.setup(setup_session=setup_session)
        this_registry.finish()

        return this_registry

    def __post_init__(self):
        super().__post_init__()

        if not self.component:
            raise EntitySetupTypeError(owner=self, msg=f"Argument component is obligatory.")

        self.has_children = self.component.has_children()

        if self.attr_node:
            if not isinstance(self.attr_node, IAttrDexpNode):
                raise EntitySetupValueError(owner=self, msg=f"Expected IAttrDexpNode, got: {type(self.attr_node)} / {self.attr_node}")
            self.attr_name = self.attr_node.name

        if self.is_items_for_each_mode:
            if not (self.component.is_subentity_items()):
                raise EntitySetupTypeError(owner=self, msg=f"is_items_for_each_mode needs to operate on List[Any], got: {self.component} ")
        else:
            if self.component.is_subentity_items():
                self.is_items_mode = True

    def setup(self, setup_session: ISetupSession) -> None:
        super().setup(setup_session)

        store = self.get_store()
        if self.attr_node:
            # This.Value == ReservedAttributeNames.VALUE_ATTR_NAME
            type_info = self.attr_node.get_type_info()
            store.register_special_attr_node(attr_name=ReservedAttributeNames.VALUE_ATTR_NAME,
                                             component=self.component,
                                             type_info=type_info)

        if self.has_children:
            if not self.is_items_for_each_mode and self.is_items_mode:
                # This.Items: List[component_fields_dataclass]
                component_fields_dataclass, _ = self.component.get_component_fields_dataclass(
                    setup_session=setup_session)
                py_type_hint = List[component_fields_dataclass]
                type_info = TypeInfo.get_or_create_by_type(py_type_hint)
                store.register_special_attr_node(attr_name=ReservedAttributeNames.ITEMS_ATTR_NAME,
                                                 component=self.component,
                                                 type_info=type_info)
            else:
                # NOTE: Includes self.is_items_for_each_mode too
                # This.<all-attribute> + This.Children: ListChildField
                self._register_all_children(
                    setup_session=setup_session,
                    component=self.component,
                )

    def _apply_to_get_root_value(self, apply_result: IApplyResult, attr_name: AttrName) -> RegistryRootValue:
        if not self.finished:
            raise EntityInternalError(owner=self, msg="Setup not called")

        root_value = None

        if self.attr_node and attr_name == ReservedAttributeNames.VALUE_ATTR_NAME.value:
            # run only if both match - otherwise will report unknown attribute .Value

            # NOTE: with 2nd return value -> instead of fetching .Value, instruct caller to
            #       fetch component's bound attribute.
            assert self.attr_name == apply_result.current_frame.value_node.name
            root_value = RegistryRootValue(value_root=apply_result.current_frame.value_node,
                                           attr_name_new=None,
                                           do_fetch_by_name=False)

        elif self.has_children:
            value_node = apply_result.current_frame.value_node

            # TODO: root_value could be already defined, should I put "elif ..." instead?
            if self.is_items_mode:
                # multiple items case
                if attr_name == ReservedAttributeNames.ITEMS_ATTR_NAME.value:
                    if not value_node.is_list():
                        raise EntityApplyTypeError(f"Items expected, got: {value_node}")

                    # assert isinstance(value_node, list), value_node
                    # value_node = apply_result.current_frame.value_node.get_self_or_items()

                    root_value = RegistryRootValue(value_root=value_node, attr_name_new=None, do_fetch_by_name=False)

                    # if not isinstance(apply_result.current_frame.instance, (list, tuple)):
                    #     raise EntityInternalError(f"Items expected, got: {apply_result.current_frame.instance}")
                    # root_value = RegistryRootValue(
                    #     value_root=apply_result.current_frame.instance,
                    #     attr_name_new=None,
                    #     do_fetch_by_name=False)

                elif attr_name == ReservedAttributeNames.CHILDREN_ATTR_NAME.value:
                    # if isinstance(apply_result.current_frame.instance, (list, tuple)):
                    #   raise EntityInternalError(f"Single item expected, got: {apply_result.current_frame.instance}")
                    if isinstance(value_node.is_list()):
                        raise EntityInternalError(f"Single item expected, got: {value_node}")

                    # if not isinstance(apply_result.current_frame.component.child_field_list, (list, tuple)):
                    if not isinstance(value_node.component.child_field_list, (list, tuple)):
                        raise EntityInternalError(owner=apply_result.current_frame.component,
                                                  msg=f"child_field_list not a list, got: {value_node.component.child_field_list}")
                    # TODO: .Children?
                    root_value = RegistryRootValue(
                        value_root=value_node.component.child_field_list,
                        attr_name_new=None,
                        do_fetch_by_name=False,
                    )
                else:
                    raise EntityInternalError(owner=self,
                                              msg=f"Expected attribute name: {ReservedAttributeNames.ITEMS_ATTR_NAME.value} or {ReservedAttributeNames.CHILDREN_ATTR_NAME.value} , got: {attr_name}")
            else:
                # single item component
                if value_node.is_list():
                    raise EntityApplyTypeError(f"Expected single item, got list instead, got: {value_node}")

                if attr_name == ReservedAttributeNames.CHILDREN_ATTR_NAME:
                    # with 2nd param == None -> do not fetch further
                    # attr_name_to_fetch = None
                    raise NotImplementedError()
                else:
                    # with 2nd param like this -> fetch further by attr_name
                    attr_name_to_fetch = attr_name

                root_value = RegistryRootValue(value_root=value_node, attr_name_new=attr_name_to_fetch)

        return root_value

    def __repr__(self):
        out: List[str] = []
        if self.attr_node:
            out.append(f"attr={self.attr_node.name}")
        if self.has_children:
            out.append(
                f"{'items' if self.is_items_mode else 'component'}="
                f"{self.component.__class__.__name__}({self.component.name})")
        else:
            out.append(
                f"{self.component.__class__.__name__}({self.component.name})/no-children")
        return f'{self.__class__.__name__}({", ".join(out)})'

    __str__ = __repr__

# ============================================================
# OBSOLETE
# ============================================================
# component = apply_result.current_frame.component
# instance = apply_result.current_frame.instance
# top_attr_accessor = ComponentAttributeAccessor(component=component, instance=instance, value_node=value_node)
# return RegistryRootValue(top_attr_accessor, None)
