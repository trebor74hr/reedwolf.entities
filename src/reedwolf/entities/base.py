# from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from contextlib import AbstractContextManager
from enum import Enum
from typing import (
    List,
    Any,
    Dict,
    Union,
    Optional,
    Tuple,
    ClassVar,
    Callable,
    Type,
)
from dataclasses import (
    dataclass,
    field,
    field as DcField,
    fields,
    MISSING as DC_MISSING,
    make_dataclass,
    asdict,
    replace as dataclass_clone,
)
from types import (
    MappingProxyType,
)

from .namespaces import (
    MultiNamespace,
    Namespace,
    FieldsNS,
)
from .utils import (
    snake_case_to_camel,
    to_repr,
    add_yaml_indent_to_strlist,
    YAML_INDENT,
    varname_to_title,
    UNDEFINED,
    UndefinedType,
    get_available_names_example,
    DumpFormatEnum,
    dump_to_format,
    NOT_APPLIABLE, MISSING, )
from .exceptions import (
    EntityInternalError,
    EntitySetupTypeError,
    EntitySetupError,
    EntitySetupNameError,
    EntitySetupValueError,
    EntitySetupNameNotFoundError,
    EntityApplyError,
    EntityInitError,
)
from .dynamic_attrs import DynamicAttrsBase
from .meta_dataclass import (
    ReedwolfDataclassBase,
    ComponentStatus,
    MAX_RECURSIONS,
    Self,
)
from .meta import (
    MetaTree,
    ComponentNameType,
    ComponentTreeWValuesType,
    NoneType,
    ModelField,
    TypeInfo,
    LiteralType,
    get_model_fields,
    is_function,
    is_method_by_name,
    is_model_klass,
    ModelKlassType,
    extract_model_field_meta,
    extract_py_type_hints,
    STANDARD_TYPE_LIST,
    TransMessageType,
    get_dataclass_fields,
    KeyString,
    AttrName,
    AttrValue,
    Index0Type,
    KeyType,
    ModelInstanceType,
    ContainerId,
    IDexpValueSource,
    ChildField,
    ListChildField,
    get_enum_values, MessageType,
)
from .expressions import (
    DotExpression,
    ExecResult,
    IDotExpressionNode,
    ISetupSession,
    IThisRegistry,
    IAttrDexpNodeStore,
    DexpValidator,
)
from .settings import (
    Settings,
    ApplySettings,
)
from .value_accessors import (
    IValueAccessor,
    ValueAccessorInputType,
)

# ------------------------------------------------------------

DEXP_PREFIX: str = "DEXP::"

DTO_STRUCT_CHILDREN_SUFFIX: str = "_children"

def warn(message: str):
    print(f"WARNING: {message}")  # noqa: T001


def repr_obj(obj, limit=100):
    out = str(obj)
    if len(out) > limit-3:
        out = out[:limit-3] + "..."
    return out



def obj_to_strlist(obj, path:Optional[List]=None):
    if path is None:
        path = []
    return obj.to_strlist(path) if not isinstance(obj, DynamicAttrsBase) and getattr(obj, "to_strlist", None) else [str(obj)]


def list_to_strlist(args, before, after):
    out = []
    if len(args) == 0:
        out.append(f"{before}{after}")
    elif len(args) == 1:
        vstr = obj_to_strlist(args[0])
        out.append(f"{before}{vstr[0]}{after}")
    else:
        out.append(before)
        for v in args:
            out.extend(add_yaml_indent_to_strlist(obj_to_strlist(v)))
        out.append(after)
    return out


def get_name_from_bind(bind_to: DotExpression):
    if len(bind_to.Path) <= 2:
        # Dexpr(Person.name) -> name
        name = bind_to._name
    else:
        # Dexpr(Person.address.street) -> address.street
        # TODO: this is messy :( - should be one simple logic ...
        name = "__".join([bit._name for bit in bind_to.Path][1:])
    assert name
    return name


# ------------------------------------------------------------


class ReservedAttributeNames(str, Enum):

    # Manual setup cases only
    # -----------------------
    # Applies to model instances, e.g. This.Instance <= Company()
    # see ChoiceField / Function arguments etc.
    INSTANCE_ATTR_NAME = "Instance" 

    # Field components
    # ----------------
    # Applies to model instance's attributes, e.g. This.Value <= Company().name
    VALUE_ATTR_NAME = "Value" 

    # Containers (Entity/SubEntity) + FieldGroup + Field with children (BooleanField)
    # -------------------------------------------------------------------------------------
    # Applies to components which have children (list of components, e.g. in
    # contains)
    CHILDREN_ATTR_NAME = "Children" 

    # SubEntityItems 
    # ---------------------------------------------------------------------------
    # Applies to components which can have list of items (SubEntities)
    ITEMS_ATTR_NAME = "Items" 

    # every Component which has Parent
    PARENT_ATTR_NAME = "Parent"

    #   CHILDREN_KEY = "__children__" 

    # TODO: traversing?:
    #   PARENT_PATH_ATTR_NAME = "ParentPath" 
    #   ITEMS_TREE_ATTR_NAME = 
    #   CHILDREN_TREE_ATTR_NAME = "ParentPath" 
    #   VALUE_TREE_ATTR_NAME = 


RESERVED_ATTRIBUTE_NAMES = get_enum_values(ReservedAttributeNames)

# ------------------------------------------------------------

# class ReservedArgumentNames(str, Enum):
#     # INJECT_COMPONENT_TREE = "inject_component_tree"
#     ...

# ------------------------------------------------------------
# Subcomponent
# ------------------------------------------------------------
@dataclass
class Subcomponent:
    # TODO: strange class - check if really required or explain if yes
    name:       str # orig: dexp_node_name
    path:       str # orig: var_path
    # TODO: can be some other types too
    component:  Union["IComponent", DotExpression]
    th_field:   Optional[ModelField]


    def __post_init__(self):
        # component can have LiteralType
        if not (self.name and self.path and self.th_field):
            raise EntityInternalError(owner=self, msg=f"name={self.name}, path={self.path}, subcomp={self.component}, th_field={self.th_field}")

        # TODO: strange - DotExpression in (None, UNDEFINED) returns True??
        if not bool(self.component) and self.component in (None, UNDEFINED):
            raise EntityInternalError(owner=self, msg=f"name={self.name}, path={self.path}, subcomp={self.component}, th_field={self.th_field}")

        # TODO: list all types available and include this check
        # if not isinstance(self.component, (IComponent, DotExpression)):
        #     raise EntityInternalError(owner=self, msg=f"wrong type of subcomponent {type(self.component)} / {self.component}")


# ------------------------------------------------------------
# IComponentFields
# ------------------------------------------------------------

class IComponentFields:
    """
    Used only for dynamically created dataclass classes that holds only fields from owner component's.
    See make_component_fields_dataclass()
    """
    ...


def make_component_fields_dataclass(class_name: str, child_field_list: ListChildField) -> Type[IComponentFields]:
    """ 
    Dynamically create Dataclass from ListChildField based on origin model.
    It is basedd for SubentitySingle and FieldGroup to create TypeInfo instance
    based only on the fields for which there are Field
    """
    # name, type, optional[field]
    # ('z', int, field(default=5))],
    children_fields = [(child_field.Name, child_field._type_info.py_type_hint)
                        for child_field in child_field_list]
    new_dataclass: Type = make_dataclass(
                cls_name=class_name, 
                fields=children_fields, 
                bases=(IComponentFields,))
    assert issubclass(new_dataclass, IComponentFields)
    return new_dataclass

# ------------------------------------------------------------


@dataclass
class AttrDexpNodeStore(IAttrDexpNodeStore):
    """
    The extraction of store from Registry was necessary, since LocalFieldsRegistry is assigned to containers only
    and store-s are assigned to each component.
    TODO: inherit standard 'dict' and remove get/set - to gain performance
    """
    namespace: Namespace = field(repr=True)

    # automatic
    _store: Dict[AttrName, "IAttrDexpNode"] = field(init=False, repr=False, default_factory=dict)
    _finished: bool = field(init=False, repr=False, default=False)

    def setup(self, setup_session: ISetupSession):
        ...

    def finish(self):
        if self._finished:
            raise EntityInternalError(owner=self, msg=f"store already finished")
        self._finished = True

    def get_items(self) -> List[Tuple[AttrName, "IAttrDexpNode"]]:
        return self._store.items()

    # def __getitem__(self, attr_name: AttrName, default=MISSING) -> "IAttrDexpNode":
    #     return self.get(attr_name=attr_name, strict=True, default=default)

    def __len__(self) -> int:
        return len(self._store)

    def get(self, attr_name: AttrName, default=MISSING) -> "IAttrDexpNode":
        # if strict and not self._finished:
        #     raise EntityInternalError(owner=self, msg=f"store not finished")
        dexp_node = self._store.get(attr_name, default)
        if default is MISSING:
            raise KeyError(attr_name)
        return dexp_node

    def set(self, attr_name: AttrName, attr_dexp_node: "IAttrDexpNode", replace_when_duplicate: bool):
        # if hasattr(self, "component") and self.component.name == "A" and attr_dexp_node.name == "Instance": print("here33")
        if self._finished:
            raise EntityInternalError(owner=self, msg=f"store finished")
        if not replace_when_duplicate and attr_name in self._store:
            raise EntitySetupNameError(owner=self, msg=f"IAttrDexpNode '{attr_dexp_node}' does not have unique name '{attr_name}' within this registry, found: {self._store[attr_name]}")
        self._store[attr_name] = attr_dexp_node

    # ------------------------------------------------------------

    def register_attr_node(self,
                           attr_node: "IAttrDexpNode",
                           replace_when_duplicate: bool = False,
                           alt_attr_node_name: Optional[AttrName] = None,
                           ):
        # --------------------------------------------------
        """
        ex. def _register_dexp_node(self, dexp_node:IDotExpressionNode,
                                    alt_dexp_node_name=None,
                                    replace_when_duplicate:bool = False):
        Data can register IFunctionDexpNode-s instances since the
        output will be used directly as data and not as a function call.
            function=[Function(name="Countries", title="Countries",
                              py_function=CatalogManager.get_countries)],
            ...
            available=(S.Countries.name != "test"),
        """
        # TODO: resolve this local import
        from reedwolf.entities.expr_attr_nodes import IAttrDexpNode

        attr_dexp_node_store: AttrDexpNodeStore = self

        if self._finished:
            raise EntityInternalError(owner=self, msg=f"Register({attr_node}) - already finished, adding not possible.")

        if not isinstance(attr_node, IAttrDexpNode):
            raise EntityInternalError(f"{type(attr_node)}->{attr_node}")

        if isinstance(attr_node.namespace, MultiNamespace):
            # namespace checked later again - on access
            # attr_node.namespace.validate_namespace(owner=self, namespace=self.NAMESPACE, is_1st_node=NOT_APPLIABLE)
            pass
        else:
            # TODO: REMOVE_THIS - test this too and later remove check:
            # if attr_node.namespace in (FieldsNS, ThisNS):
            if attr_node.namespace in (FieldsNS,):
                raise EntityInternalError(owner=self, msg=f"FieldsNS should be handled with MultiNamespace case")

            # TODO: consider if I need this check?
            # if self.NAMESPACE != attr_node.namespace:
            #     raise EntityInternalError(owner=self, msg=f"Method register({attr_node}) - namespace mismatch: {self.NAMESPACE} != {attr_node.namespace}")

        if not isinstance(attr_node, IDotExpressionNode):
            raise EntityInternalError(f"{type(attr_node)}->{attr_node}")
        dexp_node_name = alt_attr_node_name if alt_attr_node_name else attr_node.name

        if not dexp_node_name.count(".") == 0:
            raise EntityInternalError(owner=self, msg=f"Node {dexp_node_name} should not contain . - only first level vars allowed")

        # if attr_dexp_node_store is None:
        #     attr_dexp_node_store = self.get_store()

        # single place to register node
        attr_dexp_node_store.set(dexp_node_name,
                                 attr_dexp_node=attr_node,
                                 replace_when_duplicate=replace_when_duplicate)

    # ------------------------------------------------------------

    def register_special_attr_node(self,
                                    attr_name: ReservedAttributeNames,
                                    # None for model_klass
                                    component: Optional["IComponent"],
                                    type_info: TypeInfo,
                                    attr_name_prefix: Optional[str] = None,
                                    th_field: Optional[Any] = None,
                                    ) -> "IAttrDexpNode":

        from reedwolf.entities.expr_attr_nodes import AttrDexpNodeForComponent, AttrDexpNodeForTypeInfo

        # NOTE: removed restriction - was too strict
        # if not (is_model_klass(model_klass) or model_klass in STANDARD_TYPE_LIST or is_enum(model_klass)):
        #     raise EntitySetupValueError(owner=self, msg=f"Expected model class (DC/PYD), got: {type(model_klass)} / {model_klass} ")

        if th_field and not (inspect.isclass(th_field) and issubclass(th_field, IComponentFields)):
            raise EntitySetupValueError(owner=self, msg=f"Expected th_field is IComponentFields, got: {type(th_field)} / {th_field} ")

        # type_info = TypeInfo.get_or_create_by_type(py_type_hint=model_klass)
        # model_klass: ModelKlassType = type_info.type_

        for_list: bool = (attr_name == ReservedAttributeNames.ITEMS_ATTR_NAME)
        attr_name:str = attr_name.value

        if attr_name_prefix:
            attr_name = f"{attr_name_prefix}{attr_name}"

        if component:
            attr_node = AttrDexpNodeForComponent(
                name=attr_name,
                component=component,
                namespace=self.namespace,
                for_list=for_list,
                # type_info=type_info,
                # type_object=th_field,
            )
        else:
            attr_node = AttrDexpNodeForTypeInfo(
                name=attr_name,
                type_info=type_info,
                namespace=self.namespace,
                for_list=for_list,
                # type_info=None,
                # type_object=th_field,
            )

        self.register_attr_node(attr_node)
        return attr_node

# ------------------------------------------------------------


@dataclass
class AttrDexpNodeStoreForComponent(AttrDexpNodeStore):
    component: "IComponent" = field(repr=True)

    # automatic
    namespace: Namespace = field(repr=True, init=False, default=FieldsNS)

    def _register_special_nodes(self, component: "IComponent", setup_session: ISetupSession):
        if component is not self.component:
            raise EntityInternalError(owner=self, msg=f"{component} != {self.component}")
        component_parent = component.parent

        # skip case when container is a child. Containers should register its own special fields.
        if component_parent and component_parent is not component:
            self.register_special_attr_node(attr_name=ReservedAttributeNames.PARENT_ATTR_NAME,
                                            component=component_parent,
                                            type_info=component_parent.get_type_info())

        if component.is_subentity_items():
            self.register_special_attr_node(attr_name=ReservedAttributeNames.ITEMS_ATTR_NAME,
                                            component=component,
                                            type_info=component.get_type_info())

        if component.is_field():
            # simple datatype as raw value (terminates ValueNode logic), e.g. int, str, date
            self.register_special_attr_node(attr_name=ReservedAttributeNames.VALUE_ATTR_NAME,
                                            component=component,
                                            type_info=component.get_type_info())

        if component.is_container() or component.is_fieldgroup():
            # TODO: boolean+enables - not covered - special datastruct needed:
            #   if isinstance(component, IField) and component.has_children():
            # complex datastructure as raw instance (terminates ValueNode logic), e.g. Fields(name="name", age=30)
            self.register_special_attr_node(attr_name=ReservedAttributeNames.INSTANCE_ATTR_NAME,
                                            component=component,
                                            type_info=component.get_type_info())

        if component.has_children() and not component.is_subentity_items():
            # for subentity_items -> need to access each item and then access children/fields
            component_fields_dataclass, _ = component.get_component_fields_dataclass(setup_session=setup_session)
            type_info = TypeInfo.get_or_create_by_type(ListChildField)
            self.register_special_attr_node(attr_name=ReservedAttributeNames.CHILDREN_ATTR_NAME,
                                            component=component,
                                            type_info=type_info,
                                            # TODO: missusing: th_field=component_fields_dataclass,
                                            )

    # ------------------------------------------------------------
    def setup_all_nodes(self, setup_session: ISetupSession):
        """
        RECURSIVE
        """
        super().setup(setup_session=setup_session)

        container = self.component

        for component_name, component in container.components.items():
            component_parent = component.parent
            # store in child's component store
            component_attr_dexp_node_store = component.get_attr_dexp_node_store()

            is_not_me = (component is not container)

            # TODO: if not is_not_me: continue

            if not (component.is_container() and is_not_me):
                # skip case when container is child. Containers should register its own special fields.
                component_attr_dexp_node_store._register_special_nodes(component=component, setup_session=setup_session)

            # register children one by one
            if is_not_me and component.has_data():
                # allow direct SubEntityItems to be accessible
                attr_node = self.create_attr_node(component)
                self.register_attr_node(attr_node)  # , is_list=False))

                # Store in Component's parent store too - to be available in This.
                # and in 2+ expression bit-nodes (e.g. F.access.name).
                # Skip registering the component in its own store.
                if (component_parent
                  and component_parent is not container
                  and component_parent is not component):
                    # TODO: this is strange case :( - simplify
                    parent_attr_dexp_node_store = component_parent.get_attr_dexp_node_store()
                    parent_attr_dexp_node_store.register_attr_node(attr_node=attr_node)

            if is_not_me and component.is_container():
                # ------- RECURSION --------
                component_attr_dexp_node_store.setup_all_nodes(setup_session=setup_session)

        # NOTE: can not finish yet, since fallback system will add later other component's fields to this store
        #       which are referenced and visible from this component
        # self.finish()


    @classmethod
    def create_attr_node(cls, component: "IComponent"):
        # TODO: remove these local imports, move this class to expr_attr_nodes
        from .expr_attr_nodes import AttrDexpNodeForComponent

        if not isinstance(component, (IField, IFieldGroup, IContainer,)):
            raise EntitySetupError(owner=cls, msg=f"Register expected ComponentBase, got {component} / {type(component)}.")

        attr_node = AttrDexpNodeForComponent(
            component=component,
            namespace=FieldsNS,
        )
        return attr_node

# ------------------------------------------------------------


@dataclass
class IComponent(ReedwolfDataclassBase, ABC):
    """ requires (Protocol):
        name
        parent_name
        parent
        _accessor
        entity

    ReedwolfDataclassBase.DENY_POST_INIT - don't allow __post_init__ methods
    since before setup() phase objects can be configured, adapted, changed as wanted.
    After setup() is done, instances are checked, configured internally and become immutable.
    """
    DENY_POST_INIT = True

    # NOTE: I wanted to skip saving parent reference/object within component - to
    #       preserve single and one-direction references.
    # NOTE: Not DRY: Entity, SubentityBase and ComponentBase

    # TODO: cleaners:       Optional[List["ICleaner"]] = field(repr=False, init=False, default=None)
    parent:         Union[Self, UndefinedType] = field(init=False, compare=False, default=UNDEFINED, repr=False)
    parent_name:    Union[str, UndefinedType] = field(init=False, default=UNDEFINED)
    entity:         Union[Self, "IEntity"] = field(init=False, compare=False, default=UNDEFINED, repr=False)
    _accessor:      IValueAccessor = field(init=False, default=UNDEFINED, repr=False)

    name_counter_by_parent_name: Dict[str, int] = field(init=False, repr=False, default_factory=dict)

    # ------------------------------------------------------------

    # ----- lazy init - done in Setup phase -----

    # Usually is input param, if not set then filled later. Put here to avoid issues with inheritance when it is input param.
    # TODO: name:           Optional[ComponentName] = field(init=False, default=None)

    # these 2 are computed together
    child_field_list: Optional[ListChildField] = field(init=False, repr=False, default=None)
    _component_fields_dataclass: Optional[Type[IComponentFields]] = field(init=False, repr=False, default=None)

    # ThisRegistry used when inside of component, lazy set in setup process() by calling
    # get_or_create_this_registry / create_and_setup_this_registry. Set for Field*/Container* objects, and is None
    # for others. For SubEntityItems holds .Items :: List[Item], and a special _this_registry_for_item :: Item
    # is used when inside of an item.
    _this_registry: Union[IThisRegistry, NoneType, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)

    #  NOTE: had 4 statuses - mutually exclusive and replaced with single Enum:
    #   _did_init: bool      = field(init=False, repr=False, default=False)
    #   _did_phase_one: bool = field(init=False, repr=False, default=False)
    #   _immutable: bool     = field(init=False, repr=False, default=False)
    _status: ComponentStatus = field(init=False, repr=False, default=ComponentStatus.draft)

    # List of Attribute DotExpression nodes available for this component (only mine) and container (my components + from all my direct components).
    # Used in LocalFieldsRegistry, TopFieldsRegistry and in evaluating 2+ Dexp attr expressions e.g. This.access.name
    # (setup: ..., apply: get_value()). Used in Registry.get_store()
    _attr_dexp_node_store: AttrDexpNodeStoreForComponent = field(repr=False, init=False, default=UNDEFINED)

    @property
    def is_finished(self) -> bool:
        # TODO: for speed perf consider having inline logic instead of this property
        return self._status == ComponentStatus.finished

    # def __post_init__(self):
    #     self.init_base()

    def init(self):
        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.register(self)
        # when not set then will be later defined - see set_parent()
        # if self._did_init:
        if self._status != ComponentStatus.draft:
            raise EntitySetupError(owner=self, msg=f"Component not in draft state: {self._status}")
        if self.name not in (None, "", UNDEFINED):
            if not self.name.isidentifier():
                raise EntitySetupValueError(owner=self, msg=f"Attribute name needs to be valid python identifier name, got: {self.name}")

        # freeze all set dc_field values which won't be changed any more. Used for copy()
        # self._did_init = True
        self._status = ComponentStatus.did_init


    def set_parent(self, parent: Optional["IComponent"]):
        if self.parent is not UNDEFINED:
            if self.parent is not parent:
                raise EntityInternalError(owner=self, msg=f"Parent already set to different, have: {self.parent}, got: {parent}")
            return

        assert parent is None or isinstance(parent, IComponent), parent
        self.parent = parent

        if self.parent_name is not UNDEFINED:
            raise EntityInternalError(owner=self, msg=f"Parent name already defined, got: {parent}")

        self.parent_name = parent.name if parent else ""

        assert hasattr(self, "name")
        assert hasattr(self, "get_path_to_first_parent_container")

        if not self.name:
            self._getset_name()
        elif hasattr(self, "title") and not self.title and self.name:
            self.title = varname_to_title(self.name)

        if not self.name:
            raise EntityInternalError(owner=self, msg=f"Name is not set: {self}")

        self.entity = self if self.parent is None else self.parent.entity
        assert self.entity

        self._setup_accessor()


    def _setup_accessor(self):
        assert not self._accessor

        if hasattr(self, "accessor") and self.accessor:
            self._accessor = self.entity.settings._get_accessor(self.accessor)
        elif self.parent is not None:
            self._accessor = self.parent._accessor
        else:
            self._accessor = self.entity.settings._accessor
        assert self._accessor


    def _getset_name(self):
        """
        recursive
        """
        assert hasattr(self, "get_path_to_first_parent_container")

        if not self.name:

            if self.parent is None:
                # top container
                parents = []
                container = self
            else:
                # suffix = self.__class__.__name__.lower()
                # self.name = f"{self.parent_name}__{suffix}"
                parents = self.get_path_to_first_parent_container(consider_self=False)
                assert parents
                container = parents[-1]

            keys = []
            for parent in reversed(parents):
                # recursion
                parent_name = parent._getset_name()
                keys.append(parent_name)

            if isinstance(self, IField):
                assert getattr(self, "bind_to", None)
                # ModelsNs.person.surname -> surname
                this_name = get_name_from_bind(self.bind_to)
            else:
                this_name =self.__class__.__name__.lower()
            keys.append(this_name)

            key = "__".join(keys)

            name_id = container._get_new_id_by_parent_name(key)
            self.name = f"{key}__{name_id}"

        return self.name

    def _get_new_id_by_parent_name(self, key: str) -> int:
        self.name_counter_by_parent_name.setdefault(key, 0)
        self.name_counter_by_parent_name[key] += 1
        return self.name_counter_by_parent_name[key]


    def get_attr_dexp_node_store(self) -> AttrDexpNodeStoreForComponent:
        if self._attr_dexp_node_store is UNDEFINED:
            self._attr_dexp_node_store = AttrDexpNodeStoreForComponent(component=self)
            # TODO: raise EntityInternalError(owner=self, msg=f"Attr _attr_dexp_node_store is not setup")
        return self._attr_dexp_node_store

    def get_first_parent_container(self, consider_self: bool) -> "IContainer":  # noqa: F821
        # TODO: replace this function and maybe calls with:
        #       return self.parent_container if consider_self else (self.parent.parent_container if self.parent else None)
        parents = self.get_path_to_first_parent_container(consider_self=consider_self)
        return parents[-1] if parents else None

    def get_path_to_first_parent_container(self, consider_self: bool) -> List["IContainer"]:  # noqa: F821
        """
        traverses up the component tree up (parents) and find first container
        including self ( -> if self is container then it returns self)
        TODO: maybe it is reasonable to cache this
        """
        if self.parent is UNDEFINED:
            raise EntitySetupError(owner=self, msg="Parent is not set. Call .setup() method first.")

        if consider_self and isinstance(self, IContainer):
            return [self]

        parents = []
        parent_container = self.parent
        while parent_container is not None:
            parents.append(parent_container)
            if isinstance(parent_container, IContainer):
                break
            parent_container = parent_container.parent

        if parent_container in (None, UNDEFINED):
            if consider_self:
                raise EntitySetupError(owner=self, msg="Did not found container in parents. Every component needs to be in some container object tree (Entity/SubEntityItems).")
            return []

        return parents

    def is_unbound(self) -> bool:
        return self.entity.is_unbound()

    def _check_cleaners(self, allowed_cleaner_base_list: List["VALIDATION_OR_EVALUATION_TYPE"]):
        allowed_cleaner_base_list = tuple(allowed_cleaner_base_list)
        if self.cleaners is not None:
            cl_names = ", ".join([cl.__name__ for cl in allowed_cleaner_base_list])
            if not isinstance(self.cleaners, (list, tuple)):
                raise EntitySetupTypeError(owner=self, msg=f"Cleaners should be None or list of {cl_names}, got: {type(self.cleaners)} / {to_repr(self.cleaners)}") 
            for cleaner in self.cleaners:
                if not isinstance(cleaner, allowed_cleaner_base_list):
                    raise EntitySetupTypeError(owner=self, msg=f"Cleaners should be instances of {cl_names}, got: {type(cleaner)} / {cleaner}") 

    # def create_this_registry(self, setup_session: ISetupSession) -> Optional[IThisRegistry]:
    #     """
    #     default implementation
    #     """
    #     # TODO: resolve circular dependency by moving this function logic to
    #     #       .registries.py/.setup.py or somewhere else
    #     from .registries import ThisRegistry

    #     if self.is_data_model():
    #         raise EntityInternalError(owner=self, msg=f"For DataModel create_this_registry() needs to be overridden.")

    #     has_children = bool(self.get_children())
    #     if not has_children:
    #         raise EntityInternalError(owner=self, msg="Non-fields should have children")

    #     this_registry = ThisRegistryForComponent(component=self)
    #     # consider:
    #     # self.is_subentity_items()  => Items -> This.Items
    #     # self.is_subentity_single() => Children -> This.Children + This.<all-attributes> + This.Instance
    #     # else:                      => Children -> This.Children + This.<all-attributes>
    #     # test repr: print(f"this_registry={this_registry}")

    #     return this_registry


    @staticmethod
    def can_apply_partial() -> bool:
        return False


    def as_str(self):
        return "\n".join(self.to_strlist())

    def __str__(self):
        if self.name:
            out = f"{self.__class__.__name__}({self.name})"
        else:
            out = []
            for dc_field in reversed(fields(self)):
                if dc_field.repr:
                    val = getattr(self, dc_field.name)
                    if type(val) not in (NoneType, UndefinedType):
                        item = f"{dc_field.name}={val}"
                        if self.name == "parent_name":
                            out.insert(0, item)
                        else:
                            out.append(item)
            if out:
                out = ", ".join(reversed(out))
                out = f"{self.__class__.__name__}({out})"
            else:
                out = super().__str__()
        return out

    def __repr__(self):
        return self.__str__()

    def to_strlist(self, path=None):
        if path is None:
            path = []
        out = []
        out.append(f"{self.__class__.__name__}(")
        # vars(self.kwargs).items():
        if len(path) > MAX_RECURSIONS:
            raise EntitySetupError(f"Maximum object tree depth reached, not allowed depth more than {MAX_RECURSIONS}.")
        # NOTE: ALT: for name, field in self.__dataclass_fields__.items():  # noqa: F402
        for field in fields(self):  # noqa: F402
            name = field.name
            # if name.startswith("_") or callable(k):
            #     continue
            value = getattr(self, name)
            if type(field) in (list, tuple):
                out.extend(
                    add_yaml_indent_to_strlist(
                        list_to_strlist(
                            value,
                            before=f"{name}=[,",
                            after="],",
                        )
                    )
                )
            elif value == self:
                out.append(f"{name}=[Self...],")
            elif name in path:
                out.append(f"{name}=[...],")
            else:
                vstr = obj_to_strlist(value, path=path+[name])
                if len(vstr) <= 1:
                    out.append(f"{name}={vstr[0]},")
                else:
                    # vstr = add_yaml_indent_to_strlist(vstr)
                    out.append(f"{name}=")
                    for v2 in vstr:
                        out.append(f"{YAML_INDENT}{v2}")
        out.append(")")
        return add_yaml_indent_to_strlist(out)

    # ------------------------------------------------------------

    def is_top_parent(self) -> bool:
        if self.parent is UNDEFINED:
            raise EntityInternalError(owner=self, msg="Parent is not yet set")
        return not bool(self.parent)


    @abstractmethod
    def get_type_info(self) -> TypeInfo:
        ...

    @staticmethod
    @abstractmethod
    def has_data() -> bool:
        ...

    @staticmethod
    def is_entity() -> bool:
        """ different from is_top_parent since is based on type not on parent setup"""
        return False

    @staticmethod
    def is_container() -> bool:
        return False

    @staticmethod
    def is_data_model() -> bool:
        return False

    @staticmethod
    def is_subentity_items() -> bool:
        return False

    @staticmethod
    def is_subentity() -> bool:
        return False

    def is_subentity_any(self) -> bool:
        return self.is_subentity_items() or self.is_subentity()

    @staticmethod
    def is_fieldgroup() -> bool:
        return False

    @staticmethod
    def is_field() -> bool:
        return False

    @staticmethod
    def can_have_children() -> bool:
        return False

    @staticmethod
    def may_collect_my_children() -> bool:
        """ 
        when collecting children - if this sub-component has children, 
        may they be collected and be referenced e.g. in This. <all-attributess>
        access (see get_children(deep_collect=True) mode)
        """
        return False

    # ------------------------------------------------------------
    def has_children(self) -> bool:
        if not hasattr(self, "_children"):
            return bool(self.get_children_direct())
        return bool(self._children)

    def get_children_direct(self) -> List[Self]:
        return self._get_children(deep_collect=False, cache=True, traverse_all=False)

    def get_children_deep(self) -> List[Self]:
        return self._get_children(deep_collect=True, cache=True, traverse_all=False)

    def _get_children(self, deep_collect: bool = False, cache: bool = True, traverse_all: bool = False) -> List[Self]:
        """
        TODO: check if traverse_all is needed - maybe child.may_collect_my_children should not be checked?
        Fills/uses undocumented attributes - internal:
            _children       - direct children only
            _children_dict  - direct children only
            _children_deep  - whole children tree flattened

        in deep_collect mode:
            CACHED
            will collect children
            and children's children for those which have
                may_collect_my_children
                Examples:
                    - FieldGroup
                    - SubEntity

        in normal mode:
            CACHED
            Get only children components. 

        To get all - components, cleaners and all other complex objects => use
        _get_subcomponents_list()

        NOTE: it will collect all children, not only fields. E.g. will include:
              including FieldGroup, SubEntity*
        """
        # TODO: cached - be careful not to add new components aferwards
        if deep_collect:
            if not cache or not hasattr(self, "_children_deep"):
                children = []
                # RECURSION - 1 level
                for child in self._get_children(deep_collect=False, cache=cache):
                    children.append(child)
                    if traverse_all or child.may_collect_my_children():
                        # RECURSION
                        child_children = child._get_children(deep_collect=True, traverse_all=traverse_all, cache=cache)
                        if child_children:
                            children.extend(child_children)
                out = children
                if cache:
                    self._children_deep = children
            else:
                out = self._children_deep
        else:
            if not cache or not hasattr(self, "_children"):
                children = getattr(self, "contains", None)
                if not children:
                    children = getattr(self, "enables", None)
                else:
                    assert not hasattr(self, "enables"), self
                out = children if children else []
                if cache:
                    self._children = out
                    self._children_dict = {comp.name: comp for comp in out}
            else:
                out = self._children

        return out


    # ------------------------------------------------------------

    def get_children_dict(self) -> Dict[ComponentNameType, Self]:
        """
        TODO: maybe not used?
        only direct children in flat dict
        """
        return self._children_dict

    # ------------------------------------------------------------

    # def get_children_tree_flatten_dict(self, depth: int=0) -> Dict[ComponentNameType, Self]:
    #     """
    #     will go recursively through every children and
    #     fetch their "children" and collect to output structure:
    #     selects not-subentity_itemss, put in flat dict (i.e. children with
    #     model same level fields), excludes self
    #     """
    #     key = "_children_tree_flatten_dict"
    #     if not hasattr(self, key):
    #         if depth > MAX_RECURSIONS:
    #             raise EntityInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

    #         children_dict_traversed = {}

    #         for comp in self.get_children():
    #             if not comp.is_subentity_items():
    #                 # recursion
    #                 comp_chidren_dict = comp.get_children_tree_flatten_dict(depth=depth+1)
    #                 children_dict_traversed.update(comp_chidren_dict)

    #             # closer children are overriding further ones
    #             # (although this will not happen - names should be unique)
    #             children_dict_traversed[comp.name] = comp

    #         setattr(self, key, children_dict_traversed)

    #     return getattr(self, key)

    # ------------------------------------------------------------

    # def get_children_tree(self) -> ComponentTreeType:
    #     """
    #     will go recursively through every children and
    #     fetch their "children" and collect to output structure.
    #     selects all nodes, put in tree, includes self
    #     """
    #     return self._get_children_tree(key="_children_tree")


    # def _get_children_tree(self, key: str, depth:int=0) -> ComponentTreeType:
    #     # Dict[ComponentNameType, Self] = Dict[ComponentNameType, TreeNode])
    #     if not hasattr(self, key):
    #         if depth > MAX_RECURSIONS:
    #             raise EntityInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

    #         children_dict_traversed = {}
    #         children_dict_traversed["name"] = self.name
    #         children_dict_traversed["component"] = self
    #         children_dict_traversed["children"] = []


    #         for comp in self.get_children():
    #             # recursion
    #             comp_chidren_dict = comp._get_children_tree(key=key, depth=depth+1)
    #             children_dict_traversed["children"].append(comp_chidren_dict)

    #         setattr(self, key, children_dict_traversed)
    #     else:
    #         children_dict_traversed = getattr(self, key)
    #     return children_dict_traversed

    # ------------------------------------------------------------

    def dump_meta(self, format: DumpFormatEnum = None) -> MetaTree:
        """
        Recursively traverse children's tree and and collect current values to
        recursive output dict structure.
        """
        # tree: ComponentTreeType = self.get_children_tree()
        out = self._dump_meta()
        if format:
            out = dump_to_format(out, format_=format)
        return out

    # ------------------------------------------------------------

    def _dump_meta(self, depth: int=0) -> MetaTree:
        # tree: ComponentTreeType
        if depth > MAX_RECURSIONS:
            raise EntityInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

        output = {} # OrderedDict()
        component = self
        # output["name"] = component.name
        output["type"] = component.__class__.__name__

        attr_dc_fields: Tuple[DcField] = get_dataclass_fields(component)


        attr_dict = {}

        for nr, attr_field in enumerate(attr_dc_fields, 1):

            if not attr_field.init:
                continue

            if attr_field.metadata \
              and isinstance(attr_field.metadata, (dict, MappingProxyType)) \
              and attr_field.metadata.get("skip_dump", None):
                continue

            attr_name = attr_field.name

            if not hasattr(component, attr_name):
                continue 

            # -- fetch value
            attr_value = getattr(component, attr_name, UNDEFINED)

            # -- fetch default value
            attr_default = attr_field.default

            if attr_default is DC_MISSING \
              and attr_field.default_factory is not DC_MISSING \
              and callable(attr_field.default_factory):
                # e.g. dict, list => initiate and compare later
                attr_default = attr_field.default_factory()
                # in immutable state - all lists are converted to tuples
                # therefore convert default value ([]) to tuple.
                if isinstance(attr_default, list) and isinstance(attr_value, tuple):
                    attr_default = tuple(attr_default)

            # -- is value supplied i.e. is it different from default
            if isinstance(attr_value, (IComponent, DotExpression)) \
              or (attr_value is not UNDEFINED 
                  and attr_value is not DC_MISSING 
                  and (attr_default is DC_MISSING or attr_value!=attr_default)
                  ):
                assert attr_name not in attr_dict

                if isinstance(attr_value, (list, tuple)):
                    new_attr_value = []
                    for attr_value_item in attr_value:
                        # can call recursion
                        attr_value_item = self._dump_meta_process_node(attr_value_item, depth=depth)
                        new_attr_value.append(attr_value_item)
                    attr_value = new_attr_value
                else:
                    # can call recursion
                    attr_value = self._dump_meta_process_node(attr_value, depth=depth)

                assert attr_value is not self

                attr_dict[attr_name] = attr_value

        if attr_dict:
            output["attrs"] = attr_dict

        return output

    # ------------------------------------------------------------

    @staticmethod
    def _dump_meta_process_node(node: Any, depth: int) -> Any:
        if isinstance(node, DotExpression):
            # string representation contains all data
            node = f"{DEXP_PREFIX}{str(node)}"
        else:
            if hasattr(node, "_dump_meta"):
                # recursion
                node = node._dump_meta(depth=depth+1)
        return node

    # ------------------------------------------------------------

    def _add_component(self, component: Self, components: Dict[str, Self]):
        if component.name in (None, UNDEFINED):
            raise EntitySetupValueError(owner=self, item=component, msg="Component's name is required.")
        if not (component.name and isinstance(component.name, str)):
            raise EntitySetupValueError(owner=self, item=component, msg=f"Component's name needs to be a string value, got: {component.name}': ")
        if component.name in components:
            raise EntitySetupNameError(owner=self, item=component, msg=f"Duplicate name '{component.name}': "
                        + "\n   " + repr(components[component.name])[:100]
                        + "\n   " + " --------- AND --------- "
                        + "\n   " + repr(component)[:100]
                        + "\n" + ". Remove duplicate 'bind_to' or use 'name' attribute to define a distinct name.")
        # Save top container too - to preserve name and for completness (if is_top)
        components[component.name] = component

    # ------------------------------------------------------------

    def _call_init(self):
        self._getset_rwf_kwargs()
        # if self._did_phase_one:
        if self._status >= ComponentStatus.did_phase_one:
            if self.parent:
                raise EntityInitError(owner=self, msg=f"Component '{self.name} : {self.__class__.__name__}' "
                                                      f"is already embedded into '{self.parent.name}: {self.parent.__class__.__name__}' -> ... {self.entity}'. "
                                                      "Use .copy() to embed the same component into other parent component.")
            else:
                raise EntityInternalError(owner=self, msg=f"Setup (phase one) for a component '{self.name} : {self.__class__.__name__}' already done.")
        self.init()

    # ------------------------------------------------------------

    def _setup_phase_one(self, components: Optional[Dict[str, Self]] = None) -> NoneType:
        """
        does following:
        - collects components:
        - set parent
        - evaluate data_model - class or M.value_expressions
        - evaluate bind_to - M.value_expressions
        - custom dataclass and fields

        collecting components
        ---------------------
        except for subentities sub-components
            it is flattened dict of subcomponents (recursive)
        i.e. it returns components: Dictionary [component_name: Component]
            will fill following:
                all mine sub-components
                if not subentity
                    all sub-component sub-components (recursively)
        it sets parent of all sub-components recursively
        """

        # for children/contains attributes - parent is set here
        if not hasattr(self, "name"):
            raise EntitySetupError(owner=self, msg=f"Component should have 'name' attribute, got class: {self.__class__.__name__}")

        if self.is_container():
            assert components is None
            assert self.components is None, self

            components = {}

            if self.is_entity():
                self._call_init()
                self.settings._setup_all_custom_functions(self.apply_settings_class)
                self.set_parent(None)
            else:
                # SubEntityItems() / SubEntity()
                assert self.is_subentity_any()

            assert not self.setup_session
            # TODO: explain when is not set
            self.create_setup_session()

            setup_session = self.setup_session
            container = self
        else:
            assert isinstance(components, dict)
            container = self.get_first_parent_container(consider_self=False)
            setup_session = container.setup_session

        # even for owner/container add self (owner itself) to self.components
        self._add_component(component=self, components=components)

        with setup_session.use_stack_frame(
                SetupStackFrame(
                    container = container,
                    component = self,
                    this_registry = None,
                )):
            if isinstance(self, IContainer):
                # ----------------------------------------
                # ModelsNS setup 1/2 phase -> data models
                # ----------------------------------------
                #   - direct data models setup (Entity)
                #   - some standard NS fields (e.g. Instance)
                # This will call complete setup() for data_model-s
                self.data_model._call_init()

                if not self.is_unbound():
                    self.data_model.set_parent(self)
                    self._register_model_attr_nodes()

            # includes components, cleaners and all other complex objects
            for subcomponent in self._get_subcomponents_list():
                component = subcomponent.component

                # if isinstance(component, Namespace):
                #     raise EntitySetupValueError(owner=self, msg=f"Subcomponents should not be Namespace instances, got: {subcomponent.name} = {subcomponent.component}")

                if isinstance(component, DotExpression):
                    # NOTE:
                    #   1) phase_one sets: Field.bind_to, Container.data_model.model_klass (DataModel) on other place
                    #      data_model.model_klass Dexp build for bound case is done before, and for unbound after
                    #   2) phase_two sets all other: evaluation.ensure, Validation.ensure etc.
                    continue

                if isinstance(component, IComponent):
                    # for IDataModel init is done before, don't do it again.
                    if not isinstance(component, IDataModel):
                        component._call_init()

                    component.set_parent(self)
                    # ALT: component.is_subentity()
                    if component.is_subentity_any():
                        # component.set_parent(parent=self)
                        # for subentity_items container don't go deeper into tree (call _fill_components)
                        # it will be called later in container.setup() method
                        component._setup_phase_one(components=None)
                        # save only container (top) object
                        self._add_component(component=component, components=components)
                    else:
                        component._setup_phase_one(components=components)
                        # delete this
                        if component.is_data_model() and not component.entity.is_unbound():
                            data_model: IDataModel = component
                            if isinstance(data_model.model_klass, DotExpression):
                                assert data_model.model_klass.IsFinished()
                            # NOTE: moved from setup_phase_two() - this is required for in_model=False cases,
                            #       to register non-model attributes
                            data_model._register_nested_models(setup_session)

                else:
                    if component.__class__.__name__ not in ("ChoiceOption", "CustomFunctionFactory", "int", "str"):
                        raise EntityInternalError(owner=component, msg=f"Strange type of component, check or add to list of ignored types for _setup_phase_one() ")
            # end for

            if isinstance(self, IContainer):
                if self.components is not None:
                    raise EntityInternalError(owner=self, msg=f"components already set: {self.components}")
                self.components = components

                setup_session.top_setup_session.top_fields_registry.register_fields_of_container(self)

                # evaluate data_model - class or M.value_expression
                # self.data_model.setup(setup_session=setup_session)
            else:
                if isinstance(self, IField):
                    # evaluate bind_to: ValueExpression
                    if self.is_unbound():
                        # do it before
                        self._setup_phase_one_set_type_info(setup_session)
                        if len(self.bind_to.Path) > 1:
                            raise EntitySetupError(owner=self, msg=f"Unbound mode currently supports only 1 level deep bind_to DotExpressions, got: {self.bind_to}")

                    self.bind_to.Setup(setup_session=setup_session, owner=self)
                else:
                    if hasattr(self, "bind_to"):
                        raise EntityInternalError(owner=self, msg=f"Only fields expected to have bind_to attr, got: {type(self)}")

            if self.has_children():
                # Set up dataclass and list of fields for later use in FieldsNS.
                # Will validate if all type_info are available
                self.get_component_fields_dataclass(setup_session=setup_session)

            if isinstance(self, IField):
                if not self.is_unbound():
                    # do it after
                    # Check that type_info is set - call will raise error if something is wrong.
                    self._setup_phase_one_set_type_info(setup_session)

                assert hasattr(self, "get_type_info")
                if not self.get_type_info():
                    raise EntityInternalError(owner=self, msg="type_info could not be retrieved in setup phase one")
            elif self.is_container():
                if self.is_unbound():
                    self._replace_modelsns_registry(setup_session)

        # self._did_phase_one = True
        self._status = ComponentStatus.did_phase_one

        return None

    # ------------------------------------------------------------

    def get_component_fields_dataclass(self, setup_session: Optional[ISetupSession]) \
            -> Tuple[Type[IComponentFields], ListChildField]:
        """
        CACHED
        RECURSIVE

        Will call SubEntity* . setup() in order to create recursive dataclass
        that contains fields only for components (not all model fields are
        available).
        """
        if self._component_fields_dataclass is not None:
            assert self.child_field_list is not None
            return self._component_fields_dataclass, self.child_field_list

        if not setup_session:
            raise EntityInternalError(owner=self, msg="_component_fields_dataclass not set and setup_session not provided")

        container = self.get_first_parent_container(consider_self=True)

        # NOTE: deep_collect is required, otherwise errors later occur
        children = self.get_children_deep()

        if not children:
            raise EntityInternalError(owner=self, msg="No children found.")

        child_field_list: ListChildField = []

        for nr, child in enumerate(children, 1):

            if child.is_data_model():
                continue

            child_type_info = None
            if isinstance(child, IField):
                # ALT: not hasattr(child, "bind_to")
                # NOTE: check that sessino is setup correctly for this field?

                if not child.bind_to.IsFinished():
                    # Can setup only fields which are inside the same container
                    # share the same data_model
                    child.bind_to.Setup(setup_session=setup_session, owner=self)

                attr_node = child.bind_to._dexp_node
                child_type_info = attr_node.get_type_info()

            elif child.is_subentity_any() or child.is_fieldgroup():
                # TODO: check if this needs rewrite? see notes below:
                # ------------------------------------------------------------
                # NOTE: this is a bit complex chain of setup() actions: 
                #           entity -> this -> subentity -> subentity ...
                #       find some easier way how to to it
                #       
                # ------------------------------------------------------------
                # ALT: when all model fields are available:
                #       if not child.data_model.model.IsFinished():
                #           # ALT: more complex way - it seems to work, but I prefer
                #           #      simplier solution:
                #           attr_node = container._setup_data_model_dot_expression(data_model=child.data_model, setup_session=setup_session)
                #
                #       # ALT: attr_node = child.data_model.model._dexp_node
                #       #      child_type_info = attr_node.get_type_info()
                #       child_type_info = child.data_model.get_type_info()

                # NOTE: "make_component_fields_dataclass()" works recuresively to
                #       enable access only to registered fields.

                with setup_session.use_stack_frame(
                        # used only to change component/container
                        SetupStackFrame(
                            container = container 
                                        if child.is_fieldgroup() 
                                        else child, 
                            component = child, 
                            # should not be used 
                            this_registry = None,
                        )):

                    # NOTE: this is not required any more since setup_phase_one() sets up all type_info-s
                    #   if child.is_subentity():
                    #       child.setup(setup_session=setup_session) # recursive

                    # RECURSION: setup_session is used to extract name from Field/Container.bind_to ...
                    child_component_fields_dataclass, _ = child.get_component_fields_dataclass(setup_session=setup_session)

                if child.is_subentity_items():
                    child_component_fields_dataclass = List[child_component_fields_dataclass]
                child_type_info = TypeInfo.get_or_create_by_type(child_component_fields_dataclass)

            else:
                raise EntityInternalError(owner=child, msg=f"child_type_info could not be extracted, unsuppoerted component's type, got: {type(child)}") 

            child_field_list.append(
                    ChildField(
                        Name=child.name,
                        _component=child,
                        _type_info=child_type_info,
                    ))

        class_name_camel = snake_case_to_camel(self.name)

        # TODO: this really needs some explanation - looks like hack
        component_fields_dataclass = make_component_fields_dataclass(
                                class_name=f"{class_name_camel}__Fields",
                                child_field_list=child_field_list,
                                )
        self._component_fields_dataclass = component_fields_dataclass
        self.child_field_list = child_field_list

        return self._component_fields_dataclass, self.child_field_list

    # ------------------------------------------------------------

    def _setup(self, setup_session: ISetupSession):  # noqa: F821
        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.setup_called(self)
        if self.is_finished:
            raise EntityInternalError(owner=self, msg="Setup already called")

        container = self.get_first_parent_container(consider_self=True)

        with setup_session.use_stack_frame(
                SetupStackFrame(
                    container = container, 
                    component = self, 
                )):
            # setup this_registry objects must be inside of stack_frame due
            # premature component.bind_to setup in some ThisRegistryFor* classes.
            if not self.is_subentity_any():
                # NOTE: for SubEntity* - this registry must be and is already
                #       created in ContainerBase.setup()
                # this_registry = container.try_create_this_registry(component=self, setup_session=setup_session)
                this_registry = self.get_or_create_this_registry(setup_session=setup_session)
                if this_registry:
                    # set only if available, otherwise use existing
                    setup_session.current_frame.set_this_registry(this_registry, force=True)

            if not self.is_finished:
                self._setup_phase_two(setup_session=setup_session)
            else:
                if not self.is_data_model():
                    raise EntityInternalError(owner=self, msg=f"_setup_phase_two() is skipped only for data models, got: {type(self)}")

    # ------------------------------------------------------------
    def _get_subcomponents_list(self) -> List[Subcomponent]:
        """
        Includes components, cleaners and all other complex objects
        to get only children components, use get_children()

        TODO: profiling showed that this is the slowest function (even tottime
              - without function calls), although in general the lib is not so slow.
              RL 230422 

        TODO: document and make it better, pretty hackish.
              the logic behind is to collect all attributes (recurseively) that
              are:
                1. component (IComponent)
                2. DotExpression

              all of them have some setup method:
                setup() # case 1. and 2.
                Setup() # case 3.

              so the logic should be:
                collect all attributes that have setup() and Setup()
              check if this is true and if so, implement.
        """

        # TODO: do caching only in-setup or after setup, while in draft mode - is not ok
        if hasattr(self, "_subcomponent_list"):
            return self._subcomponent_list

        # returns name, subcomponent
        fields = get_model_fields(self.__class__)

        # ==== Order to have data_model first, then components, then value
        #      expressions. In the same group order by name

        # NOTE: with vars() not the best way, other is to put metadata in field()
        sub_component_items = []
        for sc_pair in vars(self).items():
            if isinstance(sc_pair[1], DotExpression):
                priority = 0
            else:
                priority = getattr(sc_pair[1], "SETUP_PRIORITY", 1)
            sub_component_items.append((priority, sc_pair))
        sub_component_items = sorted(sub_component_items, reverse=True)
        sub_component_items = [sc_pair for _, sc_pair in sub_component_items]


        # ==== Iterate all components and setup() each

        subcomponent_list = []

        for sub_component_name, sub_component in sub_component_items:

            if not isinstance(sub_component, DotExpression):
                if sub_component_name.startswith("_"):
                    continue

                if sub_component in (None, (), {}, []):
                    continue

                # TODO: can be class or some simple type too - remove them from sub_component list 
                if inspect.isclass(sub_component):
                    continue 

                # TODO: see in meta.py there is a standard types ... use this
                if isinstance(sub_component, STANDARD_TYPE_LIST):
                    continue

            # Skip procesing only following
            # TODO: do this better - check type hint (init=False) and then decide
            # TODO: collect names once and store it internally on class instance level?
            # type_hint = type_hints.get(sub_component_name)
            th_field = fields.get(sub_component_name)

            if th_field and th_field.metadata.get("skip_setup", False):
                continue

            if is_function(sub_component):
                # TODO: check that this is not class too
                raise EntitySetupValueError(owner=sub_component, msg=f"Functions/callables could not be used directly - wrap with Function() and try again. Details: ({sub_component_name}: {th_field})")

            # -------------------------------------------------------------
            # Attribute names that are not included in a subcomponents list
            # (standard attribute types like lists/dicts/int/str etc.).
            # -------------------------------------------------------------
            # "autocomplete", "evaluate",
            # TODO: not the smartest way how to do this ...
            if (sub_component_name in ("parent", "parent_name", "parent_container",
                                      # "parent_setup_session",
                                      "name", "title", "datatype", "components", "type", "autocomputed",
                                      "setup_session", "meta",
                                      # NOTE: maybe in the future will have value expressions too
                                      "error", "description", "hint", 
                                      # now is evaluated from data_model, data_model is processed
                                      "models", "py_type_hint", "type_info",
                                      "bound_attr_node",
                                      "read_handler",
                                      "function", "apply_settings_class", "settings",
                                      "min", "max", "allow_none", "ignore_none",
                                      "keys",
                                      "entity",
                                      "_accessor",
                                      "name_counter_by_parent_name",
                                      "containers_dict",
                                      "container_id",
                                      "container_id_counter",
                                      "containers_id_path",
                                      # "value",
                                      # "enum",
                                      )
              or sub_component_name[0] == "_") \
              or sub_component_name.startswith("rwf_"):
                # TODO: this should be main way how to check ...
                if sub_component_name not in ("parent", "parent_container", "entity") and \
                  (hasattr(sub_component, "setup") or hasattr(sub_component, "setup")):
                    raise EntityInternalError(f"ignored attribute name '{sub_component_name}' has setup()/Setup(). Is attribute list ok or value is not proper class (got component='{sub_component}').")
                continue


            if not th_field or getattr(th_field, "init", True) is False \
              and sub_component_name not in ("data_model", ):
                # warn(f"TODO: _get_subcomponents_list: {self} -> {sub_component_name} -> {th_field}")
                raise EntityInternalError(owner=sub_component, msg=f"Should '{sub_component_name}' field be excluded from processing: {th_field}")

            # ----------------------------------------------------------------------
            # Classes of attribute members that are included in a subcomponents list 
            # each will be "Setup" if they have appropriate class and method
            # (see _setup_phase_two()).
            # ----------------------------------------------------------------------
            # TODO: this is not nice - do it better
            # TODO: models should not be dict()
            # TODO: added Self for DataModelWithHandlers
            if sub_component_name not in ("models", "data", "enum") \
                    and th_field \
                    and "Component" not in str(th_field.type) \
                    and "Container" not in str(th_field.type) \
                    and "DotExpression" not in str(th_field.type) \
                    and "Validation" not in str(th_field.type) \
                    and "Evaluation" not in str(th_field.type) \
                    and "DataModel" not in str(th_field.type) \
                    and "[Self]" not in str(th_field.type) \
                    and not sub_component_name.startswith("rwf_"):
                # TODO: Validation should be extended to test isinstance(.., ValidationBase) ... or similar to include Required(), MaxLength etc.
                raise EntityInternalError(owner=sub_component, msg=f"Should '{sub_component_name}' attribute be excluded from processing." 
                        + f"\n  == {th_field})"
                        + f"\n  == parent := {self}"
                        + f"\n  == {type(sub_component)}"
                        + f"\n  == {sub_component}")

            if isinstance(sub_component, (list, tuple)):
                for nr, sub_sub_component in enumerate(sub_component):
                    subcomponent_list.append(
                            Subcomponent(
                                name=f"{sub_component_name}__{nr}", 
                                path=f"{sub_component_name}[{nr}]", 
                                component=sub_sub_component, 
                                th_field=th_field))
            elif isinstance(sub_component, (dict,)):
                for ss_name, sub_sub_component in sub_component.items():
                    # NOTE: bind_to_models case - key value will be used as
                    #       attr_node name - should be setup_session unique
                    subcomponent_list.append(
                            Subcomponent(
                                name=ss_name, 
                                path=f"{sub_component_name}.{ss_name}", 
                                component=sub_sub_component, 
                                th_field=th_field))
            else:
                subcomponent_list.append(
                        Subcomponent(
                            name=sub_component_name,
                            path=sub_component_name, 
                            component=sub_component, 
                            th_field=th_field))

        self._subcomponent_list = subcomponent_list

        return self._subcomponent_list


    # ------------------------------------------------------------


    def _setup_phase_two(self, setup_session: ISetupSession):  # noqa: F821
        """
        phase two -  calls *setup* methods for all components
        fields, cleaners, and all value expression and
        """
        if self.parent is UNDEFINED:
            raise EntityInternalError(owner=self, msg="Parent not set")

        if self.is_finished:
            raise EntityInternalError(owner=self, msg="Setup already called")

        for subcomponent in self._get_subcomponents_list():
            component = subcomponent.component
            if isinstance(component, IComponent) \
              and (component.is_data_model() or component.is_subentity_any()) \
              and component.is_finished:
                # raise EntityInternalError(owner=self, msg=f"DataModel.setup() should have been called before ({component})")
                continue

            component_name = subcomponent.name
            if isinstance(component, (DotExpression,)):
                dexp: DotExpression = component
                # namespace = dexp._namespace
                if dexp.IsFinished():
                    # Setup() was called in container.setup() before or in some other dependency
                    # called = False
                    pass
                else:
                    dexp.Setup(setup_session=setup_session, owner=self)
                    # called = True
            elif isinstance(component, IComponent):
                assert not component.is_entity()
                assert not component.is_finished
                component._setup(setup_session=setup_session)  # , parent=self)
                # NOTE: in some rare cases .finish() is not called - so there is an additional .finish() call later.
                component.finish()
                # called = True
            elif isinstance(component, (dict, list, tuple)):
                raise EntitySetupValueError(owner=self, msg=f"components should not be dictionaries, lists or tuples, got: {type(component)} / {component}")
            else:
                if hasattr(component, "Setup"):
                    raise EntityInternalError(owner=self, msg=f"{self.name}.{component_name} has attribute that is not DotExpression: {type(component)}")
                if hasattr(component, "setup"):
                    raise EntityInternalError(owner=self, msg=f"{self.name}.{component_name} has attribute that is not Component: {type(component)}")

        # self.finish()

        return self

    # ------------------------------------------------------------

    def finish(self):
        """
        Mark component as finished / setup - no change allowed later.
        """
        if self.is_finished:
            raise EntitySetupError(owner=self, msg="finish() should be called only once.")

        if self._attr_dexp_node_store is UNDEFINED:
            raise EntitySetupError(owner=self, msg="Attr _attr_dexp_node_store not set / used.")

        # self.is_finished = True
        # self._status = ComponentStatus.finished
        # for some classes - need to cache
        if self.has_data():
            type_info = self.get_type_info()
            if not type_info:
                raise EntityInternalError(owner=self, msg=f"get_type_info() failed to produce result in finish()")

        self._make_immutable_and_finish()

    # ------------------------------------------------------------

    def get_dexp_result_from_instance(self, apply_result: "IApplyResult", strict:bool = True) -> Optional[ExecResult]:
        """ Fetch ExecResult from component.bind_to from INSTANCE (storage)
            by executing bind_to._evaluator.execute() fetch value process
            Work on stored fields only.
            A bit slower due getattr() logic and more complex structs.
            Does not work on initial values - when cache is not yet initialized
            with initial value for the component.
        """
        # TODO: put in common function
        bind_dexp = self.bind_to
        if not bind_dexp:
            if strict:
                # TODO: move this to Setup phase
                raise EntityApplyError(owner=self, msg=f"Component '{self.name}' has no bind_to")
            return None
        bind_dexp_result = bind_dexp._evaluator.execute_dexp(apply_result=apply_result)
        return bind_dexp_result


    def get_this_registry(self) -> Optional[IThisRegistry]:
        if self._this_registry is UNDEFINED:
            raise EntityInternalError(owner=self, msg=f"_this_registry not setup, seems that get_or_create_this_registry() was not called before")
        return self._this_registry

    @abstractmethod
    def create_this_registry(self, setup_session: ISetupSession) -> Optional[IThisRegistry]:
        ...

    def get_or_create_this_registry(self, setup_session: ISetupSession) -> Optional[IThisRegistry]:
        # TODO: maybe DRY is needed - similar logic found in functions:: setup_this_registry
        if not self._this_registry is not UNDEFINED:
            if self.is_unbound():
                this_registry = None
            else:
                this_registry = self.create_this_registry(setup_session=setup_session)
            if this_registry:
                this_registry.setup(setup_session=setup_session)
                this_registry.finish()
                # test repr: print(f"this_registry={this_registry}")
            self._this_registry = this_registry

        return self._this_registry


# ------------------------------------------------------------

class ICleaner(IComponent, ABC):

    @staticmethod
    def has_data() -> bool:
        return False

    def get_type_info(self) -> TypeInfo:
        raise EntityInternalError(owner=self, msg=f"Method should not be called")


class IValidation(ICleaner, ABC):
    ensure:     DotExpression
    available:  Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    name:       Optional[str] = field(default=None)
    error:      Union[MessageType, TransMessageType, NoneType] = field(repr=False, default=None)


class IEvaluation(ICleaner, ABC):
    ...

VALIDATION_OR_EVALUATION_TYPE = Union[Type[IValidation], Type[IEvaluation]]

# ------------------------------------------------------------

class IField(IComponent, ABC):
    # to Model attribute
    bind_to: DotExpression

    @staticmethod
    def has_data() -> bool:
        return True

    @abstractmethod
    def get_type_info(self) -> TypeInfo:
        ...

    @abstractmethod
    def _setup_phase_one_set_type_info(self, setup_session: ISetupSession):
        ...

    @abstractmethod
    def validate_type(self, apply_result: "IApplyResult", value: Any = UNDEFINED) \
        -> Optional["ValidationFailure"]:
        ...

# ------------------------------------------------------------


class IFieldGroup(IComponent, ABC):

    @staticmethod
    def has_data() -> bool:
        return True

# ------------------------------------------------------------
# IContainer
# ------------------------------------------------------------

class IContainer(IComponent, ABC):
    # TODO: this is not dataclass - makes some problems

    bind_to:        Optional[Union["IDataModel", ModelKlassType, DotExpression]] = field(repr=False, default=None)
    settings:       Optional[Settings] = field(repr=False, default=None)
    contains:       List[IComponent] = field(repr=False, init=False, default_factory=list)
    meta:           Optional[Dict[str, Any]] = field(repr=False, default=None)

    # evaluated later
    # Components contain container: itself AND all other attributes - single or lists which are of type component
    # e.g. contains + data_model + cleaners + ...
    components:     Optional[Dict[str, IComponent]] = field(init=False, repr=False, default=None)

    # will be used in apply phase to get some already computed metadata
    setup_session:  Optional[ISetupSession] = field(init=False, repr=False, default=None)
    data_model:     "IDataModel" = field(init=False, repr=False)

    # used for automatic component's naming, <parent_name/class_name>__<counter>
    # TODO: there is some logic overlap with container_id and Entity.container_id_counter
    name_counter_by_parent_name: Dict[str, int] = field(init=False, repr=False, default_factory=dict)

    # unique name in Entity space - set in set_parent()
    container_id:        Union[ContainerId, UndefinedType] = field(init=False, default=UNDEFINED)

    # path of container parents from Entity to this Containers. List of container_id, containing Entity and self.
    containers_id_path:   Union[List[ContainerId], UndefinedType] = field(init=False, default=UNDEFINED)


    @staticmethod
    def has_data() -> bool:
        return True

    @abstractmethod
    def _register_model_attr_nodes(self) -> Dict[str, "IDataModel"]:
        ...

    @abstractmethod
    def get_this_registry_for_item(self) -> IThisRegistry:
        """
        The same as get_this_registry() except for SubEntityItems which returns inner ThisRegistry
        that should be used for individual Item (is_items_for_each_mode).
        """
        ...

    @abstractmethod
    def get_type_info(self) -> TypeInfo:
        ...

    # @abstractmethod
    # def add_fieldgroup(self, fieldgroup:IFieldGroup):  # noqa: F821
    #     ...

    @abstractmethod
    def __getitem__(self, name: str) -> IComponent:
        ...

    @abstractmethod
    def get_component(self, name: str) -> IComponent:
        ...

    @abstractmethod
    def _setup(self, setup_session: ISetupSession):
        ...

    @abstractmethod
    def get_data_model_attr_node(self) -> "IAttrDexpNode":  # noqa: F821
        ...

    @abstractmethod
    def pprint(self):
        """ pretty print - prints to stdout all components """
        ...

class IEntity(IContainer, ABC):
    settings: Settings = field(repr=False, )
    apply_settings_class: Optional[Type[ApplySettings]] = field(repr=False, default=None)

    # ------------------------------------------------------------
    # evaluated / computed later
    # ------------------------------------------------------------

    # will be filled in set_parent(). Key is Container.container_id
    containers_dict: Optional[Dict[ContainerId, IContainer]] = field(init=False, repr=False, default=None)

    # will be used in setting Container.container_id unique name for each container in Entity space
    # TODO: there is some logic overlap with Container.name_counter_by_parent_name
    container_id_counter: int = field(init=False, repr=False, default=None)


# ------------------------------------------------------------
# IDataModel
# ------------------------------------------------------------

@dataclass
class IDataModel(IComponent, ABC):

    model_klass: ModelKlassType = field(init=False, repr=False)
    type_info:   Optional[TypeInfo] = field(init=False, default=None, repr=False)

    # def _setup(self, setup_session:ISetupSession):
    #     if self.is_finished:
    #         raise EntityInternalError(owner=self, msg="Setup already called")
    #     super()._setup(setup_session=setup_session)

    @staticmethod
    def has_data() -> bool:
        return False

    @abstractmethod
    def get_type_info(self) -> TypeInfo:
        ...

    @staticmethod
    def is_data_model() -> bool:
        return True

    def is_top_parent(self):
        return False

    def get_full_name(self, parent: Optional[Self] = None, depth: int = 0, init: bool = False):
        if not hasattr(self, "_name"):
            if depth > MAX_RECURSIONS:
                raise EntityInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")
            if not init:
                raise EntityInternalError(owner=self, msg=f"Expectedc init mode")

            names = []
            if parent:
                # recusion
                names.append(parent.get_full_name(parent=self, depth=depth+1, init=init))
            names.append(self.name)
            self._name = ".".join(names)
        return self._name


    def fill_models(self, models: Dict[str, Self] = None, parent: Self = None) -> Dict[str, Self]:
        """
        Recursion
        """
        if models is None:
            models = {}
        if self.name in models:
            raise EntitySetupNameError(owner=self, msg=f"Currently model names in tree dependency should be unique. Model name {self.name} is not, found: {models[self.name]}")

        name = self.get_full_name(parent=parent, init=True)
        models[name] = self
        if hasattr(self, "contains"):
            for dep_data_model in self.contains:
                dep_data_model._call_init()
                # recursion
                dep_data_model.set_parent(parent=self)
                dep_data_model.fill_models(models=models, parent=self)
        return models

# ------------------------------------------------------------


class IBoundDataModel(IDataModel, ABC):

    @abstractmethod
    def _register_nested_models(self, setup_session:ISetupSession):
        ...

    @abstractmethod
    def _apply_nested_models(self, apply_result: "IApplyResult", instance: ModelInstanceType):
        ...

# ------------------------------------------------------------

class IUnboundDataModel(IDataModel, ABC):
    """
    see data_models.py:: class UnboundModel
    """

# ============================================================

@dataclass
class IStackFrame:

    @abstractmethod
    def post_clean(self):
        ...

    @abstractmethod
    def copy_from_previous_frame(self, previous_frame: Self):
        ...

    def _copy_attr_from_previous_frame(self,
                                       previous_frame: Self,
                                       attr_name: str,
                                       may_be_copied: bool = True,
                                       if_set_must_be_same: bool = True):

        if not hasattr(self, attr_name):
            raise EntityInternalError(owner=self, msg=f"This frame {self}.{attr_name} not found")
        if not hasattr(previous_frame, attr_name):
            raise EntityInternalError(owner=self, msg=f"Previous frame {previous_frame}.{attr_name} not found")

        this_frame_attr_value = getattr(self, attr_name)
        prev_frame_attr_value = getattr(previous_frame, attr_name)

        if this_frame_attr_value is None or this_frame_attr_value is UNDEFINED:
            # if prev_frame_attr_value not in (None, UNDEFINED):
            if not (prev_frame_attr_value is None or prev_frame_attr_value is UNDEFINED):
                if not may_be_copied:
                    raise EntityInternalError(owner=self,
                                              msg=f"Attribute '{attr_name}' value in previous frame is non-empty and current frame has empty value:\n  {previous_frame}\n    = {prev_frame_attr_value}\n<>\n  {self}\n    = {this_frame_attr_value} ")
                    # Copy from previous frame
                # apply_result.settings.loggeer.debugf"setattr '{attr_name}' current_frame <= previous_frame := {prev_frame_attr_value} (frame={self})")
                setattr(self, attr_name, prev_frame_attr_value)
        else:
            # in some cases id() / is should be used?
            if if_set_must_be_same and prev_frame_attr_value != this_frame_attr_value:
                raise EntityInternalError(owner=self,
                                          msg=f"Attribute '{attr_name}' value in previous frame is different from current:\n  {previous_frame}\n    = {prev_frame_attr_value}\n<>\n  {self}\n    = {this_frame_attr_value} ")


@dataclass
class UseStackFrameCtxManagerBase(AbstractContextManager):
    """
    'onwer_session' must implement following:
    # ALT: from contextlib import contextmanager

    """
    owner_session: "IStackOwnerSession"
    frame: IStackFrame
    add_to_stack: bool = True
    cleanup_callback: Optional[Callable[[], None]] = None

    def __post_init__(self):
        if self.add_to_stack:
            if self.owner_session.stack_frames:
                previous_frame = self.owner_session.stack_frames[0]
                self.frame.copy_from_previous_frame(previous_frame)
            self.frame.post_clean()

    def __enter__(self):
        # settings manager (with xxx as yy: )...
        if self.add_to_stack:
            self.owner_session.push_frame_to_stack(self.frame)
        return self.frame


    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.owner_session.current_frame != self.frame:
            raise EntityInternalError(owner=self, msg=f"Something wrong with frame stack (2), got {self.owner_session.current_frame}, expected {self.frame}")

        if self.cleanup_callback:
            self.cleanup_callback()

        if self.add_to_stack:
            frame_popped = self.owner_session.pop_frame_from_stack()
        else:
            frame_popped = self.owner_session.current_frame
        if not exc_type and frame_popped != self.frame:
            raise EntityInternalError(owner=self, msg=f"Something wrong with frame stack, got {frame_popped}, expected {self.frame}")


    # def _copy_attr_from_previous_frame(self,
    #         previous_frame: Self,
    #         attr_name: str,
    #         may_be_copied: bool = True,
    #         if_set_must_be_same: bool = True):

    #     if not hasattr(self.frame, attr_name):
    #         raise EntityInternalError(owner=self, msg=f"This frame {self.frame}.{attr_name} not found")
    #     if not hasattr(previous_frame, attr_name):
    #         raise EntityInternalError(owner=self, msg=f"Previous frame {previous_frame}.{attr_name} not found")

    #     this_frame_attr_value = getattr(self.frame, attr_name)
    #     prev_frame_attr_value = getattr(previous_frame, attr_name)

    #     if this_frame_attr_value is None or this_frame_attr_value is UNDEFINED:
    #         # if prev_frame_attr_value not in (None, UNDEFINED):
    #         if not (prev_frame_attr_value is None or prev_frame_attr_value is UNDEFINED):
    #             if not may_be_copied:
    #                 raise EntityInternalError(owner=self,
    #                     msg=f"Attribute '{attr_name}' value in previous frame is non-empty and current frame has empty value:\n  {previous_frame}\n    = {prev_frame_attr_value}\n<>\n  {self.frame}\n    = {this_frame_attr_value} ")
    #             # Copy from previous frame
    #             # apply_result.settings.loggeer.debugf"setattr '{attr_name}' current_frame <= previous_frame := {prev_frame_attr_value} (frame={self.frame})")
    #             setattr(self.frame, attr_name, prev_frame_attr_value)
    #     else:
    #         # in some cases id() / is should be used?
    #         if if_set_must_be_same and prev_frame_attr_value != this_frame_attr_value:
    #             raise EntityInternalError(owner=self,
    #                 msg=f"Attribute '{attr_name}' value in previous frame is different from current:\n  {previous_frame}\n    = {prev_frame_attr_value}\n<>\n  {self.frame}\n    = {this_frame_attr_value} ")


class IStackOwnerSession(ABC):
    """
    to override:
        stack_frames
        current_frame
    """
    stack_frames: List[IStackFrame] = field(repr=False, init=False, default_factory=list)
    current_frame: Optional[IStackFrame] = field(repr=False, init=False, default=None)

    STACK_FRAME_CLASS: ClassVar[type] = IStackFrame
    STACK_FRAME_CTX_MANAGER_CLASS: ClassVar[type] = UseStackFrameCtxManagerBase

    def use_stack_frame(self, frame: IStackFrame) -> UseStackFrameCtxManagerBase:
        if not isinstance(frame, self.STACK_FRAME_CLASS):
            raise EntityInternalError(owner=self, msg=f"Expected {self.STACK_FRAME_CLASS}, got frame: {frame}") 
        return self.STACK_FRAME_CTX_MANAGER_CLASS(owner_session = self, frame=frame)

    # def use_changed_current_stack_frame(self, **change_attrs) -> "UseStackFrameCtxManagerBase":
    #     """
    #     Removed from use - I prefer mutation-free / functional style instead. Was used only once.
    #     More info in @abstractmethod declaration on how this should work.
    #     """
    #     if not self.current_frame:
    #         raise EntityInternalError(owner=self, msg="No current frame")
    #     if not change_attrs:
    #         raise EntityInternalError(owner=self, msg="Expecting change_attrs")

    #     orig_attrs: Dict[str, Any] = {}
    #     for attr_name, attr_value in change_attrs.items():
    #         if not hasattr(self.current_frame, attr_name):
    #             raise EntityInternalError(owner=self, msg=f"Unknown attribute: {attr_name} in frame: {self.current_frame}")
    #         orig_attrs[attr_name] = getattr(self.current_frame, attr_name)
    #         setattr(self.current_frame, attr_name, attr_value)

    #     def restore_args_to_orig_callback():
    #         for attr_name, attr_value in orig_attrs.items():
    #             setattr(self.current_frame, attr_name, attr_value)

    #     return self.STACK_FRAME_CTX_MANAGER_CLASS(
    #                 owner_session = self,
    #                 frame=self.current_frame,
    #                 add_to_stack=False,
    #                 cleanup_callback=restore_args_to_orig_callback,
    #     )


    def push_frame_to_stack(self, frame: IStackFrame):
        self.stack_frames.insert(0, frame)
        self.current_frame = frame

    def pop_frame_from_stack(self) -> IStackFrame:
        # TODO: DRY - apply.py
        assert self.stack_frames
        ret = self.stack_frames.pop(0)
        self.current_frame = self.stack_frames[0] if self.stack_frames else None
        return ret




# ============================================================


@dataclass
class SetupStackFrame(IStackFrame):
    # current container
    container: IContainer = field(repr=False)

    # current component - can be DataModel too
    component: IComponent

    # # used for ThisNS in some cases
    # local_setup_session: Optional[ISetupSession] = field(repr=False, default=None)
    this_registry:  Optional[IThisRegistry] = field(repr=False, default=None)

    # Computed from container/component
    # used for DataModelWithHandlers cases (read_handlers()),
    data_model:    Optional[IDataModel] = field(init=False, repr=False, default=None)

    dexp_validator: Optional[DexpValidator] = field(repr=False, default=None)

    # -- autocomputed
    # data_model_root: Optional[IDataModel] = field(repr=False, init=False, default=None)
    # current container data instance which will be procesed: changed/validated/evaluated
    # type_info: TypeInfo


    # def __post_init__(self):
    #     self.clean()


    def copy_from_previous_frame(self, previous_frame: Self):
        assert previous_frame
        self._copy_attr_from_previous_frame(previous_frame, "this_registry",
                                            if_set_must_be_same=False)
        # check / init again
        # self.clean()


    def post_clean(self):
        if not isinstance(self.container, IContainer):
            raise EntityInternalError(owner=self, msg=f"Expected IContainer, got: {self.container}")
        if not isinstance(self.component, IComponent):
            raise EntityInternalError(owner=self, msg=f"Expected IComponent, got: {self.component}")

        if isinstance(self.component, IDataModel):
            self.data_model = self.component
        else:
            self.data_model = self.container.data_model

        if self.this_registry:
            assert isinstance(self.this_registry, IThisRegistry)

    # ------------------------------------------------------------

    # def set_local_setup_session(self, local_setup_session: ISetupSession, force: bool = False):
    #     if not force and self.local_setup_session:
    #         raise EntityInternalError(owner=self, msg=f"local_setup_session already set to '{self.local_setup_session}', got: '{local_setup_session}'")
    #     self.local_setup_session = local_setup_session

    def set_this_registry(self, this_registry: Optional[IThisRegistry], force: bool = False):
        # NOTE: in some cases existing and new this_registry can be None. Did not cover such combinations
        if not force and self.this_registry:
            raise EntityInternalError(owner=self, msg=f"this_registry already set to '{self.this_registry}', got: '{this_registry}'")
        self.this_registry = this_registry


# ------------------------------------------------------------

# TODO: put in Settings and use only in IApplyResult.settings ...
class GlobalConfig:
    ID_NAME_SEPARATOR: ClassVar[str] = "::"


    # ID_PREFIX_WHEN_INTERNAL: ClassVar[str] = "_IID_"

    # ID_KEY_COUNTER: ClassVar[int] = 0 # field(repr=False, init=False, default=0)
    # ID_KEY_COUNTER: ClassVar[ThreadSafeCounter] = ThreadSafeCounter()
    ID_KEY_PREFIX_FOR_MISSING: ClassVar[str] = "_MK_"


def get_instance_key_string_attrname_pair(key_string: str) -> Tuple[str, str]:
    idx = key_string.rfind(GlobalConfig.ID_NAME_SEPARATOR)
    if idx < 0:
        raise EntityInternalError(f"Invalid key_string='{key_string}'")
    instance_key_string = key_string[:idx]
    attrname = key_string[idx + len(GlobalConfig.ID_NAME_SEPARATOR):]
    return instance_key_string, attrname, 


# --------------------------------------------------
# Key types aliases
# --------------------------------------------------

class ChangeOpEnum(str, Enum):
    ADDED  = "ADDED"
    DELETE = "DELETE"
    UPDATE = "UPDATE"

@dataclass
class InstanceChange:
    """
    Used only for displaying detected changes
    use ApplyResult.get_changes() to get list of all changes.
    Use intesively in unit tests.
    """
    key_string: str # KeyString
    change_op: Union[ChangeOpEnum, str]
    # Next two are set when ChangeOpEnum.UPDATE
    init_value: Optional[AttrValue]
    value: Optional[AttrValue]

    def __post_init__(self):
        if isinstance(self.change_op, ChangeOpEnum):
            self.change_op = self.change_op.value

@dataclass
class InstanceAttrValue:
    # NOTE: value could be adapted version of dexp_result.value (can be different)
    value: LiteralType

    # * first / initial record in parent list is from bind_to, first
    #   this .value could be unadapted value version (field.try_adapt_value())
    #   therefore value is in a special field.
    # * second+ - are from evaluation results
    # Only in UNIT TESTS None is used
    dexp_result: Optional[ExecResult] = field(repr=False, compare=False, default=None)

    # TODO: this is not filled good - allways component.name
    # just to have track who read/set this value
    # Only in UNIT TESTS "" is used
    value_parent_name: str = field(repr=False, compare=False, default="")

    # is from bind_to or some other phase
    # Only in UNIT TESTS "UNDEFINED" is used
    value_set_phase: "ValueSetPhase" = field(repr=False, compare=False, default="UNEDFINED")

    def as_dict(self) -> Dict[str, Any]:
        return {
                "value": self.value,
                "dexp_result": self.dexp_result,
                "value_from_parent": self.value_parent_name,
                "is_from_bind": self.is_from_bind,
                }

# ------------------------------------------------------------

class ValueSetPhase(str, Enum):
    """
    Two phases:
        - initial - fill from bind_to, update by instance_new and adapt type
        - evaluation - run evaluations and change / fill the value
    It is important to preserve the same index INIT_ / EVAL_, since some logic depends on it,
    see IValueNode.set_value()
    """
    # initial value - filling from evaluation of model's .bind_to Dexp
    INIT_BY_BIND = "INIT_BY_BIND"

    # udpate by instance_new + adapting this new value with type adapter
    INIT_BY_NEW_INSTANCE = "INIT_BY_NEW_INSTANCE"

    # no changes by instance_new - change by adapting to type of an initial value
    INIT_ADAPT_TYPE_ONLY = "INIT_ADAPT_TYPE_ONLY"

    # direct use of set_value - to set value to NA_IN_PROGRESS - to detect circular dependency
    # done when traversing all components, before running evaluations
    # TODO: after proper order of evaluation sis setup already in setup phase, this detection will become obsolete.
    INIT_NA_IN_PROGRESS = "INIT_NA_IN_PROGRESS"

    # running evaluations and applying results
    EVAL_PHASE = "EVAL_PHASE"

    # in finish phase if defaults mode - setting to reasonable default => None
    EVAL_SET_NONE = "EVAL_SET_NONE"

    # used only for unit tests
    UNDEFINED = "UNDEFINED"

# ------------------------------------------------------------

@dataclass
class IValueNode(IDexpValueSource):

    instance: ModelInstanceType = field(repr=False)
    instance_none_mode: bool = field(repr=False)

    # ------------------------------------------------------------
    # Following are all accessible by . operator in ValueExpressions
    # ------------------------------------------------------------
    # TODO: fields? - only for container - includes all children's children
    #       except Items's children
    # should collect value history or not? defined by Settings.trace
    trace_value_history: bool = field(repr=False)

    # it is  init=False in EntityValueNode
    container: IContainer = field(repr=False)

    # init=False for EntityValueNode
    # <field>.Parent
    # - empty only on top tree node (Entity component)
    parent_node:  Optional[Self] = field(repr=False)

    # can be overridden - so must be last in the list of "obligatory" arguments
    component:  IComponent = field(repr=False)

    # use is_list() instead has_items
    #   when SubentityItemsValueNode (container=SubentityItems)
    #     has_items=True  - container - has filled .items[], has no children
    #   otherwise:
    #     has_items=False - individual item - with values, has children (some with values)

    # UPDATE by this instance - can be ENTITY_LIKE or MODELS_LIKE (same dataclass) struct
    # required but is None when not update
    instance_new: Union[ModelInstanceType, NoneType, UndefinedType] = field(repr=False, default=UNDEFINED)

    # set only when item of parent.items (parent is SubentityItems)
    # set only when item is deleted (DELETE) or item is added (ADDED)
    change_op: Optional[ChangeOpEnum] = field(repr=True, default=None)

    # set only when item of parent.items (parent is SubentityItems)
    # i.e. filled only when container list member
    index0: Optional[Index0Type] = field(repr=False, default=None)

    key: Optional[KeyType] = field(repr=False, default=None)

    # used when applying instance_new - will be copied from current node but with instance=instance_new
    # in this case it is not added to parent_node children
    copy_mode: bool = field(repr=False, default=False)

    # ------------------------------------------------------------
    # AUTOCOMPUTED
    # ------------------------------------------------------------
    # top of the tree - automatically computed, must be EntityValueNode
    top_node:  Self = field(repr=False, init=False, compare=False)
    key_string: KeyString = field(init=False, repr=True, default=UNDEFINED)

    # copy of component's name
    name:       AttrName = field(init=False, repr=True)

    # used on locating F.<field-value-node> from other containers
    parent_container_node: Union[Self, UndefinedType] = field(init=False, repr=False, compare=False, default=UNDEFINED)

    @abstractmethod
    def set_value(self, value: AttrValue, dexp_result: Optional[ExecResult], value_set_phase: ValueSetPhase) -> NoneType:
        ...

    @abstractmethod
    def get_instance_new_attr_value(self, component: IComponent) -> ExecResult:
        ...

    @abstractmethod
    def set_instance_attr_value(self):
        ...

    @abstractmethod
    def is_changed(self) -> bool:
        ...

    # @abstractmethod
    # def add_item(self, value_node: Self):
    #     ...

    @abstractmethod
    def add_child(self, value_node: Self):
        ...

    @abstractmethod
    def clean(self):
        ...

    @abstractmethod
    def dump_to_dict(self, depth: int=0) -> ComponentTreeWValuesType:
        ...

# ------------------------------------------------------------

@dataclass
class ApplyStackFrame(IStackFrame):
    """

    IMPORTANT NOTE: When adding new fields, consider adding to 

        UseApplyStackFrame .compare_with_previous_frame()

    """
    # current container data instance which will be procesed: changed/validated/evaluated
    # main container - class of instance - can be copied too but is a bit complicated
    # instance: InitVar[ModelInstanceType]
    # container: InitVar[IContainer] = field(repr=False)
    # component: InitVar[IComponent]

    # in value-node-tree current node - holds component/instance/container
    # can be NOT_APPLIABLE
    value_node: Union[IValueNode, UndefinedType] = field(repr=False, default=None)

    # ---------------------------------------------------------------------------------
    # Following if not provideed are copied from parent frame (previous, parent frame)
    # check UseApplyStackFrame. copy_from_previous_frame
    # ---------------------------------------------------------------------------------

    # UPDATE by this instance - can be ENTITY_LIKE or MODELS_LIKE (same dataclass) struct
    # required but is None when not update
    # instance_new_tmp: InitVar[Union[ModelInstanceType, NoneType, UndefinedType]] = field(repr=False, default=UNDEFINED)

    # partial mode - component subtree in processing
    # component is found and its subtree is being processed
    in_component_only_tree: bool = field(repr=False, default=False)

    # recursion depth - used for calling ._apply() in dependency tree branches 
    # e.g. DotExpression -> .get_current_value() -> ._apply()
    #
    # See doc for _apply() / mode_dexp_dependency -> recursion depth 
    # does not correspond with component's tree depth.
    depth: Optional[int] = field(repr=False, default = None)

    # used for cache only - holds current subtree
    # parent_values_subtree: Optional[ComponentTreeWValuesType] = field(init=False, repr=False, default=None)

    # TODO: this is ugly 
    # set only in single case: partial mode, in_component_only_tree, component=component_only, instance_new
    on_component_only: Optional[IComponent] = field(repr=False, default=None)

    # this is currently created on 1st level and then copied to each next level
    # instance_new_struct_type: Union[StructEnum, NoneType, UndefinedType] = field(repr=False)

    # filled only when container list member
    # index0_tmp: InitVar[Optional[Index0Type]] = None

    # parent instance / parent_instance_new - currenntly used for key_string logic
    # parent_instance: Optional[ModelInstanceType] = field(repr=False, default=None)
    # parent_instance_new: Optional[ModelInstanceType] = field(repr=False, default=None)

    # Usually used for ThisNS / This. namespace 
    # local_setup_session: Optional[ISetupSession] = field(repr=False, default=None)
    this_registry: Optional[IThisRegistry] = field(repr=False, default=None)

    # currently used only in cleaners - by default None to enable "automatic-copy",
    # if not set - then set to False in post_clean()
    # instance_is_list: Optional[bool] = field(repr=False, default=None)

    # --------------------
    # -- autocomputed
    # --------------------
    # internal - filled in __post_init__
    # key_string: Optional[str] = field(init=False, repr=False, default=None)

    # _component: IComponent = field(repr=False, init=False)
    # _container: IContainer = field(repr=False, init=False)
    # _instance: ModelInstanceType = field(repr=False, init=False)
    # _index0: Optional[Index0Type] = None
    # _instance_new: ModelInstanceType = field(repr=False, init=False)

    # used to check root value in models registry 
    data_model_root: Optional[IDataModel] = field(repr=False, init=False, default=None)

    # def __post_init__(self):
    #     # , instance_new_tmp: Any, index0: Optional[Index0Type], instance: ModelInstanceType, container: IContainer, component: IComponent
    #     # self._instance_new = instance_new_tmp if instance_new_tmp is not UNDEFINED else None
    #     # self._component = component
    #     # self._container = container
    #     # self._instance = instance
    #     # self._index0 = index0
    #     # NOTE: must be called only after copy() is done:
    #     # self.clean()

    @property
    def instance_new(self) -> ModelInstanceType:
        if not self.value_node:
            raise EntityInternalError(owner=self, msg="Attribute 'instance_new' can not be read, value_node not set.")
        return self.value_node.instance_new

    @property
    def index0(self) -> Optional[Index0Type]:
        if not self.value_node:
            raise EntityInternalError(owner=self, msg="Attribute 'index0' can not be read, value_node not set.")
        # NOTE: in some cases this was different, but value_node.index0 seemed correct.
        # assert self.value_node.index0 == self._index0:
        return self.value_node.index0

    @property
    def instance(self) -> ModelInstanceType:
        if not self.value_node:
            raise EntityInternalError(owner=self, msg="Attribute 'instance' can not be read, value_node not set.")
        return self.value_node.instance

    @property
    def container(self) -> IContainer:
        if not self.value_node:
            raise EntityInternalError(owner=self, msg="Attribute 'container' can not be read, value_node not set.")
        return self.value_node.container

    @property
    def component(self) -> IComponent:
        if not self.value_node:
            raise EntityInternalError(owner=self, msg="Attribute 'component' can not be read, value_node not set.")
        return self.value_node.component

    def clone(self) -> Self:
        # , instance_new_tmp=self._instance_new, index0=self._index0, instance=self._instance)  # , container=self._container, component=self._component)
        return dataclass_clone(self)

    def copy_from_previous_frame(self, previous_frame: Self):
        """
        if the instance is the same - consider from last frame
        container (copy/check), index0 (copy/check), component ...
        """
        assert previous_frame

        self._copy_attr_from_previous_frame(previous_frame, "in_component_only_tree",
                                            if_set_must_be_same=False)
        self._copy_attr_from_previous_frame(previous_frame, "depth",
                                            if_set_must_be_same=False)
        # self._copy_attr_from_previous_frame(previous_frame, "parent_values_subtree",
        #                                     if_set_must_be_same=False)
        self._copy_attr_from_previous_frame(previous_frame, "value_node",
                                            if_set_must_be_same=False)

        # do not use ==, compare by instance (ALT: use id(instance) )
        if self.instance is previous_frame.instance:
            # self._copy_attr_from_previous_frame(previous_frame, "container", may_be_copied=False)
            # will be autocomputed
            # self._copy_attr_from_previous_frame(previous_frame, "data_model_root", may_be_copied=False)
            # self._copy_attr_from_previous_frame(previous_frame, "instance_new")
            # self._copy_attr_from_previous_frame(previous_frame, "instance_is_list")
            # self._copy_attr_from_previous_frame(previous_frame, "_index0")

            # only these can be copied
            # self._copy_attr_from_previous_frame(previous_frame, "parent_instance")
            # self._copy_attr_from_previous_frame(previous_frame, "parent_instance_new")

            # do not use ==, compare by instance (ALT: use id(instance) )
            if self.component is previous_frame.component:
                self._copy_attr_from_previous_frame(previous_frame, "on_component_only", may_be_copied=False)
                # self._copy_attr_from_previous_frame(previous_frame, "key_string", may_be_copied=False)
                self._copy_attr_from_previous_frame(previous_frame, "this_registry",
                                                    # for Apply -> ChildrenValidation setup can be different
                                                    if_set_must_be_same=False)
        # check / init again
        # self.clean()


    def post_clean(self):
        # new logic - must be sure
        # if not isinstance(self._container, IContainer):
        #     raise EntityInternalError(owner=self, msg=f"Expected IContainer, got: {self.container}")
        # if not isinstance(self.component, IComponent):
        #     raise EntityInternalError(owner=self, msg=f"Expected IComponent, got: {self.component}")
        # if self.index0 is not None and self.index0 < 0:
        #     raise EntityInternalError(owner=self, msg=f"index0 must be integer >= 1, got: {self.index0}")

        if self.value_node is not NOT_APPLIABLE:
            if not self.value_node:
                raise EntityInternalError(owner=self, msg="value_node not set")

            if not self.value_node.component is self.component:
                raise EntityInternalError(owner=self, msg=f"value_node component is different:\n {self.value_node.component} \n !=\n {self.component}")

            self.value_node.clean()

        # if self.instance_is_list is None:
        #     self.instance_is_list = False

        # TODO: DRY - similar logic in ApplyResult._detect_instance_new_struct_type()
        if self.on_component_only:
            self.data_model_root = (self.on_component_only 
                                     if self.on_component_only.is_subentity_items()
                                     else self.on_component_only.get_first_parent_container(consider_self=True)
                                    ).data_model
            # can be list in this case
            # instance_to_test = self.instance[0] \
            #                    if isinstance(self.instance, (list, tuple)) \
            #                    else self.instance
        else:
            self.data_model_root = self.container.data_model
            component = self.value_node.component
            if component.is_entity() or component.is_subentity_any():
                # check instance type only once per container
                assert self.instance is not None
                # if self.instance_is_list and self.instance is not None:
                if self.value_node.is_list():
                    if not isinstance(self.instance, (list, tuple)):
                        raise EntityInternalError(owner=self, msg=f"Expected list of model instances in 'instance', got: {self.instance}")
                    if self.instance_new is not None and not isinstance(self.instance_new, (list, tuple)):
                        raise EntityInternalError(owner=self, msg=f"Expected list of model instances in 'instance_new', got: {self.instance_new}")
                #     instance_to_test = self.instance[0] if self.instance else None
                # else:
                #     instance_to_test = self.instance

        # if instance_to_test is NA_DEFAULTS_MODE:
        #     # defaults_mode
        #     pass
        # elif instance_to_test is None:
        #     # if instance is a list and it is empty, then nothing to check
        #     pass
        # else:
        #     pass
        #     # TODO: this should be checked only once per container, and I am not sure what is good way to do it
        #     # # if not is_model_klass(instance_to_test.__class__):
        #     # #     raise EntityInternalError(owner=self, msg=f"Expected model instance: {instance_to_test.__class__} or list[instances], got: {self.instance}")
        #     # self._accessor.validate_instance_type(owner_name=self.component.name,
        #     #                                                      instance=instance_to_test,
        #     #                                                      model_klass=self.container.data_model.model_klass )

        if self.this_registry:
            assert isinstance(self.this_registry, IThisRegistry)

        assert self.data_model_root


    # def set_parent_values_subtree(self, parent_values_subtree: Union[Dict, List]) -> Union[Dict, List]:
    #     " returns original values "
    #     # NOTE: didn't want to introduce a new stack frame layer for just changing one attribute
    #     # part of ComponentTreeWValuesType]
    #     parent_values_subtree_orig = self.parent_values_subtree
    #     self.parent_values_subtree = parent_values_subtree
    #     return parent_values_subtree_orig


    # def set_local_setup_session(self, local_setup_session: ISetupSession):
    #     if self.local_setup_session:
    #         raise EntityInternalError(owner=self, msg=f"local_setup_session already set to '{self.local_setup_session}', got: '{local_setup_session}'")
    #     self.local_setup_session = local_setup_session

    def set_this_registry(self, this_registry: IThisRegistry):
        if self.this_registry:
            raise EntityInternalError(owner=self, msg=f"this_registry already set to '{self.this_registry}', got: '{this_registry}'")
        self.this_registry = this_registry



@dataclass
class ValidationFailure:
    component_key_string: str
    error: TransMessageType
    validation_name: str
    validation_title: str
    details: Optional[str]

# ------------------------------------------------------------

class StructEnum(str, Enum):
    # models like - follows flat, storage/db like structure
    MODELS_LIKE = "MODELS"
    # entity like - follows hierachical structure like defined in entity
    ENTITY_LIKE  = "ENTITY"

# TODO: consider using classes instead, e.g.
#       class InputStructBase:
#           pass
#       class ModelInputStruct(InputStructBase):
#           # models like - follows flat, storage/db like structure
#           pass
#       class EntityInputStruct(InputStructBase):
#           # entity like - follows hierachical structure like defined in entity
#           pass


class ApplyExecPhasesEnum(str, Enum):
    EVALUATIONS_PHASE = "EVALUATIONS_PHASE"
    VALIDATIONS_PHASE = "VALIDATIONS_PHASE"
    # FINISH_COMPONENTS_PHASE= "FINISH_COMPONENTS_PHASE"


@dataclass
class ApplyExecCleaners:
    """
    contains all cleaners for single component + stack frame
    """
    stack_frame: IStackFrame = field(repr=False)
    cleaner_list: List[ICleaner]
    # executor_fn: ExecutorFnType = field(repr=False)

@dataclass
class ApplyExecCleanersRegistry:
    exec_cleaners_dict: Dict[ApplyExecPhasesEnum, List[ApplyExecCleaners]] = field(init=False, repr=False, default_factory=dict)
    finished: bool = field(init=False, repr=False, default=False)

    def finish(self):
        assert not self.finished
        self.finished = True

    def get_cleaners(self, apply_exec_phase: ApplyExecPhasesEnum) -> List[ApplyExecCleaners]:
        return self.exec_cleaners_dict.setdefault(apply_exec_phase, [])

    # def register_stack_frame_only(self,
    #                               apply_exec_phase: ApplyExecPhasesEnum,
    #                               stack_frame: IStackFrame):
    #     assert not self.finished
    #     exec_list = self.exec_cleaners_dict.setdefault(apply_exec_phase, [])
    #     exec_cleaner = ApplyExecCleaners(cleaner_list=[], stack_frame=stack_frame)
    #     exec_list.append(exec_cleaner)

    def register_cleaner_list(self,
                              apply_exec_phase: ApplyExecPhasesEnum,
                              stack_frame: IStackFrame,
                              cleaner_list: List[ICleaner]):
        assert not self.finished
        exec_list = self.exec_cleaners_dict.setdefault(apply_exec_phase, [])
        exec_cleaner = ApplyExecCleaners(cleaner_list=cleaner_list, stack_frame=stack_frame)
        exec_list.append(exec_cleaner)


@dataclass
class IApplyResult(IStackOwnerSession):
    entity: IEntity = field(repr=False)
    instance: Any = field(repr=False)
    # TODO: consider: instance_new: Union[ModelInstanceType, UndefinedType] = UNDEFINED,
    instance_new: Optional[ModelInstanceType] = field(repr=False)

    settings: Optional[Settings] = field(repr=False)
    # used in apply_partial
    component_name_only: Optional[str] = field(repr=False, default=None)

    # used in dump_defaults() - to get defaults dictionary
    defaults_mode: bool = field(repr=False, default=False)

    accessor: Optional[ValueAccessorInputType] = field(repr=False, default=None)

    # ---- automatically computed -----

    # top of value tree - head of the tree
    top_value_node: IValueNode = field(repr=False, init=False, default=UNDEFINED)

    # list of all value trees - currently no need to have Dict
    value_node_list: List[IValueNode] = field(repr=False, init=False, default_factory=list)

    # extracted from component
    data_model: IDataModel = field(repr=False, init=False)

    # will be extracted from self.settings.accessor, None if not set
    _apply_accessor: Optional[IValueAccessor] = field(init=False, default=UNDEFINED, repr=False)

    # final status
    finished: bool = field(repr=False, init=False, default=False)

    # ---- internal attrs and structs ----

    # stack of frames - first frame is current. On the end of the process the
    # stack must be empty
    stack_frames: List[ApplyStackFrame] = field(repr=False, init=False, default_factory=list)
    current_frame: Optional[ApplyStackFrame] = field(repr=False, init=False, default=None)

    # I do not prefer attaching key_string to instance i.e. setattr(instance, key_string)
    # but having instance not expanded, but to store in a special structure
    # key_string_container_cache: Dict[InstanceId, KeyString] = \
    #                         field(repr=False, init=False, default_factory=dict)

    # used when when collecting list of instances/attributes which are updated
    # instance_by_key_string_cache: Dict[KeyString, ModelInstanceType] = \
    #                         field(repr=False, init=False, default_factory=dict)

    # TODO: consider this:
        # Instance can be dataclass, dictionary and so on.
        # Accessing instance attribute values are done in 2 places:
        #   a) getting/setting in processing children
        #   b) getting in executing DotExpression's AttributeNode-s
        # for a) _accessor is used in all cases,
        # for b) not-attribute-instance it is too complicated to get values
        # with _accessor, so input instance is copied to this shadow
        # dataclass instance which is used for AttributeNode's DotExpression-s.
        # instance_shadow_dc: ModelInstanceType = field(init=False, repr=False)

    # Registry of history of attribute values for each IValueNode.
    # Filled only when Settings.trace =True (see .register_instance_attr_change()).
    # Use .get_value_history() to fetch values.
    # Used only for analytical purposes and unit testing.
    # For any serious jobs IValueNode-s are used i.e. value_node_list and top_value_node members.
    #
    # Contains initial (first member), intermediate and
    # last values of each instance field/attribute.
    # If initial/first is different from final/last, then this value is changed.
    #
    # See .get_changes() for simplier struct - it does not require special Settings param set.
    _value_history_dict: Dict[KeyString, List[InstanceAttrValue]] = \
                            field(repr=False, init=False, default_factory=dict)

    # Final list of created/deleted/updated list of changes - should be used only for display
    # Do not use directly, use get_changes() - list is lazy initialized and cached.
    # For any serious job use value_node_list or top_value_node
    #
    # See _value_history_dict for more complex struct - it requires special Settings param previously set.
    _instance_change_list: Union[List[InstanceChange], UndefinedType] = field(repr=False, init=False, default=UNDEFINED)

    # When first argument is <type> and second is <type> then call function Callable
    # which will adapt second type to first one. Example: <string> + <int> -> <string> + str(<int>)
    binary_operations_type_adapters: Dict[Tuple[type, type], Callable[[Any], Any]] = field(repr=False, init=False, default_factory=dict)

    # on validation errors this will be filled
    errors: Dict[KeyString, ValidationFailure] = field(repr=False, init=False, default_factory=dict)

    # for instance_new (root struct), is set, in normal mode set in init, in partial mode set on component detection
    instance_new_struct_type: Optional[StructEnum] = field(repr=False, init=False, default=None)

    # computed from component_name_only
    component_only: Optional[IComponent] = field(repr=False, init=False, default=None)

    # used for cache only
    _component_children_upward_dict: \
            Optional[
                Dict[
                    ComponentNameType,
                    Dict[ComponentNameType, IComponent]
                ]
            ]= field(init=False, repr=False, default_factory=dict)

    # used for cache only - holds complete tree
    _values_tree: Optional[ComponentTreeWValuesType] = field(init=False, repr=False, default=None)
    # values_tree_by_key_string: Optional[Dict[KeyString, ComponentTreeWValuesType]] = field(init=False, repr=False, default=None)

    new_id_counter: int = field(init=False, repr=False, default=0)

    exec_cleaners_registry: ApplyExecCleanersRegistry = field(init=False, repr=False, default_factory=ApplyExecCleanersRegistry)

    def _get_new_id(self) -> int:
        self.new_id_counter+=1
        return self.new_id_counter

    def get_setup_session(self) -> ISetupSession:
        if self.current_frame is None:
            raise EntityInternalError(owner=self, msg="Setup session can't be get, no current frame set")
        return self.current_frame.container.setup_session

    @abstractmethod
    def get_current_value(self, strict: bool) -> LiteralType:
        ...
    @abstractmethod
    def get_key_string(self) -> KeyString:
        ...

    @abstractmethod
    def get_values_tree(self, key_string: Optional[KeyString] = None) -> ComponentTreeWValuesType:
        ...

    @abstractmethod
    def get_changes(self) -> List[InstanceChange]:
        ...

    @property
    def value_history_dict(self) -> Dict[KeyString, List[InstanceAttrValue]]:
        # NOTE: trace can be set in apply_settings. if similar check will be required, then use some other method.
        #   if not self.entity.settings.is_trace():
        #       raise EntityApplyError(owner=self, msg="Value history is not collected. Pass Settings(..., trace=True) and try again.")
        # assert self._value_history_dict
        return self._value_history_dict

    # def get_current_component_bind_value(self):
    #     component = self.current_frame.component
    #     bind_dexp_result = component.get_dexp_result_from_instance(apply_result=self)
    #     return bind_dexp_result.value

    def validate_type(self, component: IComponent, value: Any = UNDEFINED):
        " only Fields can have values - all others components are ignored "
        validation_failure = None
        if isinstance(component, IField):
            validation_failure = component.validate_type(apply_result=self, value=value)
            if validation_failure:
                if self.defaults_mode:
                    raise EntityInternalError(owner=self,
                                          msg="TODO: Type validation failed in defaults_mode - probably should default value to None ")
                self.register_instance_validation_failed(component, validation_failure)

        return validation_failure


    @abstractmethod
    def _register_instance_attr_change(self,
                                       component: IComponent,
                                       dexp_result: ExecResult,
                                       new_value: Any,
                                       value_set_phase: ValueSetPhase,
                                       ) -> InstanceAttrValue:
        ...


    def register_value_node(self, value_node: IValueNode):
        assert not self.finished
        self.value_node_list.append(value_node)


    def finish(self):
        if self.finished:
            raise EntitySetupError(owner=self, msg="Method finish() already called.")

        if not len(self.stack_frames) == 0:
            raise EntityInternalError(owner=self, msg=f"Stack frames not released: {self.stack_frames}")

        self.exec_cleaners_registry.finish()

        for value_node in self.value_node_list:
            value_node.mark_initialized()

        self.finished = True

    def register_instance_validation_failed(self, component: IComponent, failure: ValidationFailure):
        if failure.component_key_string not in self.errors:
            self.errors[failure.component_key_string] = []
        self.errors[failure.component_key_string].append(failure)
        # TODO: mark invalid all children and this component


    # def get_upward_components_dict(self, component: IComponent) \
    #         -> Dict[ComponentNameType, IComponent]:
    #     # CACHE
    #     if component.name not in self._component_children_upward_dict:
    #         components_tree = []
    #         curr_comp = component
    #         while curr_comp is not None:
    #             if curr_comp in components_tree:
    #                 raise EntityInternalError(owner=component, msg=f"Issue with hierarchy tree - duplicate node: {curr_comp.name}")
    #             components_tree.append(curr_comp)
    #             curr_comp = curr_comp.parent

    #         children_dict = {}
    #         # Although no name clash could happen, reverse to have local scopes
    #         # overwrite parent's scopes.
    #         for curr_comp in reversed(components_tree):
    #             comp_children_dict = curr_comp.get_children_tree_flatten_dict()
    #             children_dict.update(comp_children_dict)

    #         self._component_children_upward_dict[component.name] = children_dict

    #     return self._component_children_upward_dict[component.name]

    def get_changes_as_list_dict(self) -> List[Dict[str, Any]]:
        """
        used only in unit-tests for now
        """
        return [asdict(instance_change) for instance_change in self.get_changes()]

# ------------------------------------------------------------

def extract_type_info_for_attr_name(
        # inspect_object: Union[IDataModel, IDotExpressionNode, DotExpression, TypeInfo, Callable, ModelKlassType],
        inspect_object: Union[TypeInfo, ModelKlassType, Any],
        attr_node_name: str
        ) -> Tuple[TypeInfo, Optional[ModelField]]:
    # Optional[IFunctionDexpNode]  # noqa: F821
    """
    TODO: this function is a mess, do something

    Main logic for extraction from parent object (dataclass, py class, dexp
    instances) member by name 'attr_node_name' -> data (struct, plain value),
    or IFunctionDexpNode instances 

    This function uses specific base interfaces/classes (IDataModel,
    IDotExpressionNode), so it can not be put in meta.py

    See 'meta. def get_or_create_by_type()' for further explanation when to use
    this function (preffered) and when directly some other lower level meta.py
    functions.
    """
    # OLD: if is_function(inspect_object):
    #     # Function - callable and not class and not pydantic?
    #     raise EntityInternalError(msg=f"Got native python function, use Function() objects instead. Got: {inspect_object}")
    # OLD: if isinstance(parent_object, DotExpression):
    #     raise EntityInternalError(item=inspect_object, msg=f"Parent object is DotExpression: '{parent_object}' / '{attr_node_name}'.")

    # expecting standard object - hopefully with with type hinting
    parent_object = inspect_object

    if isinstance(parent_object, TypeInfo):
        # go one level deeper
        # if not isinstance(parent_object.type_, type):
        #     raise EntitySetupValueError(item=inspect_object, msg=f"Inspected object's type hint is not a class object/type: {parent_object.type_}: {parent_object.type_}, got: {type(parent_object.type_)} ('.{attr_node_name}' process)")
        # Can be method call, so not pydantic / dataclass are allowed too
        parent_object = parent_object.type_

    if is_method_by_name(parent_object, attr_node_name):
        raise EntitySetupNameError(item=inspect_object,
                    msg=f"Inspected object's is a method: {parent_object}.{attr_node_name}. "
                         "Calling methods on instances are not allowed.")

    if is_model_klass(parent_object):
        # --- Model klass - try to extract attribute by name from py-type-hint
        # if not hasattr(parent_object, "__annotations__"):
        #     raise EntityInternalError(item=inspect_object, msg=f"'DC/PYD class {parent_object}' has no metadata (__annotations__ / type hints), can't read '{attr_node_name}'. Add type-hints or check names.")
        model_klass: ModelKlassType = parent_object
        parent_py_type_hints = extract_py_type_hints(model_klass, f"setup_session->{attr_node_name}:DC/PYD")
        py_type_hint = parent_py_type_hints.get(attr_node_name, None)
        if py_type_hint:
            type_info = TypeInfo.get_or_create_by_type(
                            py_type_hint=py_type_hint,
                            caller=parent_object,
                            )
        else:
            # --- Attribute not found - will trigger error report in the next section
            type_info = None

        # === Dataclass / pydantic field metadata, fields used in error report only
        th_field, fields = extract_model_field_meta(inspect_object=model_klass, attr_node_name=attr_node_name)
    else:
        # --- Primitive type (e.g. int) or some other non-model klass - will trigger error in next section
        type_info = None
        th_field = None
        fields = []

    if not type_info:
        if not fields:
            fields = get_model_fields(parent_object, strict=False)
        field_names = list(fields.keys())
        # (m) := model instance - to differentiate from AttrDexpNode / ValueNode / Component's members
        valid_names = f"Valid attributes (m): {get_available_names_example(attr_node_name, field_names)}" if field_names else "Type has no attributes at all."
        raise EntitySetupNameNotFoundError(msg=f"Type object {parent_object} has no attribute '{attr_node_name}'. {valid_names}")

    if not isinstance(type_info, TypeInfo):
        raise EntityInternalError(msg=f"{type_info} is not TypeInfo")

    return type_info, th_field


# ------------------------------------------------------------
# OBSOLETE
# ------------------------------------------------------------
#
#     def _invoke_component_setup(self,
#                                 component_name: str,
#                                 component: Union[Self, DotExpression],
#                                 setup_session: ISetupSession):  # noqa: F821
#         called = False
#
#         if isinstance(component, (DotExpression,)):
#             dexp: DotExpression = component
#             # namespace = dexp._namespace
#             if dexp.IsFinished():
#                 # Setup() was called in container.setup() before or in some other dependency
#                 called = False
#             else:
#                 dexp.Setup(setup_session=setup_session, owner=self)
#                 called = True
#         elif isinstance(component, IComponent):
#             assert "Entity(" not in repr(component)
#             component.setup(setup_session=setup_session)  # , parent=self)
#             called = True
#         elif isinstance(component, (dict, list, tuple)):
#             raise EntitySetupValueError(owner=self, msg=f"components should not be dictionaries, lists or tuples, got: {type(component)} / {component}")
#         else:
#             if hasattr(component, "Setup"):
#                 raise EntityInternalError(owner=self, msg=f"{self.name}.{component_name} has attribute that is not DotExpression: {type(component)}")
#             if hasattr(component, "setup"):
#                 raise EntityInternalError(owner=self, msg=f"{self.name}.{component_name} has attribute that is not Component: {type(component)}")
#         return called
#
#
#
