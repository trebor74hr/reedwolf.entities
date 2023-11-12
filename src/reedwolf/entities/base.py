# from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from contextlib import AbstractContextManager
from enum import Enum
from collections import OrderedDict
from typing import (
    List,
    Any,
    Dict,
    Union,
    Optional,
    Tuple,
    ClassVar,
    Callable,
    Type as TypingType, Type,
)
from dataclasses import (
    dataclass,
    field,
    field as DcField,
    fields,
    MISSING as DC_MISSING,
    make_dataclass,
    replace as dataclass_clone,
    asdict, is_dataclass,
)
from types import (
    MappingProxyType,
)

from .utils import (
    snake_case_to_camel,
    to_repr,
    add_yaml_indent_to_strlist,
    YAML_INDENT,
    varname_to_title,
    UNDEFINED,
    NA_DEFAULTS_MODE,
    UndefinedType,
    get_available_names_example,
    DumpFormatEnum,
    dump_to_format, NOT_APPLIABLE, NA_IN_PROGRESS,
)
from .exceptions import (
    EntityInternalError,
    EntitySetupTypeError,
    EntitySetupError,
    EntitySetupNameError,
    EntitySetupValueError,
    EntitySetupNameNotFoundError,
    EntityApplyError, 
    EntityApplyValueError,
)
from .namespaces import (
    DynamicAttrsBase,
)
from .meta import (
    Self,
    MetaTree,
    ComponentNameType,
    ComponentTreeType,
    ComponentTreeWValuesType,
    NoneType,
    ModelField,
    TypeInfo,
    LiteralType,
    get_model_fields,
    is_function,
    is_method_by_name,
    is_model_class,
    ModelType,
    DataclassType,
    extract_model_field_meta,
    extract_py_type_hints,
    STANDARD_TYPE_LIST,
    TransMessageType,
    get_dataclass_fields,
    InstanceId,
    KeyString,
    AttrName,
    AttrValue,
    Index0Type,
    KeyType, get_dataclass_field_type_info, make_dataclass_with_optional_fields,
)
from .expressions import (
    DotExpression,
    ExecResult,
    IDotExpressionNode,
    IFunctionDexpNode,
    ISetupSession,
    IThisRegistry,
    DexpValidator,
)
from .contexts import (
    IContext,
)
from .config import (
    Config,
)
from .values_accessor import IValueAccessor

# ------------------------------------------------------------

MAX_RECURSIONS: int = 30
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


def get_name_from_bind(bind: DotExpression):
    if len(bind.Path) <= 2:
        # Dexpr(Person.name) -> name
        name = bind._name
    else:
        # Dexpr(Person.address.street) -> address.street
        # TODO: this is messy :( - should be one simple logic ...
        name = "__".join([bit._name for bit in bind.Path][1:])
    assert name
    return name


# ------------------------------------------------------------

class AttrDexpNodeTypeEnum(str, Enum):
    CONTAINER = "CONTAINER"
    FIELD     = "FIELD"
    COMPONENT = "COMPONENT"
    DATA      = "DATA"
    FUNCTION  = "FUNCTION"

    VEXP      = "VEXP"
    VEXP_FUNC = "VEXP_FUNC"
    # VALIDATOR = "VALIDATOR"
    # EVALUATOR = "EVALUATOR"
    MODEL_CLASS = "MODEL_CLASS"
    TH_FUNC   = "TH_FUNC"
    TH_FIELD  = "TH_FIELD"

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

    # Containers (Entity/SubEntitySingle) + FieldGroup + Field with children (BooleanField)
    # -------------------------------------------------------------------------------------
    # Applies to components which have children (list of components, e.g. in
    # contiains)
    CHILDREN_ATTR_NAME = "Children" 

    # SubEntityItems 
    # ---------------------------------------------------------------------------
    # Applies to components which can have list of items (SubEntities)
    ITEMS_ATTR_NAME = "Items" 

    # TODO: reach 1 level deep:
    #   PARENT_ATTR_NAME = "Parent" 
    #   CHILDREN_KEY = "__children__" 

    # TODO: traversing?:
    #   PARENT_PATH_ATTR_NAME = "ParentPath" 
    #   ITEMS_TREE_ATTR_NAME = 
    #   CHILDREN_TREE_ATTR_NAME = "ParentPath" 
    #   VALUE_TREE_ATTR_NAME = 

# ------------------------------------------------------------

# class ReservedArgumentNames(str, Enum):
#     # INJECT_COMPONENT_TREE = "inject_component_tree"
#     ...

# ------------------------------------------------------------
# Message functions
# ------------------------------------------------------------

def _(message: str) -> TransMessageType:
    return TransMessageType(message)

# TODO: add type hint: TransMessageType -> TranslatedMessageType
# TODO: accept "{dot_node}" - can be a security issue, attr_nodes() should not make any logic
#       use .format() ... (not f"", btw. should not be possible anyway)

def msg(message: Union[str, TransMessageType]) -> Union[str, TransMessageType]:
    # TODO: should fill arguments {} and {name}
    return message


# ------------------------------------------------------------
# IBaseComponent
# ------------------------------------------------------------

@dataclass
class IBaseComponent:
    """ requires (Protocol):
        name
        parent_name
        parent
        value_accessor
        entity
    """
    parent        : Union[Self, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    parent_name   : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)
    entity        : Union[Self, "IContainer"] = field(init=False, default=UNDEFINED, repr=False)
    value_accessor: Union[IValueAccessor, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    # TODO: name          : Optional[str] = field(init=False, default=None)

    name_counter_by_parent_name: Dict[str, int] = field(init=False, repr=False, default_factory=dict)
    # value_accessor_default: IValueAccessor = field(init=False, repr=False)

    # ------------------------------------------------------------

    def set_parent(self, parent: Optional["IComponent"]):
        if self.parent is not UNDEFINED:
            raise EntityInternalError(owner=self, msg=f"Parent already defined, got: {parent}")

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

        if self.parent is None:
            self.entity = self
            if not self.value_accessor:
                self.value_accessor = self.entity.value_accessor_default
        else:
            # just copy the same object
            self.entity = self.parent.entity
            if not self.value_accessor:
                self.value_accessor = self.parent.value_accessor

        assert self.value_accessor and self.entity


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
                assert getattr(self, "bind", None)
                # ModelsNs.person.surname -> surname
                this_name = get_name_from_bind(self.bind)
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

    # ------------------------------------------------------------

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

# ------------------------------------------------------------
# Subcomponent
# ------------------------------------------------------------
@dataclass
class Subcomponent:
    # TODO: strange class - check if really required or explain if yes
    name        : str # orig: dexp_node_name
    path        : str # orig: var_path
    # TODO: can be some other types too
    component: Union["IComponent", DotExpression]
    th_field    : Optional[ModelField]


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

@dataclass
class ChildField:
    """ 
    Will be used and exposed in ChildrenValidation - where List[ChildField]
    will be put available to dot-chain and validation functions.
    Following fields will be available to these functions using This.
    namespacee:
        - Name - model class attribute name
        - Value - current model instance attribute value
        - Type - model class attribute python type
    """
    Name: AttrName
    _type_info: TypeInfo = field(repr=False)
    Type: TypingType = field(init=False)
    Value: AttrValue = field(init=False, default=UNDEFINED)

    def __post_init__(self):
        self.Type = self._type_info.type_

    def set_value(self, value: AttrValue):
        if self.Value != UNDEFINED:
            raise EntityApplyValueError(owner=self, msg=f"Value already set to '{self.Value}', got: '{value}'")
        self.Value = value


class IComponentFields:
    """
    Used only for dynamically created dataclass classes that holds only fields from owner component's.
    See make_component_fields_dataclass()
    """
    ...


def make_component_fields_dataclass(class_name: str, child_field_list: List[ChildField]) -> TypingType[IComponentFields]:
    """ 
    Dynamically create Dataclass from List[ChildField] based on origin model.
    It is basedd for SubentitySingle and FieldGroup to create TypeInfo instance
    based only on the fields for which there are Field
    """
    # name, type, optional[field]
    # ('z', int, field(default=5))],
    children_fields = [(child_field.Name, child_field.Type)
                        for child_field in child_field_list]
    new_dataclass: TypingType = make_dataclass(
                cls_name=class_name, 
                fields=children_fields, 
                bases=(IComponentFields,))
    assert issubclass(new_dataclass, IComponentFields)
    return new_dataclass


# ------------------------------------------------------------
# IComponent
# ------------------------------------------------------------

@dataclass
class IComponent(IBaseComponent, ABC):

    # NOTE: I wanted to skip saving parent reference/object within component - to
    #       preserve single and one-direction references.
    # NOTE: Not DRY: Entity, SubentityBase and ComponentBase

    # TODO: name          : Optional[str] = field(init=False, default=None)
    # TODO: cleaners:       Optional[List["ICleaner"]] = field(repr=False, init=False, default=None)

    # lazy init - done in Setup phase
    child_field_list: Optional[List[ChildField]] = field(init=False, repr=False, default=None)
    _component_fields_dataclass: Optional[TypingType[IComponentFields]] = field(init=False, repr=False, default=None)
    _this_registry: Union[IThisRegistry, NoneType, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)
    _setup_phase_one_called: bool = field(init=False, repr=False, default=False)


    def __post_init__(self):
        self.init_clean_base()
        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.register(self)

    def init_clean_base(self):
        # when not set then will be later defined - see set_parent()
        if self.name not in (None, "", UNDEFINED):
            if not self.name.isidentifier():
                raise EntitySetupValueError(owner=self, msg="Attribute name needs to be valid python identifier name")


    def _check_cleaners(self, allowed_cleaner_base_list: List[TypingType]):
        allowed_cleaner_base_list = tuple(allowed_cleaner_base_list)
        if self.cleaners is not None:
            cl_names = ", ".join([cl.__name__ for cl in allowed_cleaner_base_list])
            if not isinstance(self.cleaners, (list, tuple)):
                raise EntitySetupTypeError(owner=self, msg=f"Cleaners should be None or list of {cl_names}, got: {type(self.cleaners)} / {to_repr(self.cleaners)}") 
            for cleaner in self.cleaners:
                if not isinstance(cleaner, allowed_cleaner_base_list):
                    raise EntitySetupTypeError(owner=self, msg=f"Cleaners should be instances of {cl_names}, got: {type(cleaner)} / {cleaner}") 

    def create_this_registry(self, setup_session: ISetupSession) -> Optional[IThisRegistry]:
        """
        default implementation
        """
        # TODO: resolve circular dependency by moving this function logic to
        #       .registries.py/.setup.py or somewhere else
        from .registries import (
            ThisRegistry,
        )

        has_children = bool(self.get_children())

        if not has_children:
            raise EntityInternalError(owner=self, msg="Non-fields should have children")

        this_registry = ThisRegistry(component=self)
        # # and not self.is_bound_model()
        # if self.is_subentity_items():
        #     # Items -> This.Items
        #     this_registry = ThisRegistry(component=self)

        # # TODO: add .Instance ==> BoundModel instance
        # # elif self.is_subentity_single():
        # #     # Children -> This.Children + This.<all-attributes> + This.Instance
        # #     this_registry = ThisRegistryForInstanceAndChildren(owner=self) # , setup_session=setup_session)
        # else:
        #     # Children -> This.Children + This.<all-attributes>
        #     this_registry = ThisRegistry(component=self) # , setup_session=setup_session

        return this_registry


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

    @staticmethod
    def is_top_container() -> bool:
        """ different from is_top_parent since is based on type not on parent setup"""
        return False

    @staticmethod
    def is_container() -> bool:
        return False

    @staticmethod
    def is_bound_model() -> bool:
        return False

    @staticmethod
    def is_subentity_items() -> bool:
        return False

    @staticmethod
    def is_subentity_single() -> bool:
        return False

    def is_subentity(self) -> bool:
        return self.is_subentity_items() or self.is_subentity_single()

    @staticmethod
    def is_fieldgroup() -> bool:
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

    def get_children(self, deep_collect: bool=False) -> List[Self]:
        """
        in deep_collect mode:
            CACHED
            will collect children
            and children's children for those which have
                may_collect_my_children
                Examples:
                    - FieldGroup
                    - SubEntitySingle

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
            if not hasattr(self, "_children_deep"):
                children = []
                # RECURSION - 1 level
                for child in self.get_children():
                    children.append(child)
                    if child.may_collect_my_children():
                        # RECURSION
                        child_children = child.get_children(deep_collect=True)
                        if child_children:
                            children.extend(child_children)
                self._children_deep = children
            out = self._children_deep
        else:
            if not hasattr(self, "_children"):
                children = getattr(self, "contains", None)
                if not children:
                    children = getattr(self, "enables", None)
                else:
                    assert not hasattr(self, "enables"), self
                self._children = children if children else []
            out = self._children
        return out


    # ------------------------------------------------------------

    def get_children_dict(self) -> Dict[ComponentNameType, Self]:
        """
        only direct children in flat dict
        """
        if not hasattr(self, "_children_dict"):
            self._children_dict = {comp.name : comp for comp in self.get_children()}
        return self._children_dict

    # ------------------------------------------------------------

    def get_children_tree_flatten_dict(self, depth:int=0) -> Dict[ComponentNameType, Self]:
        """
        will go recursively through every children and
        fetch their "children" and collect to output structure:
        selects not-subentity_itemss, put in flat dict (i.e. children with
        model same level fields), excludes self 
        """
        key = "_children_tree_flatten_dict"
        if not hasattr(self, key):
            if depth > MAX_RECURSIONS:
                raise EntityInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

            children_dict_traversed = {}

            for comp in self.get_children():
                if not comp.is_subentity_items():
                    # recursion
                    comp_chidren_dict = comp.get_children_tree_flatten_dict(depth=depth+1)
                    children_dict_traversed.update(comp_chidren_dict)

                # closer children are overriding further ones
                # (although this will not happen - names should be unique)
                children_dict_traversed[comp.name] = comp

            setattr(self, key, children_dict_traversed)

        return getattr(self, key)

    # ------------------------------------------------------------

    def get_children_tree(self) -> ComponentTreeType:
        """
        will go recursively through every children and
        fetch their "children" and collect to output structure.
        selects all nodes, put in tree, includes self
        """
        return self._get_children_tree(key="_children_tree")


    def _get_children_tree(self, key: str, depth:int=0) -> ComponentTreeType:
        # Dict[ComponentNameType, Self] = Dict[ComponentNameType, TreeNode])
        if not hasattr(self, key):
            if depth > MAX_RECURSIONS:
                raise EntityInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

            children_dict_traversed = {}
            children_dict_traversed["name"] = self.name
            children_dict_traversed["component"] = self
            children_dict_traversed["children"] = []


            for comp in self.get_children():
                # recursion
                comp_chidren_dict = comp._get_children_tree(key=key, depth=depth+1)
                children_dict_traversed["children"].append(comp_chidren_dict)

            setattr(self, key, children_dict_traversed)
        else:
            children_dict_traversed = getattr(self, key)
        return children_dict_traversed

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
                        + "\n" + ". Remove duplicate 'bind' or use 'name' attribute to define a distinct name.")
        # Save top container too - to preserve name and for completness (if is_top)
        components[component.name] = component

    # ------------------------------------------------------------

    def _setup_phase_one(self, components: Optional[Dict[str, Self]] = None,
                         parent: Optional[Self] = None) \
                        -> NoneType:
        """
        does following:
        - collects components:
        - set parent
        - evaluate bound_model - class or M.value_expressions
        - evaluate bind - M.value_expressions
        - custom dataclass and fields

        collecting components
        ---------------------
        except for subentities sub-components
            it is flattened dict of subcomponents (recursive)
        i.e. it returns components : Dictionary [component_name: Component]
            will fill following:
                all mine sub-components
                if not subentity
                    all sub-component sub-components (recursively)
        it sets parent of all sub-components recursively
        """
        if self._setup_phase_one_called:
            raise EntityInternalError(owner=self, msg="fill_components already called")

        # for children/contains attributes - parent is set here
        if not hasattr(self, "name"):
            raise EntitySetupError(owner=self, msg=f"Component should have 'name' attribute, got class: {self.__class__.__name__}")

        if self.is_container():
            assert components is None
            assert self.components is None, self

            components = {}

            if self.is_top_container():
                # Entity()
                if not parent is None:
                    raise EntityInternalError(owner=self, msg=f"Parent should be None, got: {parent}")
                assert parent is None, parent
                self.set_parent(None)
            else:
                # SubEntityItems()?
                assert self.is_subentity()
                assert parent, parent
                self.set_parent(parent)

            if not self.setup_session:
                self.create_setup_session()

            setup_session = self.setup_session
            container = self
        else:
            assert isinstance(components, dict)
            assert parent is not None
            assert parent != self
            self.set_parent(parent)

            container = self.get_first_parent_container(consider_self=False)
            setup_session = container.setup_session

        self._add_component(component=self, components=components)

        with setup_session.use_stack_frame(
                SetupStackFrame(
                    container = container,
                    component = self,
                    this_registry = None,
                )):
            if self.is_container():
                # ----------------------------------------
                # ModelsNS setup 1/2 phase -> bound models
                # ----------------------------------------
                #   - direct data models setup (Entity)
                #   - some standard NS fields (e.g. Instance)
                # This will call complete setup() for bound_model-s
                self._register_model_attr_nodes()

            # includes components, cleaners and all other complex objects
            for subcomponent in self._get_subcomponents_list():
                component = subcomponent.component
                # if isinstance(component, Namespace):
                #     raise EntitySetupValueError(owner=self, msg=f"Subcomponents should not be Namespace instances, got: {subcomponent.name} = {subcomponent.component}")

                if isinstance(component, DotExpression):
                    continue

                if isinstance(component, IComponent):
                    if component.is_subentity():
                        # component.set_parent(parent=self)
                        # for subentity_items container don't go deeper into tree (call _fill_components)
                        # it will be called later in container.setup() method
                        component._setup_phase_one(components=None, parent=self)
                        # save only container (top) object
                        self._add_component(component=component, components=components)
                    else:
                        component._setup_phase_one(components=components, parent=self)
                        # delete this
                        if component.is_bound_model():
                            if isinstance(component.model, DotExpression):
                                assert component.model.IsFinished()
                            # NOTE: moved from setup_phase_two() - this is required for in_model=False cases,
                            #       to register non-model attributes
                            component._register_nested_models(setup_session)

                elif isinstance(component, IBaseComponent):
                    component.set_parent(parent=self)
                    self._add_component(component=component, components=components)
                    # e.g. BoundModel.model - can be any custom Class(dataclass/pydantic)
                else:
                    if component.__class__.__name__ not in ("ChoiceOption", "CustomFunctionFactory", "int", "str"):
                        raise EntityInternalError(owner=component, msg=f"Strange type of component, check or add to list of ignored types for _setup_phase_one() ")

            if self.is_container():
                if self.components is not None:
                    raise EntityInternalError(owner=self, msg=f"components already set: {self.components}")
                self.components = components

                # evaluate bound_model - class or M.value_expression
                # self.bound_model.setup(setup_session=setup_session)
            else:
                if isinstance(self, IField):
                    # evaluate bind: ValueExpression
                    self.bind.Setup(setup_session=setup_session, owner=self)
                else:
                    if hasattr(self, "bind"):
                        raise EntityInternalError(owner=self, msg=f"Only fields expected to have bind attr, got: {type(self)}")

            # Set up dataclass and list of fields for later use in FieldsNS.
            # Will validate if all type_info are available
            self.get_component_fields_dataclass(setup_session=setup_session)

            if isinstance(self, IField):
                self._setup_phase_one_set_type_info(setup_session)

                # Check that type_info is set - call will raise error if something is wrong.
                assert hasattr(self, "get_type_info")
                if not self.get_type_info():
                    raise EntityInternalError(owner=self, msg="type_info could not be retrieved in setup phase one")

        self._setup_phase_one_called = True

        return None

    # ------------------------------------------------------------

    @abstractmethod
    def setup(self, setup_session: ISetupSession):
        ...

    # ------------------------------------------------------------

    def _invoke_component_setup(self, 
                    component_name: str, 
                    component: Union[Self, DotExpression], 
                    setup_session: ISetupSession):  # noqa: F821
        called = False

        if isinstance(component, (DotExpression,)):
            dexp: DotExpression = component
            # namespace = dexp._namespace
            if dexp.IsFinished():
                # Setup() was called in container.setup() before or in some
                #         other dependency
                called = False
            else:
                dexp.Setup(setup_session=setup_session, owner=self)
                called = True
        # elif isinstance(component, IBoundModel):
        #     called = False
        elif isinstance(component, IComponent):
            assert "Entity(" not in repr(component)
            component.setup(setup_session=setup_session)  # , parent=self)
            called = True
        elif isinstance(component, (dict, list, tuple)):
            raise EntitySetupValueError(owner=self, msg=f"components should not be dictionaries, lists or tuples, got: {type(component)} / {component}")
        else:
            if hasattr(component, "Setup"):
                raise EntityInternalError(owner=self, msg=f"{self.name}.{component_name} has attribute that is not DotExpression: {type(component)}")
            if hasattr(component, "setup"): 
                raise EntityInternalError(owner=self, msg=f"{self.name}.{component_name} has attribute that is not Component: {type(component)}")
        return called


    # ------------------------------------------------------------

    def get_component_fields_dataclass(self, setup_session: ISetupSession) \
            -> Tuple[TypingType[IComponentFields], List[ChildField]]:
        # TODO: not happy with this
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

        # TODO: to have deep_collect or not??
        children = self.get_children(deep_collect=True)

        container = setup_session.current_frame.container

        assert setup_session and setup_session.current_frame
        if not setup_session.current_frame.component == self:
            raise EntityInternalError(owner=self, msg=f"setup_session.current_frame.component={setup_session.current_frame.component} <> component={self}")

        child_field_list: List[ChildField] = []

        for nr, child in enumerate(children, 1):

            if child.is_bound_model():
                continue

            child_type_info = None
            if isinstance(child, IField):
                # ALT: not hasattr(child, "bind")
                # NOTE: check that sessino is setup correctly for this field?

                if not child.bind.IsFinished():
                    # Can setup only fields which are inside the same container
                    # share the same bound_model 
                    child.bind.Setup(setup_session=setup_session, owner=self)

                attr_node = child.bind._dexp_node
                child_type_info = attr_node.get_type_info()

            elif child.is_subentity() or child.is_fieldgroup():
                # ------------------------------------------------------------
                # NOTE: this is a bit complex chain of setup() actions: 
                #           entity -> this -> subentity -> subentity ...
                #       find some easier way how to to it
                #       
                # ------------------------------------------------------------
                # ALT: when all model fields are available:
                #       if not child.bound_model.model.IsFinished():
                #           # ALT: more complex way - it seems to work, but I prefer
                #           #      simplier solution:
                #           attr_node = container._setup_bound_model_dot_expression(bound_model=child.bound_model, setup_session=setup_session)
                #
                #       # ALT: attr_node = child.bound_model.model._dexp_node
                #       #      child_type_info = attr_node.get_type_info()
                #       child_type_info = child.bound_model.get_type_info()

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

                    # RECURSION:
                    child_component_fields_dataclass, _ = child.get_component_fields_dataclass(setup_session=setup_session)

                child_type_info = TypeInfo.get_or_create_by_type(child_component_fields_dataclass)

            # elif child.is_fieldgroup():
            #     # TODO: currently not supported, need custom type_info
            #     continue
            else:
                raise EntityInternalError(owner=child, msg=f"child_type_info could not be extracted, unsuppoerted component's type, got: {type(child)}") 

            child_field_list.append(
                    ChildField(
                        Name=child.name,
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

    def setup(self, setup_session: ISetupSession):  # noqa: F821
        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.setup_called(self)

        container = self.get_first_parent_container(consider_self=True)

        with setup_session.use_stack_frame(
                SetupStackFrame(
                    container = container, 
                    component = self, 
                )):
            # setup this_registry objects must be inside of stack_frame due
            # premature component.bind setup in some ThisRegistryFor* classes.
            if not self.is_subentity():
                # NOTE: for SubEntity* - this registry must be and is already
                #       created in ContainerBase.setup()
                # this_registry = container.try_create_this_registry(component=self, setup_session=setup_session)
                this_registry = self.get_or_create_this_registry(setup_session=setup_session)
                if this_registry:
                    # set only if available, otherwise use existing
                    setup_session.current_frame.set_this_registry(this_registry, force=True)

            if not self.is_finished():
                ret = self._setup_phase_two(setup_session=setup_session)
            else:
                if not self.is_bound_model():
                    raise EntityInternalError(owner=self, msg=f"_setup_phase_two() is skipped only for bound models, got: {type(self)}")
                ret = self

        return ret

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
                2. have parent set (IBaseComponent)
                3. DotExpression

              all of them have some setup method:
                setup() # case 1. and 2.
                Setup() # case 3.

              so the logic should be:
                collect all attributes that have setup() and Setup()
              check if this is true and if so, implement.
        """

        # caching
        if hasattr(self, "_subcomponent_list"):
            return self._subcomponent_list

        # returns name, subcomponent
        fields = get_model_fields(self.__class__)

        # ==== Order to have bound_model first, then components, then value
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

            if th_field and th_field.metadata.get("skip_traverse", False):
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
                                      # now is evaluated from bound_model, bound_model is processed
                                      "models", "py_type_hint", "type_info",
                                      "bound_attr_node",
                                      "read_handler",
                                      "function", "context_class", "config",
                                      "min", "max", "allow_none", "ignore_none",
                                      "keys",
                                      "entity",
                                      "value_accessor",
                                      "value_accessor_default",
                                      "value_accessor_class_registry",
                                      # "value",
                                      # "enum",
                                      )
               or sub_component_name[0] == "_"):
                # TODO: this should be main way how to check ...
                if sub_component_name not in ("parent", "parent_container", "entity") and \
                  (hasattr(sub_component, "setup") or hasattr(sub_component, "setup")):
                    raise EntityInternalError(f"ignored attribute name '{sub_component_name}' has setup()/Setup(). Is attribute list ok or value is not proper class (got component='{sub_component}').")
                continue

            if not th_field or getattr(th_field, "init", True) is False:
                # warn(f"TODO: _get_subcomponents_list: {self} -> {sub_component_name} -> {th_field}")
                raise EntityInternalError(owner=sub_component, msg=f"Should '{sub_component_name}' field be excluded from processing: {th_field}")

            # ----------------------------------------------------------------------
            # Classes of attribute members that are included in a subcomponents list 
            # each will be "Setup" if they have appropriate class and method
            # (see _invoke_component_setup).
            # ----------------------------------------------------------------------
            # TODO: this is not nice - do it better
            # TODO: models should not be dict()
            # TODO: added Self for BoundModelWithHandlers
            if sub_component_name not in ("models", "data", "functions", "enum") \
                    and th_field \
                    and "Component" not in str(th_field.type) \
                    and "Container" not in str(th_field.type) \
                    and "DotExpression" not in str(th_field.type) \
                    and "Validation" not in str(th_field.type) \
                    and "Evaluation" not in str(th_field.type) \
                    and "BoundModel" not in str(th_field.type) \
                    and "[Self]" not in str(th_field.type) \
                    :
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

        if self.is_finished():
            raise EntityInternalError(owner=self, msg="Setup already called")

        for subcomponent in self._get_subcomponents_list():
            component = subcomponent.component
            if isinstance(component, IComponent) \
              and (component.is_bound_model() or component.is_subentity()) \
              and component.is_finished():
                # raise EntityInternalError(owner=self, msg=f"BoundModel.setup() should have been called before ({component})")
                continue

            self._invoke_component_setup(
                    subcomponent.name, 
                    component=component, 
                    setup_session=setup_session)

        # if not self.is_finished():
        self.finish()

        return self

    # ------------------------------------------------------------

    def finish(self):
        if self.is_finished():
            raise EntitySetupError(owner=self, msg="finish() should be called only once.")
        self._finished = True

    # ------------------------------------------------------------

    def is_finished(self):
        return getattr(self, "_finished", False)



    # ------------------------------------------------------------

    def get_dexp_result_from_instance(self, apply_result: "IApplyResult", strict:bool = True) -> Optional[ExecResult]:
        """ Fetch ExecResult from component.bind from INSTANCE (storage)
            by executing bind._evaluator.execute() fetch value process
            Work on stored fields only.
            A bit slower due getattr() logic and more complex structs.
            Does not work on initial values - when cache is not yet initialized
            with initial value for the component.
        """
        # TODO: put in common function
        bind_dexp = self.bind
        if not bind_dexp:
            if strict:
                # TODO: move this to Setup phase
                raise EntityApplyError(owner=self, msg=f"Component '{self.name}' has no bind")
            return None
        bind_dexp_result = bind_dexp._evaluator.execute_dexp(apply_result=apply_result)
        return bind_dexp_result


    def get_this_registry(self) -> Optional[IThisRegistry]:
        if self._this_registry is UNDEFINED:
            raise EntityInternalError(owner=self, msg=f"_this_registry not setup, seems that get_or_create_this_registry() was not called before")
        return self._this_registry


    def get_or_create_this_registry(self, setup_session: ISetupSession) -> Optional[IThisRegistry]:
        # TODO: maybe DRY is needed - similar logic found in functions:: create_this_registry
        if not self._this_registry is not UNDEFINED:
            this_registry = self.create_this_registry(setup_session=setup_session)
            if this_registry:
                this_registry.setup(setup_session=setup_session)
                this_registry.finish()
                # test repr: print(f"this_registry={this_registry}")
            self._this_registry = this_registry

        return self._this_registry


# ------------------------------------------------------------

class ICleaner(IComponent, ABC):
    ...

class IValidation(ICleaner, ABC):
    ...

class IEvaluation(ICleaner, ABC):
    ...

# ------------------------------------------------------------

class IField(IComponent, ABC):
    # to Model attribute
    bind: DotExpression

    @abstractmethod
    def validate_type(self, apply_result: "IApplyResult", value: Any = UNDEFINED) \
        -> Optional["ValidationFailure"]:
        ...

    # @abstractmethod
    # def get_attr_node(self, setup_session: ISetupSession) -> "AttrDexpNode":  # noqa: F821
    #     ...

    # @abstractmethod
    # def get_bound_attr_node(self, setup_session: ISetupSession) -> "AttrDexpNode":  # noqa: F821
    #     ...

# ------------------------------------------------------------

class IFieldGroup(ABC):
    ...

# ------------------------------------------------------------
# IContainer
# ------------------------------------------------------------

class IContainer(IComponent, ABC):

    bound_model: "IBoundModel" = field(repr=False)

    @abstractmethod
    def add_fieldgroup(self, fieldgroup:IFieldGroup):  # noqa: F821
        ...

    @abstractmethod
    def __getitem__(self, name: str) -> IComponent:
        ...

    @abstractmethod
    def get_component(self, name: str) -> IComponent:
        ...

    @abstractmethod
    def setup(self, setup_session: ISetupSession) -> Self:
        ...

    @abstractmethod
    def get_bound_model_attr_node(self) -> "AttrDexpNode":  # noqa: F821
        ...

    @abstractmethod
    def pprint(self):
        """ pretty print - prints to stdout all components """
        ...

    # @staticmethod
    # @abstractmethod
    # def create_this_registry_for_model_class(
    #         setup_session: ISetupSession,
    #         model_class: ModelType,
    # ) -> IThisRegistry:
    #     """
    #     This is @staticmethod
    #     """
    #     ...

class IEntity(IContainer, ABC):
    config: Config = field(repr=False, )
    context_class: Optional[Type[IContext]] = field(repr=False, default=None)

# ------------------------------------------------------------
# IBoundModel
# ------------------------------------------------------------

@dataclass
class IBoundModel(IComponent, ABC):

    model: ModelType = field(init=False, repr=False)
    _finished: bool = field(init=False, repr=False, default=False)


    def setup(self, setup_session:ISetupSession):
        if self._finished:
            raise EntityInternalError(owner=self, msg="Setup already called")
        super().setup(setup_session=setup_session)

    @abstractmethod
    def get_type_info(self) -> TypeInfo:
        ...

    @staticmethod
    def is_bound_model() -> bool:
        return True

    def is_top_parent(self):
        return False

    def get_full_name(self, parent: Optional[Self] = None, depth: int = 0, init: bool = False):
        if not hasattr(self, "_name"):
            if depth > MAX_RECURSIONS:
                raise EntityInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

            assert init
            names = []
            if parent:
                # recusion
                names.append(parent.get_full_name(parent=self, depth=depth+1, init=init))
            names.append(self.name)
            self._name = ".".join(names)
        return self._name


    def fill_models(self, models: Dict[str, Self] = None, parent : Self = None):
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
            for dep_bound_model in self.contains:
                # recursion
                dep_bound_model.fill_models(models=models, parent=self)
        return models

    # Not used:
    # def get_attr_node(self, setup_session: ISetupSession) -> Union["AttrDexpNode", UndefinedType]:  # noqa: F821
    #     return setup_session.models_registry.get_attr_node_by_bound_model(bound_model=self)


# ============================================================

@dataclass
class IStackFrame:

    @abstractmethod
    def clean(self):
        ...


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
                # apply_result.config.loggeer.debugf"setattr '{attr_name}' current_frame <= previous_frame := {prev_frame_attr_value} (frame={self})")
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
        # context manager (with xxx as : ...
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
    #             # apply_result.config.loggeer.debugf"setattr '{attr_name}' current_frame <= previous_frame := {prev_frame_attr_value} (frame={self.frame})")
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

    def use_changed_current_stack_frame(self, **change_attrs) -> "UseStackFrameCtxManagerBase":
        if not self.current_frame:
            raise EntityInternalError(owner=self, msg="No current frame")
        if not change_attrs:
            raise EntityInternalError(owner=self, msg="Expecting change_attrs")

        orig_attrs: Dict[str, Any] = {}
        for attr_name, attr_value in change_attrs.items():
            if not hasattr(self.current_frame, attr_name):
                raise EntityInternalError(owner=self, msg=f"Unknown attribute: {attr_name} in frame: {self.current_frame}")
            orig_attrs[attr_name] = getattr(self.current_frame, attr_name)
            setattr(self.current_frame, attr_name, attr_value)

        def restore_args_to_orig_callback():
            for attr_name, attr_value in orig_attrs.items():
                setattr(self.current_frame, attr_name, attr_value)

        return self.STACK_FRAME_CTX_MANAGER_CLASS(
                    owner_session = self,
                    frame=self.current_frame,
                    add_to_stack=False,
                    cleanup_callback=restore_args_to_orig_callback,
        )


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

    # current component - can be BoundModel too
    component: IComponent

    # # used for ThisNS in some cases
    # local_setup_session: Optional[ISetupSession] = field(repr=False, default=None)
    this_registry: Optional[IThisRegistry] = field(repr=False, default=None)

    # Computed from container/component
    # used for BoundModelWithHandlers cases (read_handlers()), 
    bound_model: Optional[IBoundModel] = field(init=False, repr=False, default=None)

    dexp_validator             : Optional[DexpValidator] = field(repr=False, default=None)

    # -- autocomputed
    # bound_model_root : Optional[IBoundModel] = field(repr=False, init=False, default=None)
    # current container data instance which will be procesed: changed/validated/evaluated
    # type_info: TypeInfo


    def __post_init__(self):
        self.clean()


    def clean(self):
        if not isinstance(self.container, IContainer):
            raise EntityInternalError(owner=self, msg=f"Expected IContainer, got: {self.container}")
        if not isinstance(self.component, IComponent):
            raise EntityInternalError(owner=self, msg=f"Expected IComponent, got: {self.component}")

        if isinstance(self.component, IBoundModel):
            self.bound_model = self.component
        else:
            self.bound_model = self.container.bound_model

    def copy_from_previous_frame(self, previous_frame: Self):
        assert previous_frame
        self._copy_attr_from_previous_frame(previous_frame, "this_registry",
                                            if_set_must_be_same=False)
        # check / init again
        self.clean()


    def post_clean(self):
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

# TODO: put in Config and use only in IApplyResult.config ...
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

    # * first / initial record in parent list is from bind, first
    #   this .value could be unadapted value version (field.try_adapt_value())
    #   therefore value is in a special field.
    # * second+ - are from evaluation results
    dexp_result: ExecResult = field(repr=False, compare=False)

    # TODO: this is not filled good - allways component.name
    # just to have track who read/set this value
    value_parent_name: str = field(repr=False, compare=False)

    # is from bind
    is_from_bind: bool = field(repr=False, compare=False, default=False)

    def as_dict(self) -> Dict[str, Any]:
        return {
                "value" : self.value,
                "dexp_result" : self.dexp_result,
                "value_from_parent": self.value_parent_name,
                "is_from_bind": self.is_from_bind,
                }

# ------------------------------------------------------------

@dataclass
class ValueNode:
    """
    Don't want to store apply_result into this class,
    so "setup_key_string()" method needs to be called
    just after object instatiation, e.g.

        ValueNode(...).setup_key_string(apply_result=...)

    """

    component:  IComponent = field(repr=False)
    instance: DataclassType = field(repr=False)
    container: IContainer = field(repr=False)
    instance_none_mode: bool = field(repr=False)

    # ------------------------------------------------------------
    # Following are all accessible by . operator in ValueExpressions
    # ------------------------------------------------------------
    # TODO: fields? - only for container - includes all children's children
    #       except Items's children

    should_collect_value_history: bool = field(repr=False)

    # <field>.Parent
    # - empty only on top tree node (Entity component)
    parent_node:  Optional[Self] = field(repr=False)

    # top of the tree - automatically computed
    top_node:  Self = field(repr=False, init=False)

    # when SubentityItems - hodls 2 types of ValueNodes:
    #   has_items=True  - container - has filled .items[], has no children
    #   has_items=False - individual item - with values, has children (some with values)
    has_items: bool = field(repr=True, default=False)

    # set only when item of parent.items (parent is SubentityItems)
    # set only when item is deleted (DELETE) or item is added (ADDED)
    change_op: Optional[ChangeOpEnum] = field(repr=True, default=None)

    # set only when item of parent.items (parent is SubentityItems)
    index0: Optional[Index0Type] = field(repr=False, default=None)
    key: Optional[KeyType] = field(repr=False, default=None)

    # F.<field-name>
    # - contains all fields accessible in. autocomputed.
    # TODO: replace with fields_dict: Dict[name, Self] - distribute to all
    container_node: Self = field(repr=False, init=False, default=None)

    # <field>.Value
    _value: Union[AttrValue, UndefinedType] = field(repr=False, init=False, default=UNDEFINED)
    # TODO: consider compare=False ?? do not compare by this attribute (for easier unit-test checks)
    finished: bool = field(repr=False, init=False, default=False)

    # initial value - Just value - no wrapper
    init_value: Union[UndefinedType, AttrValue] = \
        field(repr=False, init=False, default=UNDEFINED)

    # NOTE: used only in ApplyResult._model_instance_attr_change_value
    #       to set new value on model instance.
    # TODO: consider ALT: Could be done with bind.Path() too.
    #       attr_name_path: Optional[List[AttrName]] = field(init=False, repr=False)
    init_dexp_result: Union[UndefinedType, NoneType, ExecResult] = \
        field(repr=False, init=False, default=UNDEFINED)

    # Should be used only for better introspection / debug / trace / log
    # set later by calling .setup(apply_result)
    key_string: KeyString = field(init=False, repr=True, default=UNDEFINED)

    # Setup later when ApplyStackFrame() is created, set with .set_apply_stack_frame()
    # this is cross reference, since stack_frame has ref to ValueNode too
    apply_stack_frame: Union[IStackFrame, UndefinedType] = field(repr=False, default=UNDEFINED)

    # <field>.Children
    # - when component has children - initialized with empty {}, later filled
    children:   Union[UndefinedType, Dict[AttrName, Self]] = field(repr=False, init=False, default=UNDEFINED)

    # <field>.Items
    # - when component has items - initialized with empty {}, later filled
    items:      Union[UndefinedType, List[Self]] = field(repr=False, init=False, default=UNDEFINED)

    # Autocomputed
    # component's name
    name:       AttrName = field(init=False, repr=False)

    # interrnal structures
    value_history: Union[UndefinedType, List[InstanceAttrValue]] = \
        field(repr=False, init=False, default=UNDEFINED)

    # cached values used in set_value() logic
    # see . _init_instance_attr_access_objects()
    _instance_parent: Union[ModelType, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)
    _attr_name_last: Union[AttrName, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)
    _value_accessor: Union[IValueAccessor, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)


    # TODO:
    # changes: List[InstanceChange] = \
    #     field(repr=False, init=False, default_factory=list)
    # TODO: value_accessor: IValueAccessor

    # TODO: self.key_pairs
    # TODO: self.attr_name = self.component.bind ...
    # TODO: later  - see register_instance_attr_change()

    def __post_init__(self):
        self.name = self.component.name

        if self.component.is_top_container():
            assert self.parent_node is None
            self.container_node = self
            self.top_node = self
        else:
            assert self.parent_node is not None
            assert self.parent_node.top_node.parent_node is None
            self.top_node = self.parent_node.top_node

        if self.component.is_subentity():
            if self.has_items:
                self.items = []
            # TODO: consider what to do
            self.container_node = self
        else:
            assert not self.has_items

        if not self.has_items and self.component.get_children():
            # TODO: with py 3.7 {} has ordered items as default behaviour,
            #       so std {} could be used instead
            self.children = OrderedDict()

        if self.container_node is None:
            assert self.parent_node and self.parent_node.container_node
            self.container_node = self.parent_node.container_node


    def setup(self, apply_result: "IApplyResult") -> Self:
        """
        will:
            1) calculate and set key_string
            2) register within apply_result

        Calculate and set key_string
        ----------------------------
        Two cases - component has .keys or not:

        a) with keys:
            For containers which have keys defined, it is assumed that one key is
            globally unique within SubEntityItems components, so no need to prefix key
            with parent key_string. Example:

                 address_set_ext[id2=1]

            if for each key that is not not set in instance, a special
            key is generated using:

                apply_result._get_new_id()

        b) without keys - index0 based:
            In other cases item index in list is used (index0), and then this is
            only locally within one instance of parent, therefore parent
            key_string is required. Example:

                 company_entity::address_set_ext[0]

        """
        if self.key_string:
            raise EntityInternalError(owner=self, msg="value node already setup")

        component = self.component

        # ---- 1) calculate and set key_string

        # default case
        key_string = component.name
        if not self.has_items and isinstance(component, IContainer):
            # Part of key_string that includes instance identificators
            # i.e. key_pairs / index in a list
            if component.keys:
                key_pairs = component.get_key_pairs(self.instance, apply_result=apply_result)
                assert key_pairs
                key_string = "{}[{}]".format(
                    component.name,
                    GlobalConfig.ID_NAME_SEPARATOR.join(
                        [f"{name}={value}" for name, value in key_pairs]
                    ))
            elif self.index0 is not None:
                key_string = f"{component.name}[{self.index0}]"

        if not self.is_top_node():
            if not self.is_top_node() and self.container_node is self:
                container_node = self.container_node.parent_node.container_node
            else:
                container_node = self.container_node
            # TODO: consider to remove_this, just to have the same logic as before
            # ignore self.parent_node.has_items parent value_node - will introduce redunant markup
            if container_node.has_items:
                container_node = container_node.parent_node.container_node

            key_string = f"{container_node.key_string}{GlobalConfig.ID_NAME_SEPARATOR}{key_string}"

        self.key_string = KeyString(key_string)

        # ---- 2) register within apply_result - regsiter node to value_node_list
        apply_result.register_value_node(value_node=self)

        return self

    def set_apply_stack_frame(self, apply_stack_frame: IStackFrame):
        """
        called when creating ApplyStackFrame
        will be used in _finish_component()
        """
        assert apply_stack_frame
        if not apply_stack_frame.value_node == self:
            raise EntityInternalError(owner=self, msg=f"apply_stack_frame has diff value_node: {self.apply_stack_frame.value_node}")
        if self.apply_stack_frame:
            raise EntityInternalError(owner=self, msg=f"apply_stack_frame is already set: {self.apply_stack_frame}")
        # TODO: make a copy to preserve original values. Maybe is not necessary.
        apply_stack_frame_copy = dataclass_clone(apply_stack_frame)
        self.apply_stack_frame = apply_stack_frame_copy

    def is_top_node(self):
        # ALT: self.container_node and self.container_node is not self:
        # ALT: self.parent_node is None
        return self is self.top_node

    def is_changed(self) -> bool:
        # if not self.finished:
        #     raise EntityInternalError(owner=self, msg="not finished yet")
        if self.change_op is not None:
            return True
        return self._value!=self.init_value

    def set_value(self, value: AttrValue, dexp_result: Optional[ExecResult]) -> NoneType:
        """
        - when setting initial value dexp_result is stored into init_dexp_result
        - dexp_result can be None too - in NA_* some undefined values
        - fills value_history objects only in "trace" mode - only in the case when value is not of type UndefinedType
        """
        if self.finished:
            raise EntityInternalError(owner=self, msg=f"Current value already finished, last value: {self._value}")

        if dexp_result is not None and self.init_dexp_result is UNDEFINED:
            if not self._value in (NA_IN_PROGRESS, UNDEFINED):
                raise EntityInternalError(owner=self, msg=f"self._value already set to: {self._value}")
            # if not dexp_result:
            #     raise EntityInternalError(owner=self, msg=f"dexp_result is empty: {dexp_result}")
            self.init_dexp_result = dexp_result
            is_from_init_bind = True
        else:
            is_from_init_bind = True

        # TODO: if config.trace - then update values_history

        self._value = value
        # if self._value is NA_DEFAULTS_MODE:
        #     # TODO: check if finish immediatelly
        #     self.mark_finished()

        # --- value_history - add new value
        # TODO: pass input arg value_parent_name - component.name does not have any purpose

        if self.should_collect_value_history and value is not NA_IN_PROGRESS \
          and value is not NOT_APPLIABLE and value is not NA_DEFAULTS_MODE:
            instance_attr_value = InstanceAttrValue(
                value_parent_name=self.component.name,
                value=value,
                dexp_result=dexp_result,
                is_from_bind = is_from_init_bind,
                # TODO: source of change ...
            )
            self.value_history.append(instance_attr_value)
        # else: instance_attr_value = None
        return  #  instance_attr_value  : Optional[InstanceAttrValue]

    def get_value(self, strict:bool) -> AttrValue:
        if strict and not self.finished:
            # print("TODO: riješi ovu iznimku")
            raise EntityInternalError(owner=self, msg=f"Current value is not finished, last value: {self._value}")
        return self._value

    # ------------------------------------------------------------

    def apply_value_to_instance_attr(self):
        """
        will set instance.<attr-name> = self._value
        attr-name could be nested (e.g. M.access.can_delete)
        """
        # TODO: not sure if this validation is ok
        model = self.container.bound_model.get_type_info().type_
        if not self.instance_none_mode \
                and not isinstance(self.instance, model):
            raise EntityInternalError(owner=self, msg=f"Parent instance {self.instance} has wrong type")

        # -- attr_name - fetch from initial bind dexp (very first)
        # TODO: save dexp_result in ValueNode or get from component.bind ?
        init_bind_dexp_result = self.init_dexp_result
        if not init_bind_dexp_result:
            raise EntityInternalError(owner=self, msg=f"init_bind_dexp_result is not set, got: {self} . {init_bind_dexp_result}")

        if self._instance_parent is UNDEFINED:
            self._init_instance_attr_access_objects()

        # Finally change instance value by last attribute name
        assert self._attr_name_last
        self._value_accessor.set_value(instance=self._instance_parent,
                                 attr_name=self._attr_name_last,
                                 attr_index=None,
                                 new_value=self._value)

    # ------------------------------------------------------------

    def _init_instance_attr_access_objects(self):
        """
        initialize:
            self._instance_parent
            self._attr_name_last
            self._value_accessor

        used in call to:
           self._value_accessor.set_value()
        """
        assert self._instance_parent is UNDEFINED
        component = self.component
        # "for" loop is required for attributes from substructure that
        # is not done as SubEntity rather direct reference, like:
        #   bind=M.access.alive
        attr_name_path = [init_raw_attr_value.attr_name
                          for init_raw_attr_value in self.init_dexp_result.value_history
                          ]
        if not attr_name_path:
            raise EntityInternalError(owner=self, msg=f"{component}: attr_name_path is empty")

        if isinstance(component, IField):
            # TODO: what about Boolean + enables? Better to check .get_children() ?
            # value accessor should be used from parent of the component
            assert component.parent
            value_accessor = component.parent.value_accessor
        else:
            # contaainers + fieldgroup
            value_accessor = component.value_accessor

        current_instance_parent = None
        parent_instance = self.instance
        current_instance = parent_instance
        attr_name_last = UNDEFINED

        # Attribute path example: "M.access.alive".
        # Only last, i.e. "alive" in this example, will be updated,
        # and this for loop reaches instance and current value in this case.
        for anr, attr_name in enumerate(attr_name_path, 0):
            attr_name_last = attr_name
            if current_instance is None:
                if self.instance_none_mode:
                    # -------------------------------------------------------
                    # Create all missing intermediate empty dataclass objects
                    # -------------------------------------------------------
                    assert anr > 0
                    attr_name_prev = attr_name_path[anr - 1]
                    temp_dataclass_model= self._make_dataclass_with_opt_fields(
                        current_instance_parent=current_instance_parent,
                        attr_name_prev=attr_name_prev,
                        attr_name=attr_name,
                    )
                    current_instance = temp_dataclass_model()
                    value_accessor.set_value(instance=current_instance_parent,
                                             attr_name=attr_name_prev,
                                             attr_index=None,
                                             new_value=current_instance)
                else:
                    attr_name_path_prev = ".".join(attr_name_path[:anr])
                    # TODO: fix this ugly validation message
                    raise EntityApplyValueError(owner=self,
                                                msg=f"Attribute '{attr_name}' can not be set while '{parent_instance}.{attr_name_path_prev}' is not set. Is '{attr_name_path_prev}' obligatory?")

            current_instance_parent = current_instance
            current_instance = value_accessor.get_value(instance=current_instance_parent,
                                                        attr_name=attr_name,
                                                        attr_index=None)
            if current_instance is UNDEFINED:
                raise EntityInternalError(owner=self,
                                          msg=f"Missing attribute:\n  Current: {current_instance}.{attr_name}\n Parent: {parent_instance}.{'.'.join(attr_name_path)}")

        self._instance_parent = current_instance_parent
        # attribute name is in the last item
        self._attr_name_last = attr_name_last
        self._value_accessor = value_accessor


    # ------------------------------------------------------------
    def _make_dataclass_with_opt_fields(self,
                        current_instance_parent: ModelType,
                        attr_name_prev: AttrName,
                        attr_name: AttrName,
                        ) -> ModelType:
        current_instance_type_info = get_dataclass_field_type_info(current_instance_parent, attr_name_prev)
        if current_instance_type_info is None:
            raise EntityInternalError(owner=self,
                                      msg=f"Attribute {attr_name} not found in dataclass definition of {current_instance_parent}.")
        if current_instance_type_info.is_list:
            raise EntityInternalError(owner=self,
                                      msg=f"Attribute {attr_name} of {current_instance_parent} is a list: {current_instance_type_info}.")

        current_instance_model = current_instance_type_info.type_
        if not is_dataclass(current_instance_model):
            raise EntityInternalError(owner=self,
                                      msg=f"Attribute {attr_name} of {type(current_instance_parent)} is not a dataclass instance, got: {current_instance_model}")

            # set new value of temp instance attribute
        # all attrs of a new instance will have None value (dc default)
        temp_dataclass_model = make_dataclass_with_optional_fields(current_instance_model)
        return temp_dataclass_model


    def mark_finished(self):
        if self.finished: # and self._value is not NA_DEFAULTS_MODE:
            raise EntityInternalError(owner=self, msg=f"Current value already finished, last value: {self._value}")
        self.finished = True

    def add_item(self, value_node: Self):
        assert self.has_items
        # TODO: assert value_node is not self
        # TODO: assert value_node.parent_node is not self
        self.items.append(value_node)

    def add_child(self, value_node: Self):
        assert not self.has_items
        if self.children is UNDEFINED:
            raise EntityInternalError(owner=self, msg=f"failed to add {value_node}, already children dict not initialized")
        # TODO: assert value_node is not self
        # TODO: assert value_node.parent_node is not self
        if value_node.name in self.children:
            raise EntityInternalError(owner=self, msg=f"failed to add {value_node}, already child added under same name, got : {self.children[value_node.name]}")
        self.children[value_node.name] = value_node

    def clean(self):
        if self.component.is_top_container():
            assert self.parent_node is None
        else:
            if self.parent_node is None:
                raise EntityInternalError(owner=self, msg=f"parent_node not set")
            # subentity item instance - with children
            if self.parent_node.has_items and not self.parent_node.component == self.component:
                # print(f"Dump:\n{self.top_node.dump_to_str()}")
                raise EntityInternalError(owner=self, msg=f"value:node.has_items, parent_node - component does not match:"
                                                          f"\n {self.component.parent}"
                                                          f"\n !=\n {self.parent_node.component}")
            elif not self.parent_node.has_items and not self.parent_node.component == self.component.parent:
                # print(f"Dump:\n{self.top_node.dump_to_str()}")
                raise EntityInternalError(owner=self, msg=f"value:node parent_node component does not match:"
                                                          f"\n {self.component.parent}"
                                                          f"\n !=\n {self.parent_node.component}")
    def dump_to_str(self) -> str:
        return "\n".join(self.dump_to_strlist())

    def dump_to_strlist(self, depth=0) -> List[str]:
        lines = []
        indent = "  " * depth
        line = f"{indent}{self.name}{'(has_items)' if self.has_items else ''}"
        value = self.get_value(strict=False)
        if isinstance(value, UndefinedType):
            line += f" = {value}"
            nr_updates = len(self.value_history) - 1
            if nr_updates > 0:
                # orig_value = self.value_history[0].value
                init_value = self.init_value
                # assert init_value!=value
                value_type = value.__class__.__name__
                init_type = init_value.__class__.__name__
                if value_type!=init_type and not isinstance(value, (NoneType, UndefinedType)) :
                    line += f":{value_type}"
                line += f"  (orig: {init_value}"
                if value_type!=init_type and not isinstance(init_value, (NoneType, UndefinedType)):
                    line += f":{init_type}"
                line += f", ch: {nr_updates})"

        lines.append(line)
        if self.has_items:
            lines.append(f"{indent}- Items[{len(self.items)}]:")
            for item_node in self.items:
                add_lines = item_node.dump_to_strlist(depth+1)
                lines.extend(add_lines)
        if self.children != UNDEFINED:
            lines.append(f"{indent}- Children[{len(self.children)}]:")
            for child_node in self.children.values():
                add_lines = child_node.dump_to_strlist(depth+1)
                lines.extend(add_lines)
        return lines

# ------------------------------------------------------------

@dataclass
class ApplyStackFrame(IStackFrame):
    """

    IMPORTANT NOTE: When adding new fields, consider adding to 

        UseApplyStackFrame .compare_with_previous_frame()

    """
    # current container data instance which will be procesed: changed/validated/evaluated
    component: IComponent
    # main container - class of instance - can be copied too but is a bit complicated
    container: IContainer = field(repr=False)
    instance: DataclassType

    # in value-node-tree current node - holds component/instance/container
    # can be NOT_APPLIABLE
    value_node: Union[ValueNode, UndefinedType] = field(repr=False, default=None)

    # ---------------------------------------------------------------------------------
    # Following if not provideed are copied from parent frame (previous, parent frame)
    # check UseApplyStackFrame. copy_from_previous_frame
    # ---------------------------------------------------------------------------------

    # UPDATE by this instance - can be ENTITY_LIKE or MODELS_LIKE (same dataclass) struct
    # required but is None when not update
    instance_new: Union[ModelType, NoneType, UndefinedType] = field(repr=False, default=UNDEFINED)

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
    parent_values_subtree: Optional[ComponentTreeWValuesType] = field(init=False, repr=False, default=None)

    # TODO: this is ugly 
    # set only in single case: partial mode, in_component_only_tree, component=component_only, instance_new
    on_component_only: Optional[IComponent] = field(repr=False, default=None)

    # this is currently created on 1st level and then copied to each next level
    # instance_new_struct_type: Union[StructEnum, NoneType, UndefinedType] = field(repr=False)

    # filled only when container list member
    index0: Optional[Index0Type] = None

    # parent instance / parent_instance_new - currenntly used for key_string logic
    parent_instance: Optional[ModelType] = field(repr=False, default=None)
    parent_instance_new: Optional[ModelType] = field(repr=False, default=None)

    # Usually used for ThisNS / This. namespace 
    # local_setup_session: Optional[ISetupSession] = field(repr=False, default=None)
    this_registry: Optional[IThisRegistry] = field(repr=False, default=None)

    # currently used only in cleaners - by default None to enable "automatic-copy",
    # if not set - then set to False in post_clean()
    instance_is_list: Optional[bool] = field(repr=False, default=None)

    # --------------------
    # -- autocomputed
    # --------------------
    # internal - filled in __post_init__
    key_string: Optional[str] = field(init=False, repr=False, default=None)

    # used to check root value in models registry 
    bound_model_root : Optional[IBoundModel] = field(repr=False, init=False, default=None)


    def __post_init__(self):
        self.clean()

    def clean(self):
        if not isinstance(self.container, IContainer):
            raise EntityInternalError(owner=self, msg=f"Expected IContainer, got: {self.container}")
        if not isinstance(self.component, IComponent):
            raise EntityInternalError(owner=self, msg=f"Expected IComponent, got: {self.component}")

        if self.index0 is not None and self.index0 < 0:
            raise EntityInternalError(owner=self, msg=f"index0 invalid value: {self.index0}")

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
        self._copy_attr_from_previous_frame(previous_frame, "parent_values_subtree",
                                            if_set_must_be_same=False)
        self._copy_attr_from_previous_frame(previous_frame, "value_node",
                                            if_set_must_be_same=False)

        # do not use ==, compare by instance (ALT: use id(instance) )
        if self.instance is previous_frame.instance:
            self._copy_attr_from_previous_frame(previous_frame, "container", may_be_copied=False)
            # will be autocomputed
            # self._copy_attr_from_previous_frame(previous_frame, "bound_model_root", may_be_copied=False)
            self._copy_attr_from_previous_frame(previous_frame, "instance_new")
            self._copy_attr_from_previous_frame(previous_frame, "instance_is_list")
            self._copy_attr_from_previous_frame(previous_frame, "index0")

            # only these can be copied
            self._copy_attr_from_previous_frame(previous_frame, "parent_instance")
            self._copy_attr_from_previous_frame(previous_frame, "parent_instance_new")

            # do not use ==, compare by instance (ALT: use id(instance) )
            if self.component is previous_frame.component:
                self._copy_attr_from_previous_frame(previous_frame, "on_component_only", may_be_copied=False)
                self._copy_attr_from_previous_frame(previous_frame, "key_string", may_be_copied=False)

                # NOTE: not this for now:
                self._copy_attr_from_previous_frame(previous_frame, "this_registry",
                                                    # for Apply -> ChildrenValidation setup can be different
                                                    if_set_must_be_same=False)
        # check / init again
        self.clean()


    def post_clean(self):
        # new logic - must be sure
        if self.value_node is not NOT_APPLIABLE:
            if not self.value_node:
                raise EntityInternalError(owner=self, msg="value_node not set")

            if not self.value_node.component == self.component:
                raise EntityInternalError(owner=self, msg=f"value_node component is different:\n {self.value_node.component} \n !=\n {self.component}")

            self. value_node.clean()

        if self.instance_is_list is None:
            self.instance_is_list = False

        # TODO: DRY - similar logic in ApplyResult._detect_instance_new_struct_type()
        if self.on_component_only:
            self.bound_model_root = (self.on_component_only 
                                     if self.on_component_only.is_subentity_items()
                                     else self.on_component_only.get_first_parent_container(consider_self=True)
                                    ).bound_model
            # can be list in this case
            # TODO: check if list only: if self.bound_model_root.type_info.is_list:
            # TODO: use self.current_frame.instance_is_list for list case
            instance_to_test = self.instance[0] \
                               if isinstance(self.instance, (list, tuple)) \
                               else self.instance
        else:
            self.bound_model_root = self.container.bound_model
            if self.instance_is_list and self.instance is not None:
                if not isinstance(self.instance, (list, tuple)):
                    raise EntityInternalError(owner=self, msg=f"Expected list of model instances, got: {self.instance}")
                instance_to_test = self.instance[0] if self.instance else None
            else:
                instance_to_test = self.instance

        if instance_to_test is NA_DEFAULTS_MODE:
            # defaults_mode
            pass
        elif instance_to_test is None:
            # if instance is a list and it is empty, then nothing to check
            pass
        else:
            pass
            # TODO: this should be checked only once per container, and I am not sure what is good way to do it
            # # if not is_model_class(instance_to_test.__class__):
            # #     raise EntityInternalError(owner=self, msg=f"Expected model instance: {instance_to_test.__class__} or list[instances], got: {self.instance}")
            # self.component.value_accessor.validate_instance_type(owner_name=self.component.name,
            #                                                      instance=instance_to_test,
            #                                                      model_type=self.container.bound_model.model)

        if self.this_registry:
            assert isinstance(self.this_registry, IThisRegistry)

        assert self.bound_model_root


    def set_parent_values_subtree(self, parent_values_subtree: Union[Dict, List]) -> Union[Dict, List]:
        " returns original values "
        # NOTE: didn't want to introduce a new stack frame layer for just changing one attribute
        # part of ComponentTreeWValuesType]
        parent_values_subtree_orig = self.parent_values_subtree
        self.parent_values_subtree = parent_values_subtree
        return parent_values_subtree_orig


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
    # TODO: consider: instance_new: Union[ModelType, UndefinedType] = UNDEFINED,
    instance_new: Optional[ModelType] = field(repr=False)


    context: Optional[IContext] = field(repr=False)
    # used in apply_partial
    component_name_only: Optional[str] = field(repr=False, default=None)

    # used in dump_defaults() - to get defaults dictionary
    defaults_mode: bool = field(repr=False, default=False)

    # ---- automatically computed -----

    # top of value tree - head of the tree
    top_value_node: ValueNode = field(repr=False, init=False, default=UNDEFINED)

    # list of all value trees - currently no need to have Dict
    value_node_list: List[ValueNode] = field(repr=False, init=False, default_factory=list)

    # extracted from component
    bound_model : IBoundModel = field(repr=False, init=False)

    # final status
    finished: bool = field(repr=False, init=False, default=False)

    # ---- internal attrs and structs ----

    # stack of frames - first frame is current. On the end of the process the
    # stack must be empty
    stack_frames: List[ApplyStackFrame] = field(repr=False, init=False, default_factory=list)
    current_frame: Optional[ApplyStackFrame] = field(repr=False, init=False, default=None)

    # I do not prefer attaching key_string to instance i.e. setattr(instance, key_string)
    # but having instance not expanded, but to store in a special structure
    key_string_container_cache : Dict[InstanceId, KeyString] = \
                            field(repr=False, init=False, default_factory=dict)

    # used when when collecting list of instances/attributes which are updated
    instance_by_key_string_cache : Dict[KeyString, ModelType] = \
                            field(repr=False, init=False, default_factory=dict)

    # TODO: consider this:
        # Instance can be dataclass, dictionary and so on.
        # Accessing instance attribute values are done in 2 places:
        #   a) getting/setting in processing children
        #   b) getting in executing DotExpression's AttributeNode-s
        # for a) value_accessor is used in all cases,
        # for b) not-attribute-instance it is too complicated to get values
        # with value_accessor, so input instance is copied to this shadow
        # dataclass instance which is used for AttributeNode's DotExpression-s.
        # instance_shadow_dc: DataclassType = field(init=False, repr=False)

    # Registry of history of attribute values for each ValueNode.
    # Filled only when Config.collect_value_history =True (see .register_instance_attr_change()).
    # Use .get_value_history() to fetch values.
    # Used only for analytical purposes and unit testing.
    # For any serious jobs ValueNode-s are used i.e. value_node_list and top_value_node members.
    #
    # Contains initial (first member), intermediate and
    # last values of each instance field/attribute.
    # If initial/first is different from final/last, then this value is changed.
    #
    # See .get_changes() for simplier struct - it does not require special Config param set.
    _value_history_dict: Dict[KeyString, List[InstanceAttrValue]] = \
                            field(repr=False, init=False, default_factory=dict)

    # Final list of created/deleted/updated list of changes - should be used only for display
    # Do not use directly, use get_changes() - list is lazy initialized and cached.
    # For any serious job use value_node_list or top_value_node
    #
    # See _value_history_dict for more complex struct - it requires special Config param previously set.
    _instance_change_list: Union[List[InstanceChange], UndefinedType] = field(repr=False, init=False, default=UNDEFINED)

    # When first argument is <type> and second is <type> then call function Callable
    # which will adapt second type to first one. Example: <string> + <int> -> <string> + str(<int>)
    binary_operations_type_adapters : Dict[Tuple[type, type], Callable[[Any], Any]] = field(repr=False, init=False, default_factory=dict)

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
    values_tree: Optional[ComponentTreeWValuesType] = field(init=False, repr=False, default=None)
    values_tree_by_key_string: Optional[Dict[KeyString, ComponentTreeWValuesType]] = field(init=False, repr=False, default=None)

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
    def apply(self) -> Self:
        ...

    @abstractmethod
    def get_current_value(self, strict: bool) -> LiteralType:
        ...

    @abstractmethod
    def get_changes(self) -> List[InstanceChange]:
        ...

    @property
    def value_history_dict(self) -> Dict[KeyString, List[InstanceAttrValue]]:
        if not self.entity.config.should_collect_value_history:
            raise EntityApplyError(owner=self, msg="Value history is not collected. Pass Config(..., collect_value_history=True) and try again.")
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

            # ORIG: if self.defaults_mode:
            # ORIG:     validation_failure = component.validate_type(apply_result=self, strict=strict, value=value)
            # ORIG:     if validation_failure:
            # ORIG:         raise EntityInternalError(owner=self, msg="TODO: Type validation failed in defaults_mode - probably should default value to None ")
            # ORIG:     validation_failure = None
            # ORIG: else:
            # ORIG:     validation_failure = component.validate_type(apply_result=self, strict=strict, value=value)
            # ORIG:     if validation_failure:
            # ORIG:         self.register_instance_validation_failed(component, validation_failure)


        return validation_failure


    @abstractmethod
    def _register_instance_attr_change(self,
                                       component: IComponent,
                                       dexp_result: ExecResult,
                                       new_value: Any,
                                       is_from_init_bind:bool=False) -> InstanceAttrValue:
        ...


    def register_value_node(self, value_node: ValueNode):
        assert not self.finished
        self.value_node_list.append(value_node)


    def finish(self):
        if self.finished:
            raise EntitySetupError(owner=self, msg="Method finish() already called.")

        if not len(self.stack_frames) == 0:
            raise EntityInternalError(owner=self, msg=f"Stack frames not released: {self.stack_frames}")

        self.exec_cleaners_registry.finish()
        self.finished = True

    def register_instance_validation_failed(self, component: IComponent, failure: ValidationFailure):
        if failure.component_key_string not in self.errors:
            self.errors[failure.component_key_string] = []
        self.errors[failure.component_key_string].append(failure)
        # TODO: mark invalid all children and this component


    def get_upward_components_dict(self, component: IComponent) \
            -> Dict[ComponentNameType, IComponent]:
        # CACHE
        if component.name not in self._component_children_upward_dict:
            components_tree = []
            curr_comp = component
            while curr_comp is not None:
                if curr_comp in components_tree:
                    raise EntityInternalError(owner=component, msg=f"Issue with hierarchy tree - duplicate node: {curr_comp.name}")
                components_tree.append(curr_comp)
                curr_comp = curr_comp.parent

            children_dict = {}
            # Although no name clash could happen, reverse to have local scopes
            # overwrite parent's scopes.
            for curr_comp in reversed(components_tree):
                comp_children_dict = curr_comp.get_children_tree_flatten_dict()
                children_dict.update(comp_children_dict)

            self._component_children_upward_dict[component.name] = children_dict

        return self._component_children_upward_dict[component.name]

    def get_changes_as_list_dict(self) -> List[Dict[str, Any]]:
        """
        used only in unit-tests for now
        """
        return [asdict(instance_change) for instance_change in self.get_changes()]

# ------------------------------------------------------------

def extract_type_info(
        attr_node_name: str,
        inspect_object: Any,
        ) -> Tuple[
                TypeInfo, 
                Optional[ModelField], 
                Optional[IFunctionDexpNode]  # noqa: F821
                ]:
    """
    Main logic for extraction from parent object (dataclass, py class, dexp
    instances) member by name 'attr_node_name' -> data (struct, plain value),
    or IFunctionDexpNode instances 

    This function uses specific base interfaces/classes (IBoundModel,
    IDotExpressionNode), so it can not be put in meta.py

    See 'meta. def get_or_create_by_type()' for further explanation when to use
    this function (preffered) and when directly some other lower level meta.py
    functions.
    """
    if isinstance(inspect_object, IBoundModel):
        inspect_object = inspect_object.model

    # function - callable and not class and not pydantic?
    parent_type_info = None
    if isinstance(inspect_object, IDotExpressionNode):
        parent_type_info = inspect_object.get_type_info()
        parent_object = parent_type_info.type_
    elif is_function(inspect_object):
        raise Exception("should not happen, use Function/Factory-ies")
    else:
        # normal object - hopefully with with type hinting
        parent_object = inspect_object

    if isinstance(parent_object, TypeInfo):
        # go one level deeper
        parent_type_info = parent_object
        if not isinstance(parent_object.type_, type):
            raise EntitySetupValueError(item=inspect_object, msg=f"Inspected object's type hint is not a class object/type: {parent_object.type_} : {parent_object.type_}, got: {type(parent_object.type_)} ('.{attr_node_name}' process)")
        # Can be method call, so not pydantic / dataclass are allowed too
        # if not is_model_class(parent_object.type_):
        #     raise EntitySetupValueError(item=inspect_object, msg=f"Inspected object's type hint type is not 'dataclass'/'Pydantic.BaseModel' type: {parent_object.type_}, got: {parent_object.type_} ('.{attr_node_name}' process)")
        parent_object = parent_object.type_


    fields = []

    type_info = None
    th_field = None
    # func_node = None

    is_parent_dot_expr = isinstance(parent_object, DotExpression)
    if is_parent_dot_expr:
        raise EntitySetupNameNotFoundError(item=inspect_object, msg=f"Attribute '{attr_node_name}' is not member of expression '{parent_object}'.")

    # when parent virtual expression is not assigned to data-type/data-attr_node,
    # then it is sent directly what will later raise error "attribute not found" 
    if not is_parent_dot_expr:
        if is_method_by_name(parent_object, attr_node_name):
            # parent_object, attr_node_name
            raise EntitySetupNameError(item=inspect_object,
                        msg=f"Inspected object's is a method: {parent_object}.{attr_node_name}. "
                        "Calling methods on instances are not allowed.")

        if is_model_class(parent_object):
            # raise EntityInternalError(item=inspect_object, msg=f"'Parent is not DC/PYD class -> {parent_object} : {type(parent_object)}'.")

            if not hasattr(parent_object, "__annotations__"):
                # TODO: what about pydantic?
                raise EntityInternalError(item=inspect_object, msg=f"'DC/PYD class {parent_object}' has no metadata (__annotations__ / type hints), can't read '{attr_node_name}'. Add type-hints or check names.")

            model_class: ModelType = parent_object

            # === parent type hint
            parent_py_type_hints = extract_py_type_hints(model_class, f"setup_session->{attr_node_name}:DC/PYD")

            py_type_hint = parent_py_type_hints.get(attr_node_name, None)
            if py_type_hint:
                type_info = TypeInfo.get_or_create_by_type(
                                py_type_hint=py_type_hint,
                                caller=parent_object,
                                )

            # === Dataclass / pydantic field metadata - only for information
            th_field, fields = extract_model_field_meta(inspect_object=model_class, attr_node_name=attr_node_name)

    if not type_info:
        if not fields:
            fields = get_model_fields(parent_object, strict=False)
        field_names = list(fields.keys())
        valid_names = f"Valid attributes: {get_available_names_example(attr_node_name, field_names)}" if field_names else "Type has no attributes at all."
        raise EntitySetupNameNotFoundError(msg=f"Type object {parent_object} has no attribute '{attr_node_name}'. {valid_names}")

    if not isinstance(type_info, TypeInfo):
        raise EntityInternalError(msg=f"{type_info} is not TypeInfo")

    return type_info, th_field # , func_node


