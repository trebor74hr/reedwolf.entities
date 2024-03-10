from abc import ABC, abstractmethod
from typing import (
    Union,
    Optional,
    List,
    Any,
)
from dataclasses import (
    dataclass,
    field,
)

from .namespaces import (
    ThisNS,
    MultiNamespace,
    FieldsNS,
    NamespaceRule,
)
from .utils import (
    UNDEFINED,
    UndefinedType,
    NA_DEFAULTS_MODE,
    to_repr,
)
from .exceptions import (
    EntitySetupValueError,
    EntityInternalError,
    EntityApplyNameError,
    EntityApplyValueError,
    EntitySetupNameError,
    EntityApplyNameNotFoundError,
)
from .namespaces import (
    Namespace,
)
from .meta_dataclass import (
    ComponentStatus,
)
from .meta import (
    TypeInfo,
    is_function,
    AttrName,
    AttrValue,
    KlassMember,
    ContainerId,
    is_list_instance_or_type,
    IDexpValueSource,
)
from .expressions import (
    IDotExpressionNode,
    RegistryRootValue,
)
from .custom_attributes import (
    Attribute,
    AttributeByMethod,
)
from .base import (
    IContainer,
    IField,
    IApplyResult,
    ExecResult,
    ReservedAttributeNames,
    IDataModel,
    IFieldGroup,
    IComponent,
    RESERVED_ATTRIBUTE_NAMES,
)

# ------------------------------------------------------------


@dataclass(repr=False)
class IAttrDexpNode(IDotExpressionNode, ABC):
    """
    TODO: ren IAttrDexpNode -> AttrDexpNodeBase, define base.py::IAttrDexpNode or expressions.py
    Name comes from dot-chaining nodes - e.g.
        M . company . name
            ADN()     ADN()
    """
    name: str

    # TODO: rename to owner
    # data: Union[IComponent, TypeInfo, IAttribute]

    # namespace node belongs to
    namespace: Union[Namespace, MultiNamespace] = None

    # if attribute node is for list - e.g. ITEMS
    for_list: bool = field(repr=False, default=False)

    # TODO: I don't like this - overlaps with data/owner - remove this one?
    # based on attr_node_type - can contain Field() or class - used later to extract details
    # type_object: Union[ModelField, KlassMember, NoneType] = field(repr=False, default=UNDEFINED)

    # ----- Later evaluated ------
    type_info: Optional[TypeInfo] = field(init=False, default=None)
    is_finished: bool = field(init=False, repr=False, default=False)
    full_name: str = field(init=False, repr=False, default=UNDEFINED)

    # data source name - used only for diagnostics - in __str__
    data_supplier_name: str = field(init=False, repr=False, default=UNDEFINED)
    denied: bool = field(init=False, repr=False, default=False)

    # attr_node_type: AttrDexpNodeTypeEnum = field(init=False)

    def __post_init__(self):
        if isinstance(self.namespace, MultiNamespace):
            ns_names = '/'.join([f"{nsr.namespace._name}" for nsr in self.namespace.rules])
            self.full_name = f"{ns_names}.{self.name}"
        else:
            if not isinstance(self.namespace, Namespace):
                raise EntityInternalError(owner=self, msg=f"Attribute 'namespace' should be Namespace, got: {self.namespace}")
            self.full_name = f"{self.namespace._name}.{self.name}"

        if not isinstance(self.name, str) or self.name in (None, UNDEFINED):
            raise EntityInternalError(owner=self, msg=f"AttrDexpNode should have string name, got: {self.name}")


    def _check_data(self, attr_name: AttrName, data: Any):
        if data is None:
            raise EntityInternalError(owner=self, msg=f"Attribute '{attr_name}' is required.")

        if is_function(data):
            raise EntitySetupValueError(owner=self, msg=f"Attribute '{attr_name}' is a function. Maybe you forgot to wrap it with 'reedwolf.entities.Function()'?")

        if attr_name is RESERVED_ATTRIBUTE_NAMES:
            # NOTE: independent if the registry/namespace is implemented in the registry or not,
            #       reserved names are forbidden in all registries/namespaces.
            #       (model={model_klass.__name__})
            raise EntitySetupNameError(owner=self.namespace, msg=f"Expression's attribute name '{attr_name} is reserved. Rename the attribute and try again (type of attribute: {self.__class__.__name__})")

    def _get_repr_extra(self) -> str:
        return ", DENIED" if self.denied else ""

    def finish(self):
        """ fill type_info, must be available for all nodes - with exceptions those with .denied don't have it """
        if self.type_info is None:
            self._fill_type_info()
        super().finish()

    def _fill_type_info(self):
        """
        Fill self.type_info, must be available for all nodes - with exceptions those with .denied don't have it
        In regular case will raise error, inherit and fill the logic
        when self.type_info could be set lazy.
        """
        raise EntityInternalError(owner=self, msg=f"Attribute .type_info can not be set [2]")

    def get_type_info(self) -> TypeInfo:
        """
        lazy init - after components are setup()
        """
        if self.type_info is None:
            self._fill_type_info()
        assert self.type_info
        return self.type_info

    def execute_node(self,
                     apply_result: IApplyResult,
                     # previous - can be undefined too
                     dexp_result: Union[ExecResult, UndefinedType],
                     namespace: Union[Namespace, UndefinedType],
                     is_1st_node: bool,
                     is_last_node: bool,
                     prev_node_type_info: Optional[TypeInfo],
                     ) -> ExecResult:

        # if is_last_node and not self.is_finished:
        if is_last_node and not self._status == ComponentStatus.finished:
            raise EntityInternalError(owner=self, msg="Last dexp node is not finished.") 

        # TODO: not nicest way - string split
        #       for subentity_items: [p._name for p in frame.container.data_model.model.Path]
        names = self.name.split(".")
        attr_name = names[-1]
        assert dexp_result not in (None,)

        if dexp_result in (UNDEFINED, None):
            # ==== Initial / 1st value - get root value from registry / namespace, e.g. M.<attr_name>
            dexp_result = ExecResult()
            frame = apply_result.current_frame

            if not len(names)==1:
                if frame.container.is_subentity_any() or frame.on_component_only:
                    raise EntityInternalError(owner=self, msg=f"Attribute node - execution initial step for SubEntityItems/SubEntity failed, expected single name members (e.g. M), got: {self.name}\n  == Compoonent: {frame.container}")
                else:
                    raise EntityInternalError(owner=self, msg=f"Initial evaluation step for non-subentity_items failed, expected single name member (e.g. M), got: {self.name}\n  == Compoonent: {frame.container}")

            # NOTE: self.namespace can be MultiNamespace
            if namespace == ThisNS:
                # TODO: DRY this - identical logic in expressions.py:: Setup()
                if not apply_result.current_frame.this_registry:
                    raise EntityApplyNameError(owner=self, msg=f"Namespace 'This.' is not available in this context, got: {self.name}\n  == Compoonent: {frame.container}")
                registry = apply_result.current_frame.this_registry
            else:
                # take from setup_session
                setup_session = apply_result.get_setup_session()
                # NOTE: self.namespace can be MultiNamespace
                registry = setup_session.get_registry(namespace)

            # Get starting instance. For FieldsNS.<field> can be ValueNode instance
            root_value: RegistryRootValue = registry.apply_to_get_root_value(apply_result=apply_result,
                                                                             attr_name=attr_name,
                                                                             caller=str(self))

            value_prev = root_value.value_root
            attr_name_new = root_value.attr_name_new

            if attr_name_new:
                # e.g. case ReservedAttributeNames.VALUE_ATTR_NAME, i.e. .Value
                attr_name = attr_name_new

            if root_value.do_fetch_by_name != UNDEFINED:
                do_fetch_by_name = root_value.do_fetch_by_name
            elif apply_result.current_frame.on_component_only and registry.ROOT_VALUE_NEEDS_FETCH_BY_NAME:
                # == M.name mode
                # TODO: not nice solution
                do_fetch_by_name = False
            else:
                do_fetch_by_name = True
        else:
            # ==== 2+ value - based on previous result and evolved one step further, e.g. M.access.alive
            # TODO: for components with value path (field from other container) - this may fail
            # if not len(names)>1:
            #     raise EntityInternalError(owner=self, msg=f"Names need to be list of at least 2 members: {names}")

            value_prev = dexp_result.value
            do_fetch_by_name = True

        if do_fetch_by_name:
            if not is_1st_node and is_list_instance_or_type(value_prev):
                raise EntityApplyValueError(owner=self, msg=f"Expected standard object, got list compatible type: {type(value_prev)}: {to_repr(value_prev)}")

            if prev_node_type_info and prev_node_type_info.is_list:
                raise EntityApplyNameError(owner=self, msg=f"Fetching attribute '{attr_name}' expected list and got: '{to_repr(value_prev)}': '{type(value_prev)}'")

            # Handles ValueNode cases too  #  apply_result = apply_result,
            value_new = self._get_value_new(value_prev=value_prev,
                                            attr_name=attr_name)

            # NOTE: dropped iterating list results - no such test example and hard to imagine which syntax to
            #       use and when to use it: Convert previous value to list, process all and convert back to
            #       single object when previous_value is not a list.
        else:
            value_new = value_prev

        value_is_list = (value_new.is_list() if isinstance(value_new, IDexpValueSource)
                         else isinstance(value_new, (list, tuple)))

        # TODO: currently this is not done, too many issues, needs further investigation
        # if root_value:
        #   self.check_root_value_type_info(root_value, value_new OR root_value.value_new)

        # TODO: check type_info match too - and put in all types of nodes - functions/operators
        if apply_result.component_name_only and apply_result.instance_new == value_new:
            # TODO: this is workaround when single instance is passed to update single item in SubEntityItems[List]
            #       not good solution
            ...
        elif value_is_list:
            if not self.is_list():
                raise EntityApplyValueError(owner=self, msg=f"Attribute '{attr_name}' should not be a list, got: '{to_repr(value_new)}': '{type(value_new)}'")
        elif value_new is None:
            pass
            # NOTE: value not checked - can be evaluated to something not-None later
            # if not self.get_type_info().is_optional:
            #     raise EntityApplyValueError(owner=self, msg=f"Attribute '{attr_name}' has 'None' value and type is not 'Optional'.")
        elif self.is_list():
            # apply_result.entity.get_component(apply_result.component_name_only)
            raise EntityApplyValueError(owner=self, msg=f"Attribute '{attr_name}' should be a list, got: '{to_repr(value_new)}': '{type(value_new)}'")

        # TODO: hm, changer_name is equal to attr_name, any problem / check / fix ...

        # ValueNode case
        if is_last_node and isinstance(value_new, IDexpValueSource):
            # evaluate last ValueNode (in unfinished state) to concrete value (e.g. int,str,date)
            value_new = value_new.get_value(strict=False)

        dexp_result.set_value(attr_name=attr_name, changer_name=attr_name, value=value_new)

        return dexp_result


    def _get_value_new(self, value_prev: AttrValue, attr_name: AttrName) -> AttrValue:
        # apply_result: IApplyResult,
        if isinstance(value_prev, IDexpValueSource):
            # ValueNode case
            value_node: IDexpValueSource = value_prev
            # try to find in children first, then in container_children if applicable
            if not getattr(value_node, "children", UNDEFINED):
                raise EntityInternalError(owner=self, msg=f"Attribute '{attr_name}' can not be found, node '{value_node.name}' has no children")

            assert hasattr(value_node, "component"), value_node
            assert value_node.component.get_attr_dexp_node_store().get(attr_name, default=UNDEFINED), value_node

            dexp_value_node = value_node.children.get(attr_name, UNDEFINED)

            if dexp_value_node is UNDEFINED and hasattr(value_node, "container_children"):
                dexp_value_node = value_node.container_children.get(attr_name, UNDEFINED)

            if dexp_value_node is UNDEFINED:
                raise EntityApplyNameNotFoundError(owner=self, msg=f"Attribute '{attr_name}' can not be found in '{value_node.name}'")

            value_new = dexp_value_node
        else:
            # model or special attribute
            # removed. idx == 0 and
            if attr_name == ReservedAttributeNames.INSTANCE_ATTR_NAME:
                value_new = value_prev
            else:
                if (value_prev is UNDEFINED
                    or value_prev is None
                    or value_prev is NA_DEFAULTS_MODE
                ):
                    # 'Maybe' monad like - previous value has NO VALUE => promote it further and further to final value
                    # TODO: capture source/attribute of the first None/<empty> value is evaluated - for diagnostics
                    value_new = value_prev
                else:
                    if isinstance(attr_name, (Attribute, AttributeByMethod)):
                        attr_name = attr_name.name

                    if not isinstance(attr_name, str):
                        raise EntityInternalError(owner=self, msg=f"Attribute name must be a string, got. {attr_name}")

                    if not hasattr(value_prev, attr_name):
                        # TODO: list which fields are available
                        # if all types match - could be internal problem?
                        raise EntityApplyNameNotFoundError(owner=self, msg=f"Attribute '{attr_name}' not found in '{to_repr(value_prev)}': '{type(value_prev)}'")

                    # finally - fetch the attribute / callable by name from value_prev
                    value_new = getattr(value_prev, attr_name)
                    if callable(value_new):
                        try:
                            # NOTE: must be method(self) or function() - no args expected
                            value_new2 = value_new()
                        except Exception as ex:
                            raise EntityApplyValueError(owner=self,
                                   msg=f"Attribute '{attr_name}' is a callable '{value_new}' which raised error when called:: '{ex}'")
                        value_new = value_new2

        return value_new

    def is_optional(self):
        return self.type_info.is_optional if self.type_info else False

    def is_list(self):
        return self.type_info.is_list if self.type_info else False

    def as_str(self) -> str:
        """
        Pretty print
        """
        extra_str = self._get_repr_extra()
        alt_name = f"{self.data_supplier_name}" if self.data_supplier_name != self.name else ""
        out = self.type_info.as_str() if self.type_info else "-"
        out += " " + (", ".join([val for val in [alt_name, extra_str] if val]))
        return out.strip()

    def __str__(self):
        extra_str = self._get_repr_extra()
        return f"{self.__class__.__name__}({self.full_name} : {self.data_supplier_name}{extra_str})"

    def __repr__(self):
        return str(self)

# ------------------------------------------------------------


@dataclass(repr=False)
class AttrDexpNodeForTypeInfo(IAttrDexpNode):
    """
    Class Attribute or Function return type
    NOTE: for other classes type_info is private attribute automatically computed.
    """
    type_info: Union[TypeInfo] = None

    def __post_init__(self):
        super().__post_init__()

        self._check_data('type_info', self.type_info)

        if not isinstance(self.type_info, TypeInfo):
            raise EntityInternalError(owner=self, msg=f"Expected data: TypeInfo, got: {self.type_info}")
        # self.type_info = self.type_info

        # if self.type_object is UNDEFINED:
        #     raise EntityInternalError(owner=self, msg=f"TypeInfo case - type_object should be set (ModelField), got: {self.type_object}")

        # TODO: put in TypeInfo this logic:
        type_name = getattr(self.type_info.type_, "__name__",
                            getattr(self.type_info.type_, "_name", repr(self.type_info.type_)))
        self.data_supplier_name = f"TH[{type_name}]"


@dataclass(repr=False)
class AttrDexpNodeForDataModel(IAttrDexpNode):
    data_model: Union[IDataModel] = None

    def __post_init__(self):
        super().__post_init__()

        self._check_data('data_model', self.data_model)

        if not isinstance(self.data_model, IDataModel):
            raise EntityInternalError(owner=self, msg=f"Expected data: IDataModel, got: {self.data_model}")

        self.type_info = self.data_model.get_type_info()

        self.data_supplier_name = f"{self.data_model.name}"

    # def get_component(self) -> Optional[IComponent]:
    #     return self.data_model


@dataclass(repr=False)
class AttrDexpNodeForKlassMember(IAttrDexpNode):

    klass_member: KlassMember = None

    # ---- automatically evaluated
    # attribute: IAttribute = field(init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()

        self._check_data('klass_member', self.klass_member)

        if not isinstance(self.klass_member, KlassMember): #  and self.klass_member.attribute is self.attribute):
            raise EntityInternalError(owner=self, msg=f"Attribute 'klass_member' must be instance of KlassMember, got: {self.klass_member}")

        # self.attribute = self.klass_member.attribute

        self.type_info = self.klass_member.attribute.output_type_info
        assert self.type_info
        type_name = str(self.klass_member)  # field name
        self.data_supplier_name = f"FN[{type_name}]"


@dataclass(repr=False)
class AttrDexpNodeForComponent(IAttrDexpNode):
    """
    TODO: cache this type of attr_dexp_node - attach to component
          currently for the same component attr_node is created 3 times:
            - localfields,
            - topfields,
            - children
          Name could differ, but name is not neccessary to hold here.
    """

    component: Union[IContainer, IFieldGroup, IField] = None

    # can be set from outside - e.g. for Children
    name: str = field(default=None)

    # initially set to Namespace, used only for check, will changed later to MultiNamespace()
    namespace: Union[Namespace, MultiNamespace] = field(repr=False, default=None)

    # ---- later evaluated ----
    # in some cases (LocalFieldsNS) some fields should not be referenced, but are registered within Registry.store,
    # in order to report to user the reason for denial - better than just to report - attribute / field name not found.
    denied: bool = field(init=False, repr=False)
    deny_reason: str = field(init=False, repr=False)



    def __post_init__(self):
        # automatically set and interpreted later
        if isinstance(self.namespace, Namespace):
            if self.namespace not in (FieldsNS, ThisNS):
                raise EntityInternalError(owner=self, msg=f"Expected F./This., got: {self.namespace}")

            deny_root = (self.name in RESERVED_ATTRIBUTE_NAMES)
            self.namespace = MultiNamespace([
                NamespaceRule(namespace=FieldsNS, deny_root=deny_root,
                              deny_root_reason_templ="Reserved attribute '{attr_name}' is not available as a first attribute in the namespace '{namespace}'. Attribute could be available in a non-first attributes (e.g. 'Fields.<field-name>.{attr_name}') or in field's context's 'This' namespace (e.g. 'This.{attr_name}')."
                                                     if deny_root else ""),
                NamespaceRule(namespace=ThisNS),
            ])

        assert isinstance(self.namespace, MultiNamespace), self.namespace

        if self.name is None:
            self.name = self.component.name
            assert "." not in self.name
        # else: assert "." not in self.name

        self._check_data('component', self.component)

        # if not isinstance(self.component, (IContainer, IField, IFieldGroup)):
        if not isinstance(self.component, (IComponent,)):
            raise EntityInternalError(owner=self, msg=f"Expected data: Union[IContainer, IField, IFieldGroup], got: {self.component}")

        if self.component.has_data():
            self.denied = False
            self.deny_reason = ""
        else:
            # stored - but should not be used
            self.denied = True
            self.deny_reason = f"Component of type {self.component.__class__.__name__} can not be referenced in DotExpressions"

        super().__post_init__()

        self.data_supplier_name = f"{self.name}"
        # NOTE: self.type_info is lazy and will be set in _fill_type_info()

        # if not self.data.has_data():
        #     raise EntityInternalError(owner=self, msg=f"Expected component that has_data, got: {self.data}")

    def _fill_type_info(self):
        assert self.type_info is None
        if not self.denied:
            # assert self.attr_node_type not in (AttrDexpNodeTypeEnum.FIELD_GROUP,)
            self.type_info = self.component.get_type_info()
            if not self.type_info:
                raise EntityInternalError(owner=self, msg=f"Attribute .type_info could not be set [1] (type={type(self.component)}).")


# ------------------------------------------------------------


@dataclass
class AttrValueContainerPath:
    """
    Used in TopFieldsRegistry as a result when
    Attribute/Field (by AttrName) is found within
    some other visible Container.
    """
    attr_name: AttrName
    container_id_from: ContainerId = field(repr=False)
    container_id_to: ContainerId = field(repr=False)
    container_node_mode: bool = field(repr=False)
    path_up: List[ContainerId]
    path_down: List[ContainerId]


@dataclass
class AttrDexpNodeWithValuePath(AttrDexpNodeForComponent):
    """
    Used in TopFieldsRegistry - contains path to the ValueNode tree container owner.
    """
    attr_value_container_path: Union[AttrValueContainerPath, UndefinedType] = field(repr=False, default=UNDEFINED)

    def __post_init__(self):
        super().__post_init__()
        if self.attr_value_container_path is UNDEFINED:
            raise EntityInternalError(owner=self, msg=f"Attribute attr_value_container_path is required")

# ============================================================
# Obsolete code
# ============================================================
# # Convert previous value to list, process all and convert back to
# # single object when previous_value is not a list
# ------------------------------------------------------------
# NOTE: dropped iterating list results - no such test example and hard to imagine which syntax to
#       use and when to use it.
# result_is_list = isinstance(value_previous, (list, tuple))
# if not result_is_list:
#   # TODO: handle None, UNDEFINED?
#   if prev_node_type_info and prev_node_type_info.is_list:
#         raise EntityApplyNameError(owner=self, msg=f"Fetching attribute '{attr_name}' expected list and got: '{to_repr(value_previous)}': '{type(value_previous)}'")
#   value_prev_as_list = [value_previous]
# else:
#   value_prev_as_list = value_previous
#   if prev_node_type_info and not prev_node_type_info.is_list:
#     raise EntityApplyNameError(owner=self, msg=f"Fetching attribute '{attr_name}' got list what is not expected, got: '{to_repr(value_previous)}': '{type(value_previous)}'")
# value_new_as_list = []
# for idx, value_prev in enumerate(value_prev_as_list, 0):
#   value_new = self._get_value_new(apply_result=apply_result,
#   value_new_as_list.append(value_new)
# if result_is_list:
#   value_new = value_new_as_list
# else:
#   assert len(value_new_as_list) == 1
#   value_new = value_new_as_list[0]

# elif isinstance(value_prev, IAttributeAccessorBase):
#     # NOTE: if this is last in chain - fetch final value
#     value_new = value_prev.get_attribute(
#         apply_result=apply_result,
#         attr_name=attr_name)
