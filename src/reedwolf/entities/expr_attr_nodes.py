from typing import (
    Any,
    Union,
    Optional, List,
)
from dataclasses import (
    dataclass,
    field,
)

from .namespaces import (
    ThisNS,
)
from .utils import (
    UNDEFINED,
    NA_DEFAULTS_MODE,
    to_repr,
)
from .exceptions import (
    EntitySetupValueError,
    EntityInternalError,
    EntityApplyNameError,
    EntityApplyValueError,
)
from .namespaces import (
    Namespace,
)
from .expressions import (
    DotExpression,
    IDotExpressionNode,
    RegistryRootValue,
)
from .meta import (
    TypeInfo,
    is_model_klass,
    is_function,
    ModelField,
    AttrName,
    AttrValue,
    NoneType,
    KlassMember,
    ContainerId,
    ComponentStatus,
    is_list_instance_or_type, IDexpValueSource,
)
from .custom_attributes import (
    Attribute,
    AttributeByMethod,
)
# TODO: remove this dependency
from .base import (
    UndefinedType,
    IComponent,
    IContainer,
    IField,
    IApplyResult,
    ExecResult,
    ReservedAttributeNames,
    AttrDexpNodeTypeEnum,
)

# ------------------------------------------------------------


@dataclass
class AttrDexpNode(IDotExpressionNode):
    """
    Name comes from dot-chaining nodes - e.g.
        M.name
          company and name are AttrDexpNodes
    """
    # TODO: too many if-s - make subclassing instead?
    name: str
    # TODO: data can be also - check each:
    #   - some dataproviding function
    # TODO: Union[TypeInfo, IFunctionDexpNode, DotExpression, 'Component', type]  #
    #       noqa: F821
    data: Any 
    namespace: Namespace

    # based on attr_node_type - can contain Field() or class - used later to extract details
    type_object: Union[ModelField, KlassMember, NoneType] = field(repr=False, default=UNDEFINED)
    type_info: Optional[TypeInfo] = field(default=None)
    # TODO: rename to: func_node or function_node
    # function: Optional[Union[IFunctionDexpNode]] = field(repr=False, default=None)

    # in some cases (LocalFieldsNS) some fields should not be referenced, but are registered within Registry.store,
    # in order to report to user the reason for denial - better than just to report - attribute / field name not found.
    denied: bool = False
    deny_reason: str = ""

    # ----- Later evaluated ------
    # could live without this attribute but ...
    attr_node_type: Union[AttrDexpNodeTypeEnum] = field(init=False)
    is_finished: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        self.full_name = f"{self.namespace._name}.{self.name}"

        if not isinstance(self.name, str) or self.name in (None, UNDEFINED):
            raise EntityInternalError(owner=self, msg=f"AttrDexpNode should have string name, got: {self.name}")

        # ---------------------------------------------
        # CASES: COMPONENTS
        # ---------------------------------------------
        # TODO: antipattern - split to diff class implementations (inherit and make diff classes)
        if self.data is None:
            raise EntityInternalError(owner=self, msg=f"Data is not set for {self}")

        if isinstance(self.data, IContainer):
            self.attr_node_type = AttrDexpNodeTypeEnum.CONTAINER
            self.data_supplier_name = f"{self.data.name}"
        elif isinstance(self.data, IField):
            self.attr_node_type = AttrDexpNodeTypeEnum.FIELD
            self.data_supplier_name = f"{self.data.name}"
        elif isinstance(self.data, IComponent):
            self.attr_node_type = AttrDexpNodeTypeEnum.COMPONENT
            self.data_supplier_name = f"{self.data.name}"

        # ---------------------------------------------
        # CASE: Value expressions and FunctinoBase 
        # ---------------------------------------------
        elif isinstance(self.data, DotExpression):
            if self.data._func_args:
                self.attr_node_type = AttrDexpNodeTypeEnum.VEXP_FUNC
            else:
                self.attr_node_type = AttrDexpNodeTypeEnum.VEXP
            self.data_supplier_name = f"{self.data!r}"

        # ---------------------------------------------
        # CASE: Concrete type
        # ---------------------------------------------
        elif is_model_klass(self.data):
            # ModelKlassType
            assert type(self.data) == type, self.data
            # ALT: inspect.isclass()
            self.attr_node_type = AttrDexpNodeTypeEnum.MODEL_CLASS
            self.data_supplier_name = f"{self.data.__name__}"

        # ---------------------------------------------
        # CASE: Class Attribute or Function return type
        # ---------------------------------------------
        elif isinstance(self.data, TypeInfo):
            if self.type_object is UNDEFINED:
                raise EntityInternalError(owner=self, msg=f"TypeInfo case - type_object should be set (ModelField), got: {self.type_object}")
            self.attr_node_type = AttrDexpNodeTypeEnum.TH_FIELD
            # .type_ could be a class/type or NewType instance
            type_name = getattr(self.data.type_, "__name__", 
                                getattr(self.data.type_, "_name", repr(self.data.type_)))
            self.data_supplier_name = f"TH[{type_name}]"
        elif isinstance(self.data, Attribute):
            # TODO: don't like this - too hackish
            if not isinstance(self.type_object, KlassMember) and self.type_object.name == self.data:
                raise EntityInternalError(owner=self, msg=f"MethodName case - type_object must be instance of KlassMember, got: {self.type_object},")
            self.attr_node_type = AttrDexpNodeTypeEnum.TH_FIELD
            type_name = str(self.data)  # field name
            self.data_supplier_name = f"FN[{type_name}]"

        elif isinstance(self.data, AttributeByMethod):
            # TODO: don't like this - too hackish
            if not isinstance(self.type_object, KlassMember) and self.type_object.name == self.data:
                raise EntityInternalError(owner=self, msg=f"MethodName case - type_object must be instance of KlassMember, got: {self.type_object},")
            self.attr_node_type = AttrDexpNodeTypeEnum.TH_FUNCTION
            type_name = str(self.data)  # method name
            self.data_supplier_name = f"MN[{type_name}]"

        else:
            if is_function(self.data):
                raise EntitySetupValueError(owner=self, msg=f"Node '.{self.name}' is a function. Maybe you forgot to wrap it with 'reedwolf.entities.Function()'?")
            raise EntitySetupValueError(owner=self, msg=f"AttrDexpNode {self.name} should be based on PYD/DC class, got: {self.data}")

        # NOTE: .type_info could be calculated later in finish() method

    def finish(self):
        """ fill type_info, must be available for all nodes - with exceptions those with .denied don't have it """

        if self.type_info is None:

            if self.attr_node_type == AttrDexpNodeTypeEnum.FIELD:
                type_info = self.data
                if not type_info.bound_attr_node:
                    raise EntityInternalError(owner=self, msg=f"AttrDexpNode {self.data} .bound_attr_node not set.")

                bound_type_info = type_info.bound_attr_node.get_type_info()
                if not bound_type_info:
                    raise EntityInternalError(owner=self, msg=f"AttrDexpNode data.bound_attr_node={self.data} -> {self.data.bound_attr_node} .type_info not set.")

                # transfer type_info from type_info.bound attr_node
                self.type_info = bound_type_info

            elif self.attr_node_type == AttrDexpNodeTypeEnum.DATA:
                self.type_info = self.data.type_info

            elif not self.denied and self.attr_node_type == AttrDexpNodeTypeEnum.CONTAINER:
                self.type_info = self.data.get_type_info()

            elif self.attr_node_type not in (
                    AttrDexpNodeTypeEnum.CONTAINER,
                    AttrDexpNodeTypeEnum.COMPONENT,
                    ):
                # all other require type_info
                raise EntityInternalError(owner=self, msg=f"For attr_node {self.attr_node_type} .type_info not set (type={type(self.data)}).")

            if not self.denied and not self.type_info:
                raise EntityInternalError(owner=self, msg=f"For attr_node {self.attr_node_type} .type_info could not not set (type={type(self.data)}).")

        super().finish()


    def get_type_info(self) -> TypeInfo:
        # if strict and self.type_info is None:
        #     raise EntityInternalError(owner=self, msg=f"Finish was not called, type_info is not set")
        return self.type_info


    def execute_node(self,
                     apply_result: IApplyResult,
                     # previous - can be undefined too
                     dexp_result: Union[ExecResult, UndefinedType],
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

            if self.namespace == ThisNS:
                # TODO: DRY this - identical logic in expressions.py:: Setup()
                if not apply_result.current_frame.this_registry:
                    raise EntityApplyNameError(owner=self, msg=f"Namespace 'This.' is not available in this context, got: {self.name}\n  == Compoonent: {frame.container}")
                registry = apply_result.current_frame.this_registry
            else:
                # take from setup_session
                setup_session = apply_result.get_setup_session()
                registry = setup_session.get_registry(self.namespace)

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
            if not len(names)>1:
                raise EntityInternalError(owner=self, msg=f"Names need to be list of at least 2 members: {names}") 
            value_prev = dexp_result.value
            do_fetch_by_name = True

        if do_fetch_by_name:
            if not is_1st_node and is_list_instance_or_type(value_prev):
                raise EntityApplyValueError(owner=self, msg=f"Expected standard object, got list compatible type: {type(value_prev)}: {to_repr(value_prev)}")

            if prev_node_type_info and prev_node_type_info.is_list:
                raise EntityApplyNameError(owner=self, msg=f"Fetching attribute '{attr_name}' expected list and got: '{to_repr(value_prev)}': '{type(value_prev)}'")

            # Handles ValueNode cases too
            value_new = self._get_value_new(apply_result=apply_result,
                                            value_prev=value_prev,
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
            if not self.islist():
                raise EntityApplyValueError(owner=self, msg=f"Attribute '{attr_name}' should not be a list, got: '{to_repr(value_new)}': '{type(value_new)}'")
        elif value_new is None:
            pass
            # NOTE: value not checked - can be evaluated to something not-None later
            # if not self.get_type_info().is_optional:
            #     raise EntityApplyValueError(owner=self, msg=f"Attribute '{attr_name}' has 'None' value and type is not 'Optional'.")
        elif self.islist():
            # apply_result.entity.get_component(apply_result.component_name_only)
            raise EntityApplyValueError(owner=self, msg=f"Attribute '{attr_name}' should be a list, got: '{to_repr(value_new)}': '{type(value_new)}'")

        # TODO: hm, changer_name is equal to attr_name, any problem / check / fix ...

        # ValueNode case
        if is_last_node and isinstance(value_new, IDexpValueSource):
            # fetch unfinished value
            value_new = value_new.get_value(strict=False)

        dexp_result.set_value(attr_name=attr_name, changer_name=attr_name, value=value_new)

        return dexp_result


    def _get_value_new(self, apply_result: IApplyResult, value_prev: AttrValue, attr_name: AttrName) -> AttrValue:
        if isinstance(value_prev, IDexpValueSource):
            # try to find in children first, then in container_children if applicable
            if not hasattr(value_prev, "children"):
                raise EntityInternalError(owner=self, msg=f"Attribute '{attr_name}' can not be found, node '{value_prev.name}' has no children")
            dexp_value_node = value_prev.children.get(attr_name, UNDEFINED)
            if dexp_value_node is UNDEFINED and hasattr(value_prev, "container_children"):
                dexp_value_node = value_prev.container_children.get(attr_name, UNDEFINED)
            if dexp_value_node is UNDEFINED:
                raise EntityInternalError(owner=self, msg=f"Attribute '{attr_name}' can not be found in '{value_prev.name}'")
            value_new = dexp_value_node
        else:
            # removed. idx == 0 and
            if attr_name == ReservedAttributeNames.INSTANCE_ATTR_NAME:
                value_new = value_prev
            else:
                if (value_prev is UNDEFINED
                        or value_prev is None
                        or value_prev is NA_DEFAULTS_MODE
                ):
                    # 'Maybe monad' like
                    value_new = value_prev
                else:
                    # if self.namespace == ThisNS:
                    #     value_new = apply_result.current_frame.component._accessor.get_value(
                    #                     instance=value_prev, attr_name=attr_name, attr_index=None)
                    # else:
                    #     # TODO: this is not easy - how to detect to which component this attribute belongs,
                    #     #       if it belongs to any. So using default _accessor and hope for the best.
                    #     #       ListByIndex can not be used since, attr_index is not available.
                    #     value_new = apply_result.current_frame.component.entity._accessor.get_value(
                    #         instance=value_prev, attr_name=attr_name, attr_index=None)
                    # if value_new is UNDEFINED:
                    #     raise EntityApplyNameError(owner=self,
                    #            msg=f"Attribute '{attr_name}' not found in '{to_repr(value_prev)}': '{type(value_prev)}'")
                    if isinstance(attr_name, (Attribute, AttributeByMethod)):
                        attr_name = attr_name.name

                    if not isinstance(attr_name, str):
                        raise EntityInternalError(owner=self, msg=f"Attribute name must be a string, got. {attr_name}")

                    if not hasattr(value_prev, attr_name):
                        # TODO: list which fields are available
                        # if all types match - could be internal problem?
                        raise EntityApplyNameError(owner=self, msg=f"Attribute '{attr_name}' not found in '{to_repr(value_prev)}': '{type(value_prev)}'")
                    value_new = getattr(value_prev, attr_name)
                    if callable(value_new):
                        try:
                            # NOTE: must be method(self) or function() - no args expected
                            value_new2 = value_new()
                        except Exception as ex:
                            raise EntityApplyValueError(owner=self,
                                   msg=f"Attribute '{attr_name}' is a callable '{value_new}' which raised error by calling:: '{ex}'")
                        value_new = value_new2
        return value_new

    # NOTE: this logic is dropped - expected_type_info is sometimes missmatched and sometimes adaptations needs to be done ...
    #       all in all - currently too complex, anyway type of final value will be checked later.
    # def check_root_value_type_info(self, ...):
    #     if root_value and value_new not in ([], {}, None, UNDEFINED):
    #         adapted_value = apply_result.current_frame.component.try_adapt_value(value_new) \
    #             if isinstance(apply_result.current_frame.component, IField) \
    #             else value_new
    #         # will be reported later
    #         ignore_list_check = True # apply_result.current_frame.on_component_only is not None

    #         # None is set in dataclasss initialization - and then this test fails. Ignore this for noew
    #         assert root_value.attr_dexp_node
    #         expected_type_info = root_value.attr_dexp_node.type_info
    #         assert  expected_type_info
    #         value_type_info = TypeInfo.get_or_create_by_value(adapted_value)
    #         err_msg = expected_type_info.check_compatible(value_type_info, ignore_list_check=ignore_list_check)
    #         if err_msg:
    #             expected_type_info.check_compatible(value_type_info, ignore_list_check=ignore_list_check)
    #             raise EntityApplyValueError(owner=self, msg=f"For attribute '{attr_name}' type is not compatible:\n  {err_msg}\n  expecting value of type:"
    #                                                         f"\n  {expected_type_info}"
    #                                                         f"\n  got:\n  {value_type_info}"
    #                                                         f"\n  value: {adapted_value}")

    def isoptional(self):
        return self.type_info.is_optional if self.type_info else False

    def islist(self):
        return self.type_info.is_list if self.type_info else False

    def as_str(self) -> str:
        # pretty print
        denied = "DENIED" if self.denied else ""
        # bound= ("BOUND[{}]".format(", ".join([f"{setup_session_name}->{ns._name}.{attr_node_name}" for setup_session_name, ns, attr_node_name in self.bound_list]))) if self.bound_list else ""
        # used = f"USED={self.refcount}" if self.refcount else ""
        altname = f"{self.data_supplier_name}" if self.data_supplier_name != self.name else ""
        # function = f"Function({self.function.name}{', custom' if self.function.is_custom else ''})" if self.function else ""
        # , used, bound, function
        out = self.type_info.as_str() if self.type_info else "-"
        out += " " + (", ".join([val for val in [altname, denied] if val]))
        return out.strip()

    def __str__(self):
        denied = ", DENIED" if self.denied else ""
        return f"AttrDexpNode({self.full_name} : {self.data_supplier_name}{denied})"
        # bound= (", BOUND={}".format(", ".join([f"{setup_session_name}->{ns._name}.{attr_node_name}" for setup_session_name, ns, attr_node_name in self.bound_list]))) if self.bound_list else ""
        # {bound}, {self.refcount}

    def __repr__(self):
        return str(self)

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
class AttrDexpNodeWithValuePath(AttrDexpNode):
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
