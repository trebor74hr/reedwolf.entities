from typing import (
        Any,
        Union,
        Optional,
        )
from dataclasses import dataclass, field

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
        IAttributeAccessorBase,
        )
from .meta import (
        TypeInfo,
        is_model_class,
        is_function,
        ModelField,
        )
# TODO: remove this dependency
from .base import (
    UndefinedType,
    ComponentBase,
    IContainerBase,
    IFieldBase,
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
    name: str
    # TODO: data can be also - check each:
    #   - some dataproviding function
    # TODO: Union[TypeInfo, IFunctionDexpNode, DotExpression, 'Component', type]  #
    #       noqa: F821
    data: Any 
    namespace: Namespace

    th_field: Optional[ModelField] = field(repr=False, default=UNDEFINED)
    type_info: Optional[TypeInfo] = field(default=None)
    # TODO: rename to: func_node or function_node
    # function: Optional[Union[IFunctionDexpNode]] = field(repr=False, default=None)

    # TODO: explain!
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
        if self.data is None:
            raise EntityInternalError(owner=self, msg=f"Data is not set for {self}")

        if isinstance(self.data, IContainerBase):
            self.attr_node_type = AttrDexpNodeTypeEnum.CONTAINER
            self.data_supplier_name = f"{self.data.name}"
        elif isinstance(self.data, IFieldBase):
            self.attr_node_type = AttrDexpNodeTypeEnum.FIELD
            self.data_supplier_name = f"{self.data.name}"
        elif isinstance(self.data, ComponentBase):
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
        elif is_model_class(self.data):
            # ModelType
            assert type(self.data) == type, self.data
            # ALT: inspect.isclass()
            self.attr_node_type = AttrDexpNodeTypeEnum.MODEL_CLASS
            self.data_supplier_name = f"{self.data.__name__}"

        # ---------------------------------------------
        # CASE: Class Attribute or Function return type
        # ---------------------------------------------
        elif isinstance(self.data, TypeInfo):
            # TODO: For Context - th_field is a method() for AttrNode, solve this more clever
            if self.th_field is UNDEFINED:
                raise EntityInternalError(owner=self, msg="TypeInfo case - expected th_field (ModelField or py_function).")

            self.attr_node_type = AttrDexpNodeTypeEnum.TH_FIELD
            self.data_supplier_name = f"TH[{self.data.type_.__name__}]"
        else:
            if is_function(self.data):
                raise EntitySetupValueError(owner=self, msg=f"Node '.{self.name}' is a function. Maybe you forgot to wrap it with 'reedwolf.entities.Function()'?")
            raise EntitySetupValueError(owner=self, msg=f"AttrDexpNode {self.name} should be based on PYD/DC class, got: {self.data}")

        # NOTE: .type_info could be calculated later in finish() method



    def finish(self):
        """ fill type_info, must be available for all nodes - with exceptions those with .denied don't have it """
        super().finish()

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
            elif self.attr_node_type not in (
                    AttrDexpNodeTypeEnum.CONTAINER,
                    AttrDexpNodeTypeEnum.COMPONENT,
                    ):
                # all other require type_info
                raise EntityInternalError(owner=self, msg=f"For attr_node {self.attr_node_type} .type_info not set (type={type(self.data)}).")

            if not self.denied and not self.type_info:
                raise EntityInternalError(owner=self, msg=f"For attr_node {self.attr_node_type} .type_info could not not set (type={type(self.data)}).")


    def get_type_info(self) -> TypeInfo:
        return self.type_info


    def execute_node(self,
                     apply_session: IApplyResult,
                     # previous - can be undefined too
                     dexp_result: Union[ExecResult, UndefinedType],
                     prev_node_type_info: Optional[TypeInfo],
                     is_last: bool,
                     ) -> ExecResult:

        if is_last and not self.is_finished:
            raise EntityInternalError(owner=self, msg="Last dexp node is not finished.") 

        # TODO: not nicest way - string split
        #       for subentity_items: [p._name for p in frame.container.bound_model.model.Path]
        names = self.name.split(".")

        attr_name = names[-1]
        # attr_name_orig = attr_name

        assert dexp_result not in (None,)

        if dexp_result in (UNDEFINED, None):
            # ==== Initial / first value - get from registry / namespace, e.g. M
            dexp_result = ExecResult()
            frame = apply_session.current_frame

            if frame.container.is_subentity() or frame.on_component_only:
                if not len(names)==1:
                    raise EntityInternalError(owner=self, msg=f"Attribute node - execution initial step for SubEntityItems/SubEntitySingle failed, expected single name members (e.g. M), got: {self.name}\n  == Compoonent: {frame.container}")
                # if not len(names)>1:
                #     raise EntityInternalError(owner=self, msg=f"Initial evaluation step for subentity_items failed, expected multiple name members (e.g. M.address_set), got: {self.name}\n  == Compoonent: {frame.container}")
            else:
                if not len(names)==1:
                    raise EntityInternalError(owner=self, msg=f"Initial evaluation step for non-subentity_items failed, expected single name member (e.g. M), got: {self.name}\n  == Compoonent: {frame.container}")

            registry = None
            if apply_session.current_frame.local_setup_session:
                # take from apply_session's current local_setup_session
                registry = apply_session.current_frame.local_setup_session.get_registry(self.namespace, strict=False)

            if not registry:
                # take from setup_session
                registry = apply_session.setup_session.get_registry(self.namespace)

            value_previous, attr_name_new = registry.get_root_value(apply_session=apply_session, attr_name=attr_name)
            if attr_name_new:
                # e.g. case ReservedAttributeNames.VALUE_ATTR_NAME, i.e. .Value
                attr_name = attr_name_new

            # == M.name mode
            if apply_session.current_frame.on_component_only and registry.ROOT_VALUE_NEEDS_FETCH_BY_NAME:
                # TODO: not nice solution
                do_fetch_by_name = False
            else:
                do_fetch_by_name = True

            # == M.company.name mode
            # do_fetch_by_name = registry.ROOT_VALUE_NEEDS_FETCH_BY_NAME

        else:
            # ==== 2+ value - based on previous result and evolved one step further

            if not len(names)>1:
                raise EntityInternalError(owner=self, msg=f"Names need to be list of at least 2 members: {names}") 
            value_previous = dexp_result.value
            do_fetch_by_name = True


        if do_fetch_by_name:
            # convert previous value to list, process all and convert back to
            # single object when previous_value is not a list
            result_is_list = isinstance(value_previous, (list, tuple))

            if not result_is_list:
                # TODO: handle None, UNDEFINED?
                if prev_node_type_info and prev_node_type_info.is_list:
                    raise EntityApplyNameError(owner=self, msg=f"Fetching attribute '{attr_name}' expected list and got: '{to_repr(value_previous)}' : '{type(value_previous)}'")
                value_prev_as_list = [value_previous]
            else:
                if prev_node_type_info and not prev_node_type_info.is_list:
                    raise EntityApplyNameError(owner=self, msg=f"Fetching attribute '{attr_name}' got list what is not expected, got: '{to_repr(value_previous)}' : '{type(value_previous)}'")
                value_prev_as_list = value_previous

            # ------------------------------------------------------------
            value_new_as_list = []

            for idx, value_prev in enumerate(value_prev_as_list, 0):
                if isinstance(value_prev, IAttributeAccessorBase):
                    # NOTE: if this is last in chain - fetch final value
                    value_new = value_prev.get_attribute(
                                    apply_session=apply_session, 
                                    attr_name=attr_name, 
                                    is_last=is_last)
                else:
                    if idx==0 and attr_name==ReservedAttributeNames.INSTANCE_ATTR_NAME:
                        value_new = value_prev
                    else:
                        if (value_prev is UNDEFINED 
                          or value_prev is None
                          or value_prev is NA_DEFAULTS_MODE
                          ):
                            # 'Maybe monad' like
                            value_new = value_prev
                        else:
                            if not hasattr(value_prev, attr_name):
                                # TODO: list which fields are available
                                # if all types match - could be internal problem?
                                raise EntityApplyNameError(owner=self, msg=f"Attribute '{attr_name}' not found in '{to_repr(value_prev)}' : '{type(value_prev)}'")
                            value_new = getattr(value_prev, attr_name)

                value_new_as_list.append(value_new)

            value_new = value_new_as_list[0] if not result_is_list else value_new_as_list
        else:
            value_new = value_previous

        # TODO: check type_info match too - and put in all types of nodes - functions/operators
        if apply_session.component_name_only and apply_session.instance_new == value_new:
            # TODO: this is workaround when single instance is passed to update single item in SubEntityItems[List]
            #       not good solution
            ...
        elif isinstance(value_new, (list, tuple)):
            if not self.islist():
                raise EntityApplyValueError(owner=self, msg=f"Attribute '{attr_name}' should not be a list, got: '{to_repr(value_new)}' : '{type(value_new)}'")
        elif value_new is None:
            pass
            # NOTE: value not checked - can be evaluated to something not-None later
            # if not self.get_type_info().is_optional:
            #     raise EntityApplyValueError(owner=self, msg=f"Attribute '{attr_name}' has 'None' value and type is not 'Optional'.")
        elif self.islist():
            # apply_session.entity.get_component(apply_session.component_name_only)
            raise EntityApplyValueError(owner=self, msg=f"Attribute '{attr_name}' should be a list, got: '{to_repr(value_new)}' : '{type(value_new)}'")

        # TODO: hm, changer_name is equal to attr_name, any problem / check / fix ... 
        dexp_result.set_value(attr_name=attr_name, changer_name=attr_name, value=value_new)

        return dexp_result


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

