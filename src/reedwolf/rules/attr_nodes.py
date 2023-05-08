from typing import (
        Any,
        Union,
        Optional,
        )
from dataclasses import dataclass, field

# fields as dc_fields
from enum import Enum

from .utils import (
        UNDEFINED,
        to_repr,
        )
from .exceptions import (
        RuleSetupValueError,
        RuleInternalError,
        RuleApplyNameError,
        )
from .namespaces import (
        Namespace,
        ModelsNS,
        )
from .expressions import (
        ValueExpression,
        IValueExpressionNode,
        IAttributeAccessorBase,
        )
from .meta import (
        TypeInfo,
        is_model_class,
        is_function,
        ModelField,
        ModelType,
        )
from .base import (
        UndefinedType,
        ComponentBase,
        IData,
        IContainerBase,
        IFieldBase,
        IApplySession,
        ExecResult,
        )

# ------------------------------------------------------------


class AttrVexpNodeTypeEnum(str, Enum):
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


@dataclass
class AttrVexpNode(IValueExpressionNode):
    """
    Name comes from dot-chaining nodes - e.g.
        M.name
          company and name are AttrVexpNodes
    """
    name: str
    # TODO: data can be also - check each:
    #   - some dataproviding function
    # TODO: Union[TypeInfo, IFunctionVexpNode, ValueExpression, 'Component', type]  #
    #       noqa: F821
    data: Any 
    namespace: Namespace

    th_field: Optional[ModelField] = field(repr=False, default=UNDEFINED)
    type_info: Optional[TypeInfo] = field(default=None)
    # TODO: rename to: func_node or function_node
    # function: Optional[Union[IFunctionVexpNode]] = field(repr=False, default=None)

    # TODO: explain!
    denied: bool = False
    deny_reason: str = ""

    # ----- Later evaluated ------
    # could live without this attribute but ...
    attr_node_type: Union[AttrVexpNodeTypeEnum] = field(init=False)
    is_finished: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        self.full_name = f"{self.namespace._name}.{self.name}"

        if not isinstance(self.name, str) or self.name in (None, UNDEFINED):
            raise RuleInternalError(owner=self, msg=f"AttrVexpNode should have string name, got: {self.name}")

        # ---------------------------------------------
        # CASES: COMPONENTS
        # ---------------------------------------------
        if self.data is None:
            raise RuleInternalError(owner=self, msg=f"Data is not set for {self}")

        if isinstance(self.data, IContainerBase):
            self.attr_node_type = AttrVexpNodeTypeEnum.CONTAINER
            self.data_supplier_name = f"{self.data.name}"
        elif isinstance(self.data, IFieldBase):
            self.attr_node_type = AttrVexpNodeTypeEnum.FIELD
            self.data_supplier_name = f"{self.data.name}"
        # elif isinstance(self.data, ValidatorBase):
        #     self.attr_node_type = AttrVexpNodeTypeEnum.VALIDATOR
        #     self.data_supplier_name = f"{self.data.name}"
        # elif isinstance(self.data, EvaluatorBase):
        #     # self.attr_node_type = AttrVexpNodeTypeEnum.EVALUATOR
        #     self.data_supplier_name = f"{self.data.name}"
        elif isinstance(self.data, IData):
            self.attr_node_type = AttrVexpNodeTypeEnum.DATA
            self.data_supplier_name = f"{self.data.name}"
        elif isinstance(self.data, ComponentBase):
            self.attr_node_type = AttrVexpNodeTypeEnum.COMPONENT
            self.data_supplier_name = f"{self.data.name}"

        # ---------------------------------------------
        # CASE: Value expressions and FunctinoBase 
        # ---------------------------------------------
        elif isinstance(self.data, ValueExpression):
            if self.data._func_args:
                self.attr_node_type = AttrVexpNodeTypeEnum.VEXP_FUNC
            else:
                self.attr_node_type = AttrVexpNodeTypeEnum.VEXP
            self.data_supplier_name = f"{self.data!r}"


        # ---------------------------------------------
        # CASE: Concrete type
        # ---------------------------------------------
        elif is_model_class(self.data):
            # ModelType
            assert type(self.data) == type, self.data
            # ALT: inspect.isclass()
            self.attr_node_type = AttrVexpNodeTypeEnum.MODEL_CLASS
            self.data_supplier_name = f"{self.data.__name__}"

        # ---------------------------------------------
        # CASE: Class Attribute or Function return type
        # ---------------------------------------------
        elif isinstance(self.data, TypeInfo):
            # TODO: For Context - th_field is a method() for AttrNode, solve this more clever
            if self.th_field is UNDEFINED:
                raise RuleInternalError(owner=self, msg="TypeInfo case - expected th_field (ModelField or py_function).")

            self.attr_node_type = AttrVexpNodeTypeEnum.TH_FIELD
            # self.data_supplier_name = f"TH[{type_info.var_type.name}: {type_info.type_.__name__}]"
            self.data_supplier_name = f"TH[{self.data.type_.__name__}]"
        else:
            if is_function(self.data):
                # self.attr_node_type = AttrVexpNodeTypeEnum.FUNCTION
                # self.data_supplier_name = f"{self.data.__name__}"
                raise RuleSetupValueError(owner=self, msg=f"Node '.{self.name}' is a function. Maybe you forgot to wrap it with 'reedwolf.rules.Function()'?")
            raise RuleSetupValueError(owner=self, msg=f"AttrVexpNode {self.name} should be based on PYD/DC class, got: {self.data}")


        # NOTE: .type_info could be calculated later in finish() method



    def finish(self):
        " fill type_info, must be available for all nodes - with exceptions those with .denied don't have it "
        super().finish()

        if self.type_info is None:

            if self.attr_node_type == AttrVexpNodeTypeEnum.FIELD:

                type_info = self.data
                if not type_info.bound_attr_node:
                    raise RuleInternalError(owner=self, msg=f"AttrVexpNode {self.data} .bound_attr_node not set.")

                bound_type_info = type_info.bound_attr_node.get_type_info()
                if not bound_type_info:
                    raise RuleInternalError(owner=self, msg=f"AttrVexpNode data.bound_attr_node={self.data} -> {self.data.bound_attr_node} .type_info not set.")

                # transfer type_info from type_info.bound attr_node
                self.type_info = bound_type_info

            elif self.attr_node_type == AttrVexpNodeTypeEnum.DATA:
                self.type_info = self.data.type_info
            elif self.attr_node_type not in (
                    AttrVexpNodeTypeEnum.CONTAINER,
                    AttrVexpNodeTypeEnum.COMPONENT,
                    ):
                # all other require type_info
                raise RuleInternalError(owner=self, msg=f"For attr_node {self.attr_node_type} .type_info not set (type={type(self.data)}).")

            if not self.denied and not self.type_info:
                raise RuleInternalError(owner=self, msg=f"For attr_node {self.attr_node_type} .type_info could not not set (type={type(self.data)}).")


    def get_type_info(self) -> TypeInfo:
        return self.type_info


    def execute_node(self, 
                 apply_session: IApplySession, 
                 # previous - can be undefined too
                 vexp_result: Union[ExecResult, UndefinedType],
                 is_last: bool,
                 ) -> ExecResult:

        if is_last and not self.is_finished:
            raise RuleInternalError(owner=self, msg=f"Last vexp node is not finished.") 

        # TODO: not nicest way - string split
        #       for extension: [p._name for p in frame.container.bound_model.model.Path]
        names = self.name.split(".")
        attr_name = names[-1]

        if vexp_result is UNDEFINED:
            # initial / first value - get from registry / namespace, e.g. M
            vexp_result = ExecResult()
            frame = apply_session.current_frame

            if frame.container.is_extension() or frame.on_component_only:
                if not len(names)==1:
                    raise RuleInternalError(owner=self, msg=f"Attribute node - execution initial step for extension failed, expected single name members (e.g. M), got: {self.name}\n  == Compoonent: {frame.container}")
                # if not len(names)>1:
                #     raise RuleInternalError(owner=self, msg=f"Initial evaluation step for extension failed, expected multiple name members (e.g. M.address_set), got: {self.name}\n  == Compoonent: {frame.container}")
            else:
                if not len(names)==1:
                    raise RuleInternalError(owner=self, msg=f"Initial evaluation step for non-extension failed, expected single name member (e.g. M), got: {self.name}\n  == Compoonent: {frame.container}")

            registry = apply_session.registries.get_registry(self.namespace)
            value_previous = registry.get_root_value(apply_session=apply_session, attr_name=attr_name)

            # == M.name mode
            if apply_session.current_frame.on_component_only and registry.ROOT_VALUE_NEEDS_FETCH_BY_NAME:
                # TODO: not nice solution
                do_fetch_by_name = False
            else:
                do_fetch_by_name = True

            # == M.company.name mode
            # do_fetch_by_name = registry.ROOT_VALUE_NEEDS_FETCH_BY_NAME
        else:
            # 2+ value - based on previous result and evolved one step further
            assert len(names)>1
            value_previous = vexp_result.value
            do_fetch_by_name = True

        if do_fetch_by_name:
            if isinstance(value_previous, (list, tuple)):
                raise RuleApplyNameError(owner=self, msg=f"Fetching attribute '{attr_name}' could not be read from list/tuple: '{to_repr(value_previous)}' : '{type(value_previous)}'")

            if isinstance(value_previous, IAttributeAccessorBase):
                # NOTE: if this is last in chain - fetch final value
                value_new = value_previous.get_attribute(
                                apply_session=apply_session, 
                                attr_name=attr_name, 
                                is_last=is_last)
            else:
                if not hasattr(value_previous, attr_name):
                    # TODO: list which fields are available
                    # if all types match - could be internal problem?
                    raise RuleApplyNameError(owner=self, msg=f"Attribute '{attr_name}' not found in '{to_repr(value_previous)}' : '{type(value_previous)}'")
                value_new = getattr(value_previous, attr_name)
        else:
            value_new = value_previous

        # TODO: if isinstance(value_new, (list, tuple))):
        # TODO:     if not self.islist() 
        # TODO:         raise RuleApplyValueError(owner=self, msg=f"Attribute '{attr_name}' should not return list, got: '{to_repr(value_new)}' : '{type(value_new)}'")
        # TODO: else:
        # TODO:     if self.islist() 
        # TODO:         raise RuleApplyValueError(owner=self, msg=f"Attribute '{attr_name}' should return list, got: '{to_repr(value_new)}' : '{type(value_new)}'")

        # TODO: hm, changer_name is equal to attr_name, any problem / check / fix ... 
        vexp_result.set_value(attr_name=attr_name, changer_name=attr_name, value=value_new)

        return vexp_result


    def isoptional(self):
        return self.type_info.is_optional if self.type_info else False
        # if isinstance(self.data, TypeInfo):
        #     out = self.data.is_optional
        # # TODO: IData and others
        # #   from .components import IData
        # else:
        #     out = False
        # return out

    def islist(self):
        # TODO: currently IData needs to be Component, needs to setup owner properly. Move IData to this module!
        # from .components import IData
        # if isinstance(self.data, TypeInfo):
        #     out = self.data.is_list
        # if isinstance(self.data, IData):
        #     # TODO: IData to have TypeInfo too
        #     return isinstance(self.data.datatype, list)
        return self.type_info.is_list if self.type_info else False


    def pp(self):
        # pretty print
        denied = "DENIED" if self.denied else ""
        # bound= ("BOUND[{}]".format(", ".join([f"{registries_name}->{ns._name}.{attr_node_name}" for registries_name, ns, attr_node_name in self.bound_list]))) if self.bound_list else ""
        # used = f"USED={self.refcount}" if self.refcount else ""
        altname = f"{self.data_supplier_name}" if self.data_supplier_name != self.name else ""
        # function = f"Function({self.function.name}{', custom' if self.function.is_custom else ''})" if self.function else ""
        # , used, bound, function
        out = self.type_info.pp() if self.type_info else "-"
        out += " " + (", ".join([val for val in [altname, denied] if val]))
        return out.strip()

    def __str__(self):
        denied = ", DENIED" if self.denied else ""
        return f"AttrVexpNode({self.full_name} : {self.data_supplier_name}{denied})"
        # bound= (", BOUND={}".format(", ".join([f"{registries_name}->{ns._name}.{attr_node_name}" for registries_name, ns, attr_node_name in self.bound_list]))) if self.bound_list else ""
        # {bound}, {self.refcount}

    def __repr__(self):
        return str(self)

