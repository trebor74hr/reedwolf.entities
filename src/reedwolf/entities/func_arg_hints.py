"""
Special function arguments helpers - custom type hints or wrappers
"""
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Type, Any, Callable, Union, List

from .utils import (
    UNDEFINED,
)
from .meta import (
    IFuncArgHint,
    IInjectFuncArgHint,
    TypeInfo,
    AttrValue,
    ModelInstanceType, IDexpValueSource,
)
from .exceptions import (
    EntityTypeError,
    EntityInternalError,
)
from .expressions import (
    DotExpression,
    ISetupSession,
    IDotExpressionNode, execute_dexp_or_node,
)
from .func_args import (
    FuncArg,
    PrepArg,
)
from .base import (
    IApplyResult,
    IValueNode,
    ApplyStackFrame,
)

# can be various types, check: functions.py::execute_arg()

ValueOrDexp = Union[
                  DotExpression,
                  IDotExpressionNode,
                  IDexpValueSource,
                  List[IDexpValueSource],
                  # TODO: InjectFuncArgValueByCallable,
              ]


@dataclass
class DotexprExecuteOnItemFactoryFuncArgHint(IInjectFuncArgHint):
    """
    TODO: name is too long and unreadable.
    """
    inner_type: Optional[Type] = field(repr=True, default=Any)
    type: Type = field(init=False, default=DotExpression)

    def setup_check(self, setup_session: "ISetupSession", caller: Optional["IDotExpressionNode"], func_arg: "FuncArg"):
        exp_type_info = TypeInfo.get_or_create_by_type(self.type)
        err_msg = exp_type_info.check_compatible(func_arg.type_info)
        if err_msg:
            raise EntityTypeError(owner=self, msg=f"Function argument {func_arg} type not compatible: {err_msg}")


    def get_type(self) -> Type:
        return self.type

    def get_inner_type(self) -> Optional[Type]:
        return self.inner_type

    def __hash__(self):
        return hash((self.__class__.__name__, self.type, self.inner_type))

    def get_apply_inject_value(self, apply_result: "IApplyResult", prep_arg: "PrepArg"
                               ) -> Callable[[DotExpression, ModelInstanceType], AttrValue]:
        """
        create this registry function that retriieves settings processor
        which will crearte this factory for providded item
        """

        def execute_dot_expr_w_this_registry_of_item(
                dot_expr:  DotExpression,
                item: Union[ModelInstanceType, IValueNode],
        ) -> AttrValue:
            # TODO: if this becommes heavy - instead of new frame, reuse existing and change instance only
            #       this_registry should be the same for same session (same type items)
            if not isinstance(item, IValueNode):
                raise EntityInternalError(owner=self, msg=f"Expecting ValueNode, got: {item}")
            if item.component.is_subentity_items():
                this_registry = item.component.get_this_registry_for_item()
            else:
                this_registry = item.component.get_this_registry()

            value_node, instance = (item, item.instance) if isinstance(item, IValueNode) else (None, item)

            with apply_result.use_stack_frame(
                    ApplyStackFrame(
                        # container = apply_result.current_frame.container,
                        # component = apply_result.current_frame.component,
                        # instance=instance,
                        value_node=value_node,
                        this_registry = this_registry,
                    )):
                dexp_result = dot_expr._evaluator.execute_dexp(
                    apply_result=apply_result,
                )
            return dexp_result.value

        return execute_dot_expr_w_this_registry_of_item


class IExecuteFuncArgHint(IFuncArgHint):

    @abstractmethod
    def get_apply_value(self, apply_result: "IApplyResult",
                        exp_arg: PrepArg,
                        arg_value: ValueOrDexp,
                        prev_node_type_info: TypeInfo,
                        ) -> AttrValue:
        ...


@dataclass
class AttrnameFuncArgHint(IExecuteFuncArgHint):
    inner_type: Optional[Type] = field(repr=True, default=Any)
    type: Type = field(init=False, default=DotExpression)

    def get_type(self) -> Type:
        return self.type

    def get_inner_type(self) -> Optional[Type]:
        return self.inner_type

    def __hash__(self):
        return hash((self.__class__.__name__, self.type, self.inner_type))

    def get_apply_value(self, apply_result: "IApplyResult",
                        exp_arg: PrepArg,
                        arg_value: ValueOrDexp,
                        prev_node_type_info: TypeInfo,
                        ) -> AttrValue:
        # TODO: check that dot expression (arg_value) is in Models/Fields namespace
        #       do not evaluate, just use ._name or GetName()
        raise NotImplementedError()


@dataclass
class JustDotexprFuncArgHint(IFuncArgHint):
    """
    will not execute dot expression - will leave the job to function
    """
    inner_type: Optional[Type] = field(repr=True, default=Any)
    type: Type = field(init=False, default=DotExpression)

    def get_type(self) -> Type:
        return self.type

    def get_inner_type(self) -> Optional[Type]:
        return self.inner_type

    def __hash__(self):
        return hash((self.__class__.__name__, self.type, self.inner_type))


@dataclass
class DotexprFuncArgHint(IExecuteFuncArgHint):
    inner_type: Optional[Type] = field(repr=True, default=Any)
    type: Type = field(init=False, default=DotExpression)

    def get_type(self) -> Type:
        return self.type

    def get_inner_type(self) -> Optional[Type]:
        return self.inner_type

    def __hash__(self):
        return hash((self.__class__.__name__, self.type, self.inner_type))

    def get_apply_value(self, apply_result: "IApplyResult",
                        exp_arg: PrepArg,
                        arg_value: ValueOrDexp,
                        prev_node_type_info: TypeInfo,
                        ) -> AttrValue:
        if isinstance(arg_value, DotExpression):
            namespace = arg_value._namespace
        # elif isinstance(arg_value, IDotExpressionNode):
        #     raise NotImplementedError()  # namespace = arg_value.namespace
        else:
            raise EntityInternalError(owner=self, msg=f"Expected DotExpression / IDotExpressionNode: {arg_value}")

        dexp_result = execute_dexp_or_node(
            dexp_or_value=arg_value,
            dexp_node=arg_value,
            namespace=namespace,
            dexp_result=UNDEFINED,
            prev_node_type_info=prev_node_type_info,
            apply_result=apply_result)
        arg_value = dexp_result.value
        return arg_value


