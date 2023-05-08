"""
Process:

    ValueExpression object:
        vexp = M.value.expressions.and.Functions().results 

    vexp.Setup converts to
        .value -> AttrVexpNode()
        .expressions -> AttrVexpNode()
        .Functions() -> FunctionVexpNode()
        .results -> AttrVexpNode()

        # there is OperationVexpNode() e.g. M.x + M.y

    what is saved inside of (last) value expression (and owner):
        vexp_evaluator = ValueExpressionEvaluator()
            [AttrVexpNode(), AttrVexpNode(), Function(), AttrVexpNode()]

    and later could be evaluated/executed with concrete data objects/structure:
        vexp_evaluator.execute(struct) 
        -> ExecResult()
"""
from __future__ import annotations

from dataclasses import (
        dataclass, 
        field,
        )
from typing import (
        List,
        )
from .utils import (
        UNDEFINED,
        )
from .exceptions import (
        RuleInternalError,
        RuleApplyError,
        )
from .expressions import (
        IValueExpressionNode,
        ExecResult,
        )

# ------------------------------------------------------------

@dataclass
class ValueExpressionEvaluator:

    attr_node_list : List[IValueExpressionNode] = field(repr=False, init=False, default_factory=list)
    finished: bool = field(repr=False, init=False, default= False)
    failed_reasons: List[str] = field(repr=False, init=False, default_factory=list)

    def __str__(self):
        ok = "OK" if self.is_all_ok() else f"FAILED={'; '.join(self.failed_reasons)}"
        return f"{self.__class__.__name__}({ok},nodes={'.'.join(str(dn) for dn in self.attr_node_list)})"
    __repr__ = __str__

    def add(self, node:IValueExpressionNode):
        if self.finished:
            raise RuleInternalError(owner=self, msg="Already finished.")
        if not isinstance(node, IValueExpressionNode):
            raise RuleInternalError(owner=self, msg=f"node not 'IValueExpressionNode', got: {node}")
        self.attr_node_list.append(node)

    def is_all_ok(self) -> bool:
        return (self.finished and bool(self.attr_node_list) and not bool(self.failed_reasons))

    def last_node(self) -> IValueExpressionNode:
        if not self.finished:
            raise RuleInternalError(owner=self, msg="Not finished.")
        return self.attr_node_list[-1]

    def failed(self, reason:str):
        if self.finished:
            raise RuleInternalError(owner=self, msg="Already finished.")
        self.failed_reasons.append(reason)

    def finish(self):
        if self.finished:
            raise RuleInternalError(owner=self, msg="Already finished.")
        if not self.attr_node_list:
            raise RuleInternalError(owner=self, msg=f"attr_node_list is empty, failed reasons: {self.failed_reasons}")

        # TODO: if not self.attr_node_list:
        # TODO:     raise RuleInternalError(owner=self, msg=f"Empty attr_node_list.")
        self.finished = True

    def execute_vexp(self, apply_session: "IApplySession") -> ExecResult: # noqa: F821
        if not self.finished:
            raise RuleInternalError(owner=self, msg="Not yet finished.")
        if self.failed_reasons:
            raise RuleApplyError(owner=self, msg=f"Failed in creation: {'; '.join(self.failed_reasons)}.")

        vexp_result = UNDEFINED
        if apply_session.current_frame.on_component_only:
            # M.address_set -> process only address_set 
            node = self.attr_node_list[-1]
            vexp_result = node.execute_node(
                                apply_session, 
                                vexp_result, 
                                is_last=True, 
                                prev_node_type_info=None)
        else:
            idx_last = len(self.attr_node_list)
            prev_node_type_info = None
            for idx, node in enumerate(self.attr_node_list, 1):
                vexp_result = node.execute_node(
                                    apply_session, 
                                    vexp_result, 
                                    is_last=(idx_last==idx),
                                    prev_node_type_info=prev_node_type_info,
                                    )
                prev_node_type_info = node.get_type_info()


        return vexp_result

