"""
Process:

    DotExpression object:
        dexp = M.value.expressions.and.Functions().results 

    dexp.Setup converts to
        .value -> IAttrDexpNode()
        .expressions -> IAttrDexpNode()
        .Functions() -> FunctionDexpNode()
        .results -> IAttrDexpNode()

        # there is OperationDexpNode() e.g. M.x + M.y

    what is saved inside (last) value expression (and owner):
        dexp_evaluator = DotExpressionEvaluator()
            [IAttrDexpNode(), IAttrDexpNode(), Function(), IAttrDexpNode()]

    and later could be evaluated/executed with concrete data objects/structure:
        dexp_evaluator.execute(struct) 
        -> ExecResult()
"""
from dataclasses import (
        dataclass, 
        field,
        )
from typing import (
        List,
        )

from .namespaces import Namespace
from .utils import (
        UNDEFINED,
        )
from .exceptions import (
        EntityInternalError,
        EntityApplyError,
        )
from .expressions import (
        IDotExpressionNode,
        ExecResult,
        )

# ------------------------------------------------------------
@dataclass
class DotExpressionNodeList:
    namespace: Namespace
    nodes: List[IDotExpressionNode] = field(init=False, repr=True, default_factory=list)

    def add(self, node: IDotExpressionNode):
        self.nodes.append(node)


@dataclass
class DotExpressionEvaluator:
    namespace: Namespace = field(repr=False)
    attr_node_list: DotExpressionNodeList = field(repr=False, init=False)
    finished: bool = field(repr=False, init=False, default=False)
    failed_reasons: List[str] = field(repr=False, init=False, default_factory=list)

    def __post_init__(self):
        self.attr_node_list = DotExpressionNodeList(namespace=self.namespace)

    def __str__(self):
        ok = "OK" if self.is_all_ok() else f"FAILED={'; '.join(self.failed_reasons)}"
        return f"{self.__class__.__name__}({ok},nodes={'.'.join(str(dn) for dn in self.attr_node_list.nodes)})"
    __repr__ = __str__

    def add(self, node: IDotExpressionNode):
        if self.finished:
            raise EntityInternalError(owner=self, msg="Already finished.")
        if not isinstance(node, IDotExpressionNode):
            raise EntityInternalError(owner=self, msg=f"node not 'IDotExpressionNode', got: {node}")
        self.attr_node_list.add(node)

    def is_all_ok(self) -> bool:
        return self.finished and bool(self.attr_node_list.nodes) and not bool(self.failed_reasons)

    def last_node(self) -> IDotExpressionNode:
        if not self.finished:
            raise EntityInternalError(owner=self, msg="Not finished.")
        return self.attr_node_list.nodes[-1]

    def failed(self, reason: str):
        if self.finished:
            raise EntityInternalError(owner=self, msg="Already finished.")
        self.failed_reasons.append(reason)

    def finish(self):
        if self.finished:
            raise EntityInternalError(owner=self, msg="Already finished.")
        if not self.attr_node_list.nodes:
            raise EntityInternalError(owner=self, msg=f"attr_node_list is empty, failed reasons: {self.failed_reasons}")

        # TODO: if not self.attr_node_list.nodes:
        # TODO:     raise EntityInternalError(owner=self, msg=f"Empty attr_node_list.")
        self.finished = True

    def execute_dexp(self, apply_result: "IApplyResult") -> ExecResult:  # noqa: F821
        if not self.finished:
            raise EntityInternalError(owner=self, msg="Not yet finished.")
        if self.failed_reasons:
            raise EntityApplyError(owner=self, msg=f"Failed in creation: {'; '.join(self.failed_reasons)}.")

        dexp_result = UNDEFINED
        if apply_result.current_frame.on_component_only:
            # M.address_set -> process only address_set 
            node = self.attr_node_list.nodes[-1]
            dexp_result = node.execute_node(
                                apply_result, 
                                dexp_result,
                                namespace=self.attr_node_list.namespace,
                                is_1st_node=(len(self.attr_node_list.nodes) == 1),
                                is_last_node=True,
                                prev_node_type_info=None)
        else:
            idx_last = len(self.attr_node_list.nodes)
            prev_node_type_info = None
            for idx, node in enumerate(self.attr_node_list.nodes, 1):
                dexp_result = node.execute_node(
                                    apply_result, 
                                    dexp_result,
                                    namespace=self.attr_node_list.namespace,
                                    is_1st_node=(idx == 1),
                                    is_last_node=(idx == idx_last),
                                    prev_node_type_info=prev_node_type_info,
                                    )
                prev_node_type_info = node.get_type_info()

        return dexp_result
