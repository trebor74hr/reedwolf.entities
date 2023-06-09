import ast
from dataclasses import dataclass, field
from typing import (
        List,
        Any,
        )
from contextlib import contextmanager

from .utils import (
        to_repr,
        get_available_names_example,
        )
from .exceptions import (
        RuleLoadTypeError,
        RuleLoadNameError,
        RuleInternalError,
        )
from .namespaces import ALL_NS_OBJECTS
from .expressions import DotExpression
from .base import ComponentBase



@dataclass
class Builder:
    # out_nodes: List[Any] = field(init=False)
    call_trace: List[str] = field(init=False, default_factory=list)

    @contextmanager
    def use_call_trace(self, call_trace_id: str):
        # Code to acquire resource, e.g.:
        self.call_trace.append(call_trace_id)
        call_repr = ".".join(self.call_trace)
        try:
            yield call_repr
        finally:
            self.call_trace.pop()

    def load_dot_expression(self, expr_code: str) -> DotExpression:
        with self.use_call_trace(f"load_dot_expr({expr_code})") as call_repr:
            try:
                start_node = ast.parse(expr_code)
            except Exception as ex:
                raise RuleLoadTypeError(owner=call_repr, msg=f"Parsing failed: {ex}")

            out_nodes = self._ast_nodes_prepare(start_node)

            out_node = self._parse_dot_expression_nodes(out_nodes)

            return out_node

    def _ast_nodes_prepare(self, start_node: ast.Module) -> List[ast.AST]:
        with self.use_call_trace(f"prepare({start_node})") as call_repr:
            if type(start_node)!=ast.Module:
                raise RuleLoadTypeError(owner=call_repr, msg=f"Expected ast.Module type, got: {type(start_node)}")

            assert len(start_node.body)==1, start_node.body
            node = start_node.body[0]

            if type(node)==ast.Expr:
                node = node.value
            else:
                raise RuleLoadTypeError(owner=call_repr, msg=f"Expected expression, got: {type(node)} / {node}")

            out_nodes = []
            nr = 0
            while True:
                nr += 1
                if nr > 200:
                    raise RuleInternalError(owner=call_repr, msg=f"Too many nodes to process...")  
                out_nodes.append(node)
                if type(node)==ast.Attribute:
                    node = node.value
                elif type(node)==ast.Call:
                    node = node.func
                elif type(node)==ast.Name:
                    # terminate
                    break
                else:
                    raise RuleLoadTypeError(owner=call_repr, msg=f"Expected expression, attribute or name, got: {type(node)} / {node}")

            out_nodes = list(reversed(out_nodes))
            return out_nodes


    def _parse_dot_expression_nodes(self, out_nodes: List[ast.AST]) -> DotExpression:
        with self.use_call_trace(f"nodes({out_nodes})") as call_repr:
            # --- check start node - must be namespace
            if len(out_nodes) <= 1:
                raise RuleLoadTypeError(owner=call_repr, msg=f"Expected at minimal <Namespace>.<attribute/function>, got: {out_nodes}")

            node = out_nodes[0]
            if not type(node)==ast.Name:
                raise RuleInternalError(owner=self, msg=f"Expected Name, got: {type(node)} / {node}") 
            ns_name = node.id
            if ns_name not in ALL_NS_OBJECTS:
                names_avail = get_available_names_example(ns_name, ALL_NS_OBJECTS.keys())
                raise RuleLoadNameError(owner=call_repr, msg=f"Starting node should be namespace, found: {ns_name}, available: {names_avail}")

            # iterate all others
            namespace = ALL_NS_OBJECTS[ns_name]
            out_node = namespace

            for node in out_nodes[1:]:
                if type(node)==ast.Attribute:
                    attr_name = node.attr
                    try:
                        out_node = getattr(out_node, attr_name)
                    except Exception as ex:
                        raise RuleLoadNameError(owner=call_repr, msg=f"Failed when creating DotExpression.attribute '{attr_name}' caused error: {ex}")
                elif type(node)==ast.Call:
                    method_name = node.func.attr
                    # TODO: recursion for node.args and node.kwargs
                    #       self._ast_nodes_prepare func.args, func.keywords
                    args, kwargs = (), {}

                    # NOTE: need to remove previous attr_node, ast works:
                    #       ast.Attribute + ast.Call, this ast.Attribute is
                    #       extra, but must be checked later 
                    if len(out_node.Path)>=2:
                        func_attr_node = out_node.Path[-1]
                        out_node_previous = out_node.Path[-2]
                    else:
                        func_attr_node = out_node.Path[-1]
                        out_node_previous = out_node.GetNamespace()
                    try:
                        method_node = getattr(out_node_previous, method_name)
                    except Exception as ex:
                        raise RuleLoadNameError(owner=call_repr, msg=f"Failed when creating DotExpression.Function '{method_name}' caused error: {ex}")

                    if func_attr_node and not method_node.Equals(func_attr_node):
                        raise RuleInternalError(owner=self, msg=f"Failed in creating function DotExpression node, duplication removal failed: {method_node} != {func_attr_node}") 

                    try:
                        out_node = method_node(*args, **kwargs)
                    except Exception as ex:
                        raise RuleLoadNameError(owner=call_repr, msg=f"Failed when creating DotExpression.Function '{method_name}' caused error: {ex}")

                else:
                    raise RuleLoadTypeError(owner=call_repr, msg=f"Expected expression, attribute or name, got: {type(node)} / {node}")

            return out_node

# ------------------------------------------------------------

def load_dot_expression(code_string: str) -> DotExpression:
    return Builder().load_dot_expression(code_string)

def load(input: str) -> ComponentBase:
    raise NotImplementedError()
