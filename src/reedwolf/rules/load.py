import ast
from dataclasses import dataclass, field
from typing import (
        List,
        Any,
        )

from .utils import (
        to_repr,
        get_available_names_example,
        )
from .exceptions import (
        RuleLoadTypeError,
        RuleLoadNameError,
        )
from .namespaces import ALL_NS_OBJECTS
from .expressions import DotExpression

def load(input: str) -> DotExpression:
    return None

@dataclass
class Builder:
    out_nodes: List[Any] = field(init=False)

    def load_dot_expression(self, expr_code: str) -> DotExpression:
        try:
            start_node = ast.parse(expr_code)
        except Exception as ex:
            raise RuleLoadTypeError(owner=self, msg=f"Parsing failed: {ex}")
        # ALL_NS_OBJECTS

        if type(start_node)!=ast.Module:
            raise RuleLoadTypeError(owner=self, msg=f"Expected ast.Module type, got: {type(start_node)}")

        assert len(start_node.body)==1, start_node.body
        node = start_node.body[0]

        if type(node)==ast.Expr:
            node = node.value
        else:
            raise RuleLoadTypeError(owner=self, msg=f"Expected expression, got: {type(node)} / {node}")

        out_nodes = []
        while node is not None:
            if type(node)==ast.Attribute:
                out_nodes.append(node.attr)
                node = node.value
            elif type(node)==ast.Name:
                out_nodes.append(node.id)
                node = None
            else:
                raise RuleLoadTypeError(owner=self, msg=f"Expected expression, attribute or name, got: {type(node)} / {node}")

        out_nodes = list(reversed(out_nodes))

        out_nodes_repr = ".".join(out_nodes)
        if len(out_nodes) <= 1:
            raise RuleLoadTypeError(owner=self, msg=f"Expected at minimal <Namespace>.<attribute/function>, got: {out_nodes_repr}")

        ns_name = out_nodes[0]
        if ns_name not in ALL_NS_OBJECTS:
            names_avail = get_available_names_example(ns_name, ALL_NS_OBJECTS.keys())
            raise RuleLoadNameError(owner=self, msg=f"Starting node should be namespace, found: {ns_name}, available: {names_avail}")

        namespace = ALL_NS_OBJECTS[ns_name]
        out_node = namespace

        for attr_name in out_nodes[1:]:
            # setattr(namespace, aname)
            try:
                out_node = getattr(out_node, attr_name)
            except Exception as ex:
                raise RuleLoadNameError(owner=self, msg=f"Failed when creating DotExpression, attribute '{attr_name}' caused error: {ex}")
        return out_node

def load_dot_expression(code_string: str) -> DotExpression:
    return Builder().load_dot_expression(code_string)

