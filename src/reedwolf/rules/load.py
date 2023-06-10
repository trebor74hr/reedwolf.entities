import ast
from dataclasses import dataclass, field
from typing import (
        List,
        Any,
        Union,
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
    # ast_node_list: List[Any] = field(init=False)
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

    # ------------------------------------------------------------

    def load_dot_expression(self, expr_code: str) -> DotExpression:
        with self.use_call_trace(f"load_dot_expr(to_repr({expr_code}))") as call_repr:
            try:
                module_node = ast.parse(expr_code)
            except Exception as ex:
                raise RuleLoadTypeError(owner=call_repr, msg=f"Parsing failed: {ex}")

            if type(module_node)!=ast.Module:
                raise RuleLoadTypeError(owner=call_repr, msg=f"Expected ast.Module type, got: {type(module_node)}")

            if len(module_node.body)!=1:
                raise RuleLoadTypeError(owner=call_repr, msg=f"Start ast.Module node should have only single body (expression) , got: {module_node.body}")

            expr_node = module_node.body[0]

            dexp_node = self._parse_expression_node(expr_node) 

            return dexp_node


    # ------------------------------------------------------------

    def _parse_expression_node(self, expr_node: ast.Expr) -> DotExpression:
        with self.use_call_trace(f"_pars_expr({to_repr(expr_node)})") as call_repr:

            ast_node_list: List[ast.AST] = self._ast_nodes_prepare(expr_node)

            # --- check start node - must be namespace
            if len(ast_node_list) <= 1:
                raise RuleLoadTypeError(owner=call_repr, msg=f"Expected at minimal <Namespace>.<attribute/function>, got: {ast_node_list}")

            node = ast_node_list[0]
            if not type(node)==ast.Name:
                raise RuleInternalError(owner=self, msg=f"Expected Name, got: {type(node)} / {node}") 
            ns_name = node.id
            if ns_name not in ALL_NS_OBJECTS:
                names_avail = get_available_names_example(ns_name, ALL_NS_OBJECTS.keys())
                raise RuleLoadNameError(owner=call_repr, msg=f"Starting node should be namespace, found: {ns_name}, available: {names_avail}")

            # iterate all others
            namespace = ALL_NS_OBJECTS[ns_name]
            dexp_node = namespace

            for node in ast_node_list[1:]:
                if type(node)==ast.Attribute:
                    attr_name = node.attr
                    try:
                        dexp_node = getattr(dexp_node, attr_name)
                    except Exception as ex:
                        raise RuleLoadNameError(owner=call_repr, msg=f"Failed when creating DotExpression.attribute '{attr_name}' caused error: {ex}")
                elif type(node)==ast.Call:
                    method_name = node.func.attr

                    args = []
                    if node.args:
                        for arg_node in node.args:
                            # recursion
                            dexp_arg = self._parse_expression_node(expr_node=arg_node)
                            args.append(dexp_arg)

                    kwargs = {}
                    if node.keywords:
                        for kwarg_node in node.keywords:
                            # recursion
                            dexp_arg = self._parse_expression_node(expr_node=kwarg_node.value)
                            kwargs[kwarg_node.arg] = dexp_arg


                    # NOTE: need to remove previous attr_node, ast works:
                    #       ast.Attribute + ast.Call, this ast.Attribute is
                    #       extra, but must be checked later 
                    if len(dexp_node.Path)>=2:
                        func_attr_node = dexp_node.Path[-1]
                        dexp_node_previous = dexp_node.Path[-2]
                    else:
                        func_attr_node = dexp_node.Path[-1]
                        dexp_node_previous = dexp_node.GetNamespace()
                    try:
                        method_node = getattr(dexp_node_previous, method_name)
                    except Exception as ex:
                        raise RuleLoadNameError(owner=call_repr, msg=f"Failed when creating DotExpression.Function '{method_name}' caused error: {ex}")

                    if func_attr_node and not method_node.Equals(func_attr_node):
                        raise RuleInternalError(owner=self, msg=f"Failed in creating function DotExpression node, duplication removal failed: {method_node} != {func_attr_node}") 

                    try:
                        dexp_node = method_node(*args, **kwargs)
                    except Exception as ex:
                        raise RuleLoadNameError(owner=call_repr, msg=f"Failed when creating DotExpression.Function '{method_name}' caused error: {ex}")

                else:
                    raise RuleLoadTypeError(owner=call_repr, msg=f"Expected expression, attribute or name, got: {type(node)} / {node}")

            return dexp_node

    # ------------------------------------------------------------

    def _ast_nodes_prepare(self, start_node: Union[ast.Expr, ast.Attribute]) -> List[ast.AST]:
        with self.use_call_trace(f"prepare({start_node})") as call_repr:
            if type(start_node)==ast.Expr:
                node = start_node.value
            elif type(start_node)==ast.Attribute:
                node = start_node
            else:
                raise RuleLoadTypeError(owner=call_repr, msg=f"Expected expression/attribute for start node, got: {type(start_node)} / {to_repr(start_node)}")

            ast_node_list = []
            nr = 0
            while True:
                nr += 1
                if nr > 200:
                    raise RuleInternalError(owner=call_repr, msg=f"Too many nodes to process...")  
                ast_node_list.append(node)
                if type(node)==ast.Attribute:
                    node = node.value
                elif type(node)==ast.Call:
                    node = node.func
                elif type(node)==ast.Name:
                    # terminate
                    break
                else:
                    raise RuleLoadTypeError(owner=call_repr, msg=f"Expected expression, attribute or name, got: {type(node)} / {to_repr(node)}")

            ast_node_list = list(reversed(ast_node_list))
            return ast_node_list

# ------------------------------------------------------------

def load_dot_expression(code_string: str) -> DotExpression:
    return Builder().load_dot_expression(code_string)

def load(input: str) -> ComponentBase:
    raise NotImplementedError()
