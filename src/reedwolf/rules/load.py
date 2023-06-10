import ast
from ast import iter_fields, iter_child_nodes
from dataclasses import dataclass, field
from typing import (
        List,
        Any,
        Union,
        )
from contextlib import contextmanager

from .utils import (
        to_str,
        get_available_names_example,
        )
from .exceptions import (
        RuleLoadTypeError,
        RuleLoadNameError,
        RuleInternalError,
        )
from .namespaces import ALL_NS_OBJECTS
from .meta import (
        STANDARD_TYPE_W_NONE_LIST,
        )
from .expressions import (
        DotExpression,
        AST_NODE_TYPE_TO_FUNCTION,
        Just,
        )
from .base import (
        ComponentBase,
        MAX_RECURSIONS,
        )


def ast_node_repr(ast_node: ast.AST, depth: int = 0) -> str:
    if depth > MAX_RECURSIONS:
        raise RuleInternalError(owner=ast_node, msg=f"Maximum recursion depth exceeded ({depth})")

    if isinstance(ast_node, (int, str,)):
        out = ast_node
    elif isinstance(ast_node, (list, tuple)):
        out = []
        for ast_subitem in ast_node:
            subitem_repr = ast_node_repr(ast_subitem, depth=depth+1)
            out.append(subitem_repr)
        out = f"[{', '.join(out)}]"
    else:
        out = []
        for fname, fval in iter_fields(ast_node):
            if fname in ("ctx", ):
                continue
            if isinstance(fval, ast.AST):
                # recursion
                fval = ast_node_repr(fval, depth=depth+1)
                fvale = f"({fval})"
            fval = str(fval)[:50]
            out.append(f"{fname}={fval}")
        if not out:
            out.append(ast_node.__class__.__name__)
        out = to_str(", ".join(out))
    return out

@dataclass
class Builder:
    # ast_node_list: List[Any] = field(init=False)
    call_trace: List[str] = field(init=False, default_factory=list)

    @contextmanager
    def use_call_trace(self, call_trace_id: str):
        # Code to acquire resource, e.g.:
        self.call_trace.append(call_trace_id)
        call_repr = [f"{'  '*nr}-> {bit}" for nr, bit in enumerate(self.call_trace)]
        call_repr = "\n".join(call_repr) 
        call_repr += f"\n{'  ' * (len(self.call_trace))}"
        try:
            yield call_repr
        finally:
            self.call_trace.pop()

    # ------------------------------------------------------------

    def load_dot_expression(self, expr_code: str) -> DotExpression:
        with self.use_call_trace(f"load_dot_expr({ast_node_repr(expr_code)})") as call_repr:
            try:
                module_node = ast.parse(expr_code)
            except Exception as ex:
                raise RuleLoadTypeError(owner=call_repr, msg=f"Parsing failed: {ex}")

            if type(module_node)!=ast.Module:
                raise RuleLoadTypeError(owner=call_repr, msg=f"Expected ast.Module type, got: {type(module_node)}")

            if len(module_node.body)!=1:
                raise RuleLoadTypeError(owner=call_repr, msg=f"Start ast.Module node should have only single body (expression) , got: {module_node.body}")

            ast_node = module_node.body[0]

            dexp_node = self._parse_expression_node(ast_node, depth=0) 

            return dexp_node


    # ------------------------------------------------------------

    def _parse_expression_node(self, ast_node: ast.Expr, depth: int) -> DotExpression:
        if depth > MAX_RECURSIONS:
            raise RuleInternalError(owner=ast_node, msg=f"Maximum recursion depth exceeded ({depth})")

        with self.use_call_trace(f"parse({ast_node_repr(ast_node)})") as call_repr:

            if type(ast_node)==ast.BinOp:
                # recursion
                dexp_node = self._process_ast_binop(node=ast_node, call_repr=call_repr, depth=depth+1)
            elif type(ast_node)==ast.Call:
                depx_node = self._process_ast_start_node_call(ast_node, call_repr=call_repr)
            else:
                # Multiple items cases, M.id.test.Call, Fn(test=1).test.test, ...
                ast_node_list: List[ast.AST] = self._ast_nodes_prepare(ast_node)
                if not ast_node_list:
                    return None

                start_node = ast_node_list[0]

                if type(start_node)==ast.Constant:
                    if len(ast_node_list) != 1:
                        # TODO: if this happens, then do not call _ast_nodes_prepare() due reverse ...
                        raise NotImplementedError()
                    dexp_node = self._process_constant(node = start_node, call_repr=call_repr)
                elif type(start_node)==ast.BinOp:
                    if len(ast_node_list) != 1:
                        # TODO: if this happens, then do not call _ast_nodes_prepare() due reverse ...
                        raise NotImplementedError()
                    # recursion
                    dexp_node = self._process_ast_binop(node=start_node, call_repr=call_repr, depth=depth+1)
                elif type(start_node)==ast.Name:
                    # --- check start node - must be namespace
                    if len(ast_node_list)>=2 and type(start_node)==ast.Name and type(ast_node_list[1])==ast.Call:
                        # Just("Peter").Lower() ...
                        ast_node = ast_node_list[1]
                        dexp_node = self._process_ast_start_node_call(ast_node, call_repr=call_repr)
                        ast_node_list = ast_node_list[2:]
                    else:
                        # Namespace.attrib/Func() case
                        if len(ast_node_list) <= 1:
                            raise RuleLoadTypeError(owner=call_repr, msg=f"Expected at minimal <Namespace>.<attribute/function>, got: {ast_node_list}")
                        ns_name = start_node.id
                        if ns_name not in ALL_NS_OBJECTS:
                            ops_avail = get_available_names_example(ns_name, ALL_NS_OBJECTS.keys())
                            raise RuleLoadNameError(owner=call_repr, msg=f"Starting node should be namespace, found: {ns_name}, available: {ops_avail}")

                        # iterate all others
                        namespace = ALL_NS_OBJECTS[ns_name]
                        dexp_node = namespace
                        ast_node_list = ast_node_list[1:]

                    for nr, node in enumerate(ast_node_list):
                        if type(node)==ast.Attribute:
                            # indirect recursion
                            dexp_node = self._process_ast_attribute(dexp_node=dexp_node, node=node, call_repr=call_repr, depth=depth+1)
                        elif type(node)==ast.Call:
                            # indirect recursion
                            dexp_node = self._process_ast_call(dexp_node=dexp_node, node=node, call_repr=call_repr, depth=depth+1)
                        else:
                            raise RuleLoadTypeError(owner=call_repr, msg=f"Expected expression, attribute or name, got: {type(node)} / {node}")
                else:
                    raise RuleInternalError(owner=self, msg=f"Expected Name/BinOp, got: {type(start_node)} / {start_node}") 

            return dexp_node

    # ------------------------------------------------------------

    def _ast_nodes_prepare(self, start_node: Union[ast.Expr, ast.Attribute]) -> List[ast.AST]:
        with self.use_call_trace(f"prepare({ast_node_repr(start_node)})") as call_repr:
            if type(start_node)==ast.Expr:
                node = start_node.value
            elif type(start_node)==ast.Attribute:
                node = start_node
            elif type(start_node)==ast.BinOp:
                node = start_node
            elif type(start_node)==ast.Constant:
                node = start_node
            else:
                raise RuleLoadTypeError(owner=call_repr, msg=f"Expected expression/attribute for start node, got: {type(start_node)} / {ast_node_repr(start_node)}")

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
                elif type(node)==ast.Constant:
                    node = node.value
                    if isinstance(node, STANDARD_TYPE_W_NONE_LIST):
                        break
                    import pdb;pdb.set_trace() 
                    # NOTE: do not terminate when Just("name").Lower()
                elif type(node)==ast.Name:
                    # terminate
                    break
                elif type(node)==ast.BinOp:
                    # terminate
                    break
                else:
                    raise RuleLoadTypeError(owner=call_repr, msg=f"Expected expression, attribute, constant, bioop or name, got: {type(node)} / {ast_node_repr(node)}")

            ast_node_list = list(reversed(ast_node_list))
            return ast_node_list

    # ------------------------------------------------------------

    def _process_ast_attribute(self, dexp_node: DotExpression, node: ast.Attribute, call_repr: str, depth: int) -> DotExpression:
        assert type(node) == ast.Attribute

        attr_name = node.attr
        try:
            dexp_node = getattr(dexp_node, attr_name)
        except Exception as ex:
            raise RuleLoadNameError(owner=call_repr, msg=f"Failed when creating DotExpression.attribute '{attr_name}' caused error: {ex}")
        return dexp_node

    # ------------------------------------------------------------

    def _process_ast_call(self, dexp_node: DotExpression, node: ast.Call, call_repr: str, depth: int) -> DotExpression:
        assert type(node)==ast.Call, node

        method_name = node.func.attr

        args = []
        if node.args:
            for arg_node in node.args:
                # recursion
                dexp_arg = self._parse_expression_node(ast_node=arg_node, depth=depth+1)
                args.append(dexp_arg)

        kwargs = {}
        if node.keywords:
            for kwarg_node in node.keywords:
                # recursion
                dexp_arg = self._parse_expression_node(ast_node=kwarg_node.value, depth=depth+1)
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

        return dexp_node

    # ------------------------------------------------------------

    def _process_ast_binop(self, node: ast.BinOp, call_repr: str, depth: int) -> DotExpression:
        assert type(node)==ast.BinOp, node

        if type(node.op) not in AST_NODE_TYPE_TO_FUNCTION:
            names_avail = get_available_names_example(
                    str(type(node.op)), 
                    [str(nt) for nt in AST_NODE_TYPE_TO_FUNCTION.keys()]),
            raise RuleLoadTypeError(owner=call_repr, msg=f"Expected expression, attribute or name, got: {type(node)} / {ast_node_repr(node)}. Available: {names_avail}")

        function = AST_NODE_TYPE_TO_FUNCTION[type(node.op)]
        # recursion
        dexp_arg_left = self._parse_expression_node(ast_node=node.left, depth=depth+1)
        # recursion
        dexp_arg_right = self._parse_expression_node(ast_node=node.right, depth=depth+1)

        dexp_node = function(dexp_arg_left, dexp_arg_right)

        return dexp_node

    def _process_ast_start_node_call(self, ast_node: ast.AST, call_repr: str) -> DotExpression:
        if ast_node.func.id == "Just":
            if not (len(ast_node.args) == 1 and len(ast_node.keywords)==0):
                raise RuleLoadTypeError(owner=call_repr, msg=f"Function 'Just' function can receive simple constant argument, got: {ast_node_repr(ast_node.args)} / {ast_node_repr(ast_node.keywords)}")
            arg_node = ast_node.args[0]
            if type(arg_node) != ast.Constant:
                raise RuleLoadTypeError(owner=call_repr, msg=f"Function 'Just' argument must be Constant, got: {ast_node_repr(arg_node)}")
            dexp_node = self._process_constant(node = arg_node, call_repr=call_repr)
        else:
            raise RuleLoadTypeError(owner=call_repr, msg=f"Only 'Just' function can be a starting point, got: {ast_node.func.id}")

        return dexp_node


    def _process_constant(self, node: ast.Constant, call_repr: str) -> Just:
        value = node.value
        if not isinstance(value, STANDARD_TYPE_W_NONE_LIST):
            raise RuleLoadTypeError(owner=call_repr, msg=f"Function 'Just' argument must be standard type: {STANDARD_TYPE_W_NONE_LIST}, got: {ast_node_repr(node)}")
        dexp_node = Just(value)
        return dexp_node



# ------------------------------------------------------------

def load_dot_expression(code_string: str) -> DotExpression:
    return Builder().load_dot_expression(code_string)

def load(input: str) -> ComponentBase:
    raise NotImplementedError()
