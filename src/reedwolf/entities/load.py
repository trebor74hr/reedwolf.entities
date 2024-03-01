# https://docs.python.org/3.7/library/ast.html
import ast
from ast import iter_fields
from dataclasses import dataclass, field
from typing import (
    List,
    Any,
    Union,
)
from contextlib import contextmanager
from copy import deepcopy

from .containers import COMPONENTS_REGISTRY

from .utils import (
    to_str,
    to_repr,
    get_available_names_example,
    DumpFormatEnum,
    load_from_format,
)
from .exceptions import (
    EntityLoadError,
    EntityLoadTypeError,
    EntityLoadNameError,
    EntityInternalError,
)
from .namespaces import ALL_NS_OBJECTS
from .meta_dataclass import MAX_RECURSIONS
from .meta import (
    STANDARD_TYPE_W_NONE_LIST,
    LiteralType,
)
from .expressions import (
    DotExpression,
    AST_NODE_TYPE_TO_FUNCTION,
    Just,
)
from .base import (
    IComponent,
    DEXP_PREFIX,
)


ConstantAstTypes = Union[ast.Constant, ast.Num, ast.Str, ast.Bytes]
CONSTANT_AST_TYPE_LIST = [ast.Constant, ast.Num, ast.Str, ast.Bytes, ast.NameConstant]


class CallTraceMixin:

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


def ast_node_repr(ast_node: ast.AST, depth: int = 0) -> str:
    if depth > MAX_RECURSIONS:
        raise EntityInternalError(owner=ast_node, msg=f"Maximum recursion depth exceeded ({depth})")

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
                fval = f"({fval})"
            fval = str(fval)[:50]
            out.append(f"{fname}={fval}")
        if not out:
            out.append(ast_node.__class__.__name__)
        out = to_str(", ".join(out))
    return out


@dataclass
class DotExpressionLoader(CallTraceMixin):
    # ast_node_list: List[Any] = field(init=False)
    call_trace: List[str] = field(init=False, default_factory=list)

    # ------------------------------------------------------------

    def load_dot_expression(self, expr_code: str) -> DotExpression:
        with self.use_call_trace(f"load_dot_expr({ast_node_repr(expr_code)})") as call_repr:
            try:
                # ATL: ast.comppile() ?
                module_node = ast.parse(expr_code)
            except Exception as ex:
                raise EntityLoadTypeError(owner=call_repr, msg=f"Parsing failed: {ex}")

            if type(module_node) != ast.Module:
                raise EntityLoadTypeError(owner=call_repr, msg=f"Expected ast.Module type, got: {type(module_node)}")

            if len(module_node.body) != 1:
                raise EntityLoadTypeError(owner=call_repr,
                                          msg=f"Start ast.Module node should have only single body (expression) , "
                                              f"got: {module_node.body}")

            ast_node = module_node.body[0]

            dexp_node = self._parse_expression_node(ast_node, depth=0) 

            return dexp_node

    # ------------------------------------------------------------

    def _parse_expression_node(self, ast_node: ast.Expr, depth: int) -> DotExpression:
        if depth > MAX_RECURSIONS:
            raise EntityInternalError(owner=ast_node, msg=f"Maximum recursion depth exceeded ({depth})")

        with self.use_call_trace(f"parse({ast_node_repr(ast_node)})") as call_repr:

            processed = False
            if type(ast_node) == ast.UnaryOp:
                # recursion
                dexp_node = self._process_ast_unaryop(node=ast_node, call_repr=call_repr, depth=depth+1)
                processed = True
            elif type(ast_node) in (ast.BinOp, ast.Compare):
                # recursion
                dexp_node = self._process_ast_binop(node=ast_node, call_repr=call_repr, depth=depth+1)
                processed = True
            elif type(ast_node) == ast.Call:
                # skip case: M.name.Length() ?
                if not (type(ast_node.func) == ast.Attribute and ast_node.func.value):
                    dexp_node = self._process_ast_start_node_call(ast_node, call_repr=call_repr)
                    processed = True

            if not processed:
                # Multiple items cases, M.id.test.Call, Ctx.Func(test=1).test.test, ...
                ast_node_list: List[ast.AST] = self._ast_nodes_prepare(ast_node)
                if not ast_node_list:
                    return None

                start_node = ast_node_list[0]

                if type(start_node) in CONSTANT_AST_TYPE_LIST:
                    if len(ast_node_list) != 1:
                        # TODO: if this happens, then do not call _ast_nodes_prepare() due reverse ...
                        raise NotImplementedError()
                    dexp_node = self._process_constant(node=start_node, call_repr=call_repr)

                elif type(start_node) == ast.UnaryOp:
                    if len(ast_node_list) != 1:
                        # TODO: if this happens, then do not call _ast_nodes_prepare() due reverse ...
                        raise NotImplementedError()
                    # recursion
                    dexp_node = self._process_ast_unaryop(node=start_node, call_repr=call_repr, depth=depth+1)

                elif type(start_node) in (ast.BinOp, ast.Compare):
                    if len(ast_node_list) != 1:
                        # TODO: if this happens, then do not call _ast_nodes_prepare() due reverse ...
                        raise NotImplementedError()
                    # recursion
                    dexp_node = self._process_ast_binop(node=start_node, call_repr=call_repr, depth=depth+1)

                elif type(start_node) == ast.Name:
                    # --- check start node - must be namespace
                    if len(ast_node_list) >= 2 and type(start_node) == ast.Name and type(ast_node_list[1]) == ast.Call:
                        # Just("Peter").Lower() ...
                        ast_node = ast_node_list[1]
                        dexp_node = self._process_ast_start_node_call(ast_node, call_repr=call_repr)
                        ast_node_list = ast_node_list[2:]
                    else:
                        # Namespace.attrib/Func() case
                        if len(ast_node_list) <= 1:
                            raise EntityLoadTypeError(owner=call_repr, msg=f"Expected at minimal <Namespace>.<attribute/function>, got: {ast_node_repr(ast_node_list)}")
                        ns_name = start_node.id
                        if ns_name not in ALL_NS_OBJECTS:
                            ops_avail = get_available_names_example(ns_name, ALL_NS_OBJECTS.keys())
                            raise EntityLoadNameError(owner=call_repr, msg=f"Starting node should be namespace, found: {ns_name}, available: {ops_avail}")

                        # iterate all others
                        namespace = ALL_NS_OBJECTS[ns_name]
                        dexp_node = namespace
                        ast_node_list = ast_node_list[1:]

                    # iterate further
                    for nr, node in enumerate(ast_node_list):
                        if type(node)==ast.Attribute:
                            # indirect recursion
                            dexp_node = self._process_ast_attribute(dexp_node=dexp_node, node=node, call_repr=call_repr, depth=depth+1)
                        elif type(node)==ast.Call:
                            # indirect recursion
                            dexp_node = self._process_ast_call(dexp_node=dexp_node, node=node, call_repr=call_repr, depth=depth+1)
                        else:
                            raise EntityLoadTypeError(owner=call_repr, msg=f"Expected expression, attribute or name, got: {type(node)} / {node}")
                else:
                    raise EntityInternalError(owner=self, msg=f"Expected Name/BinOp, got: {type(start_node)} / {start_node}") 

            return dexp_node

    # ------------------------------------------------------------

    def _ast_nodes_prepare(self, start_node: Union[ast.Expr, ast.Attribute]) -> List[ast.AST]:
        with self.use_call_trace(f"prepare({ast_node_repr(start_node)})") as call_repr:
            if type(start_node)==ast.Call and type(start_node.func)==ast.Attribute:
                node = start_node.func
            elif type(start_node)==ast.Expr:
                node = start_node.value
            elif type(start_node)==ast.Attribute:
                node = start_node
            elif type(start_node) in (ast.BinOp, ast.Compare):
                node = start_node
            elif type(start_node)==ast.UnaryOp:
                node = start_node
            elif type(start_node) in CONSTANT_AST_TYPE_LIST:
                node = start_node
            else:
                raise EntityLoadTypeError(owner=call_repr, msg=f"Expected expression/attribute for start node, got: {type(start_node)} / {ast_node_repr(start_node)}")

            ast_node_list = []
            nr = 0
            while True:
                nr += 1
                if nr > 200:
                    raise EntityInternalError(owner=call_repr, 
                            msg="Too many nodes to process...")  
                ast_node_list.append(node)
                if type(node)==ast.Attribute:
                    node = node.value
                elif type(node)==ast.Call:
                    node = node.func
                elif type(node) in CONSTANT_AST_TYPE_LIST:
                    node = self.get_constant_value(node, call_repr=call_repr)
                    if isinstance(node, STANDARD_TYPE_W_NONE_LIST):
                        break
                    # NOTE: do not terminate when Just("name").Lower()
                elif type(node)==ast.Name:
                    # terminate
                    break
                elif type(node) in (ast.BinOp, ast.UnaryOp, ast.Compare):
                    # terminate
                    break
                else:
                    raise EntityLoadTypeError(owner=call_repr, msg=f"Expected expression, attribute, constant, bioop or name, got: {type(node)} / {ast_node_repr(node)}")

            ast_node_list = list(reversed(ast_node_list))
            return ast_node_list

    # ------------------------------------------------------------

    def _process_ast_attribute(self, dexp_node: DotExpression, node: ast.Attribute, call_repr: str, depth: int) -> DotExpression:
        assert type(node) == ast.Attribute

        attr_name = node.attr
        try:
            dexp_node = getattr(dexp_node, attr_name)
        except Exception as ex:
            raise EntityLoadNameError(owner=call_repr, msg=f"Failed when creating DotExpression.attribute '{attr_name}' caused error: {ex}")
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
            dexp_node_previous = dexp_node._namespace
        try:
            method_node = getattr(dexp_node_previous, method_name)
        except Exception as ex:
            raise EntityLoadNameError(owner=call_repr, msg=f"Failed when creating DotExpression.Function '{method_name}' caused error: {ex}")

        if func_attr_node and not method_node.Equals(func_attr_node):
            raise EntityInternalError(owner=self, msg=f"Failed in creating function DotExpression node, duplication removal failed: {method_node} != {func_attr_node}") 

        try:
            dexp_node = method_node(*args, **kwargs)
        except Exception as ex:
            raise EntityLoadNameError(owner=call_repr, msg=f"Failed when creating DotExpression.Function '{method_name}' caused error: {ex}")

        return dexp_node

    # ------------------------------------------------------------

    def _process_ast_binop(self, node: ast.BinOp, call_repr: str, depth: int) -> DotExpression:
        if type(node) == ast.BinOp:
            left = node.left
            right = node.right
            node_op = node.op
        elif type(node) == ast.Compare:
            left = node.left
            assert len(node.comparators)==1
            right = node.comparators[0]
            assert len(node.ops)==1
            node_op = node.ops[0]
        else:
            raise EntityLoadTypeError(owner=call_repr, msg=f"Expected binop/compare, got: {node}")

        if type(node_op) not in AST_NODE_TYPE_TO_FUNCTION:
            names_avail = get_available_names_example(
                    str(type(node_op)), 
                    [str(nt) for nt in AST_NODE_TYPE_TO_FUNCTION.keys()]),
            raise EntityLoadTypeError(owner=call_repr, msg=f"Operation '{ast_node_repr(node_op)}' not supported. Available: {names_avail}")

        operation = AST_NODE_TYPE_TO_FUNCTION[type(node_op)]
        function = operation.load_function
        # recursion
        dexp_arg_left = self._parse_expression_node(ast_node=left, depth=depth+1)
        # recursion
        dexp_arg_right = self._parse_expression_node(ast_node=right, depth=depth+1)
        dexp_node = function(dexp_arg_left, dexp_arg_right)
        return dexp_node

    def _process_ast_unaryop(self, node: ast.UnaryOp, call_repr: str, depth: int) -> DotExpression:
        assert type(node)==ast.UnaryOp, node

        if type(node.op) not in AST_NODE_TYPE_TO_FUNCTION:
            names_avail = get_available_names_example(
                    str(type(node.op)), 
                    [str(nt) for nt in AST_NODE_TYPE_TO_FUNCTION.keys()]),
            raise EntityLoadTypeError(owner=call_repr, msg=f"Operation '{ast_node_repr(node.op)}' not supported. Available: {names_avail}")

        operation = AST_NODE_TYPE_TO_FUNCTION[type(node.op)]
        function = operation.load_function
        # recursion
        dexp_arg_operand = self._parse_expression_node(ast_node=node.operand, depth=depth+1)
        dexp_node = function(dexp_arg_operand)
        return dexp_node

    def _process_ast_start_node_call(self, ast_node: ast.AST, call_repr: str) -> DotExpression:
        if type(ast_node.func)==ast.Attribute:
            # func_name = ast_node.func.attr
            raise EntityLoadTypeError(owner=call_repr, msg=f"Function '{ast_node.func}' is a function with link to Attribute. This should have been processed before ...")

        func_name = ast_node.func.id
        if func_name == "Just":
            if not (len(ast_node.args) == 1 and len(ast_node.keywords)==0):
                raise EntityLoadTypeError(owner=call_repr, msg=f"Function 'Just' function can receive simple constant argument, got: {ast_node_repr(ast_node.args)} / {ast_node_repr(ast_node.keywords)}")
            dexp_node = self._process_constant(node = ast_node.args[0], call_repr=call_repr)
        else:
            raise EntityLoadTypeError(owner=call_repr, msg=f"Only 'Just' function can be a starting point, got: {func_name}")

        return dexp_node

    @staticmethod
    def get_constant_value(node: ConstantAstTypes, call_repr: str) -> LiteralType:
        if type(node)==ast.NameConstant:
            value = node.value # e.g. None
        elif type(node)==ast.Constant:
            value = node.value
        elif type(node)==ast.Num:
            value = node.n
        elif type(node) in (ast.Str, ast.Bytes):
            value = node.s
        else:
            raise EntityLoadTypeError(owner=call_repr, msg=f"Function 'Just' argument must be Constant/Num/Str, got: {ast_node_repr(node)}")

        if not isinstance(value, STANDARD_TYPE_W_NONE_LIST):
            raise EntityLoadTypeError(owner=call_repr, msg=f"Function 'Just' argument must be standard type: {STANDARD_TYPE_W_NONE_LIST}, got: {ast_node_repr(node)}")

        return value


    def _process_constant(self, node: ast.AST, call_repr: str) -> Just:
        value = self.get_constant_value(node, call_repr=call_repr)
        dexp_node = Just(value)
        return dexp_node


# ------------------------------------------------------------


@dataclass
class ComponentsLoader(CallTraceMixin):
    call_trace: List[str] = field(init=False, default_factory=list)

    def load_component(self, input_dict: dict) -> IComponent:
        out = self._load_component(input_dict, depth=0)
        return out

    def _load_component(self, input_dict: dict, depth: int) -> IComponent:
        if depth>MAX_RECURSIONS:
            raise EntityInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

        with self.use_call_trace(f"load({to_repr(input_dict)})") as call_repr:
            if not isinstance(input_dict, dict):
                raise EntityLoadTypeError(owner=call_repr, msg=f"Expecting dict, got: {to_repr(input_dict)}")
            input_dict = deepcopy(input_dict)

            comp_class_name = input_dict.pop("type", None)
            if not comp_class_name:
                raise EntityLoadTypeError(owner=call_repr, msg=f"Expecting dict key 'type', not found between: {to_repr(input_dict.keys())}")

            comp_klass = COMPONENTS_REGISTRY.get(comp_class_name, None)
            if comp_klass is None:
                klass_names_avail = get_available_names_example(comp_class_name, COMPONENTS_REGISTRY.keys())
                raise EntityLoadTypeError(owner=call_repr, msg=f"Class name '{comp_class_name}' is unknown. Available: {klass_names_avail}")

            comp_class_attrs = input_dict.pop("attrs", None)
            if not comp_class_attrs:
                raise EntityLoadTypeError(owner=call_repr, msg=f"Expecting dict key 'attrs', not found between: {to_repr(input_dict.keys())}")

            if input_dict.keys():
                raise EntityLoadTypeError(owner=call_repr, msg=f"Found unrecognized dict key(s): {to_repr(input_dict.keys())}")

            kwargs = {}
            for attr_name, attr_value in comp_class_attrs.items():
                value = self._load_process_attr_value(attr_value=attr_value, depth=depth)
                kwargs[attr_name] = value

            try:
                component = comp_klass(**kwargs)
            except TypeError as ex:
                # constructor error 
                raise EntityLoadError(owner=call_repr, msg=f"Failed to load '{comp_klass}', error: {ex}. kwargs={kwargs}")

        return component


    def _load_process_attr_value(self, attr_value: Any, depth: int) -> Any:
        if depth>MAX_RECURSIONS:
            raise EntityInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

        if isinstance(attr_value, list):
            attr_value_new = []
            for attr_value_item in attr_value:
                # recursion
                attr_value_item_new = self._load_process_attr_value(attr_value=attr_value_item, depth=depth+1)
                attr_value_new.append(attr_value_item_new)
            value = attr_value_new
        elif isinstance(attr_value, dict) and "type" in attr_value:
            # recursion
            value: IComponent = self._load_component(attr_value, depth=depth + 1)
        elif isinstance(attr_value, str) and attr_value.startswith(DEXP_PREFIX):
            code_string = attr_value[len(DEXP_PREFIX):]
            value: DotExpression = load_dot_expression(code_string)
        else:
            value: Any = attr_value
        return value

# ------------------------------------------------------------

def load_dot_expression(code_string: str) -> DotExpression:
    return DotExpressionLoader().load_dot_expression(code_string)


def load(input: Union[dict, str], format_: DumpFormatEnum = None) -> IComponent:
    if format_ is not None:
        assert isinstance(input , str)
        input_dict = load_from_format(input, format_=format_)
        assert isinstance(input, dict)
    else:
        assert isinstance(input, dict)
        input_dict = input

    return ComponentsLoader().load_component(input_dict)

