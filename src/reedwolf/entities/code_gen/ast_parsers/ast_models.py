from __future__ import annotations

import ast
import os.path
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

from ...utils import MISSING
from ..utils import ensure_python_version, info_debug, right_strip, warn

def extract_code(node: ast.AST, code_lines: list[str], verbose: bool = False) -> str:
    # NOTE: 3.9 ast.object.end_*lineno - required
    # NOTE: 3.10 has ast.unparse - interesting
    ensure_python_version((3, 9))

    code: list[str] = []
    assert node.end_lineno
    lineno_end: int = node.end_lineno
    for lnr in range(node.lineno, lineno_end + 1):
        line = code_lines[lnr - 1]
        if lnr == node.end_lineno:
            line = line[: node.end_col_offset]
        if lnr == node.lineno:
            line = line[node.col_offset :]
        if verbose:
            line = "%03d: %s" % (lnr, line)
        code.append(line)
    return "\n".join(code)


def process_node(node: ast.AST, lines: list[str], depth: int = 0) -> list[Any]:  # noqa: C901
    """returns list of values/AST nodes/sourcecode: str, list and similar"""
    ensure_python_version((3, 9))

    if isinstance(node, ast.Constant):
        val = node.value
        if not isinstance(val, (str, int, float, bool, type(None))):
            warn(ast.dump(val))
            raise Exception("Unknown constant type: " + extract_code(val, lines, verbose=True))
        return [val]
    elif isinstance(node, ast.keyword):
        return [node.arg] + process_node(node.value, lines, depth + 1)
    elif isinstance(node, ast.Lambda):
        if isinstance(node.body, ast.Constant):
            # lambda: None, lambda: True, lambda: False, lambda: ''
            # fix: was not a list
            return [node.body.value]
        return [extract_code(node.body, lines, verbose=False)]
    elif isinstance(node, ast.Call):
        return (
            process_node(node.func, lines, depth + 1)
            + [process_node(arg, lines, depth + 1) for arg in node.args]
            + [process_node(kwarg, lines, depth + 1) for kwarg in node.keywords]
        )
    elif isinstance(node, ast.Name):
        if node.id == "_":
            return []  # ignore
        return [node.id]
    elif isinstance(node, ast.Attribute):
        return process_node(node.value, lines, depth + 1) + [node.attr]
    elif isinstance(node, (ast.Tuple, ast.List, ast.Starred, ast.Dict)):
        py_code_str: str = ast.unparse(node)
        # noinspection PyBroadException
        try:
            # NOTE: -------- EVAL() --------
            # TODO: explain better why eval() is required
            py_object = eval(py_code_str, {}, {})
            return [py_object]
        except Exception:
            return [py_code_str]

    warn(ast.dump(node))
    raise Exception(
        "Unknown node type: " + ast.dump(node) + " -> " + extract_code(node, lines, verbose=True)
    )


def ast_call_node_create_kw_arg_if_not_exists(
    ast_call_node: ast.Call, kw_name: str, kw_value: Union[ast.AST, str]
) -> bool:
    """returns if attribute was found and updated (True) or was not found and nothing was done (False)"""
    return _ast_call_node_add_or_update_kw_arg(
        ast_call_node=ast_call_node,
        kw_name=kw_name,
        kw_value=kw_value,
        can_create=True,
        can_update=False,
    )


def ast_call_node_add_or_update_kw_arg(
    ast_call_node: ast.Call, kw_name: str, kw_value: Union[ast.AST, str]
) -> bool:
    """returns if attribute was found and updated (True) or created and added (False)"""
    return _ast_call_node_add_or_update_kw_arg(
        ast_call_node=ast_call_node,
        kw_name=kw_name,
        kw_value=kw_value,
        can_create=True,
        can_update=True,
    )


def _ast_call_node_add_or_update_kw_arg(
    ast_call_node: ast.Call,
    kw_name: str,
    kw_value: Union[ast.AST, str],
    can_create: bool,
    can_update: bool,
) -> bool:
    """returns if attribute was found and updated (True) or created and added (False)"""
    kw_ast_node: ast.AST = (
        ast_single_expr_from_code(kw_value) if isinstance(kw_value, str) else kw_value
    )
    ast_keyword_list = [
        ast_fn_kw for ast_fn_kw in ast_call_node.keywords if ast_fn_kw.arg == kw_name
    ]
    if ast_keyword_list:
        assert len(ast_keyword_list) == 1
        if can_update:
            ast_keyword = ast_keyword_list[0]
            assert isinstance(kw_ast_node, ast.expr)
            kw_ast_expr: ast.expr = kw_ast_node
            ast_keyword.value = kw_ast_expr
        updated = True
    else:
        if can_create:
            ast_keyword = ast.keyword(kw_name, kw_ast_node)
            ast_call_node.keywords.append(ast_keyword)
        updated = False
    return updated


def ast_single_expr_from_code(code: str) -> ast.AST:
    # ALT: ast.compile() ?
    ast_module: ast.Module = ast.parse(code)
    assert len(ast_module.body) == 1, code
    assert isinstance(ast_module.body[0], ast.Expr), f"{code} -> {ast_module.body[0]}"
    ast_expr: ast.Expr = ast_module.body[0]
    ast_result: ast.AST = ast_expr.value
    return ast_result


def ast_call_node_update_positional_arg(
    ast_call_node: ast.Call, arg_number: int, arg_value: Union[ast.AST, str]
) -> None:
    ast_node: ast.AST = (
        ast_single_expr_from_code(arg_value) if isinstance(arg_value, str) else arg_value
    )
    assert arg_number >= 1
    assert len(ast_call_node.args) >= arg_number
    # old_ast_arg = ast_call_node.args[arg_number-1]
    assert isinstance(ast_node, ast.expr)
    ast_call_node.args[arg_number - 1] = ast_node


@dataclass
class FunctionFromCode:
    name: str
    args: list[Any]
    kwargs: list[tuple[str, Any]]
    kwargs_dict: dict[str, Any] = field(init=False, repr=False)
    name_path: list[str]
    node: ast.Call

    def __post_init__(self) -> None:
        self.kwargs_dict = dict(self.kwargs)

    def get_kwarg(self, name: str, default: Any = MISSING) -> Any:
        return self.kwargs_dict.get(name, default)

    @classmethod
    def from_ast_node(cls, func_node: ast.Call, lines: list[str]) -> FunctionFromCode:
        assert isinstance(func_node, ast.Call)
        # attr_node.value.func
        func_path = process_node(func_node.func, lines, True)

        func_name = ".".join(func_path)
        func_args = []
        for arg in func_node.args:
            val = process_node(arg, lines)
            if isinstance(val, list) and len(val) == 1:
                # ['CustomUser'] -> 'CustomUser'
                val = val[0]
            func_args.append(val)

        func_kwargs = []
        for kwarg in func_node.keywords:
            kw_val = process_node(kwarg.value, lines)
            if isinstance(kw_val, list) and len(kw_val) == 1:
                # HACK: [True] -> True
                kw_val = kw_val[0]
            assert kwarg.arg
            kwarg_name: str = kwarg.arg
            func_kwargs.append((kwarg_name, kw_val))

        func = FunctionFromCode(
            name=func_name, args=func_args, kwargs=func_kwargs, name_path=func_path, node=func_node
        )
        return func


@dataclass
class AnnotationFromCode:
    type: Any
    default: Optional[Any] = MISSING

    def has_default(self) -> bool:
        return self.default is not MISSING


@dataclass
class ModelAttrFromCode:
    name: str
    node: ast.AST
    ann: Optional[AnnotationFromCode] = None
    func: Optional[FunctionFromCode] = None

    def __str__(self) -> str:
        msg = [self.name]
        if self.ann and self.ann.type:
            msg.append(f" : {self.ann.type}")
        if self.func:
            msg.append(" = ")
            msg.append(f"{self.func.name}(")
            if self.func.args:
                msg.append(f"args={self.func.args},")
            if self.func.kwargs:
                msg.append(f"kws={self.func.kwargs}")
            msg.append(")")
        if self.ann and self.ann.default is not MISSING:
            msg.append(f" = {self.ann.default}")
        return "".join(msg)

    __repr__ = __str__


@dataclass
class PatchedLine:
    lines: list[str]
    lineno: int
    patched: bool = field(init=False, default=False)


@dataclass
class SourceCode:
    module_path_list: list[str] = field(repr=False)
    file_path: str
    lines: list[str] = field(repr=False)

    # --- COMPUTED
    line_count: int = field(init=False)
    # line of lines - in order to delete, update or add new lines on original positions
    # leaving original line numbers. First argument is Updated or not
    lines_patched: list[PatchedLine] = field(init=False, repr=False)
    patched_count: int = field(init=False, default=0)
    module_path_list_temp: list[str] = field(init=False, repr=False)
    file_path_temp: str = field(init=False, repr=False)

    TEMP_FILE_SUFFIX: ClassVar[str] = "_TO_DELETE.py"

    def __post_init__(self) -> None:
        for ident in self.module_path_list:
            assert ident.isidentifier(), self.module_path_list

        self.line_count = len(self.lines)
        assert self.lines
        self.lines_patched = []
        for lineno, line in enumerate(self.lines, 1):
            self.lines_patched.append(PatchedLine([line], lineno))
        assert len(self.lines_patched) == len(self.lines)

        # ---- set temporary file path and module path
        file_path_obj = Path(self.file_path)
        file_name_no_ext = right_strip(file_path_obj.name, ".py", strict=True)
        file_name_temp = file_name_no_ext + self.TEMP_FILE_SUFFIX
        self.file_path_temp = str(file_path_obj.parent / Path(file_name_temp))

        assert (
            self.module_path_list[-1] == file_name_no_ext
        ), f"{self.module_path_list[-1]} <> {file_name_no_ext}"
        self.module_path_list_temp = self.module_path_list[:-1] + [
            right_strip(file_name_temp, ".py", strict=True)
        ]
        for ident in self.module_path_list_temp:
            assert ident.isidentifier(), self.module_path_list_temp

    def get_lines(self, lineno_start: int, lineno_end: int) -> list[str]:
        assert lineno_start >= 1 and lineno_end <= len(self.lines_patched)
        out = []
        for lineno in range(lineno_start, lineno_end + 1):
            out.extend(self.lines_patched[lineno - 1].lines)
        return out

    def patch_lines(self, lineno_start: int, lineno_end: int, patched_lines: list[str]) -> None:
        assert lineno_start >= 1 and lineno_end <= len(self.lines_patched)
        for nr, lineno in enumerate(range(lineno_start, lineno_end + 1), 1):
            line_idx = lineno - 1
            patched_line = self.lines_patched[line_idx]
            if patched_line.patched:
                raise ValueError(f"{self}: Line {line_idx+1} already patched.")
            patched_line.patched = True
            # put content only in first line, other are left empty
            patched_line.lines = patched_lines if nr == 1 else []
        self.patched_count += 1

    def get_patched_content(self, debug: bool = False) -> str:
        new_lines = []
        for patched_line in self.lines_patched:
            lines = patched_line.lines
            if debug:
                lines = [
                    (f"{line}  # PATCHED" if patched_line.patched else line)
                    for idx, line in enumerate(lines)
                ]
            new_lines.extend(lines)
        return "\n".join(new_lines)

    def dump_changes_to_file(self, file_path: str, debug: bool = False, verbose: int = 1) -> None:
        if not self.patched_count:
            raise ValueError(f"{self.file_path}: Source could not be dumped, no change detected.")
        assert (
            file_path != self.file_path
        ), f"Internal issue, direct write to {file_path} is not allowed"
        file_path = os.path.abspath(file_path)
        content = self.get_patched_content(debug=debug)
        with open(file_path, "w") as file_out:
            file_out.write(content)
        if verbose >= 2:
            info_debug(f"Generated output {self.patched_count} changed attributes in {file_path}")
