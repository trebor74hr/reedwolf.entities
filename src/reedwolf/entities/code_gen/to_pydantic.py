import os
import inspect
from collections import OrderedDict
from contextlib import AbstractContextManager
from dataclasses import (
        dataclass, 
        field,
        )
from typing import (
        List, 
        Set, 
        Optional,
        Union,
        Dict,
        Any,
        )

from ..exceptions import (
        EntityInternalError,
        )
from ..utils import (
        snake_case_to_camel,
        get_available_names_example,
        )
from ..meta import (
        TypeInfo,
        STANDARD_TYPE_LIST,
        )
from ..base import (
        IStackOwnerSession,
        IStackFrame,
        UseStackFrameCtxManagerBase,
        ComponentBase,
        PY_INDENT,
        MAX_RECURSIONS,
        )
from ..fields import (
        FieldBase, 
        FieldGroup, 
        ChoiceField,
        BooleanField,
        EnumField,
        )
from ..containers import (
        SubEntityBase, 
        SubEntityItems, 
        SubEntitySingle, 
        Entity,
        )

from . import base

DelarationCodeLinesType = List[str]
FieldCodeLinesType = List[str]

THIS_PACKAGE = inspect.getmodule(base).__package__
THIS_MODULE = os.path.basename(__file__).split(".")[0]

# ------------------------------------------------------------

@dataclass
class CodegenStackFrame(IStackFrame):
    component: ComponentBase = field(repr=False)
    filename: str = field()
    path_names: List[str] = field()
    depth: Optional[int] = field(repr=False) # 0 based
    # autocomputed
    indent: int = field(repr=False, init=False)
    component_name: str = field(init=False)

    def __post_init__(self):
        assert isinstance(self.component, ComponentBase)
        self.component_name = self.component.name
        # make a copy and add component name
        self.path_names = self.path_names[:]
        self.path_names.append(self.component.name)
        assert len(self.path_names) == self.depth+1

        self.indent = PY_INDENT * self.depth

        if self.depth > MAX_RECURSIONS:
            raise EntityInternalError(owner=component, msg=f"Maximum recursion depth exceeded ({self.depth})")

# ------------------------------------------------------------

@dataclass
class ComponentPydanticDump:
    " dumped lines of code and declarations for a component "
    name:str
    filename: str
    lines: List[str]
    vars_declarations: List[str]

@dataclass 
class FilePydanticDump:
    " one filename - can have several ComponentPydanticDump "
    filename: str
    component_dump: Dict[str, ComponentPydanticDump] = field(init=False, default_factory=OrderedDict)

    # internal
    types_by_lib : Dict[str, Set[str]] = field(init=False, default_factory=dict)


    def dump_to_str(self) -> str:
        all_lines = []
        all_lines.extend([
            "from __future__ import annotations",
            "# --------------------------------------------------------------------------------",
            "# IMPORTANT: DO NOT EDIT!!! The code is generated by ",
           f"#            {THIS_PACKAGE}.{THIS_MODULE}",
            "#            Change 'Entity' object and regenerate the code .",
            "# --------------------------------------------------------------------------------",
            "from pydantic import BaseModel, Field",
            ])

        for module_name in sorted(self.types_by_lib.keys()):
            # "from datetime import date, datetime  # noqa: F401",
            # "from decimal import Decimal",
            all_lines.append(f"from {module_name} import (")
            for class_name in sorted(self.types_by_lib[module_name]):
                all_lines.append(f"    {class_name},")
            all_lines.append(")")

        for commponent_dump in self.component_dump.values():
            all_lines.extend(commponent_dump.lines)

        all_lines.append("")
        all_lines.append("")
        all_lines.append("# from typing import get_type_hints; print(get_type_hints(VendingCompanyDTO)); print('---- ALL OK -----')")
        all_lines.append("")

        out = "\n".join(all_lines)

        return out

# ------------------------------------------------------------

@dataclass
class DumpToPydantic(IStackOwnerSession):

    # all internal
    # component_dump: Dict[str, ComponentPydanticDump] = field(init=False, default_factory=OrderedDict)
    file_dump_dict: Dict[str, FilePydanticDump] = field(init=False, default_factory=OrderedDict)

    # obligatory for stack handling
    stack_frames: List[CodegenStackFrame] = field(repr=False, init=False, default_factory=list)
    # autocomputed
    current_frame: Optional[CodegenStackFrame] = field(repr=False, init=False, default=None)

    def use_stack_frame(self, frame: CodegenStackFrame) -> UseStackFrameCtxManagerBase:
        if not isinstance(frame, CodegenStackFrame):
            raise EntityInternalError(owner=self, msg=f"Expected CodegenStackFrame, got frame: {frame}") 
        return UseStackFrameCtxManagerBase(owner_session = self, frame=frame)


    def use_type(self, klass: type) -> Union[str, str]:
        " 2nd return param is lib_name when is not part of standard library "
        lib_name, _, type_name = str(klass).rpartition(".")

        lib_name_out = None

        if lib_name != "typing":
            type_name = klass.__name__
            module = inspect.getmodule(klass)
            lib_name = module.__name__
            if lib_name != "builtins":
                lib_name_out = lib_name

        if lib_name != "builtins":
            file_dump = self.get_or_create_file_dump()

            if lib_name not in file_dump.types_by_lib:
                file_dump.types_by_lib[lib_name] = set()
            file_dump.types_by_lib[lib_name].add(type_name)

        return type_name, lib_name_out


    def get_or_create_file_dump(self) -> FilePydanticDump:
        filename = self.current_frame.filename
        if filename not in self.file_dump_dict:
            self.file_dump_dict[filename] = FilePydanticDump(filename=filename)
        return self.file_dump_dict[filename]

    def get_file_dump(self, filename: str) -> FilePydanticDump:
        if filename not in self.file_dump_dict:
            raise EntityInternalError(owner=self, msg=f"File dump '{filename}' not found") 
        return self.file_dump_dict[filename]


    @staticmethod
    def get_component_dump_full_name(path_names: List[str]) -> str:
        # no need to have first parent in name
        path_name = ".".join([p for p in path_names])
        return path_name


    def set_component_dump(self, commponent_dump:ComponentPydanticDump):
        assert commponent_dump.name
        assert commponent_dump.vars_declarations

        file_dump = self.get_or_create_file_dump()

        path_names: List[str] = self.current_frame.path_names

        if not path_names[-1]==commponent_dump.name:
            raise EntityInternalError(owner=self, msg=f"not valid: {commponent_dump.name} vs {path_names}")

        full_name = self.get_component_dump_full_name(path_names)

        if full_name in file_dump.component_dump:
            raise EntityInternalError(owner=self, msg=f"Code under name '{full_name}' already set: {file_dump.component_dump[full_name]}")

        file_dump.component_dump[full_name] = commponent_dump


    def get_current_commponent_dump(self) -> ComponentPydanticDump:
        file_dump = self.get_file_dump(self.current_frame.filename)

        full_name = self.get_component_dump_full_name(self.current_frame.path_names)

        commponent_dump = file_dump.component_dump.get(full_name, None)
        if not commponent_dump:
            vars_avail = get_available_names_example(full_name, file_dump.component_dump.keys(), max_display=10)
            raise EntityInternalError(owner=self, msg=f"Code object for '{full_name}' not available, available: {vars_avail}")
        return commponent_dump



    @staticmethod
    def create_pydantic_var_declaration(name, type_hint_str, comment="", title=""):
        out = f"{name}: {type_hint_str}"
        if title:
            out = f'{out} = Field(title="{title}")'
        if comment:
            out = f"{out}  # {comment}"
        return out


    @staticmethod
    def create_pydantic_class_declaration(indent_level, name, title=""):
        out = []
        out.append(f"{PY_INDENT*indent_level}class {name}(BaseModel):")
        if title:
            out.append(f'{PY_INDENT*(indent_level+1)}""" {title} """')
            out.append("")
        return out

    # ------------------------------------------------------------

    def dump_field(self) -> Union[DelarationCodeLinesType, FieldCodeLinesType]:

        component = self.current_frame.component
        indent = self.current_frame.indent
        depth = self.current_frame.depth

        py_name = component.name
        todo_comment = ""
        children = component.get_children()

        lines = []
        vars_declarations = []

        if component.bound_attr_node:
            assert isinstance(component.bound_attr_node.data, TypeInfo)

            # py_type_klass = component.bound_attr_node.data.type_

            py_type_klass = component.type_info.type_
            assert py_type_klass

            if isinstance(component, EnumField):
                py_type_klass = component.enum
                py_type_name, _ = self.use_type(py_type_klass)
            elif isinstance(component, ChoiceField):
                py_type_klass = component.python_type
                py_type_name, _ = self.use_type(py_type_klass)

            # elif isinstance(component, ChoiceField) \
            #         and component.choice_title_attr_node is not None \
            #         and component.choice_title_attr_node.type_info is not None:
            #     self.dump_choice_field_w_custom_option_type(
            #             component=component,
            #             indent=indent,
            #             depth=depth,
            #             )
            elif py_type_klass in STANDARD_TYPE_LIST:
                py_type_name, _ = self.use_type(py_type_klass)
            else:
                py_type_name, _ = self.use_type(Any)
                todo_comment=f"TODO: domain_dataclass {py_type_name}"

            # type hint options
            if component.bound_attr_node.data.is_list:
                self.use_type(List)
                py_type_name = f"List[{py_type_name}]"
            if component.bound_attr_node.data.is_optional:
                self.use_type(Optional)
                py_type_name = f"Optional[{py_type_name}]"

            # NOTE: currently not implemented for:
            #   default, required, readonly, max-length, etc.

        else:
            # todo_comment = f"  # TODO: unbound {component.bind}"
            todo_comment = f"TODO: unbound {component.bind}"
            py_type_name, _ = self.use_type(Any)

        var_declaration = DumpToPydantic.create_pydantic_var_declaration(
                    py_name, 
                    py_type_name, 
                    todo_comment, 
                    title=component.title)
        vars_declarations.append(f'{indent}{var_declaration}')

        if children:
            lines.append("")
            # lines.append("")
            class_py_name = f"{component.name}_fieldgroup"
            class_py_type = f"{snake_case_to_camel(component.name)}DetailsDTO"
            # lines.append(f"{indent}class {class_py_type}(BaseModel):")
            title = f"Component beneath type {component.name}"
            lines.extend(DumpToPydantic.create_pydantic_class_declaration(
                            depth,
                            class_py_type,
                            title=title))

            # self.vars_declarations.append(f'{indent}{class_py_name}: {class_py_type}')
            vars_declarations.append(f'{indent}{DumpToPydantic.create_pydantic_var_declaration(class_py_name, class_py_type)}')

        return lines, vars_declarations

    # ------------------------------------------------------------

    def dump_class_with_children(self) -> Union[DelarationCodeLinesType, FieldCodeLinesType]:

        path_names = self.current_frame.path_names
        component  = self.current_frame.component 
        indent     = self.current_frame.indent    
        depth      = self.current_frame.depth     

        # if isinstance(component, BooleanField) and component.enables:
        #     import pdb;pdb.set_trace() 
        assert isinstance(component, (FieldGroup, Entity, SubEntityBase))
        lines = []
        vars_declarations = []
        py_type_name = f"{snake_case_to_camel(component.name)}DTO"

        # py_type_name_ext = snake_case_to_camel(".".join(map(snake_case_to_camel, path_names)))
        py_type_name_ext = snake_case_to_camel(py_type_name)

        # vars_declarations will be consumed in owner, so
        # only for top object it won't be consumed

        if isinstance(component, SubEntityBase):
            subitem_type_info = component.bound_model.type_info

            if isinstance(component, SubEntityItems):
                py_type_name_ext = f"List[{py_type_name_ext}]"

            if subitem_type_info.is_optional:
                py_type_name_ext = f"Optional[{py_type_name_ext}]"

        # self.vars_declarations.append(f'{indent}{component.name}: {py_type_name_ext}')
        vars_declarations.append(f'{indent}{DumpToPydantic.create_pydantic_var_declaration(component.name, py_type_name_ext, title=component.title)}')

        lines.append("")
        # lines.append("")
        # lines.append(f"{indent}class {py_name}(BaseModel):")
        lines.extend(DumpToPydantic.create_pydantic_class_declaration(
                        depth,
                        py_type_name,
                        title=component.title))
        return lines, vars_declarations

    # ------------------------------------------------------------

    def dump_all(self) -> Dict[str, str]:
        component   = self.current_frame.component
        path_names  = self.current_frame.path_names
        depth       = self.current_frame.depth
        indent      = self.current_frame.indent
        filename    = self.current_frame.filename
        if depth == 0:
            filename = f"{component.name}"

        lines : List[str] = []
        vars_declarations: List[str] = []

        indent_next = PY_INDENT * (depth+1)

        any_dump = False
        children = component.get_children()

        if isinstance(component, (FieldGroup, Entity, SubEntityBase)):
            lines_, vars_declarations_ = self.dump_class_with_children()
            lines.extend(lines_)
            vars_declarations.extend(vars_declarations_)
            any_dump = True

        if isinstance(component, (FieldBase,)):
            lines_, vars_declarations_ = self.dump_field()
            lines.extend(lines_)
            vars_declarations.extend(vars_declarations_)
            any_dump = True

        if not any_dump:
            raise EntityInternalError(owner=self, msg=f"No dump for: {component}")

        self.set_component_dump(ComponentPydanticDump(
                                    name=component.name,
                                    filename=filename,
                                    lines=lines, 
                                    vars_declarations=vars_declarations))

        for sub_component in children:
            with self.use_stack_frame(
                    CodegenStackFrame(
                        depth=depth+1,
                        path_names = path_names,
                        component = sub_component,
                        filename = filename,
                        # indent = indent,
                        )):

                # recursion
                self.dump_all()

                commponent_dump = self.get_current_commponent_dump()
                # print(f"RT: {component.name} -> {sub_component.name}")
                if commponent_dump.vars_declarations:
                    # remove last (simple) cluss dump and dump it here
                    # in order to preserve structure:
                    #   attributes
                    #   custom_class definitions
                    lines.extend(commponent_dump.vars_declarations)


        if depth==0:
            lines_by_file = OrderedDict()
            for file_dump in self.file_dump_dict.values():
                lines_by_file[file_dump.filename] = file_dump.dump_to_str()
        else:
            lines_by_file = None

        return lines_by_file

    
# ------------------------------------------------------------

def dump_to_pydantic_models_as_dict(component:ComponentBase) -> Dict[str, str]:
    dumper = DumpToPydantic()
    with dumper.use_stack_frame(
            CodegenStackFrame(
                depth=0,
                path_names = [],
                component = component,
                filename = component.name,
                )):
        return dumper.dump_all()

# ------------------------------------------------------------

def dump_to_pydantic_models(component:ComponentBase, fname_or_dname:str) -> str:
    lines_by_file = dump_to_pydantic_models_as_dict(component=component)

    lines_by_file_out = OrderedDict()

    if len(lines_by_file)==1:
        # dump in a single file
        if not fname_or_dname.endswith(".py"):
            fname_or_dname += ".py"
        if os.path.exists(fname_or_dname) and not os.path.isfile(fname_or_dname):
            raise EntityCodegenError(fmsg=f"Found '{fname_or_dname}' and is not a file")

        code = list(lines_by_file.values())[0]
        fname_abs = os.path.abspath(fname_or_dname)
        with open(fname_abs, "w") as fout:
            fout.write(code)
        lines_by_file_out[fname_abs] = code
    else:
        # dump in a single file
        root = os.path.abspath(fname_or_dname)
        if not os.path.exists(root):
            os.makedirs(root)
        else:
            if not os.path.isdir(fname_or_dname):
                raise EntityCodegenError(fmsg=f"Found '{fname_or_dname}' and is not a folder")

        for fname, code in lines_by_file.items():
            if not fname.endswith(".py"):
                fname += ".py"
            fname_abs = os.path.join(root, fname)
            with open(fname_abs, "w") as fout:
                fout.write(code)
            lines_by_file_out[fname_abs] = code

    return lines_by_file_out


# ------------------------------------------------------------
# OBSOLETE
# ------------------------------------------------------------

# def dump_choice_field_w_custom_option_type(self, 
#         component: ChoiceField, 
#         indent: int,
#         depth: int,
#         ) -> Union[DelarationCodeLinesType, FieldCodeLinesType]:
#     lines = []
#     title_type_info = component.choice_title_attr_node.type_info

#     py_type_name, lib_name = self.use_type(component.python_type)

#     # parent_klass_full_name = title_type_info.parent_object.__name__
#     # parent_klass_name = parent_klass_full_name.split(".")[-1]
#     # # value_klass_name = py_type_klass.__name__
#     # value_klass_name = component.choice_value_type_info.type_.__name__
#     # should be string
#     # title_klass_name = title_type_info.type_

#     py_type_name = f"{snake_case_to_camel(parent_klass_name)}ChoiceDTO"
#     lines.append("")
#     # lines.append(f"{indent}class {py_type_name}(BaseModel):")
#     title = f"Choice type for {component.name}"
#     lines.extend(DumpToPydantic.create_pydantic_class_declaration(
#                     depth,
#                     py_type_name,
#                     title=title))
#     lines.append(f"{indent_next}value: {value_klass_name}")
#     lines.append(f"{indent_next}title: str")
#     # lines.append("")
#     return lines

