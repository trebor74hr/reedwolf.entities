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
        EntityCodegenError,
        EntityCodegenValueError,
        EntityCodegenNameError,
        EntityCodegenNameNotFoundError,
        EntityCodegenTypeError,
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

DelarationCodeLinesType = List[str]
FieldCodeLinesType = List[str]
# ------------------------------------------------------------

@dataclass
class CodegenStackFrame(IStackFrame):
    path_names: List[str] = field(repr=False)
    depth: Optional[int] = field(repr=False) # 0 based
    component: ComponentBase = field(repr=False)
    indent: int = field(repr=False)

    def __post_init__(self):
        assert isinstance(self.component, ComponentBase)
        assert len(self.path_names) == self.depth+1

# ------------------------------------------------------------

@dataclass
class KlassPydanticDump:
    " dumped lines of code and declarations for a component "
    name:str
    lines: List[str]
    vars_declarations: List[str]

# ------------------------------------------------------------

@dataclass
class DumpToPydantic(IStackOwnerSession):

    # all internal
    class_dumps: Dict[str, KlassPydanticDump] = field(init=False, default_factory=OrderedDict)
    types_by_lib : Dict[str, Set[str]] = field(init=False, default_factory=dict)
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
            if lib_name not in self.types_by_lib:
                self.types_by_lib[lib_name] = set()
            self.types_by_lib[lib_name].add(type_name)

        return type_name, lib_name_out


    @staticmethod
    def get_cd_full_name(path_names: List[str], name: Optional[str]) -> str:
        # no need to have first parent in name
        path_name = ".".join([p for p in path_names])
        if name is not None:
            assert name
            path_name = f'{path_name}.{name}'
        return path_name


    def get_cd_by_name(self, path_names: List[str], name:str) -> KlassPydanticDump:
        assert name
        full_name = self.get_cd_full_name(path_names, name=name)
        cd = self.class_dumps.get(full_name, None)
        if not cd:
            vars_avail = get_available_names_example(full_name, self.class_dumps.keys(), max_display=10)
            raise Exception(f"Code object for '{full_name}' not available, available: {vars_avail}")
        return cd

        # " 2nd param is index in list "
        # out = [(cd, nr) for nr, cd in enumerate(self.class_dumps, 0) if cd.name==name]
        # if len(out)==0:
        #     return None
        # assert len(out)==1
        # return out[0]


    def add_cd(self, path_names: List[str], class_dump:KlassPydanticDump):
        assert class_dump.name
        assert class_dump.vars_declarations

        if not path_names[-1]==class_dump.name:
            raise Exception(f"not valid: {class_dump.name} vs {path_names}")

        full_name = self.get_cd_full_name(path_names, name=None)

        if full_name in self.class_dumps:
            raise Exception(f"Code under name '{full_name}' already set: {self.class_dumps[full_name]}")
        self.class_dumps[full_name] = class_dump


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

    def dump_class_with_children(self, 
            path_names: List[str], 
            component: Union[FieldGroup, Entity, SubEntityBase],
            indent: int, 
            depth: int,
            ) -> Union[DelarationCodeLinesType, FieldCodeLinesType]:

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

    def dump_to_str(self,
                    component:ComponentBase,
                    # internal params
                    path_names: List[str] = None,
                    depth:int=0) -> List[str]:
        if depth==0:
            path_names = []
        else:
            if depth > MAX_RECURSIONS:
                raise EntityInternalError(owner=component, msg=f"Maximum recursion depth exceeded ({depth})")
            path_names = path_names[:]

        path_names.append(component.name)
        indent = PY_INDENT * depth

        with self.use_stack_frame(
                CodegenStackFrame(
                    depth=depth,
                    path_names = path_names,
                    component = component,
                    indent = indent,
                    )):

            lines : List[str] = []
            vars_declarations: List[str] = []

            indent_next = PY_INDENT * (depth+1)

            any_dump = False
            children = component.get_children()

            if isinstance(component, (FieldGroup, Entity, SubEntityBase)):
                lines_, vars_declarations_ = self.dump_class_with_children(path_names, component, indent, depth)
                lines.extend(lines_)
                vars_declarations.extend(vars_declarations_)
                any_dump = True

            if isinstance(component, (FieldBase,)):
                lines_, vars_declarations_ = self.dump_field()
                lines.extend(lines_)
                vars_declarations.extend(vars_declarations_)
                any_dump = True

            if not any_dump:
                raise Exception(f"No dump for: {component}")

            self.add_cd(path_names, KlassPydanticDump(
                                    name=component.name,
                                    lines=lines, 
                                    vars_declarations=vars_declarations))

            for sub_component in children:
                # recursion
                self.dump_to_str(sub_component, 
                                 path_names=path_names, 
                                 depth=depth+1)
                cd = self.get_cd_by_name(path_names, name=sub_component.name)
                # print(f"RT: {component.name} -> {sub_component.name}")
                if cd.vars_declarations:
                    # remove last (simple) cluss dump and dump it here
                    # in order to preserve structure:
                    #   attributes
                    #   custom_class definitions
                    lines.extend(cd.vars_declarations)


        if depth==0:
            all_lines = []
            all_lines.extend([
                "from __future__ import annotations",
                "# --------------------------------------------------------------------------------",
                "# IMPORTANT: DO NOT EDIT!!! The code is generated by ",
                "#            reedwolf.entities.generators.to_pydantic",
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

            for cd in self.class_dumps.values():
                all_lines.extend(cd.lines)

            all_lines.append("")
            all_lines.append("")
            all_lines.append("# from typing import get_type_hints; print(get_type_hints(VendingCompanyDTO)); print('---- ALL OK -----')")
            all_lines.append("")

            out = "\n".join(all_lines)
        else:
            out = None # ignored (alt: return lines)

        return out

    
# ------------------------------------------------------------

def dump_to_pydantic_models_as_str(component:ComponentBase) -> str:
    dumper = DumpToPydantic()
    return dumper.dump_to_str(component=component)

# ------------------------------------------------------------

def dump_to_pydantic_models(component:ComponentBase, fname:str) -> str:
    code = dump_to_pydantic_models_as_str(component=component)
    with open(fname, "w") as fout:
        fout.write(code)
    return code


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

