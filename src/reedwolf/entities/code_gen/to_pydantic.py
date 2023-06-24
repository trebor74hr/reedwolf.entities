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
        ClassVar,
        )

from ..exceptions import (
        EntityInternalError,
        )
from ..utils import (
        snake_case_to_camel,
        to_repr,
        get_available_names_example,
        add_py_indent_to_strlist,
        PY_INDENT,
        )
from ..meta import (
        TypeInfo,
        STANDARD_TYPE_LIST,
        Self,
        )
from ..base import (
        IStackOwnerSession,
        IStackFrame,
        UseStackFrameCtxManagerBase,
        ComponentBase,
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

THIS_PACKAGE = inspect.getmodule(base).__package__
THIS_MODULE = os.path.basename(__file__).split(".")[0]



@dataclass
class ClassDeclaration:
    name : str
    title : str =""
    # class_declaration_owner: Optional[Self]

    def __post_init__(self):
        ...

    def dump_to_strlist(self) -> List[str]:
        lines = []
        # lines.append("")
        lines.append(f"class {self.name}(BaseModel):")
        if self.title:
            lines.append(f'{PY_INDENT}""" {self.title} """')
            # lines.append("")
        return lines


@dataclass
class VariableDeclaration:
    name : str
    class_name_base: str
    # decoration template
    class_name_deco_templ: str
    # NOTE: belongs to - not used for now
    class_declaration: Optional[ClassDeclaration]

    title : str = ""
    comment: str = ""

    # later computed when assigned to component
    owner_comp_dump : Optional["ComponentPydanticDump"] = field(repr=False, init=False, default=None)
    class_name_full_path: Optional[str ]= field(repr=False, init=False, default=None)

    def __post_init__(self):
        ...

    def set_owner_comp_dump(self, owner_comp_dump: "ComponentPydanticDump"):
        assert not self.owner_comp_dump
        self.owner_comp_dump = owner_comp_dump
        if self.class_declaration:
            self.class_name_full_path = ".".join(self.owner_comp_dump.owners_path_class_names + [self.class_declaration.name])

    def dump_to_str(self) -> str:
        if self.class_name_full_path:
            class_name = self.class_name_deco_templ.format(self.class_name_full_path)
        else:
            class_name = self.class_name_deco_templ.format(self.class_name_base)
        out = f"{self.name}: {class_name}"
        if self.title:
            out = f'{out} = Field(title="{self.title}")'
        if self.comment:
            out = f"{out}  # {self.comment}"
        return out


# ------------------------------------------------------------

@dataclass
class ComponentPydanticDump:
    """ dumped lines of code and declarations for a component 
    class_declaration
        vars_declarations

        DumpAll (
            List[ComponentPydanticDump]
            )

    """
    name:str
    filename: str

    # class <comp-name>DTO:
    class_declaration: ClassDeclaration = field(repr=False)

    owner_comp_dump : Optional[Self] = field(repr=False)

    # variable declarations - can have:
    #   - 1 item for normal case: 
    #     comp_name: CompType
    #   - 2 items for boolean + enables: 
    #     comp_name: bool
    #     comp_name_details: CompTypeDetails
    vars_declarations: List[VariableDeclaration]  = field(repr=False, init=False, default_factory=list)

    nested_comp_dumps: List[Self] = field(repr=False, init=False, default_factory=list)

    # autocomputed
    owners_path_class_names : List[str] = field(repr=False, init=False)
    dumped: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        self.owners_path_class_names = []

        if self.owner_comp_dump:
            if self.owner_comp_dump.owners_path_class_names:
                self.owners_path_class_names.extend(self.owner_comp_dump.owners_path_class_names[:])

        if self.class_declaration:
            self.owners_path_class_names.append(self.class_declaration.name)


    def add_vars_declarations(self, vars_declarations: List[VariableDeclaration]):
        for variable_declaration in vars_declarations:
            variable_declaration.set_owner_comp_dump(self)
            self.vars_declarations.append(variable_declaration)

    def dump_to_strlist(self) -> List[str]:
        " object can have nothing - e.g. plain std. type attribute, e.g. String(M.id) "
        if self.dumped:
            raise EntityInternalError(owner=self, msg=f"Component dump already done") 

        lines = []

        if self.class_declaration:
            assert self.vars_declarations
            lines.append("")
            lines.extend(self.class_declaration.dump_to_strlist())
            lines.append("")

            lines.extend(
                add_py_indent_to_strlist(1, 
                    [vd.dump_to_str() for vd in self.vars_declarations]))

            if self.nested_comp_dumps:
                for comp_dump in self.nested_comp_dumps:
                    nested_lines = comp_dump.dump_to_strlist()
                    lines.extend(
                        add_py_indent_to_strlist(1, 
                            nested_lines))
        else:
            " can be single attribute that is already dumped before in some class declaration (leaf) "
            assert not self.vars_declarations
            assert not self.nested_comp_dumps

        self.dumped = True

        return lines


@dataclass 
class FilePydanticDump:
    " one filename - can have several ComponentPydanticDump "
    filename: str

    # internal
    comp_dump_dict: Dict[str, ComponentPydanticDump] = field(init=False, repr=False, default_factory=OrderedDict)
    types_by_lib : Dict[str, Set[str]] = field(init=False, repr=False, default_factory=dict)

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

        comp_dump_first = None
        for comp_dump in self.comp_dump_dict.values():
            if not comp_dump_first:
                comp_dump_first = comp_dump
            if comp_dump.dumped:
                continue
            all_lines.extend(comp_dump.dump_to_strlist())

        if not comp_dump_first:
            raise EntityInternalError(owner=self, msg=f"comp_dump_first not set") 

        all_lines.append("")
        all_lines.append("")
        all_lines.append( "def _check_hints():")
        all_lines.append( "    from typing import get_type_hints")
        all_lines.append(f"    return get_type_hints({comp_dump_first.class_declaration.name})")
        all_lines.append("")

        out = "\n".join(all_lines)

        return out

# ------------------------------------------------------------

@dataclass
class CodegenStackFrame(IStackFrame):
    owner_comp_dump: Optional[ComponentPydanticDump] = field(repr=False)
    component: ComponentBase = field(repr=False)
    filename: str = field()
    path_names: List[str] = field()
    owner_class_name_path: List[str] = field(repr=False)
    depth: Optional[int] = field(repr=False) # 0 based
    indent_level: int = field(repr=False)

    # autocomputed
    component_name: str = field(init=False)

    def __post_init__(self):
        if self.depth==0:
            assert not self.owner_comp_dump
        else:
            assert self.owner_comp_dump
        assert isinstance(self.component, ComponentBase)
        self.component_name = self.component.name
        # make a copy and add component name
        self.path_names = self.path_names[:]
        self.path_names.append(self.component.name)
        assert len(self.path_names) == self.depth+1

        if self.depth > MAX_RECURSIONS:
            raise EntityInternalError(owner=component, msg=f"Maximum recursion depth exceeded ({self.depth})")

        self.owner_class_name_path = self.owner_class_name_path[:]
        if self.owner_comp_dump:
            self.owner_class_name_path.append(self.owner_comp_dump.class_declaration.name)

    def set_new_filename(self, filename:str): 
        assert self.filename != filename
        self.filename = filename
        self.indent_level = 0

    # def get_indent(self): 
    #     return f"{PY_INDENT * self.indent_level}"


# ------------------------------------------------------------

@dataclass
class DumpToPydantic(IStackOwnerSession):

    filenames_up_to_level: Optional[int] = field(default=1)

    # all internal
    file_dump_dict: Dict[str, FilePydanticDump] = field(init=False, repr=False, default_factory=OrderedDict)

    # obligatory for stack handling
    stack_frames: List[CodegenStackFrame] = field(repr=False, init=False, default_factory=list)
    # autocomputed
    current_frame: Optional[CodegenStackFrame] = field(repr=False, init=False, default=None)

    STACK_FRAME_CLASS: ClassVar[type] = CodegenStackFrame
    STACK_FRAME_CTX_MANAGER_CLASS: ClassVar[type] = UseStackFrameCtxManagerBase

    # ------------------------------------------------------------

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

    # ------------------------------------------------------------

    def get_or_create_file_dump(self) -> FilePydanticDump:
        filename = self.current_frame.filename
        if filename not in self.file_dump_dict:
            self.file_dump_dict[filename] = FilePydanticDump(filename=filename)
        return self.file_dump_dict[filename]

    def get_file_dump(self, filename: str) -> FilePydanticDump:
        if filename not in self.file_dump_dict:
            raise EntityInternalError(owner=self, msg=f"File dump '{filename}' not found") 
        return self.file_dump_dict[filename]

    # ------------------------------------------------------------

    @staticmethod
    def get_comp_dump_full_name(path_names: List[str]) -> str:
        # no need to have first parent in name
        path_name = ".".join([p for p in path_names])
        return path_name


    def set_comp_dump(self, comp_dump:ComponentPydanticDump) -> ComponentPydanticDump:
        assert comp_dump.name

        file_dump = self.get_or_create_file_dump()

        path_names: List[str] = self.current_frame.path_names

        if not path_names[-1]==comp_dump.name:
            raise EntityInternalError(owner=self, msg=f"not valid: {comp_dump.name} vs {path_names}")

        full_name = self.get_comp_dump_full_name(path_names)

        if full_name in file_dump.comp_dump_dict:
            raise EntityInternalError(owner=self, msg=f"Code under name '{full_name}' already set: {file_dump.comp_dump_dict[full_name]}")

        file_dump.comp_dump_dict[full_name] = comp_dump

        return comp_dump


    def get_current_comp_dump(self) -> ComponentPydanticDump:
        file_dump = self.get_file_dump(self.current_frame.filename)

        full_name = self.get_comp_dump_full_name(self.current_frame.path_names)

        comp_dump = file_dump.comp_dump_dict.get(full_name, None)
        if not comp_dump:
            vars_avail = get_available_names_example(full_name, file_dump.comp_dump_dict.keys(), max_display=10)
            raise EntityInternalError(owner=self, msg=f"Code object for '{full_name}' not available, available: {vars_avail}")
        return comp_dump

    # ------------------------------------------------------------

    def dump_field(self) -> Union[List[VariableDeclaration], Optional[ClassDeclaration]]:
        """ can return 1 or 2 items:
            1) attribute (no class decl) 
            2) optional - var with complex type with class decl.
        """
        component = self.current_frame.component
        indent_level = self.current_frame.indent_level

        todo_comment = ""
        children = component.get_children()

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
            #             depth=depth,
            #             )
            elif py_type_klass in STANDARD_TYPE_LIST:
                py_type_name, _ = self.use_type(py_type_klass)
            else:
                py_type_name, _ = self.use_type(Any)
                todo_comment=f"TODO: domain_dataclass {py_type_name}"

            # type hint options - decorated type name
            py_type_name_deco_templ = "{}"
            if component.bound_attr_node.data.is_list:
                self.use_type(List)
                py_type_name_deco_templ = f"List[{py_type_name_deco_templ}]"
            if component.bound_attr_node.data.is_optional:
                self.use_type(Optional)
                py_type_name_deco_templ = f"Optional[{py_type_name_deco_templ}]"

            # NOTE: currently not implemented for:
            #   default, required, readonly, max-length, etc.

        else:
            # todo_comment = f"  # TODO: unbound {component.bind}"
            todo_comment = f"TODO: unbound {component.bind}"
            py_type_name, _ = self.use_type(Any)
            py_type_name_deco_templ = "{}"


        var_declaration = VariableDeclaration(
                    name=component.name, 
                    class_name_base = py_type_name, 
                    class_name_deco_templ = py_type_name_deco_templ, 
                    comment = todo_comment, 
                    title=component.title,
                    class_declaration=None,
                    )
        vars_declarations.append(var_declaration)

        if children:
            class_py_type = f"{snake_case_to_camel(component.name)}ChildrenDTO"
            class_declaration = ClassDeclaration(
                                    name=class_py_type,
                                    title=f"Children of '{component.name}'")

            # class_py_type_ext = ".".join(self.current_frame.owner_class_name_path))
            var_py_name = f"{component.name}_children"
            var_declaration = VariableDeclaration(
                                    name = var_py_name, 
                                    class_name_base = class_py_type,
                                    class_name_deco_templ = "{}",
                                    class_declaration=class_declaration,
                                    )
            vars_declarations.append(var_declaration)
        else:
            class_declaration = None

        return vars_declarations, class_declaration

    # ------------------------------------------------------------

    def dump_composite_class(self) -> Union[List[VariableDeclaration], Optional[ClassDeclaration]]:
        " returns single variable decl "

        path_names = self.current_frame.path_names
        component  = self.current_frame.component 
        indent_level = self.current_frame.indent_level
        assert isinstance(component, (FieldGroup, Entity, SubEntityBase))

        py_type_name = f"{snake_case_to_camel(component.name)}DTO"
        class_declaration = ClassDeclaration(
                                name=py_type_name,
                                title=component.title)

        vars_declarations = []

        # py_type_name_ext2 = snake_case_to_camel(".".join(map(snake_case_to_camel, path_names)))
        class_name_deco_templ = "{}"

        if isinstance(component, SubEntityBase):
            subitem_type_info = component.bound_model.type_info

            if isinstance(component, SubEntityItems):
                self.use_type(List)
                class_name_deco_templ = f"List[{class_name_deco_templ}]"

            if subitem_type_info.is_optional:
                self.use_type(Optional)
                class_name_deco_templ = f"Optional[{class_name_deco_templ}]"

        var_declaration = VariableDeclaration(
                                name=component.name, 
                                class_name_base=py_type_name, 
                                class_name_deco_templ=class_name_deco_templ, 
                                title=component.title,
                                class_declaration=class_declaration,
                                )
        vars_declarations.append(var_declaration)

        return vars_declarations, class_declaration

    # ------------------------------------------------------------

    def dump_all(self) -> Dict[str, str]:
        component   = self.current_frame.component
        path_names  = self.current_frame.path_names
        depth       = self.current_frame.depth

        is_composite_component = isinstance(component, (FieldGroup, Entity, SubEntityBase))

        children = component.get_children()

        # set_filename_for_children = False
        # if depth!=0 and self.filenames_up_to_level >= (depth+1) and children: 
        #     # is_composite_component:
        #     set_filename_for_children = True

        indent_level= self.current_frame.indent_level
        filename    = self.current_frame.filename

        if is_composite_component:
            vars_declarations, class_declaration = self.dump_composite_class()
        elif isinstance(component, (FieldBase,)):
            vars_declarations, class_declaration = self.dump_field()
        else:
            raise EntityInternalError(owner=self, msg=f"No dump for: {component}")

        # if class_declaration and not set_filename_for_children:
        #     lines.extend(add_py_indent_to_strlist(indent_level, class_declaration))

        if self.current_frame.owner_comp_dump:
            # belongs to owner
            assert vars_declarations, component
            self.current_frame.owner_comp_dump.add_vars_declarations(vars_declarations)
        else:
            # NOTE: vars_declarations won't be consumed in top level component
            assert depth==0

        comp_dump = self.set_comp_dump(
                            ComponentPydanticDump(
                                    name=component.name,
                                    filename=filename,
                                    class_declaration=class_declaration,
                                    owner_comp_dump = self.current_frame.owner_comp_dump,
                                    ))

        if children:
            for nr, sub_component in enumerate(children,0):
                with self.use_stack_frame(
                        CodegenStackFrame(
                            owner_comp_dump=comp_dump,
                            depth=depth+1,
                            path_names = path_names,
                            component = sub_component,
                            filename = filename,
                            indent_level = indent_level+1,
                            owner_class_name_path = self.current_frame.owner_class_name_path,
                            )):
                    # if set_filename_for_children:
                    #     self.current_frame.set_new_filename(component.name)
                    #     if nr==0:
                    #         file_dump = self.get_or_create_file_dump()
                    #         assert class_declaration
                    #         file_dump.set_class_declaration(
                    #              add_py_indent_to_strlist(0, class_declaration))

                    # RECURSION
                    self.dump_all()

                    sub_comp_dump = self.get_current_comp_dump()

                    comp_dump.nested_comp_dumps.append(sub_comp_dump)

                    # if set_filename_for_children or sub_comp_dump.filename == comp_dump.filename:
                    #   # or sub_comp_dump.filename == component.name:
                    #     comp_dump.lines.extend(sub_comp_dump.vars_declarations)
                    # else:
                    #     sub_comp_dump.lines.extend(
                    #         add_py_indent_to_strlist(
                    #             self.current_frame.indent_level + 1, 
                    #             sub_comp_dump.vars_declarations))
        else:
            assert not class_declaration


        if depth==0:
            lines_by_file = OrderedDict()
            for file_dump in self.file_dump_dict.values():
                lines_by_file[file_dump.filename] = file_dump.dump_to_str()
        else:
            lines_by_file = None

        return lines_by_file

    
# ------------------------------------------------------------

def dump_to_pydantic_models_as_dict(
        component:ComponentBase, 
        filenames_up_to_level: Optional[int] = 1,
        ) -> Dict[str, str]:

    assert filenames_up_to_level >= 1, filenames_up_to_level
    dumper = DumpToPydantic(
                filenames_up_to_level=filenames_up_to_level,
                )
    with dumper.use_stack_frame(
            CodegenStackFrame(
                owner_comp_dump=None,
                component = component,
                filename = component.name,
                depth=0,
                indent_level=0,
                path_names = [],
                owner_class_name_path=[],
                )):
        return dumper.dump_all()

# ------------------------------------------------------------

def dump_to_pydantic_models(
        component:ComponentBase, 
        fname_or_dname:str,
        filenames_up_to_level: Optional[int] = 1,
        ) -> str:
    lines_by_file = dump_to_pydantic_models_as_dict(
                        component=component,
                        filenames_up_to_level=filenames_up_to_level,
                        )
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
#     # lines.append(f"class {py_type_name}(BaseModel):")
#     title = f"Choice type for {component.name}"
#     lines.extend(DumpToPydantic.dump_to_strlist(
#                     depth,
#                     py_type_name,
#                     title=title))
#     lines.append(f"{indent_next}value: {value_klass_name}")
#     lines.append(f"{indent_next}title: str")
#     # lines.append("")
#     return lines

