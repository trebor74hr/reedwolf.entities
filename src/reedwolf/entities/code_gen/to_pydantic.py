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
        Tuple,
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
    file_dump: "FilePydanticDump"
    title : str =""

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
    # cross-reference
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

        register_local_import = True
        if self.class_declaration:
            if self.class_declaration.file_dump is self.owner_comp_dump.file_dump:
                self.class_name_full_path = ".".join(
                        self.owner_comp_dump.owners_path_class_names \
                        + [self.class_declaration.name])
            else:
                self.class_name_full_path = self.class_declaration.name
                self.owner_comp_dump\
                        .file_dump\
                        .use_type_w_lib(
                                type_name=self.class_declaration.name, 
                                lib_name=f".{self.class_declaration.file_dump.filename}")

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
    # filename: str
    file_dump: "FilePydanticDump"

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

    # all registered classes are nested under owner class
    nested_comp_dumps: List[Self] = field(repr=False, init=False, default_factory=list)

    # autocomputed
    owners_path_class_names : List[str] = field(repr=False, init=False)
    dumped: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        self.owners_path_class_names = []

        # if self.name == "address_set":
        #     import pdb;pdb.set_trace() 

        if self.owner_comp_dump:
            if self.owner_comp_dump.file_dump is self.file_dump \
              and self.owner_comp_dump.owners_path_class_names:
                self.owners_path_class_names.extend(self.owner_comp_dump.owners_path_class_names[:])

        if self.class_declaration:
            self.owners_path_class_names.append(self.class_declaration.name)



    def add_vars_declarations(self, vars_declarations: List[VariableDeclaration]):
        for variable_declaration in vars_declarations:
            variable_declaration.set_owner_comp_dump(self)
            self.vars_declarations.append(variable_declaration)

    def add_nested_comp_dump(self, comp_dump: Self):
        # if self.current_frame.owner_comp_dump.file_dump is self.current_frame.file_dump:
        if self.file_dump is not comp_dump.file_dump:
            raise EntityInternalError(owner=self, msg=f"File dump must be the same of nested class and this one:\n  {self.file_dump}\n  !=\n  {comp_dump.file_dump}") 
        self.nested_comp_dumps.append(comp_dump)

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

    # ------------------------------------------------------------

    def use_type(self, klass: type) -> Tuple[str, str]:
        " 2nd return param is lib_name when is not part of standard library "
        lib_name, _, type_name = str(klass).rpartition(".")
        lib_name_out = None
        if lib_name != "typing":
            type_name = klass.__name__
            module = inspect.getmodule(klass)
            lib_name = module.__name__
            if lib_name != "builtins":
                lib_name_out = lib_name
        type_name = self.use_type_w_lib(type_name=type_name, lib_name=lib_name)
        return type_name, lib_name_out

    def use_type_w_lib(self, type_name : str, lib_name: str) -> str:
        if lib_name != "builtins":
            if lib_name not in self.types_by_lib:
                self.types_by_lib[lib_name] = set()
            self.types_by_lib[lib_name].add(type_name)
        return type_name

    # ------------------------------------------------------------

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
            # if "can_user_change" in str(comp_dump): import pdb;pdb.set_trace() 

            if not comp_dump_first:
                comp_dump_first = comp_dump
            if comp_dump.dumped:
                continue
            all_lines.extend(comp_dump.dump_to_strlist())

        if not (comp_dump_first and comp_dump_first.class_declaration):
            raise EntityInternalError(owner=self, msg=f"comp_dump_first='{comp_dump_first}' or its 'class_declaration' not set") 

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
    # filename: str = field(kw_only=False)
    file_dump: FilePydanticDump = field(init=True)
    path_names: List[str] = field()
    owner_class_name_path: List[str] = field(repr=False)
    depth: Optional[int] = field(repr=False) # 0 based

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

    def set_new_file_dump(self, file_dump: FilePydanticDump): 
        assert self.file_dump.filename != file_dump.filename
        self.file_dump = file_dump


# ------------------------------------------------------------


@dataclass
class DumpToPydantic(IStackOwnerSession):

    file_split_to_depth: Optional[int] = field(default=1)

    # all internal
    file_dump_dict: Dict[str, FilePydanticDump] = field(init=False, repr=False, default_factory=OrderedDict)

    # obligatory for stack handling
    stack_frames: List[CodegenStackFrame] = field(repr=False, init=False, default_factory=list)
    # autocomputed
    current_frame: Optional[CodegenStackFrame] = field(repr=False, init=False, default=None)

    STACK_FRAME_CLASS: ClassVar[type] = CodegenStackFrame
    STACK_FRAME_CTX_MANAGER_CLASS: ClassVar[type] = UseStackFrameCtxManagerBase

    # ------------------------------------------------------------

    def use_type(self, klass: type) -> Tuple[str, str]:
        file_dump = self.current_frame.file_dump
        return file_dump.use_type(klass)

    # ------------------------------------------------------------

    def get_or_create_file_dump(self, filename: str) -> FilePydanticDump:
        # filename = self.current_frame.filename
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

        # file_dump = self.get_or_create_file_dump()
        file_dump = self.current_frame.file_dump

        path_names: List[str] = self.current_frame.path_names

        if not path_names[-1]==comp_dump.name:
            raise EntityInternalError(owner=self, msg=f"not valid: {comp_dump.name} vs {path_names}")

        full_name = self.get_comp_dump_full_name(path_names)

        if full_name in file_dump.comp_dump_dict:
            raise EntityInternalError(owner=self, msg=f"Code under name '{full_name}' already set: {file_dump.comp_dump_dict[full_name]}")

        file_dump.comp_dump_dict[full_name] = comp_dump

        return comp_dump


    def get_current_comp_dump(self) -> ComponentPydanticDump:
        # file_dump = self.get_file_dump(self.current_frame.filename)
        file_dump = self.current_frame.file_dump

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
                                    title=f"Children of '{component.name}'",
                                    file_dump = self.current_frame.file_dump,
                                    )

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
        assert isinstance(component, (FieldGroup, Entity, SubEntityBase))

        py_type_name = f"{snake_case_to_camel(component.name)}DTO"
        class_declaration = ClassDeclaration(
                                name=py_type_name,
                                title=component.title,
                                file_dump = self.current_frame.file_dump,
                                )

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

        # if component.name == "can_user_change": import pdb;pdb.set_trace() 

        if depth!=0 and self.file_split_to_depth >= (depth+1) and children: 
            file_dump = self.get_or_create_file_dump(component.name)
            self.current_frame.set_new_file_dump(file_dump)
        else:
            file_dump     = self.current_frame.file_dump

        if is_composite_component:
            vars_declarations, class_declaration = self.dump_composite_class()
        elif isinstance(component, (FieldBase,)):
            vars_declarations, class_declaration = self.dump_field()
        else:
            raise EntityInternalError(owner=self, msg=f"No dump for: {component}")

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
                                    file_dump=file_dump,
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
                            file_dump = file_dump,
                            owner_class_name_path = self.current_frame.owner_class_name_path,
                            )):
                    # RECURSION
                    self.dump_all()

                    sub_comp_dump = self.get_current_comp_dump()

                    assert self.current_frame.owner_comp_dump is comp_dump

                    if self.current_frame.owner_comp_dump.file_dump is self.current_frame.file_dump:
                        # in this case classes are nested under owner class
                        comp_dump.add_nested_comp_dump(sub_comp_dump)


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
        file_split_to_depth: Optional[int] = 1,
        ) -> Dict[str, str]:

    assert file_split_to_depth >= 1, file_split_to_depth
    dumper = DumpToPydantic(
                file_split_to_depth=file_split_to_depth,
                )
    with dumper.use_stack_frame(
            CodegenStackFrame(
                owner_comp_dump=None,
                component = component,
                file_dump = dumper.get_or_create_file_dump(component.name),
                depth=0,
                path_names = [],
                owner_class_name_path=[],
                )):
        return dumper.dump_all()

# ------------------------------------------------------------

def dump_to_pydantic_models(
        component:ComponentBase, 
        fname_or_dname:str,
        file_split_to_depth: Optional[int] = 1,
        ) -> str:
    lines_by_file = dump_to_pydantic_models_as_dict(
                        component=component,
                        file_split_to_depth=file_split_to_depth,
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


