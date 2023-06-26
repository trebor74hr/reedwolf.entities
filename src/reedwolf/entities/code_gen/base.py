import os
import inspect
from abc import abstractmethod
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
from . import dummy


DelarationCodeLinesType = List[str]

THIS_PACKAGE = inspect.getmodule(dummy).__package__

PyType = type


@dataclass
class ClassDeclarationBase:

    name : str
    file_dump: "FileDumpBase"
    title : str =""

    def __post_init__(self):
        ...

    @abstractmethod
    def dump_to_strlist(self) -> List[str]:
        ...


@dataclass
class VariableDeclarationBase:

    name : str

    flatten: bool
    deps_order: bool

    class_name_base: str
    # decoration template
    class_name_deco_templ: str
    # cross-reference
    class_declaration: Optional[ClassDeclarationBase]

    title : str = ""
    comment: str = ""

    # later computed when assigned to component
    owner_comp_dump : Optional["ComponentDumpBase"] = field(repr=False, init=False, default=None)
    class_name_full_path: Optional[str ]= field(repr=False, init=False, default=None)

    def __post_init__(self):
        ...

    def set_owner_comp_dump(self, owner_comp_dump: "ComponentDumpBase"):
        assert not self.owner_comp_dump
        self.owner_comp_dump = owner_comp_dump

        register_local_import = True
        if self.class_declaration:
            if self.class_declaration.file_dump is self.owner_comp_dump.file_dump:
                # not top level in file split - used in nested case (not flatten)
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


    @abstractmethod
    def dump_to_str(self) -> str:
        ...


@dataclass
class ComponentDumpBase:
    """ dumped lines of code and declarations for a component 
    class_declaration
        vars_declarations

        DumpAll (
            List[ComponentDumpBase]
            )
    """

    name:str
    # filename: str
    file_dump: "FileDumpBase"

    deps_order: bool = field(repr=False)

    # class <comp-name>DTO:
    class_declaration: ClassDeclarationBase = field(repr=False)

    owner_comp_dump : Optional[Self] = field(repr=False)

    # Later added - variable declarations - can have:
    #   - 1 item for normal case: 
    #     comp_name: CompType
    #   - 2 items for boolean + enables: 
    #     comp_name: bool
    #     comp_name_details: CompTypeDetails
    vars_declarations: List[VariableDeclarationBase]  = field(repr=False, init=False, default_factory=list)

    # all registered classes are nested under owner class
    nested_comp_dumps: List[Self] = field(repr=False, init=False, default_factory=list)

    # autocomputed
    owners_path_class_names : List[str] = field(repr=False, init=False)
    dumped: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        self.owners_path_class_names = []

        if self.owner_comp_dump:
            if self.owner_comp_dump.file_dump is self.file_dump \
              and self.owner_comp_dump.owners_path_class_names:
                self.owners_path_class_names.extend(self.owner_comp_dump.owners_path_class_names[:])

        if self.class_declaration:
            self.owners_path_class_names.append(self.class_declaration.name)



    def add_vars_declarations(self, vars_declarations: List[VariableDeclarationBase]):
        for variable_declaration in vars_declarations:
            variable_declaration.set_owner_comp_dump(self)
            self.vars_declarations.append(variable_declaration)

    def add_nested_comp_dump(self, comp_dump: Self):
        # if self.current_frame.owner_comp_dump.file_dump is self.current_frame.file_dump:
        if self.file_dump is not comp_dump.file_dump:
            raise EntityInternalError(owner=self, msg=f"File dump must be the same of nested class and this one:\n  {self.file_dump}\n  !=\n  {comp_dump.file_dump}") 
        self.nested_comp_dumps.append(comp_dump)


    def _dump_to_strlist_nested_comps(self, lines: List[str]):
        if self.nested_comp_dumps:
            comp_dump_list = self.nested_comp_dumps
            # if self.deps_order:
            #     comp_dump_list = reversed(comp_dump_list)

            for comp_dump in comp_dump_list:
                if comp_dump.dumped:
                    # TODO: this should not happen
                    continue
                nested_lines = comp_dump.dump_to_strlist()
                lines.extend(
                    add_py_indent_to_strlist(1, 
                        nested_lines))


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

            if self.deps_order:
                self._dump_to_strlist_nested_comps(lines)

            lines.extend(
                add_py_indent_to_strlist(1, 
                    [vd.dump_to_str() for vd in self.vars_declarations]))

            if self.deps_order:
                lines.append("")
            else:
                self._dump_to_strlist_nested_comps(lines)

        else:
            " can be single attribute that is already dumped before in some class declaration (leaf) "
            assert not self.vars_declarations
            assert not self.nested_comp_dumps

        self.dumped = True

        return lines


# ------------------------------------------------------------


@dataclass 
class FileDumpBase:
    " one filename - can have several ComponentDumpBase "

    filename: str
    flatten: bool
    deps_order: bool

    # internal
    comp_dump_dict: Dict[str, ComponentDumpBase] = field(init=False, repr=False, default_factory=OrderedDict)
    types_by_lib : Dict[str, Set[str]] = field(init=False, repr=False, default_factory=dict)


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


    @abstractmethod
    def dump_to_str(self) -> str:
        ...


# ------------------------------------------------------------


@dataclass
class CodegenStackFrame(IStackFrame):

    owner_comp_dump: Optional[ComponentDumpBase] = field(repr=False)
    component: ComponentBase = field(repr=False)
    # filename: str = field(kw_only=False)
    file_dump: FileDumpBase = field(init=True)
    path_names: List[str] = field()
    owner_class_name_path: List[str] = field(repr=False)
    depth: Optional[int] = field(repr=False) # 0 based

    # autocomputed
    component_name: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.component, ComponentBase):
            raise EntityInternalError(owner=self, msg=f"Expected Component, got: {self.component}") 

        if self.depth==0:
            assert not self.owner_comp_dump
        else:
            assert self.owner_comp_dump

        self.component_name = self.component.name
        # make a copy and add component name
        self.path_names = self.path_names[:]
        self.path_names.append(self.component.name)
        if not len(self.path_names) == self.depth+1:
            raise EntityInternalError(owner=self, msg=f"Depth is {self.depth} and length of path-names not matched: {self.path_names}") 

        if self.depth > MAX_RECURSIONS:
            raise EntityInternalError(owner=component, msg=f"Maximum recursion depth exceeded ({self.depth})")

        self.owner_class_name_path = self.owner_class_name_path[:]
        if self.owner_comp_dump:
            self.owner_class_name_path.append(self.owner_comp_dump.class_declaration.name)

    def set_new_file_dump(self, file_dump: FileDumpBase): 
        if self.file_dump.filename == file_dump.filename:
            raise EntityInternalError(owner=self, msg=f"Expected diff self.file_dump.filename == file_dump.filename, got: {file_dump}") 
        self.file_dump = file_dump


# ------------------------------------------------------------


@dataclass
class DumpToBase(IStackOwnerSession):

    # file_split_to_depth:
    #     1 - (default) no split at all, single file is produced
    #     2 - all components on level depth 2 will have own file
    #     3 - ...
    #     None - split on all levels, every component -> own file
    file_split_to_depth: Optional[int] = field(default=1)

    # flatten:
    #     False - (default) dependent classes are nested inside parent classes (except
    #         when component is top component in the file - see previous
    #         paremeter
    #     True - no nesting, all classes are on module level
    flatten: bool = field(default=False)

    # deps_order:
    #     False - (default) classes are in order from top to bottom
    #     True - classes are ordered in dependency order so all dependent classes 
    #            are above class that needs them.
    deps_order: bool = field(default=False)


    # all internal
    file_dump_dict: Dict[str, FileDumpBase] = field(init=False, repr=False, default_factory=OrderedDict)

    # obligatory for stack handling
    stack_frames: List[CodegenStackFrame] = field(repr=False, init=False, default_factory=list)
    # autocomputed
    current_frame: Optional[CodegenStackFrame] = field(repr=False, init=False, default=None)

    STACK_FRAME_CLASS: ClassVar[type] = CodegenStackFrame
    STACK_FRAME_CTX_MANAGER_CLASS: ClassVar[type] = UseStackFrameCtxManagerBase

    # needs to be set in inherited class
    KlassClassDeclaration       : ClassVar[type] = None
    KlassVariableDeclaration    : ClassVar[type] = None
    KlassComponentDump  : ClassVar[type] = None
    KlassFileDump       : ClassVar[type] = None

    def __post_init__(self):

        if not (self.__class__.KlassClassDeclaration and issubclass(self.__class__.KlassClassDeclaration, ClassDeclarationBase)):
            raise EntityInternalError(owner=self, msg=f"For KlassClassDeclaration expecting ClassDeclarationBase class, got: {self.__class__.KlassClassDeclaration} ")
        if not (self.__class__.KlassVariableDeclaration and issubclass(self.__class__.KlassVariableDeclaration, VariableDeclarationBase)):
            raise EntityInternalError(owner=self, msg=f"For KlassVariableDeclaration expecting VariableDeclarationBase class, got: {self.__class__.KlassVariableDeclaration}")
        if not (self.__class__.KlassComponentDump and issubclass(self.__class__.KlassComponentDump, ComponentDumpBase)):
            raise EntityInternalError(owner=self, msg=f"For KlassComponentDump expecting ComponentDumpBase class, got: {self.__class__.KlassComponentDump}")
        if not (self.__class__.KlassFileDump and issubclass(self.__class__.KlassFileDump, FileDumpBase)):
            raise EntityInternalError(owner=self, msg=f"For KlassFileDump expecting FileDumpBase class, got: {self.__class__.KlassFileDump }")

    # ------------------------------------------------------------

    def get_or_create_file_dump(self, filename: str) -> FileDumpBase:
        # filename = self.current_frame.filename
        if filename not in self.file_dump_dict:
            self.file_dump_dict[filename] = \
                    self.KlassFileDump(
                        filename=filename, 
                        flatten=self.flatten,
                        deps_order=self.deps_order)
        return self.file_dump_dict[filename]

    def get_file_dump(self, filename: str) -> FileDumpBase:
        if filename not in self.file_dump_dict:
            raise EntityInternalError(owner=self, msg=f"File dump '{filename}' not found") 
        return self.file_dump_dict[filename]

    # ------------------------------------------------------------

    @staticmethod
    def get_comp_dump_full_name(path_names: List[str]) -> str:
        # no need to have first parent in name
        path_name = ".".join([p for p in path_names])
        return path_name


    def set_comp_dump(self, comp_dump:ComponentDumpBase) -> ComponentDumpBase:
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


    def get_current_comp_dump(self) -> ComponentDumpBase:
        # file_dump = self.get_file_dump(self.current_frame.filename)
        file_dump = self.current_frame.file_dump

        full_name = self.get_comp_dump_full_name(self.current_frame.path_names)

        comp_dump = file_dump.comp_dump_dict.get(full_name, None)
        if not comp_dump:
            vars_avail = get_available_names_example(full_name, file_dump.comp_dump_dict.keys(), max_display=10)
            raise EntityInternalError(owner=self, msg=f"Code object for '{full_name}' not available, available: {vars_avail}")
        return comp_dump

    # ------------------------------------------------------------

    def dump_field(self) -> Tuple[
                List[VariableDeclarationBase], 
                Optional[ClassDeclarationBase], 
                List[PyType]]:
        """ Can return 1 - 3 items:
            1) attribute (no class decl) 
            2) optional - var with complex type with class decl.
            3) List of python-types to be used in this or owner file_dump
               what needs to be done in caller
        """
        component = self.current_frame.component

        todo_comment = ""
        children = component.get_children()

        vars_declarations = []

        file_dump = self.current_frame.file_dump

        if component.type_info:
            type_info = component.type_info

            py_type_klass = component.type_info.type_
            if not py_type_klass:
                raise EntityInternalError(owner=self, msg=f"component.type_info.type_ is not set: {to_repr(component.type_info)}") 


            if isinstance(component, EnumField):
                py_type_klass = component.enum
                py_type_name, _ = file_dump.use_type(py_type_klass)
            elif isinstance(component, ChoiceField):
                py_type_klass = component.python_type
                py_type_name, _ = file_dump.use_type(py_type_klass)
            elif py_type_klass in STANDARD_TYPE_LIST:
                py_type_name, _ = file_dump.use_type(py_type_klass)
            else:
                py_type_name, _ = file_dump.use_type(Any)
                todo_comment=f"TODO: domain_dataclass {py_type_name}"

            # type hint options - decorated type name
            py_type_name_deco_templ = "{}"
            if type_info.is_list:
                file_dump.use_type(List)
                py_type_name_deco_templ = f"List[{py_type_name_deco_templ}]"

            if type_info.is_optional:
                file_dump.use_type(Optional)
                py_type_name_deco_templ = f"Optional[{py_type_name_deco_templ}]"

            # NOTE: currently not implemented for:
            #   default, required, readonly, max-length, etc.

        else:
            todo_comment = f"TODO: unknown type for bind: {component.bind}"
            py_type_name, _ = file_dump.use_type(Any)
            py_type_name_deco_templ = "{}"


        var_declaration = self.KlassVariableDeclaration(
                    name=component.name, 
                    flatten=self.flatten,
                    deps_order=self.deps_order,
                    class_name_base = py_type_name, 
                    class_name_deco_templ = py_type_name_deco_templ, 
                    comment = todo_comment, 
                    title=component.title,
                    class_declaration=None,
                    )
        vars_declarations.append(var_declaration)

        if children:
            class_py_type = f"{snake_case_to_camel(component.name)}ChildrenDTO"
            class_declaration = self.KlassClassDeclaration(
                                    name=class_py_type,
                                    title=f"Children of '{component.name}'",
                                    file_dump = self.current_frame.file_dump,
                                    )
            var_py_name = f"{component.name}_children"
            var_declaration = self.KlassVariableDeclaration(
                                    name = var_py_name, 
                                    flatten=self.flatten,
                                    deps_order=self.deps_order,
                                    class_name_base = class_py_type,
                                    class_name_deco_templ = "{}",
                                    class_declaration=class_declaration,
                                    )
            vars_declarations.append(var_declaration)
        else:
            class_declaration = None

        return vars_declarations, class_declaration, []

    # ------------------------------------------------------------

    def dump_composite_class(self) -> Tuple[
                List[VariableDeclarationBase], 
                ClassDeclarationBase, 
                List[PyType]]:

        " see return values description in dump_field() "

        path_names = self.current_frame.path_names
        component  = self.current_frame.component 
        if not isinstance(component, (FieldGroup, Entity, SubEntityBase)):
            raise EntityInternalError(owner=self, msg=f"Invalid type of component, expected [FieldGroup, Entity, SubEntityBase], got: {to_repr(component)}") 

        py_type_name = f"{snake_case_to_camel(component.name)}DTO"
        class_declaration = self.KlassClassDeclaration(
                                name=py_type_name,
                                title=component.title,
                                file_dump = self.current_frame.file_dump,
                                )

        vars_declarations = []

        class_name_deco_templ = "{}"

        py_types_to_use = []
        if isinstance(component, SubEntityBase):
            subitem_type_info = component.bound_model.type_info

            if isinstance(component, SubEntityItems):
                py_types_to_use.append(List)
                class_name_deco_templ = f"List[{class_name_deco_templ}]"

            if subitem_type_info.is_optional:
                py_types_to_use.append(Optional)
                class_name_deco_templ = f"Optional[{class_name_deco_templ}]"


        var_declaration = self.KlassVariableDeclaration(
                                name=component.name, 
                                flatten=self.flatten,
                                deps_order=self.deps_order,
                                class_name_base=py_type_name, 
                                class_name_deco_templ=class_name_deco_templ, 
                                title=component.title,
                                class_declaration=class_declaration,
                                )
        vars_declarations.append(var_declaration)

        return vars_declarations, class_declaration, py_types_to_use


    # ------------------------------------------------------------

    def dump_all(self) -> Dict[str, str]:
        component   = self.current_frame.component
        path_names  = self.current_frame.path_names
        depth       = self.current_frame.depth


        is_composite_component = isinstance(component, (FieldGroup, Entity, SubEntityBase))

        children = component.get_children()

        if children and depth!=0 and (
          self.file_split_to_depth is None 
          or self.file_split_to_depth >= (depth+1)): 
            file_dump = self.get_or_create_file_dump(component.name)
            self.current_frame.set_new_file_dump(file_dump)
        else:
            file_dump     = self.current_frame.file_dump
        assert file_dump == self.current_frame.file_dump

        if is_composite_component:
            vars_declarations, class_declaration, py_types_to_use = \
                    self.dump_composite_class()
        elif isinstance(component, (FieldBase,)):
            vars_declarations, class_declaration, py_types_to_use = \
                    self.dump_field()
        else:
            raise EntityInternalError(owner=self, msg=f"No dump for: {component}")

        if self.current_frame.owner_comp_dump:
            # belongs to owner
            assert vars_declarations, component
            self.current_frame.owner_comp_dump.add_vars_declarations(vars_declarations)

            file_dump_for_imports = self.current_frame.owner_comp_dump.file_dump
            for py_type in py_types_to_use:
                file_dump_for_imports.use_type(py_type)

        else:
            # NOTE: vars_declarations won't be consumed in top level component
            if depth!=0:
                raise EntityInternalError(owner=self, msg=f"Comp dump owner not set and depth is not 0, got: {depth}") 


        comp_dump = self.set_comp_dump(
                            self.KlassComponentDump(
                                    name=component.name,
                                    file_dump=file_dump,
                                    deps_order=self.deps_order,
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

                    if not self.current_frame.owner_comp_dump is comp_dump:
                        raise EntityInternalError(owner=self, msg=f"self.current_frame.owner_comp_dump={self.current_frame.owner_comp_dump} is not comp_dump={comp_dump}") 

                    if not self.flatten \
                      and self.current_frame.owner_comp_dump.file_dump is self.current_frame.file_dump:
                        # in this case classes are nested under owner class
                        comp_dump.add_nested_comp_dump(sub_comp_dump)


        else:
            if class_declaration:
                raise EntityInternalError(owner=self, msg=f"class_declaration should be None, got: {class_declaration}") 

        if depth==0:
            lines_by_file = OrderedDict()
            for file_dump in self.file_dump_dict.values():
                lines_by_file[file_dump.filename] = file_dump.dump_to_str()
        else:
            lines_by_file = None

        return lines_by_file





