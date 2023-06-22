import inspect
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
        )
from ..meta import (
        TypeInfo,
        is_enum,
        STANDARD_TYPE_LIST,
        )
from ..base import (
        ComponentBase,
        PY_INDENT,
        MAX_RECURSIONS,
        )
from ..fields import (
        FieldBase, 
        FieldGroup, 
        ChoiceField,
        )
from ..containers import (
        SubEntityItems, 
        Entity,
        )

# ------------------------------------------------------------

@dataclass
class DumpPydanticClassLines:
    name:str
    lines: List[str]
    vars_declarations: List[str]
    # py_name:str
    # py_type:str
    # class_py_name:Optional[str]
    # class_py_type:Optional[str]

# ------------------------------------------------------------

@dataclass
class DumpPydanticClassLinesStore:

    class_dumps:List[DumpPydanticClassLines] = field(init=False, default_factory=list)
    class_dumps_names:Set[str] = field(init=False, default_factory=set)
    # enums : Set[str] = field(init=False, default_factory=set)
    types_by_lib : Dict[str, Set[str]] = field(init=False, default_factory=dict)

    # def use_enum(self, enum_name: str) -> str:
    #     self.enums.add(enum_name)
    #     module_name = inspect.getmodule(py_type_klass).__name__
    #     return enum_name

    def use_type(self, klass: type) -> str:
        lib_name, _, type_name = str(klass).rpartition(".")

        if lib_name != "typing":
            type_name = klass.__name__
            module = inspect.getmodule(klass)
            lib_name = module.__name__

        if lib_name != "builtins":
            if lib_name not in self.types_by_lib:
                self.types_by_lib[lib_name] = set()
            self.types_by_lib[lib_name].add(type_name)

        return type_name

    def get(self, name:str) -> Optional[Union[DumpPydanticClassLines, int]]:
        " 2nd param is index in list "
        out = [(cd, nr) for nr, cd in enumerate(self.class_dumps, 0) if cd.name==name]
        if len(out)==0:
            return None
        assert len(out)==1
        return out[0]

    # def remove_last(self, name:str):
    #     class_dump = self.class_dumps[-1]
    #     assert class_dump.name==name
    #     del self.class_dumps[-1]
    #     self.class_dumps_names.remove(class_dump.name)

    def add(self, class_dump:DumpPydanticClassLines):
        assert class_dump.name
        assert class_dump.vars_declarations
        # assert class_dump.lines - contain fieldgroup
        assert class_dump.name not in self.class_dumps_names
        self.class_dumps.append(class_dump)
        self.class_dumps_names.add(class_dump.name)


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

def _dump_to_pydantic_models_as_str(
                         component:ComponentBase,
                         # internal params
                         class_dump_store:Optional[List[DumpPydanticClassLines]]=None,
                         path:List[str]=None,
                         depth:int=0) -> List[str]:
    self = component
    if depth==0:
        assert class_dump_store is None
        class_dump_store = DumpPydanticClassLinesStore()
        assert path is None
        path = []
    else:
        assert class_dump_store is not None

    if depth > MAX_RECURSIONS:
        raise EntityInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

    indent = PY_INDENT * depth
    indent_next = PY_INDENT * (depth+1)

    lines = []
    vars_declarations = []

    children = self.get_children()

    if isinstance(self, (FieldGroup, Entity, SubEntityItems)):
        py_type_name = f"{snake_case_to_camel(self.name)}DTO"
        # make a copy
        path = path[:] + [py_type_name]
        py_type_name_ext = snake_case_to_camel(".".join(path))

        # vars_declarations will be consumed in owner, so
        # only for top object it won't be consumed

        if isinstance(self, SubEntityItems):
            assert isinstance(self.bound_attr_node.data, TypeInfo), self.bound_attr_node.data
            if self.bound_attr_node.data.is_list:
                py_type_name_ext = f"List[{py_type_name_ext}]"
            if self.bound_attr_node.data.is_optional:
                py_type_name_ext = f"Optional[{py_type_name_ext}]"
        # vars_declarations.append(f'{indent}{self.name}: {py_type_name_ext}')
        vars_declarations.append(f'{indent}{DumpPydanticClassLinesStore.create_pydantic_var_declaration(self.name, py_type_name_ext, title=self.title)}')

        lines.append("")
        # lines.append("")
        # lines.append(f"{indent}class {py_name}(BaseModel):")
        lines.extend(DumpPydanticClassLinesStore.create_pydantic_class_declaration(
                        depth,
                        py_type_name,
                        title=self.title))

    elif isinstance(self, (FieldBase,)):
        py_name = self.name
        todo_comment = ""
        if self.bound_attr_node:
            assert isinstance(self.bound_attr_node.data, TypeInfo)
            py_type_klass = self.bound_attr_node.data.type_
            if is_enum(py_type_klass):
                py_type_name = class_dump_store.use_type(py_type_klass)
            elif isinstance(self, ChoiceField) and self.choice_title_type_info is not None:
                parent_klass_full_name = self.choice_title_type_info.parent_object.__name__
                parent_klass_name = parent_klass_full_name.split(".")[-1]
                # value_klass_name = py_type_klass.__name__
                value_klass_name = self.choice_value_type_info.type_.__name__
                # should be string
                # title_klass_name = self.choice_title_type_info.type_
                py_type_name = f"{snake_case_to_camel(parent_klass_name)}ChoiceDTO"
                lines.append("")
                # lines.append(f"{indent}class {py_type_name}(BaseModel):")
                title = f"Choice type for {self.name}"
                lines.extend(DumpPydanticClassLinesStore.create_pydantic_class_declaration(
                                depth,
                                py_type_name,
                                title=title))
                lines.append(f"{indent_next}value: {value_klass_name}")
                lines.append(f"{indent_next}title: str")
                # lines.append("")
            elif py_type_klass in STANDARD_TYPE_LIST:
                py_type_name = class_dump_store.use_type(py_type_klass)
            else:
                py_type_name = class_dump_store.use_type(Any)
                todo_comment=f"TODO: domain_dataclass {py_type_name}"

            # type hint options
            if self.bound_attr_node.data.is_list:
                class_dump_store.use_type(List)
                py_type_name = f"List[{py_type_name}]"
            if self.bound_attr_node.data.is_optional:
                class_dump_store.use_type(Optional)
                py_type_name = f"Optional[{py_type_name}]"
        else:
            # todo_comment = f"  # TODO: unbound {self.bind}"
            todo_comment = f"TODO: unbound {self.bind}"
            py_type_name = class_dump_store.use_type(Any)

        # vars_declarations.append(f"{indent}{py_name}: {py_type_name}{todo_comment}")
        var_declaration = DumpPydanticClassLinesStore.create_pydantic_var_declaration(
                    py_name, 
                    py_type_name, 
                    todo_comment, 
                    title=self.title)
        vars_declarations.append(f'{indent}{var_declaration}')

        if children:
            lines.append("")
            # lines.append("")
            class_py_name = f"{self.name}_fieldgroup"
            class_py_type = f"{snake_case_to_camel(self.name)}DetailsDTO"
            # lines.append(f"{indent}class {class_py_type}(BaseModel):")
            title = f"Component beneath type {self.name}"
            lines.extend(DumpPydanticClassLinesStore.create_pydantic_class_declaration(
                            depth,
                            class_py_type,
                            title=title))

            # vars_declarations.append(f'{indent}{class_py_name}: {class_py_type}')
            vars_declarations.append(f'{indent}{DumpPydanticClassLinesStore.create_pydantic_var_declaration(class_py_name, class_py_type)}')

            # TODO: za dokumentaciju/title/description - dodatno:
            #       - možda markdown
            #       - Field(description="...")
            #       - class Config:
            #           title = "dokumentacija"
            #           description = "dokumentacija"
            #           fields = [] # "samo za njih dokumentaciju"

    else:
        assert False, self

    class_dump_store.add(DumpPydanticClassLines(
                            name=self.name,
                            lines=lines, 
                            vars_declarations=vars_declarations))


    for component in children:
        # recursion
        _dump_to_pydantic_models_as_str(component, class_dump_store=class_dump_store, path=path[:], depth=depth+1)
        cd, cd_nr = class_dump_store.get(component.name)
        # print(f"RT: {self.name} -> {component.name}")
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
            "# IMPORTANT: DO NOT EDIT!!! The code is generated by "
            "#            reedwolf.entities.generators.to_pydantic",
            "#            Change 'Entity' object and regenerate the code .",
            "# --------------------------------------------------------------------------------",
            "from pydantic import BaseModel, Field",
            ])

        for module_name in sorted(class_dump_store.types_by_lib.keys()):
            # "from datetime import date, datetime  # noqa: F401",
            # "from decimal import Decimal",
            all_lines.append(f"from {module_name} import (")
            for class_name in sorted(class_dump_store.types_by_lib[module_name]):
                all_lines.append(f"    {class_name},")
            all_lines.append(")")

        # if class_dump_store.enums:
        #     all_lines.append("from domain.cloud.enum import (")
        #     for enum_name in sorted(class_dump_store.enums):
        #         all_lines.append(f"    {enum_name},")
        #     all_lines.append(")")
        # all_lines.append(f"")

        for cd in class_dump_store.class_dumps:
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
    return _dump_to_pydantic_models_as_str(component=component)

# ------------------------------------------------------------

def dump_to_pydantic_models(component:ComponentBase, fname:str) -> str:
    code = _dump_to_pydantic_models_as_str(component=component)
    with open(fname, "w") as fout:
        fout.write(code)
    return code

