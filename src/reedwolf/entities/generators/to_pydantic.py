# Copied and adapted from Reedwolf project (project by robert.lujo@gmail.com - git@bitbucket.org:trebor74hr/reedwolf.git)
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Optional

from .meta import (
        TypeInfo,
        )
from ..base import (
        ComponentBase,
        PY_INDENT,
        )
from ..types import STANDARD_TYPE_LIST
from ..utils import (
        snake_case_to_camel,
        )
from ..meta import (
        is_enum,
        )
from ..components import Field, FieldGroup, ChoiceField
from ..containers import SubEntityItems, Entity

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
    enums : Set[str] = field(init=False, default_factory=set)


    def get(self, name:str) -> Optional[(DumpPydanticClassLines, int)]:
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
    def create_pydantic_class_declaration(indent_level, name, label=""):
        out = []
        out.append(f"{PY_INDENT*indent_level}class {name}(BaseModel):")
        if label:
            out.append(f'{PY_INDENT*(indent_level+1)}""" {label} """')
            out.append("")
        return out


# ------------------------------------------------------------

def dump_pydantic_models(component:ComponentBase, fname:str):
    code = dump_pydantic_models_to_str(component=component)
    with open(fname, "w") as fout:
        fout.write(code)
        len_lines = len(code.splitlines())
        print(f"Output in {fname}, {len_lines} lines.")
    return

# ------------------------------------------------------------

def dump_pydantic_models_to_str(
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

    indent = PY_INDENT * depth
    indent_next = PY_INDENT * (depth+1)


    # class_py_name = None
    # class_py_type = None

    lines = []
    vars_declarations = []

    children = self.get_children()

    if isinstance(self, (FieldGroup, Entity, SubEntityItems)):
        py_name = f"{snake_case_to_camel(self.name)}DTO"
        py_type = py_name
        # make copy
        path = path[:] + [py_type]
        py_type_ext = snake_case_to_camel(".".join(path))
        # vars_declarations will be consumed in owner, so
        # only for top object it won't be consumed

        if isinstance(self, SubEntityItems):
            assert isinstance(self.bound_attr_node.data, TypeInfo), self.bound_attr_node.data
            if self.bound_attr_node.data.is_list:
                py_type_ext = f"List[{py_type_ext}]"
            if self.bound_attr_node.data.is_optional:
                py_type_ext = f"Optional[{py_type_ext}]"
        # vars_declarations.append(f'{indent}{self.name}: {py_type_ext}')
        vars_declarations.append(f'{indent}{DumpPydanticClassLinesStore.create_pydantic_var_declaration(self.name, py_type_ext, title=self.label)}')

        lines.append("")
        # lines.append("")
        # lines.append(f"{indent}class {py_name}(BaseModel):")
        lines.extend(DumpPydanticClassLinesStore.create_pydantic_class_declaration(
                        depth,
                        py_name,
                        label=self.label))

    elif isinstance(self, (Field,)):
        py_name = self.name
        todo_comment = ""
        if self.bound_attr_node:
            assert isinstance(self.bound_attr_node.data, TypeInfo)
            py_type_klass = self.bound_attr_node.data.type_
            if is_enum(py_type_klass):
                py_type = f"{py_type_klass.__name__}"
                class_dump_store.enums.add(py_type)
            elif isinstance(self, ChoiceField) and self.choice_label_type_info is not None:
                parent_klass_full_name = self.choice_label_type_info.parent_object.__name__
                parent_klass_name = parent_klass_full_name.split(".")[-1]
                # value_klass_name = py_type_klass.__name__
                value_klass_name = self.choice_value_type_info.type_.__name__
                # should be string
                # label_klass_name = self.choice_label_type_info.type_
                py_type = f"{snake_case_to_camel(parent_klass_name)}ChoiceDTO"
                lines.append("")
                # lines.append(f"{indent}class {py_type}(BaseModel):")
                label = f"Choice type for {self.name}"
                lines.extend(DumpPydanticClassLinesStore.create_pydantic_class_declaration(
                                depth,
                                py_type,
                                label=label))
                lines.append(f"{indent_next}value: {value_klass_name}")
                lines.append(f"{indent_next}label: str")
                # lines.append("")
            elif py_type_klass in STANDARD_TYPE_LIST:
                py_type = f"{py_type_klass.__name__}"
            else:
                todo_comment=f"TODO: domain_dataclass {py_type_klass.__name__}"
                py_type = "Any"

            # type hint options
            if self.bound_attr_node.data.is_list:
                py_type = f"List[{py_type}]"
            if self.bound_attr_node.data.is_optional:
                py_type = f"Optional[{py_type}]"
        else:
            # todo_comment = f"  # TODO: unbound {self.bind}"
            todo_comment = f"TODO: unbound {self.bind}"
            py_type = "Any"

        # vars_declarations.append(f"{indent}{py_name}: {py_type}{todo_comment}")
        vars_declarations.append(f'{indent}{DumpPydanticClassLinesStore.create_pydantic_var_declaration(py_name, py_type, todo_comment, title=self.label)}')

        if children:
            lines.append("")
            # lines.append("")
            class_py_name = f"{self.name}_fieldgroup"
            class_py_type = f"{snake_case_to_camel(self.name)}DetailsDTO"
            # lines.append(f"{indent}class {class_py_type}(BaseModel):")
            label = f"Component beneath type {self.name}"
            lines.extend(DumpPydanticClassLinesStore.create_pydantic_class_declaration(
                            depth,
                            class_py_type,
                            label=label))

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
        # py_name=py_name, py_type=py_type,
        # class_py_name=class_py_name,
        # class_py_type=class_py_type,
        lines=lines, vars_declarations=vars_declarations))


    for component in children:
        # recursion
        dump_pydantic_models_to_str(component, class_dump_store=class_dump_store, path=path[:], depth=depth+1)
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
            "# IMPORTANT: DO NOT EDIT!!! The code is generated by reedwolf.entities system,",
            "#            rather change entities.py and regenerate the code.",
            "# --------------------------------------------------------------------------------",
            "from datetime import date, datetime  # noqa: F401",
            "from decimal import Decimal",
            "from typing import Any, List, Optional",
            "from pydantic import BaseModel, Field",
            # '',
            # '# to allow classes as type_hints used before actually declared',
            # '# At least Python 3.7 version is required, got: {sys.version_info[:2]}',
            # '# see: https://stackoverflow.com/questions/61544854/from-future-import-annotations',
            ])
        all_lines.append("from domain.cloud.enum import (")
        for enum_name in sorted(class_dump_store.enums):
            all_lines.append(f"    {enum_name},")
        all_lines.append(")")
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


