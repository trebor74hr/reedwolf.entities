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

from .base import (
        DelarationCodeLinesType,
        THIS_PACKAGE,
        PyType,
        ClassDeclarationBase,
        VariableDeclarationBase,
        ComponentDumpBase,
        FileDumpBase,
        CodegenStackFrame,
        DumpToBase,
        )


THIS_MODULE = os.path.basename(__file__).split(".")[0]

# ------------------------------------------------------------

class ClassDeclaration(ClassDeclarationBase):

    def dump_to_strlist(self) -> List[str]:
        lines = []
        lines.append(f"class {self.name}(BaseModel):")
        if self.title:
            lines.append(f'{PY_INDENT}""" {self.title} """')
        return lines

# ------------------------------------------------------------

class VariableDeclaration(VariableDeclarationBase):

    def dump_to_str(self) -> str:
        # NOTE: when including:
        #           from __future__ import annotations
        #       then adding term here:
        #           'not self.deps_order and'
        #       will make get_type_hints() fail, since it does not recognize
        #       (must be class full path).
        if not self.deps_order and not self.flatten and self.class_name_full_path:
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

class ComponentPydanticDump(ComponentDumpBase):
    ...

# ------------------------------------------------------------

@dataclass 
class FilePydanticDump(FileDumpBase):

    def dump_to_str(self) -> str:
        all_lines = []
        if not self.deps_order:
            all_lines.append("from __future__ import annotations")

        all_lines.extend([
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

        comp_dump_list = self.comp_dump_dict.values()

        comp_dump_first = list(comp_dump_list)[0]

        if self.flatten and self.deps_order:
            comp_dump_list = reversed(comp_dump_list)

        for comp_dump in comp_dump_list:
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
class DumpToPydantic(DumpToBase):

    KlassClassDeclaration       : ClassVar[type] = ClassDeclaration       
    KlassVariableDeclaration    : ClassVar[type] = VariableDeclaration    
    KlassComponentDump          : ClassVar[type] = ComponentPydanticDump  
    KlassFileDump               : ClassVar[type] = FilePydanticDump       
    
# ------------------------------------------------------------

def dump_to_pydantic_models_as_dict(
        component:ComponentBase, 
        file_split_to_depth: Optional[int] = 1,
        flatten: bool = False,
        deps_order: bool = False,
        ) -> Dict[str, str]:
    """
    file_split_to_depth - more details in base.DumpToBase
    flatten - more details in base.DumpToBase
    deps_order - more details in base.DumpToBase
    """

    if not (file_split_to_depth is None or file_split_to_depth >= 1):
        raise EntityInternalError(owner=self, msg=f"file_split_to_depth argument needs to be integer >= 1 or None (for all levels). Got: {file_split_to_depth}") 

    dumper = DumpToPydantic(
                file_split_to_depth=file_split_to_depth,
                flatten=flatten,
                deps_order=deps_order,
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
        flatten: bool = False,
        deps_order: bool = False,
        ) -> str:
    """
    file_split_to_depth - more details in base.DumpToBase
    flatten - more details in base.DumpToBase
    deps_order - more details in base.DumpToBase
    """
    lines_by_file = dump_to_pydantic_models_as_dict(
                        component=component,
                        file_split_to_depth=file_split_to_depth,
                        flaten=flatten,
                        deps_order=deps_order,
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
        # dump in several files
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


