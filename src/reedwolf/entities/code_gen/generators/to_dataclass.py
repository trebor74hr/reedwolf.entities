from dataclasses import (
        dataclass, 
        )
from typing import (
        List, 
        Optional,
        Dict,
        ClassVar,
        Tuple,
        )

from ...utils import (
        PY_INDENT,
        )
from ...base import (
        ComponentBase,
        )

from .base import (
        ClassDeclarationBase,
        VariableDeclarationBase,
        ComponentDumpBase,
        FileDumpBase,
        DumpToBase,
        dump_to_models_as_dict,
        dump_to_models,
        CodeStringType,
        )

# ------------------------------------------------------------

class ClassDeclaration(ClassDeclarationBase):

    def dump_to_strlist(self) -> List[str]:
        lines = []
        lines.append("@dataclass")
        lines.append(f"class {self.name}:")
        if self.title:
            lines.append(f'{PY_INDENT}""" {self.title} """')
        return lines

# ------------------------------------------------------------

class VariableDeclaration(VariableDeclarationBase):

    def dump_to_str(self) -> str:
        out = f"{self.name}: {self.class_name}"
        if self.title:
            out = f'{out} = field(metadata=dict(title="{self.title}"))'
        if self.comment:
            out = f"{out}  # {self.comment}"
        return out


# ------------------------------------------------------------

class ComponentDataclassDump(ComponentDumpBase):
    ...

# ------------------------------------------------------------

@dataclass 
class FileDataclassDump(FileDumpBase):

    def dump_to_str_fill_imports(self) -> List[str]:
        return [
            "from dataclasses import dataclass, field",
            ]


# ------------------------------------------------------------

@dataclass
class DumpToDataclass(DumpToBase):

    KlassClassDeclaration       : ClassVar[type] = ClassDeclaration       
    KlassVariableDeclaration    : ClassVar[type] = VariableDeclaration    
    KlassComponentDump          : ClassVar[type] = ComponentDataclassDump  
    KlassFileDump               : ClassVar[type] = FileDataclassDump       

# ------------------------------------------------------------

def dump_to_dataclass_models_as_dict(
        component:ComponentBase, 
        file_split_to_depth: Optional[int] = 1,
        flatten: bool = False,
        deps_order: bool = False,
        ) -> Tuple[DumpToBase, Dict[str, CodeStringType]]:
    return dump_to_models_as_dict(
        KlassDumpTo=DumpToDataclass,
        component=component,
        file_split_to_depth=file_split_to_depth,
        flatten=flatten,
        deps_order=deps_order,
        )


# ------------------------------------------------------------

def dump_to_dataclass_models(
        component:ComponentBase, 
        fname_or_dname:str,
        file_split_to_depth: Optional[int] = 1,
        flatten: bool = False,
        deps_order: bool = False,
        add_init_py: bool = False,
        ) -> Tuple[DumpToBase, Dict[str, CodeStringType]]:
    return dump_to_models(
        KlassDumpTo=DumpToDataclass,
        fname_or_dname=fname_or_dname,
        component=component,
        file_split_to_depth=file_split_to_depth,
        flatten=flatten,
        deps_order=deps_order,
        add_init_py=add_init_py,
        )
