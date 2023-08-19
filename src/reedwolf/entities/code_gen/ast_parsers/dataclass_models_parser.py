import ast
from dataclasses import dataclass
from typing import Optional

from ..utils import warn
from .ast_models import (
    AnnotationFromCode,
    FunctionFromCode,
    ModelAttrFromCode,
    extract_code,
    process_node,
)
from .base import ModelsParserBase


@dataclass
class DataclassModelsParser(ModelsParserBase):
    @classmethod
    def should_process_class(cls, cls_node: ast.ClassDef, lines: list[str]) -> tuple[bool, bool]:
        dc_model_allowed_name_paths: list[str] = [
            "Base",
            "IsoHeader",
        ]
        found_model_base, ignored_base = cls.match_base_class(
            cls_node,
            class_bases_to_match=dc_model_allowed_name_paths,
            ignore_bases=None,
            lines=lines,
        )
        if not found_model_base:
            found_model_base = cls.has_dataclass_decorator(cls_node, lines=lines)

        return found_model_base, ignored_base

    # ------------------------------------------------------------

    @classmethod
    def parse_cls_attr(
        cls, attr_node: ast.AST, class_name: str, lines: list[str]
    ) -> Optional[list[ModelAttrFromCode]]:
        """
        Parses CWA Domain @dataclass class attributes.

        Example:

            @dataclass(eq=False)
            class CompanyVatNumber(Base):
                ''' CompanyVatNumber model '''

                max_character: int = 0
                min_character : int = 0
                example : Optional[str] = field(default_factory=lambda: None)
                prefix : Optional[str] = field(default_factory=lambda: None)
                country : Optional[str] = field(default_factory=lambda: None)
                id: int = field(default_factory=lambda: None)

        """
        if not (isinstance(attr_node, ast.AnnAssign) and isinstance(attr_node.target, ast.Name)):
            return None  # ignore

        attr_name = attr_node.target.id
        if isinstance(attr_node.annotation, ast.Name):
            # AnnAssign(target=Name(id='first_name', ctx=Store()),
            #           annotation=Name(id='str', ctx=Load()),
            attr_ann_type = attr_node.annotation.id
        elif isinstance(attr_node.annotation, ast.Attribute):
            # AnnAssign(target=Name(id='last_login', ctx=Store()),
            #           annotation=Attribute(value=Name(id='datetime', ctx=Load()), attr='datetime', ctx=Load()),
            #           value=Call(func=Name(id='field', ctx=Load()), args=[], keywords=[keyword(arg='default_factory', value=Lambda(args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=Constant(value=None)))]),
            result_list = process_node(attr_node.annotation, lines)
            assert len(result_list) == 1
            attr_ann_type = result_list[0]
        elif isinstance(attr_node.annotation, ast.Subscript):
            # AnnAssign(target=Name(id='company_packing_models', ctx=Store()),
            #           annotation=Subscript(value=Name(id='list', ctx=Load()), slice=Name(id='CompanyPackingModel', ctx=Load()), ctx=Load()),
            #           value=Call(func=Name(id='field', ctx=Load()), args=[], keywords=[keyword(arg='default_factory', value=Name(id='list', ctx=Load()))]),
            attr_ann_type = extract_code(attr_node.annotation, lines, verbose=False)
        elif isinstance(attr_node.annotation, ast.Constant):
            # annotation=Constant(value='Company'), simple=1)
            attr_ann_type = attr_node.annotation.value
        else:
            warn(ast.dump(attr_node.annotation))
            raise Exception(
                "AnnotationFromCode -> Unknown node type: "
                + extract_code(attr_node.annotation, lines, verbose=True)
            )

        attr_func_node = attr_ann_default = None
        if isinstance(attr_node.value, ast.Call):
            # = field()
            attr_func_node = attr_node.value
        elif isinstance(attr_node.value, ast.Constant):
            # = 0
            attr_ann_default = attr_node.value.value

        func = FunctionFromCode.from_ast_node(attr_func_node, lines) if attr_func_node else None

        if func and func.name_path[-1] not in ("field",):
            # TODO: report warning or parse this other function too?
            func = None

        return [
            ModelAttrFromCode(
                name=attr_name,
                ann=AnnotationFromCode(type=attr_ann_type, default=attr_ann_default)
                if attr_ann_type
                else None,
                func=func,
                node=attr_node,
            )
        ]
