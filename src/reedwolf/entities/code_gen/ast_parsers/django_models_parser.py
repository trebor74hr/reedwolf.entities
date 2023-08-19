import ast
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional, List, Dict, Tuple

from ...utils import fix_print
from ..utils import error, info, right_strip
from .ast_models import FunctionFromCode, ModelAttrFromCode
from .base import ModelFromCode, ModelsParserBase


@dataclass
class DjangoModelsParser(ModelsParserBase):
    # TODO: this should be on model_path base
    CLASS_BASES_TO_MATCH: ClassVar[List[str]] = [
        "models.Model",
        "Model",
    ]

    IGNORE_BASES: ClassVar[List[str]] = [
        "Enum",
        "Exception",
        "Manager",
        "models.Manager",
    ]

    # ALT: for path in Path(django_project_root_path).glob('**/models.py'):
    # NOTE: helper: find . -name models.py # as of RL 2203xx
    MODELS_LIST: ClassVar[List[List[str]]] = [
        ["models.py"],
    ]

    @classmethod
    def extract_all_dj_model_model_source_dict_from_code(
        cls, project_root_path: str, verbose: bool = False
    ) -> Dict[str, ModelFromCode]:
        models: Dict[str, ModelFromCode] = {}
        info(f"Processing django project repo at: {project_root_path}")
        for model_path_list in cls.MODELS_LIST:
            path = Path(project_root_path, *model_path_list)
            file_path: str = str(path)
            if not path.exists():
                error(
                    f"Model {file_path} does not exists, check and update MODELS_LIST in {__file__}."
                )
            # file_path=file_path,
            with open(file_path) as file_in:
                code = file_in.read()

            # adjust 2to3 - print statement -> print() function
            #   e.g. administration/models.py
            #       print "wrong type {}".format(...)
            code = "\n".join([fix_print(line) for line in code.splitlines()])

            module_path_list: List[str] = list(model_path_list)
            module_path_list[-1] = right_strip(module_path_list[-1], ".py", strict=True)
            models_for_file = cls.extract_class_model_source_dict_from_code(
                module_path_list=module_path_list,
                code=code,
                file_path=file_path,
                verbose=verbose,
            )
            # TODO: check for duplicates
            in_both = set(models_for_file.keys()).intersection(set(models.keys()))
            if in_both:
                raise NameError(
                    f"In {file_path} found some duplicate class/table/model names: {in_both}"
                )
            models.update(models_for_file)

        return models

    @classmethod
    def should_process_class(cls, cls_node: ast.ClassDef, lines: List[str]) -> Tuple[bool, bool]:
        found_model_base, ignored_base = cls.match_base_class(
            cls_node,
            class_bases_to_match=cls.CLASS_BASES_TO_MATCH,
            ignore_bases=cls.IGNORE_BASES,
            lines=lines,
        )
        return found_model_base, ignored_base

    # ------------------------------------------------------------

    @classmethod
    def parse_cls_attr(
        cls, attr_node: ast.AST, class_name: str, lines: List[str]
    ) -> Optional[List[ModelAttrFromCode]]:
        """
        Parses django model class attributes.

        Example - administration/models.py:

            class CompanyVatNumber(AdminSetter, SoftDeletionModel):
                country = models.CharField(max_length=100, null=True, blank=True)
                prefix = models.CharField(max_length=10, null=True, blank=True)
                example = models.CharField(max_length=90, null=True, blank=True)
                min_character = models.SmallIntegerField(blank=True, default=0)
                max_character = models.SmallIntegerField(blank=True, default=0)

                class Meta:
                    db_table = 'vat_number_examples'
                    ordering = ["country"]

        """

        if not (
            isinstance(attr_node, (ast.Assign,))
            and isinstance(attr_node.targets[0], ast.Name)
            and isinstance(attr_node.value, ast.Call)
        ):
            return None  # ignore

        target_first: ast.expr = attr_node.targets[0]
        assert isinstance(target_first, ast.Name)
        ast_name: ast.Name = target_first
        attr_name: str = ast_name.id
        attr_func_node = attr_node.value

        func = FunctionFromCode.from_ast_node(attr_func_node, lines) if attr_func_node else None
        if func and not (
            func.name_path[0] in ("models",)
            or func.name_path[-1].endswith("Field")
            or func.name_path[-1] in ("HistoricalRecords", "DeviceStatusManager")
        ):
            return None  # ignore function

        return [
            ModelAttrFromCode(
                name=attr_name,
                ann=None,
                func=func,
                node=attr_node,
            )
        ]
