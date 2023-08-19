import ast
from dataclasses import dataclass
from typing import Optional

from ..utils import error, info
from .ast_models import FunctionFromCode, ModelAttrFromCode, SourceCode
from .base import ModelFromCode, ModelsParserBase


@dataclass
class SqlalchemyTablesParser(ModelsParserBase):
    @classmethod
    def extract_sa_table_model_source_dict_from_code(  # noqa: C901
        cls,
        model_path_list: list[str],
        file_path: str,
        verbose: bool = False,
    ) -> dict[str, ModelFromCode]:
        """
        Example:
        COMPANY_TABLE = Table(
            "company",
            metadata_obj,
            Column("id", Integer, primary_key=True, index=True, unique=True),
            Column("alive", Boolean, default=True, nullable=False),
            ...
            )
        """
        assert model_path_list
        assert file_path

        head_node, lines = cls.parse_code_by_filepath(file_path)
        model_package = f"{cls.__name__}::{'.'.join(model_path_list)}"
        if verbose:
            info(f"={model_package}")
        source = SourceCode(model_path_list, file_path, lines)
        class_model_source_dict = {}

        for assign_node in head_node.body:
            if not (
                isinstance(assign_node, (ast.Assign,))
                and isinstance(assign_node.targets[0], ast.Name)
                and isinstance(assign_node.value, ast.Call)
            ):
                continue

            table_func = FunctionFromCode.from_ast_node(assign_node.value, lines)
            table_instance_name = table_func.name_path[0]
            if not (
                # TODO: this is specific ...
                table_func.name_path[-1] == "Table"
                and len(table_func.args) > 0
                and len(table_func.kwargs) == 0
            ):
                continue

            if verbose:
                info(f"  {table_instance_name}")

            attr_list = []
            sa_table_name = None
            for nr, table_arg in enumerate(table_func.node.args, 1):
                if nr == 1:
                    if not (
                        isinstance(table_arg, ast.Constant) and isinstance(table_arg.value, str)
                    ):
                        raise ValueError(
                            f"{table_func}: Expecting table name string argument ast first Table() argument, got: {table_arg}.\n  File: {file_path} "
                        )
                    sa_table_name = table_arg.value.lower()
                    continue

                if not isinstance(table_arg, ast.Call):
                    continue
                func = FunctionFromCode.from_ast_node(table_arg, lines)
                col_name = func.args and func.args[0]
                if not (func and func.name_path[-1] in ("Column",)):
                    continue
                assert col_name and isinstance(col_name, str)
                attr = ModelAttrFromCode(
                    name=col_name,
                    ann=None,
                    func=func,
                    node=table_arg,
                )
                if verbose:
                    info(f"    {attr}")
                attr_list.append(attr)

            if not sa_table_name:
                raise ValueError(f"{table_func}: Table name not found.\n  File: {file_path}")
            if not attr_list:
                raise ValueError(f"{table_func}: No attributes found.\n  File: {file_path}")

            model_from_source = ModelFromCode(
                name=sa_table_name, node=table_func.node, sourcecode=source
            )
            for attr in attr_list:
                model_from_source.add_attr(attr)

            model_from_source.finish()

            if model_from_source.name in class_model_source_dict:
                error(f"In {file_path} class {model_from_source.name} defined at least twice")

            class_model_source_dict[model_from_source.name] = model_from_source

        if verbose:
            info("=" * 80)
            info("ALL OK")
            info("=" * 80)

        return class_model_source_dict

    @classmethod
    def should_process_class(cls, cls_node: ast.ClassDef, lines: list[str]) -> tuple[bool, bool]:
        """
        returns found_model_base, ignored_base
        return True, False
        """
        raise Exception("Should not be used")

    @classmethod
    def parse_cls_attr(
        cls, node: ast.AST, class_name: str, lines: list[str]
    ) -> Optional[list[ModelAttrFromCode]]:
        raise Exception("Should not be used")
