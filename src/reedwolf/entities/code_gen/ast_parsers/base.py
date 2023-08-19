import ast
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..utils import error, info, warn
from .ast_models import FunctionFromCode, ModelAttrFromCode, SourceCode, process_node


@dataclass
class ModelFromCode:
    name: str
    sourcecode: SourceCode
    node: ast.AST
    attrs: list[ModelAttrFromCode] = field(default_factory=list, init=False, repr=False)
    attrs_dict: dict[str, ModelAttrFromCode] = field(init=False, repr=False, default_factory=dict)
    finished: bool = field(init=False, default=False, repr=False)

    def add_attr(self, attr: ModelAttrFromCode) -> None:
        assert not self.finished
        assert attr
        self.attrs.append(attr)
        assert attr.name not in self.attrs_dict, attr.name
        self.attrs_dict[attr.name] = attr

    def finish(self) -> None:
        assert not self.finished
        self.finished = True

    def get_attr(self, name: str) -> Optional[ModelAttrFromCode]:
        assert self.finished
        return self.attrs_dict.get(name, None)  # noqa: SIM910


class ModelsParserBase:
    @classmethod
    @abstractmethod
    def parse_cls_attr(
        cls, attr_node: ast.AST, class_name: str, lines: list[str]
    ) -> Optional[list[ModelAttrFromCode]]:
        pass

    @classmethod
    def extract_class_model_source_dict_from_code_by_filepath(  # noqa: C901
        cls,
        module_path_list: list[str],
        file_path: str,
        verbose: bool = False,
    ) -> dict[str, ModelFromCode]:
        assert file_path
        with open(file_path) as fout:
            code = fout.read()
        return cls.extract_class_model_source_dict_from_code(
            module_path_list=module_path_list,
            code=code,
            file_path=file_path,
            verbose=verbose,
        )

    @classmethod
    def extract_class_model_source_dict_from_code(  # noqa: C901
        cls,
        module_path_list: list[str],
        code: str,
        file_path: str,
        verbose: bool = False,
    ) -> dict[str, ModelFromCode]:
        assert module_path_list

        # head_node, lines = cls.parse_code_by_filepath(file_path)
        head_node, lines = cls.parse_code(code=code, name=file_path)

        model_package = f"{cls.__name__}::{'.'.join(module_path_list)}"
        if verbose:
            info(f"={model_package}")

        source = SourceCode(module_path_list=module_path_list, file_path=file_path, lines=lines)

        class_model_source_dict = {}

        for cls_node in head_node.body:
            if not isinstance(cls_node, ast.ClassDef):
                # only classes are parsed
                continue
            class_name = f"{cls_node.name}"
            if not cls_node.bases and class_name.endswith("Mixin"):
                continue

            if verbose:
                info(f"  {class_name}")

            do_process, ignored_base = cls.should_process_class(cls_node, lines=lines)
            if not do_process:
                if not ignored_base:
                    warn(f"Model {model_package}.{class_name} has no valid base class.")
                continue

            model_from_source = ModelFromCode(name=class_name, node=cls_node, sourcecode=source)

            for attr_node in cls_node.body:
                attr_list = cls.parse_cls_attr(attr_node, class_name, lines)
                if attr_list:
                    for attr in attr_list:
                        if verbose:
                            info(f"    {attr}")
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
    def parse_code_by_filepath(cls, file_path: str) -> tuple[ast.Module, list[str]]:
        """returns head AST node and sourcecode as list of strings"""
        with open(file_path) as fout:
            code = fout.read()
        return cls.parse_code(code, name=file_path)

    @staticmethod
    def parse_code(code: str, name: str) -> tuple[ast.Module, list[str]]:
        """returns head AST node and sourcecode as list of strings"""
        try:
            # ALT: ast.compile()
            head_node = ast.parse(code)
        except Exception as ex:
            raise Exception(f"Model {name} not ast parsable, got: {ex}")
        lines = code.splitlines()
        return head_node, lines

    @staticmethod
    def match_base_class(
        cls_node: ast.ClassDef,
        class_bases_to_match: list[str],
        ignore_bases: Optional[list[str]],
        lines: list[str],
    ) -> tuple[bool, bool]:
        """
        Returns:
            found_model_base
            ignored_base - should be interpreted only when found_model_base
        """
        assert class_bases_to_match
        for acnp in class_bases_to_match:
            assert isinstance(acnp, str)

        found_model_base = False
        ignored_base = False

        for base in cls_node.bases:
            # searching only classes that inherit models.Model or Model
            result_list = process_node(base, lines)
            name_path = ".".join(result_list)
            if ignore_bases and name_path and name_path in ignore_bases:
                ignored_base = True
                break

            if name_path and name_path in class_bases_to_match:
                found_model_base = True
                break

        return found_model_base, ignored_base

    @staticmethod
    def has_dataclass_decorator(cls_node: ast.ClassDef, lines: list[str]) -> bool:
        # check @dataclass decorator exists?
        found_model_base = False
        if cls_node.decorator_list:
            for dec_node in cls_node.decorator_list:
                if not isinstance(dec_node, ast.Call):
                    continue
                func = FunctionFromCode.from_ast_node(dec_node, lines)
                if func.name_path[-1] == "dataclass":
                    found_model_base = True
                    break
        return found_model_base

    @staticmethod
    @abstractmethod
    def should_process_class(cls_node: ast.ClassDef, lines: list[str]) -> tuple[bool, bool]:
        """
        Returns:
            found_model_base
            ignored_base - should be interpreted only when found_model_base
        """
        raise NotImplementedError("abstract function")
