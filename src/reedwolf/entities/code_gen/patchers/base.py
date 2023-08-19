from abc import abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Any, Dict, Tuple, List

try:
    import sqlalchemy
except ImportError:
    # used just for type hints
    SAColumn = Any
    SATable = Any
    SATypeEngine = Any
else:
    from sqlalchemy.sql.schema import Column as SAColumn, Table as SATable
    from sqlalchemy.sql.type_api import TypeEngine as SATypeEngine

from ...utils import list_to_str_limited

from ..ast_parsers.ast_models import ModelAttrFromCode
from ..ast_parsers.base import ModelFromCode
from ..utils import DataclassModel, error, info, info_debug

from .table_models import Column, Table


@dataclass
class PatcherDiff:
    name: str
    same: List[str]
    source_only: List[str]
    target_only: List[str]
    diff_dict: Dict[str, Tuple[Column, Column]]
    # computed
    columns_count: int = field(init=False)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, {', '.join(self.report_as_str_list())}) "

    def report_as_str_list(self) -> List[str]:
        out: List[str] = [f"columns    = {self.columns_count:3d}"]
        if self.source_only or self.target_only or self.diff_dict:
            out.append(f"same       = {len(self.same):3d}")
            if self.source_only:
                out.append(f"source_only= {len(self.source_only):3d}")
            if self.target_only:
                out.append(f"target_only= {len(self.target_only):3d}")
            if self.diff_dict:
                out.append(f"different  = {len(self.diff_dict):3d}")
        else:
            out.append(f"ALL SAME= {len(self.same)}")
        return out

    def __post_init__(self) -> None:
        self.columns_count = (
            len(self.same) + len(self.source_only) + len(self.target_only) + len(self.diff_dict)
        )


@dataclass
class PatcherBase:
    name: str
    # set for all - from sqlalchemy registry / mappings
    # (although not used in all Patcher* classes)
    sa_table: SATable
    dc_model: DataclassModel
    sa_name_to_dc_model_dict: Dict[str, DataclassModel]
    dj_name_to_dc_model_dict: Dict[str, DataclassModel]

    verbose: int = field(repr=False, default=1)

    TARGET_CLASS_DECLARED_ATTRS: ClassVar[bool] = True

    def compare(
        self,
        source_name: str,
        source_table: Table,
        target_name: str,
        target_table: Table,
        verbose: int = 1,
    ) -> PatcherDiff:
        source_col_names = set(source_table.columns_dict.keys())
        target_col_names = set(target_table.columns_dict.keys())

        source_only: List[str] = list(source_col_names - target_col_names)
        target_only: List[str] = list(target_col_names - source_col_names)
        same: List[str] = []
        diff_dict: Dict[str, Tuple[Column, Column]] = {}

        for col_name in target_col_names.intersection(source_col_names):
            # Column("access_id", ForeignKey("administration_companyaccess.id", deferrable=True, initially="DEFERRED"), nullable=False),
            source_column = source_table.columns_dict[col_name]
            target_column = target_table.columns_dict[col_name]
            # NOTE: source must be first, target must be second
            if source_column.equals_to_target(target_column, verbose=verbose):
                same.append(col_name)
            else:
                diff_dict[col_name] = (source_column, target_column)

        if verbose:
            msg: List[str] = []
            max_cols_str_len = 80
            if verbose >= 2 and same:
                msg.append(
                    f"Same    ({len(same):2d}): {list_to_str_limited(same, max_cols_str_len)}"
                )
            if source_only:
                msg.append(
                    f"{source_name} only ({len(source_only):2d}): {list_to_str_limited(source_only, max_cols_str_len)}"
                )
            if target_only:
                msg.append(
                    f"{target_name} only ({len(target_only):2d}): {list_to_str_limited(target_only, max_cols_str_len)}"
                )
            if diff_dict:
                msg.append(
                    f"DIFFS** ({len(diff_dict.keys()):2d}): {list_to_str_limited(list(diff_dict.keys()), max_cols_str_len)}"
                )
            if msg:
                msg.insert(0, f"==== Model '{self.name}' - diff result:")
            else:
                msg.insert(0, f"==== Model '{self.name}' - no differences")
            info("\n".join(msg))

            # NOTE: not possible anymore - needs attr_from_code. ModelAttrFromCode
            # if verbose >= 2:
            #     for col_name, (source_column, target_column) in diff_dict.items():
            #         info(
            #             f" {source_name}: {self.column_to_target_pycode_str(source_column)}"
            #             + f"\n       {target_name}: {self.column_to_target_pycode_str(target_column, None)}"
            #         )

        return PatcherDiff(
            name=self.name,
            same=same,
            source_only=source_only,
            target_only=target_only,
            diff_dict=diff_dict,
        )

    @abstractmethod
    def column_to_target_pycode_str(
        self, column: Column, attr_from_code: ModelAttrFromCode
    ) -> str:
        raise NotImplementedError()

    def patch_differences(
        self, patcher_diff: PatcherDiff, target_model_source: ModelFromCode, verbose: int = 1
    ) -> int:
        cnt_columns_patched = 0  # noqa: SIM113
        assert patcher_diff.diff_dict

        for col_name, (target_column, source_column) in patcher_diff.diff_dict.items():
            col_name_to_find = col_name
            if col_name_to_find not in target_model_source.attrs_dict:
                if target_column.fk_column_raw:
                    col_name_to_find = target_column.fk_column_raw.name
                elif source_column.fk_column_raw:
                    col_name_to_find = source_column.fk_column_raw.name
                if col_name_to_find not in target_model_source.attrs_dict:
                    error(
                        f"column {col_name_to_find} not found in source, available: "
                        f"{', '.join(target_model_source.attrs_dict.keys())}"
                    )
            src_attr = target_model_source.attrs_dict[col_name_to_find]
            source = target_model_source.sourcecode

            lineno_start: int = src_attr.node.lineno
            assert src_attr.node.end_lineno
            lineno_end: int = src_attr.node.end_lineno

            col_src_lines_orig = source.get_lines(lineno_start, lineno_end)
            assert col_src_lines_orig

            # new python code
            col_src_line_new = self.column_to_target_pycode_str(
                column=target_column, attr_from_code=src_attr
            )

            col_src_line1_stripped = " ".join(col_src_lines_orig[:2]).lstrip(" ")
            col_src_orig_indent = len(col_src_lines_orig[0]) - len(col_src_line1_stripped)
            col_src_line_new = f"{' ' * col_src_orig_indent}{col_src_line_new}"
            if self.TARGET_CLASS_DECLARED_ATTRS:
                if not col_src_line1_stripped.startswith(target_column.name):
                    error(
                        f"{self.name}: {col_src_line1_stripped} does not start with {target_column.name}"
                    )
            else:
                if col_name_to_find not in col_src_line1_stripped:
                    error(
                        f"{self.name}: {col_src_line1_stripped} does not contain {col_name_to_find}"
                    )

            if verbose >= 2:
                info_debug(
                    f"About to update: {src_attr} / {col_src_lines_orig} "
                    f"==> {col_src_orig_indent} + {col_src_line_new}"
                )

            source.patch_lines(
                lineno_start=lineno_start,
                lineno_end=lineno_end,
                patched_lines=[col_src_line_new],
            )
            cnt_columns_patched += 1

        return cnt_columns_patched

    @abstractmethod
    def compare_and_patch_diffs(
        self,
        target_model_source_dict: Dict[str, Dict[str, ModelFromCode]],
        source_model_source_dict: Dict[str, ModelFromCode],
        verbose: int = 1,
    ) -> PatcherDiff:
        """returns number of column processed"""
        raise NotImplementedError()

@dataclass()
class PatcherStats:
    files_processed: int
    models_processed: int
    columns_processed: int

    files_patched: int
    models_patched: int
    columns_patched: int

    def report_as_str_list(self) -> List[str]:
        out: List[str] = []
        out.append("-------- patched / processed")
        out.append(f"Files  : {self.files_patched} / {self.files_processed}")
        out.append(f"Models : {self.models_patched} / {self.models_processed}")
        out.append(f"Columns: {self.columns_patched} / {self.columns_processed}")
        return out

