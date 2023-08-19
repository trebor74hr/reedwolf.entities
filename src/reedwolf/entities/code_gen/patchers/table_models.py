from __future__ import annotations

import ast
import inspect
from dataclasses import MISSING as DC_MISSING
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, ClassVar, Iterable, Optional, Type

from ...meta import TypeInfo, extract_py_type_hints, is_enum
from ...utils import MISSING, get_available_names_example

from ..ast_parsers.ast_models import (
    FunctionFromCode,
    ModelAttrFromCode,
    ast_call_node_add_or_update_kw_arg,
    ast_call_node_create_kw_arg_if_not_exists,
    ast_call_node_update_positional_arg,
)
from ..ast_parsers.base import ModelFromCode
from ..utils import DataclassModel, error, info, warn
from .base import SATable, SAColumn, SATypeEngine

SA_TO_PY_TYPE_MAP: dict[str, str] = {
    "SmallInteger": "int",
    "String": "str",
    "DateTime": "datetime",
}

PY_INDENT = "    "


def literal_to_str(value: Any) -> str:
    return repr(value) if isinstance(value, str) else str(value)


def map_sa_to_py_type_as_str(sa_type: Any) -> str:
    return (
        SA_TO_PY_TYPE_MAP[sa_type.__class__.__name__]
        if sa_type.__class__.__name__ in SA_TO_PY_TYPE_MAP
        else str(sa_type.__name__)
    )


@dataclass()
class TypeMap:
    dj_type: str
    sa_type: str
    py_type: type
    is_sa_main: bool = True
    is_py_main: bool = True


class TableSourceEnum(str, Enum):
    FROM_SQLALCHEMY = "SA"
    FROM_DJANGO = "DJ"
    FROM_DATACLASS = "DC"


@dataclass
class Column:
    name: str
    source: TableSourceEnum
    meta: dict[str, Any]

    # -- type
    py_type: type
    type_map: Optional[TypeMap]
    # Inner type for ARRAY or similar
    item_type_map: Optional[TypeMap] = None

    # -- accepts None/Null?
    nullable: bool = True

    is_primary: Optional[bool] = field(default=False)
    length: Optional[int] = field(default=None)
    scale: Optional[int] = field(default=None, repr=False)

    # -- default value when not set
    default: Any = field(default=MISSING)
    # TODO: Optional[Callable]
    default_factory: Any = None

    # -- foreign key
    fk_table: Optional[str] = field(default=None, repr=False)
    fk_column: Optional[str] = field(default=None, repr=False)

    # -- e.g. access -> access_id column link
    fk_column_raw: Optional[Column] = field(init=False, default=None, repr=False)

    verbose: int = field(repr=False, default=1)

    # -- computed
    pytype_name: str = field(init=False)
    # item_type_name: str = field(init=False)

    # -- class variables
    TYPE_MAP_LIST: ClassVar[list[TypeMap]] = [
        TypeMap("FloatField", "Float", float),
        TypeMap("DateField", "Date", date),
        TypeMap("DateTimeField", "DateTime", datetime),
        TypeMap("TimeField", "Time", time),
        TypeMap("CharField", "String", str),
        TypeMap("ImageField", "String", str, is_sa_main=False, is_py_main=False),
        TypeMap("FileField", "String", str, is_sa_main=False, is_py_main=False),
        TypeMap("CountryField", "String", str, is_sa_main=False, is_py_main=False),
        TypeMap("TextField", "Text", str, is_py_main=False),
        TypeMap("IntegerField", "Integer", int),
        TypeMap("PositiveIntegerField", "Integer", int, is_sa_main=False, is_py_main=False),
        TypeMap("SmallIntegerField", "SmallInteger", int, is_py_main=False),
        TypeMap(
            "PositiveSmallIntegerField", "SmallInteger", int, is_sa_main=False, is_py_main=False
        ),
        TypeMap("BooleanField", "Boolean", bool),
        TypeMap("NullBooleanField", "Boolean", bool, is_sa_main=False, is_py_main=False),
        TypeMap("DecimalField", "DECIMAL", Decimal),
        TypeMap("CustomDecimalField", "DECIMAL", Decimal, is_sa_main=False, is_py_main=False),
        TypeMap("ArrayField", "ARRAY", list),
        # TypeMap("ArrayField",               "postgresql.ARRAY", list, is_sa_main=False, is_py_main=False),
    ]
    TYPE_MAP_BY_DJ_TYPE_DICT: ClassVar[dict[str, TypeMap]] = {
        type_map.dj_type: type_map for type_map in TYPE_MAP_LIST
    }
    TYPE_MAP_BY_SA_TYPE_DICT: ClassVar[dict[str, TypeMap]] = {
        type_map.sa_type: type_map for type_map in TYPE_MAP_LIST if type_map.is_sa_main
    }
    # NOTE: this one is by type, not by str
    TYPE_MAP_BY_PY_TYPE_DICT: ClassVar[dict[type, TypeMap]] = {
        type_map.py_type: type_map for type_map in TYPE_MAP_LIST if type_map.is_py_main
    }

    FK_DJ_TYPE_NAMES: ClassVar[list[str]] = [
        "ForeignKey",
        # TODO: "HistoricalRecords",
        "ManyToManyField",
        "OneToOneField",
    ]

    def __post_init__(self) -> None:  # noqa: C901
        assert " " not in self.name, self
        if self.is_primary:
            self.nullable = True
            self.default = None

        # if self.item_type_map:
        #     self.item_type_name = (
        #         self.item_type_map.__name__
        #         if inspect.isclass(self.item_type_map)
        #         else self.item_type_map.__class__.__name__
        #     )
        # else:
        #     self.item_type_name = ""

        self.pytype_name = self.py_type.__name__

        if self.fk_table:
            if not self.fk_column:
                raise Exception(f"{self}: fk {self.fk_table} -> self.fk_column is not set")
            if not isinstance(self.fk_table, str):
                raise TypeError(f"{self.name}.fk_table not a string: {self.fk_table}")
            if not is_dataclass(self.py_type):
                raise TypeError(f"{self.name}.fk_table -> py_type not a dataclass: {self.py_type}")
            if self.name.endswith("_id"):
                raise NameError(f"{self}: fk field does not end with _id: {self.name}")

        if (isinstance(self.default, str) and self.default == "list") or (
            isinstance(self.default, list) and len(self.default) == 0
        ):
            if self.default == "list":
                assert self.item_type_map
            else:
                warn(
                    f"{self.name}: bad default: {self.default}, converting to default_factory=list"
                )
            self.default = MISSING
            self.default_factory = list

        if self.default_factory is not None:
            if not callable(self.default_factory):
                raise ValueError(
                    f"{self.name}: bad default_factory: {self.default_factory}, expecting callable."
                )
            # NOTE: strange list is not list
            if self.default_factory not in (list,) and self.default_factory.__name__ == "list":
                self.default_factory = list

        if self.default is not MISSING:
            if self.default_factory:
                raise ValueError(
                    f"{self.name}: invalid defaults, default/default_factory - both defined: {self.default} / {self.default_factory}"
                )
            if (
                not isinstance(self.default, (type(None), int, str))
                and self.source != TableSourceEnum.FROM_DJANGO
            ):
                raise ValueError(
                    f"{self.name}: invalid default, expecting simple types, got: {self.default} : {type(self.default)}"
                )

    def set_fk_column_raw(self, column: Column) -> None:
        assert self.fk_table and not self.fk_column_raw
        self.fk_column_raw = column

    def equals_to_target(self, target: Column, verbose: int = 1) -> bool:  # noqa: C901
        if self.name != target.name:
            raise NameError(
                f"{self.name}: Only same named columns can be compared, got: {self.name} != {target.name}"
            )
        # fk_table / fk_column is not set in dataclass case
        if (self.fk_table and target.fk_table) and not (
            self.py_type == target.py_type and self.fk_column == target.fk_column
        ):
            raise ValueError(
                f"{self.name}: both sides FK - fk attrs don't not match: Only same named columns can be compared, "
                f"\n got: {self.fk_table   if self.fk_table   else 'none':30s}:{self.py_type} . {self.fk_column} "
                f"\n != : {target.fk_table if target.fk_table else 'none':30s}:{target.py_type} . {target.fk_column}"
            )

        # type check
        out = bool(
            self.pytype_name == target.pytype_name
            # compare item type only by py_type
            and (self.item_type_map.py_type if self.item_type_map else None)
            == (target.item_type_map.py_type if target.item_type_map else None)
        )

        # TODO: DRY this
        if self.default_factory != target.default_factory:
            if self.default_factory is not None and target.default_factory is None:
                out = False  # do update
            elif not target.meta.get("sm_default_factory_ok", False):
                warn(
                    f"{self.name}: default_factory is diff but defined on both sides: {self.default_factory} != {target.default_factory}, ignored."
                )

        if (
            target.source == TableSourceEnum.FROM_DATACLASS
            and self.nullable
            and self.default is MISSING
            and target.default is None
        ):
            # ignore this case: DC generates default=None, when missing and nullable
            pass
        elif self.default != target.default:
            if self.default is not MISSING and target.default is MISSING:
                out = False  # do update
            elif not target.meta.get("sm_default_ok", False):
                warn(
                    f"{self.name}: default is diff but defined on both sides: {self.default} != {target.default}, ignored."
                )

        if self.nullable != target.nullable:
            if (target.source == TableSourceEnum.FROM_DATACLASS) or (
                self.nullable is not MISSING and target.nullable is MISSING
            ):
                out = False  # do update
            elif not target.meta.get("sm_nullable_ok", False):
                warn(
                    f"{self.name}: nullable is diff but defined on both sides: {self.nullable} != {target.nullable}, ignored."
                )

        if not out and verbose >= 2:
            msg: list[str] = []
            if self.pytype_name != target.pytype_name:
                msg.append(f"pytype_name: {self.pytype_name} != { target.pytype_name}")
            if self.item_type_map != target.item_type_map:
                msg.append(f"item_type_map: {self.item_type_map} != {target.item_type_map}")
            if self.default_factory != target.default_factory:
                msg.append(f"default_factory: {self.default_factory} != {target.default_factory}")
            if self.nullable != target.nullable:
                msg.append(f"nullable(*): {self.nullable} != {target.nullable}")
            if self.default != target.default:
                msg.append(f"default(*): {self.default} != {target.default}")
            info(f"{self.name}: diff detected: {', '.join(msg)}")

        return out

    @classmethod
    def columns_from_dc_model(cls, dc_model: Any, verbose: int = 1) -> list[Column]:
        if not is_dataclass(dc_model):
            raise TypeError(f"Model {dc_model} is not a dataclass")
        columns = []
        py_type_hints = extract_py_type_hints(dc_model)

        for dc_field in fields(dc_model):
            default = MISSING if dc_field.default is DC_MISSING else dc_field.default
            default_factory = (
                None if dc_field.default_factory is DC_MISSING else dc_field.default_factory
            )
            py_hint = py_type_hints[dc_field.name]
            type_info = TypeInfo.get_or_create_by_type(py_hint)
            nullable = type_info.is_optional
            if type_info.is_list:
                py_type = list
                item_type_map = cls._get_type_map_by_py_type(
                    model_name=dc_model.__name__, name=dc_field.name, py_type=type_info.type_
                )
            else:
                py_type = type_info.type_
                item_type_map = None

            type_map = cls._get_type_map_by_py_type(
                model_name=dc_model.__name__, name=dc_field.name, py_type=py_type
            )
            column = Column(
                name=dc_field.name,
                meta=dict(dc_field.metadata),
                verbose=verbose,
                source=TableSourceEnum.FROM_DATACLASS,
                py_type=py_type,
                type_map=type_map,
                item_type_map=item_type_map,
                nullable=nullable,
                default=default,
                default_factory=default_factory,
                is_primary=None,
                length=None,
                fk_table=None,
                fk_column=None,
            )
            columns.append(column)
        return columns

    @classmethod
    def from_sa_column(
        cls,
        model_name: str,
        sa_column: SAColumn,
        sa_name_to_dc_model_dict: dict[str, DataclassModel],
        verbose: int = 1,
        fk_raw_column: bool = False,
    ) -> Column:
        name: str = sa_column.name
        if not fk_raw_column and sa_column.foreign_keys:
            assert len(sa_column.foreign_keys) == 1
            fk = list(sa_column.foreign_keys)[0]
            fk_table, fk_column = fk.target_fullname.split(".")
            # py_type = fk_table
            if fk_table not in sa_name_to_dc_model_dict:
                raise NameError(
                    f"SqlAlchemy column {sa_column} references {fk_table} which is not found."
                )
            pytype = sa_name_to_dc_model_dict[fk_table]
            assert name.endswith("_id"), name
            name = name[: -len("_id")]
        else:
            fk_table, fk_column = None, None
            try:
                pytype = sa_column.type.python_type
            except Exception as ex:
                raise TypeError(
                    f"{sa_column}: Not implemented python_type for {sa_column.type}: {ex}"
                )
        default = MISSING
        default_factory = None
        if sa_column.default:
            if sa_column.default.is_scalar and not sa_column.default.is_callable:
                default = sa_column.default.arg
            elif not sa_column.default.is_scalar and sa_column.default.is_callable:
                default_factory = sa_column.default.arg
            else:
                assert False, f"{sa_column} / {sa_column.default}"

        length: Optional[int] = getattr(sa_column, "length", None)
        if not length:
            # https://docs.sqlalchemy.org/en/14/dialects/mysql.html#sqlalchemy.dialects.mysql.DECIMAL.__init__
            length = getattr(sa_column, "precision", None)
        scale: Optional[int] = getattr(sa_column, "scale", None)

        type_map = cls._get_type_map_by_sa_type(
            model_name,
            name,
            sa_type=sa_column.type,
            length=length,
            scale=scale,
        )
        assert type_map

        item_type = getattr(sa_column.type, "item_type", None)
        if item_type:
            item_type_map = cls._get_type_map_by_sa_type(
                model_name,
                name,
                sa_type=item_type,
                length=None,
                scale=None,
            )
            assert item_type_map
        else:
            item_type_map = None

        assert sa_column.info is None or isinstance(sa_column.info, dict)
        meta = sa_column.info if sa_column.info else {}

        return Column(
            name=name,
            meta=meta,
            verbose=verbose,
            source=TableSourceEnum.FROM_SQLALCHEMY,
            py_type=pytype,
            type_map=type_map,
            item_type_map=item_type_map,
            nullable=sa_column.nullable,
            default=default,
            default_factory=default_factory,
            is_primary=sa_column.primary_key,
            length=length,
            scale=scale,
            fk_table=fk_table,
            fk_column=fk_column,
        )

    @classmethod
    def _get_type_map_by_py_type(
        cls,
        model_name: str,
        name: str,
        py_type: type,
    ) -> Optional[TypeMap]:
        if is_dataclass(py_type) or is_enum(py_type):
            type_map = None
        else:
            if py_type not in cls.TYPE_MAP_BY_PY_TYPE_DICT:
                raise TypeError(
                    f"{model_name}.{name}: Type {py_type} is not mapped, update TYPE_MAP_BY_PY_TYPE_DICT"
                )
            type_map = cls.TYPE_MAP_BY_PY_TYPE_DICT[py_type]
        return type_map

    @classmethod
    def _get_type_map_by_sa_type(
        cls,
        model_name: str,
        name: str,
        sa_type: SATypeEngine,
        length: Optional[int],
        scale: Optional[int],
    ) -> TypeMap:
        assert not inspect.isclass(sa_type), sa_type
        # NOTE: can not compare when SATypeEngine is Any
        # assert isinstance(sa_type, SATypeEngine), sa_type

        sa_type_name: str = sa_type.__class__.__name__

        type_map = cls.TYPE_MAP_BY_SA_TYPE_DICT.get(sa_type_name)
        if not type_map:
            error(
                f"{model_name}.{name}: unknown SA type {sa_type_name}, check and update TYPE_MAP_LIST"
            )
        assert type_map

        # Decimal with no length -> float
        if sa_type_name.upper().startswith("DECIMAL") and not length and not scale:
            type_map = cls.TYPE_MAP_BY_SA_TYPE_DICT["Float"]

        return type_map

    @classmethod
    def _get_type_map_by_dj_type(
        cls,
        model_name: str,
        name: str,
        dj_type_name: str,
        verbose: int = 1,
    ) -> Optional[TypeMap]:
        type_map = cls.TYPE_MAP_BY_DJ_TYPE_DICT.get(dj_type_name)
        if not type_map:
            if dj_type_name in ("HistoricalRecords",):
                if verbose >= 2:
                    info(f"{model_name}.{name} dj type {dj_type_name}, not processed, ignored.")
                return None

            if dj_type_name in cls.FK_DJ_TYPE_NAMES:
                error(f"{model_name}.{name} dj type {dj_type_name}, should not be processed here")
            else:
                error(f"{model_name}.{name}: unknown dj type {dj_type_name}, update TYPE_MAP_LIST")
        # Decimal with no length -> float
        # if "DECIMAL" in dj_type_name.upper() and not length and not scale:
        #     type_map = cls.TYPE_MAP_BY_SA_TYPE_DICT.get("Float")

        return type_map

    @classmethod
    def _from_dj_model_attr_fk(
        cls,
        model_name: str,
        name: str,
        fk_func: FunctionFromCode,
        dj_name_to_dc_model_dict: dict[str, DataclassModel],
        verbose: int = 1,
    ) -> Optional[tuple[str, str, DataclassModel, TypeMap]]:
        """returns:
        fk_table
        fk_column
        ...
        """
        if len(fk_func.args) != 1:
            raise TypeError(
                f"{model_name}.{name} -> FK expected single positional arg, got: {fk_func.args}"
            )

        fk_type_name: str = fk_func.args[0]
        fk_type_name = fk_type_name.lower()

        if fk_type_name not in dj_name_to_dc_model_dict:
            names_avail = get_available_names_example(
                fk_type_name, list(dj_name_to_dc_model_dict.keys())
            )
            if verbose >= 2:
                info(
                    f"{model_name}.{name} -> Ignoring FK unknown referenced type: {fk_type_name}, available: {names_avail}"
                )
            return None

        fk_dc_type = dj_name_to_dc_model_dict[fk_type_name]
        fk_table = fk_dc_type.__name__
        fk_column = "id"
        # find python type -> type_map
        all_fields = fields(fk_dc_type)
        id_cols = [field for field in all_fields if field.name == fk_column]
        if len(id_cols) == 0:
            py_type: Type = int
        else:
            if len(id_cols) != 1:
                raise NameError(
                    f"{model_name}: {name} -> FK -> {fk_type_name} - expected single id field, got: {id_cols}"
                )
            dj_py_type_hint = id_cols[0].type
            type_info: TypeInfo = TypeInfo.get_or_create_by_type(dj_py_type_hint)
            py_type = type_info.type_

        raw_type_map = cls._get_type_map_by_py_type(model_name, name, py_type)
        assert raw_type_map
        return fk_table, fk_column, fk_dc_type, raw_type_map

    @classmethod
    def from_dj_model_attr(
        cls,
        model_name: str,
        dj_model_attr: ModelAttrFromCode,
        dj_name_to_dc_model_dict: dict[str, DataclassModel],
        verbose: int = 1,
    ) -> Optional[Column]:
        name = dj_model_attr.name
        assert dj_model_attr.func and dj_model_attr.func.name_path
        dj_type_name = dj_model_attr.func.name_path[-1]

        max_length = dj_model_attr.func.get_kwarg("max_length", None)
        scale = None
        if max_length is None:
            # Decimal
            max_length = dj_model_attr.func.get_kwarg("max_digits", None)
            scale = dj_model_attr.func.get_kwarg("decimal_places", None)

        fk_table: Optional[str] = None
        fk_column: Optional[str] = None

        type_map: Optional[TypeMap] = None

        if dj_type_name in cls.FK_DJ_TYPE_NAMES:
            result = cls._from_dj_model_attr_fk(
                verbose=verbose,
                model_name=model_name,
                name=name,
                dj_name_to_dc_model_dict=dj_name_to_dc_model_dict,
                fk_func=dj_model_attr.func,
            )
            if result is None:
                return None
            # py_type is DataclassModel, type_map is for raw table type (usually integer)
            fk_table, fk_column, py_type, type_map = result
        else:
            type_map = cls._get_type_map_by_dj_type(
                model_name, name, dj_type_name, verbose=verbose
            )
            if type_map is None:
                # TODO: column not considered - ignored
                return None
            py_type = type_map.py_type

        if type_map.dj_type == "NullBooleanField":
            nullable = True
        else:
            nullable = dj_model_attr.func.get_kwarg("null", False)

        default = dj_model_attr.func.get_kwarg("default", MISSING)

        if type_map.dj_type == "ArrayField":
            assert len(dj_model_attr.func.args) >= 1
            arg0_list = dj_model_attr.func.args[0]
            assert (
                arg0_list and isinstance(arg0_list, list) and isinstance(arg0_list[0], str)
            ), arg0_list
            if arg0_list[0] == "models":
                arg0_list = arg0_list[1:]
            item_type_name = arg0_list[0]
            item_type_map = cls._get_type_map_by_dj_type(
                model_name,
                name,
                item_type_name,
                verbose=verbose,
            )
            assert item_type_map, f"{model_name}..{name} : {item_type_name}"
        else:
            item_type_map = None

        # is_unique = dj_model_attr.func.get_kwarg("unique", False)
        # indexed = dj_model_attr.func.get_kwarg("db_index", False)

        column = Column(
            name=name,
            meta={},
            verbose=verbose,
            source=TableSourceEnum.FROM_DJANGO,
            py_type=py_type,
            type_map=type_map,
            item_type_map=item_type_map,
            default_factory=None,
            nullable=nullable,
            default=default,
            length=max_length,
            scale=scale,
            fk_table=fk_table,
            fk_column=fk_column,
            # is_primary=sa_column.primary_key,
            # is_unique=is_unique,
            # indexed=indexed,
            # fk_table=fk_table,
            # fk_column=fk_column,
        )
        return column

    @staticmethod
    def _get_col_str(name: str, pyhint: str, default: Optional[str]) -> str:
        col_str: list[str] = [f"{name}: {pyhint}"]
        if default is not None:
            col_str.append(default)
        return "".join(col_str)

    def to_dataclass_pycode(self) -> str:
        name = self.name
        if self.fk_table:
            pyhint = self.pytype_name
        else:
            pyhint = map_sa_to_py_type_as_str(self.py_type)
            if self.item_type_map:
                item_type = map_sa_to_py_type_as_str(self.item_type_map.py_type)
                pyhint = f"{pyhint}[{item_type}]"

        default = None
        if self.is_primary or self.nullable:
            # for nullable columns - use None as default
            pyhint = f"Optional[{pyhint}]"
            default = " = None"

        if self.default_factory is not None:
            default = self.default_factory.__name__
            default = f" = field(default_factory={default})"
        elif self.default is not MISSING:
            default = literal_to_str(self.default)
            default = f" = {default}"

        output = self._get_col_str(name=name, pyhint=pyhint, default=default)
        return output

    # NOTE: use to_sqlalchemy_pycode_from_ast instead. Do not delete this function until all logic is transferred
    #
    #   def to_sqlalchemy_pycode(self) -> str:  # noqa: C901
    #       """
    #       returns code for the column + needs_sa_utils_types
    #       copied and adapted from intis/codegen/gen_models_all.py
    #           def generate_sa_models(app:App, verbose:bool=False):
    #       """
    #       # TODO: inout param: enums_needed: list[str]
    #       # needs_sa_utils_types = False

    #       col_spec = []
    #       indent_2_col = PY_INDENT * 1
    #       # 1st positional - name
    #       # TODO: _assigned_clients -> assigned_clients
    #       # https://www.attrs.org/en/latest/init.html#private-attributes
    #       # models:           assigned_clients
    #       # sa_models:        Column("_assigned_clients",
    #       # sa_models.mapper: properties={"assigned_clients" : CustomUserTable.c._assigned_clients,

    #       cname = self.name  # TODO: self.db_name if self.db_name else self.name
    #       col_spec.append(f"'{cname}'")

    #       # 2nd positional - column type
    #       assert self.type_map
    #       sa_type: str = self.type_map.sa_type

    #       # TODO: fk-s
    #       # if self.fk_column and not self.fk_circular_dep:
    #       #     fk = self.fk_column
    #       #     col_type = f'\nForeignKey("{fk.table.table_name}.{fk.name}", deferrable=True, initially="DEFERRED")\n'
    #       if sa_type in ("DateRangeType", "EmailType", "NumericRangeType"):
    #           # needs_sa_utils_types = True
    #           pass
    #       elif sa_type == "ARRAY":
    #           assert self.item_type_map is not None, self
    #           sa_type = "postgresql.ARRAY"
    #           assert inspect.isclass(self.item_type_map.sa_type)
    #           item_type_name = self.item_type_map.sa_type.__name__
    #           sa_type = f"{sa_type}({item_type_name})"
    #       # elif sa_type == "Enum":
    #       #     assert self.enum, self
    #       #     enum_name, use_to_value_list = self.enum
    #       #     # TODO: probably are identical, use only one  - e.g. only function .to_value_list
    #       #     if use_to_value_list:
    #       #         sa_type = f"Enum(*[i.value for i in iter({enum_name})])"
    #       #     else:
    #       #         sa_type = f"Enum(*{enum_name}.to_value_list())"
    #       #     # if enum_name not in enums_needed:
    #       #     #     enums_needed.append(enum_name)
    #       elif sa_type == "String" and self.length:
    #           sa_type = f"{sa_type}({self.length})"
    #       elif sa_type == "Numeric" and self.length:
    #           if self.scale:
    #               sa_type = f"{sa_type}({self.length}, {self.scale})"
    #           else:
    #               sa_type = f"{sa_type}({self.length})"
    #       if sa_type:
    #           col_spec.append(f"{sa_type}")

    #       # kwargs
    #       if self.is_primary:  # SA has True by default
    #           col_spec.append(f"primary_key={self.is_primary}")
    #       elif not self.nullable:  # SA has True by default
    #           # https://docs.sqlalchemy.org/en/20/orm/mapping_api.html#sqlalchemy.orm.mapped_column.params.nullable
    #           # nullable - defaults to True, except for primary keys
    #           col_spec.append(f"nullable={self.nullable}")

    #       # default
    #       default = self.default
    #       if default is not None:
    #           if isinstance(default, str):
    #               default = f"'{default}'"
    #           elif callable(default):
    #               if default == datetime.now:
    #                   default = "datetime.now"
    #               else:
    #                   default = f"{default.__name__}"
    #           col_spec.append(f"default={default}")

    #       # if self.is_unique:
    #       #     col_spec.append(f"unique={self.is_unique}")

    #       # if self.indexed:
    #       #     col_spec.append(f"index={self.indexed}")

    #       col_str = ", ".join(col_spec)
    #       if "\n" in col_str:
    #           # move , to previous line and indent properly
    #           col_str = col_str.replace("\n, ", ",\n").replace("\n", f"\n{indent_2_col}")

    #       # comment = (
    #       #     " # App circular dependency: {}".format(self.fk_circular_dep)
    #       #     if self.fk_circular_dep
    #       #     else ""
    #       # )
    #       # # disabled = "# TODO: sqlite ARRAY/JSON: " if self.type=="ARRAY" else ""
    #       # disabled = ""
    #       # out = f"{disabled}Column({col_str}),{comment}"

    #       out = f"Column({col_str})"

    #       # TODO: return needs_sa_utils_types
    #       return out

    def to_sqlalchemy_pycode_from_ast(  # noqa: C901
        self, attr_from_code: ModelAttrFromCode
    ) -> str:  # noqa: C901
        """
        returns code for the column + needs_sa_utils_types
        copied and adapted from intis/codegen/gen_models_all.py
            def generate_sa_models(app:App, verbose:bool=False):
        inspired by:
            https://stackoverflow.com/questions/9187837/how-to-modify-the-python-source-code-in-order-to-add-a-new-ast-node
        """
        # TODO: inout param: enums_needed: list[str]
        # needs_sa_utils_types = False

        assert isinstance(attr_from_code.node, ast.Call)
        ast_call_node: ast.Call = attr_from_code.node

        # # 1st positional
        # cname = self.name  # TODO: self.db_name if self.db_name else self.name
        # col_spec.append(f"'{cname}'")
        # # 2nd positional - column type
        # assert self.type_map
        # sa_type: str = self.type_map.sa_type

        # TODO: currently type and belonging args/kwargs (e.g. String(45)
        #       is not changed. Could check or update
        assert self.type_map
        sa_type_str = self.type_map.sa_type
        if sa_type_str == "ARRAY":
            assert self.item_type_map is not None, self
            sa_type_str = "postgresql.ARRAY"
            # assert not inspect.isclass(self.item_type_map.sa_type)
            assert isinstance(self.item_type_map.sa_type, str), self.item_type_map.sa_type
            item_type_name = self.item_type_map.sa_type
            sa_type_str = f"{sa_type_str}({item_type_name})"
        elif sa_type_str == "String" and self.length:
            sa_type_str = f"{sa_type_str}({self.length})"
        elif sa_type_str.upper() in ("DECIMAL", "NUMERIC") and self.length:
            if self.scale:
                sa_type_str = f"{sa_type_str}({self.length}, {self.scale})"
            else:
                sa_type_str = f"{sa_type_str}({self.length})"
        else:
            if self.length or self.scale:
                raise TypeError(
                    f"{self}: Did not processed (length, scale) = ({self.length}, {self.scale})"
                )

        if self.fk_table:
            # TODO: currently no update ...
            pass
        elif sa_type_str:
            ast_call_node_update_positional_arg(
                ast_call_node=ast_call_node, arg_number=2, arg_value=sa_type_str
            )

        if self.is_primary:  # SA has True by default
            ast_call_node_add_or_update_kw_arg(
                ast_call_node=ast_call_node,
                kw_name="primary_key",
                kw_value=ast.Constant(self.is_primary),
            )
        elif not self.nullable:  # SA has True by default
            # https://docs.sqlalchemy.org/en/20/orm/mapping_api.html#sqlalchemy.orm.mapped_column.params.nullable
            # nullable - defaults to True, except for primary keys
            ast_call_node_add_or_update_kw_arg(
                ast_call_node=ast_call_node,
                kw_name="nullable",
                kw_value=ast.Constant(self.nullable),
            )

        # default
        default = self.default
        if default not in (MISSING,):
            ast_default: ast.AST
            if callable(default):
                # TODO: any import needed
                if default == datetime.now:
                    default = "datetime.now"
                else:
                    default = default.__name__
                ast_default = ast.Name(default)
            else:
                ast_default = ast.Constant(default)
            # do not update if exists
            ast_call_node_create_kw_arg_if_not_exists(
                ast_call_node=ast_call_node, kw_name="default", kw_value=ast_default
            )

        # if self.is_unique:
        #     col_spec.append(f"unique={self.is_unique}")

        # if self.indexed:
        #     col_spec.append(f"index={self.indexed}")

        # TODO: fk-s
        # if self.fk_column and not self.fk_circular_dep:
        #     fk = self.fk_column
        #     col_type = f'\nForeignKey("{fk.table.table_name}.{fk.name}", deferrable=True, initially="DEFERRED")\n'

        pycode = ast.unparse(ast_call_node)
        pycode = f"{pycode},"
        # TODO: return needs_sa_utils_types
        return pycode


@dataclass
class Table:
    name: str
    sa_name: Optional[str]
    columns: list[Column] = field(repr=False)
    verbose: int = field(repr=False, default=1)
    columns_dict: dict[str, Column] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.columns_dict = {col.name: col for col in self.columns}

    @staticmethod
    def from_sa_table(
        sa_table: SATable,
        sa_name_to_dc_model_dict: dict[str, DataclassModel],
        verbose: int = 1,
    ) -> Table:
        # if sa_table.name not in FK_SA_TO_DC_MAP:
        #     raise NameError(f"Not found: {sa_table.name}")
        # name = FK_SA_TO_DC_MAP[sa_table.name]
        name = sa_table.name
        columns = []
        # noinspection PyTypeChecker
        sa_columns: Iterable = sa_table.columns
        for sa_column in sa_columns:
            column = Column.from_sa_column(
                name, sa_column, sa_name_to_dc_model_dict, verbose=verbose
            )
            columns.append(column)
            if column.fk_column:
                # FK access_id => access + access_id
                fk_raw_column = Column.from_sa_column(
                    model_name=name,
                    sa_column=sa_column,
                    sa_name_to_dc_model_dict=sa_name_to_dc_model_dict,
                    fk_raw_column=True,
                    verbose=verbose,
                )
                columns.append(fk_raw_column)
                column.set_fk_column_raw(fk_raw_column)

        table = Table(name=name, sa_name=sa_table.name, columns=columns, verbose=verbose)
        return table

    @staticmethod
    def from_dc_model(dc_model: DataclassModel, verbose: int = 1) -> Table:
        columns = Column.columns_from_dc_model(dc_model, verbose=verbose)
        table = Table(name=dc_model.__name__, sa_name=None, columns=columns, verbose=verbose)
        return table

    @staticmethod
    def from_dj_model_source(
        dj_model_source: ModelFromCode,
        dj_name_to_dc_model_dict: dict[str, DataclassModel],
        verbose: int = 1,
    ) -> Table:
        columns: list[Column] = []
        for dj_model_attr in dj_model_source.attrs:  # [::-1]:
            column = Column.from_dj_model_attr(
                model_name=dj_model_source.name,
                dj_model_attr=dj_model_attr,
                dj_name_to_dc_model_dict=dj_name_to_dc_model_dict,
                verbose=verbose,
            )
            if not column:
                continue
            columns.append(column)

        table = Table(name=dj_model_source.name, sa_name=None, columns=columns, verbose=verbose)
        return table

    def dump_to_dataclass_as_str_list(self, sort: bool = False) -> list[str]:
        out = [f"class {self.name}(Base):"]
        if sort:
            columns = [self.columns_dict[name] for name in sorted(self.columns_dict.keys())]
        else:
            columns = self.columns
        for column in columns:
            for col_str in column.to_dataclass_pycode():
                out.append(f"    {col_str}")
        return out

    def dump_to_dataclass_to_file(self, filename: str, sort: bool = False) -> None:
        str_list = self.dump_to_dataclass_as_str_list(sort=sort)
        with open(filename, "w") as file_out:
            file_out.write("\n".join(str_list))
