from dataclasses import dataclass, field
from typing import Optional

from .meta import ExpressionsAttributesDict, FieldName
from .contexts import IContext
from .values_accessor import IValueAccessor


# ------------------------------------------------------------
# IConfig
# ------------------------------------------------------------


@dataclass
class Config(IContext):
    """
    The Config instances contain general predefined Entity configuration parameters (settings).
    One can add custom config params.
    Config will be available in ConfigNS namespace (Cfg.).
    For values only literal / plain callables (python functions) are accepted,
    no DotExpression or Function() instances allowed.
    and belonging.
    This is plain class, no setup()/Setup() process.
    """
    # if not set will use default ValueExpress
    value_accessor: Optional[IValueAccessor] = field(default=None, metadata={"dexp_exposed": False})

    # -----------------------------------------------------------
    # Tracing, analytics, debug ...
    # -----------------------------------------------------------
    debug: bool = False

    # Should all changes of values are collected in
    # ApplyResult.value_history_dict and ValueNode.value_history
    trace: bool = False

    # def set_value_accessor(self, value_accessor: IValueAccessor) -> None:
    #     assert isinstance(value_accessor, IValueAccessor)
    #     assert self.value_accessor is None
    #     self.value_accessor = value_accessor

    def is_trace(self) -> bool:
        return self.debug or self.trace

    # def is_debug(self) -> bool:
    #     return self.debug

    @classmethod
    def get_expressions_attributes(cls) -> ExpressionsAttributesDict:
        return {
            "Debug": FieldName("debug"),
            # ALT: "Debug": cls.is_debug,
            # "Trace": cls.is_trace,
        }
