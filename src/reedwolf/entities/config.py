from dataclasses import dataclass, field
from typing import Optional

from .values_accessor import IValueAccessor


# ------------------------------------------------------------
# IConfig
# ------------------------------------------------------------


@dataclass
class Config:
    """
    The Config instances contain general predefined Entity configuration parameters (settings).
    One can add custom config params.
    Config will be available in ConfigNS namespace (Cfg.).
    For values only literal / plain callables (python functions) are accepted,
    no DotExpression or Function() instances allowed.
    and belonging.
    This is plain class, no setup()/Setup() process.
    """
    debug: bool = False
    # if not set will use default ValueExpress
    value_accessor: Optional[IValueAccessor] = field(default=None, metadata={"dexp_exposed": False})

    # def set_value_accessor(self, value_accessor: IValueAccessor) -> None:
    #     assert isinstance(value_accessor, IValueAccessor)
    #     assert self.value_accessor is None
    #     self.value_accessor = value_accessor
