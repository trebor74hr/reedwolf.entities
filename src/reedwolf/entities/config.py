from dataclasses import dataclass

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
