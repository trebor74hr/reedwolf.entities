import os
from dataclasses import dataclass, field

from .utils import UNDEFINED, to_bool

@dataclass
class GlobalSettings:
    is_production: bool = field(init=False, repr=False, default=UNDEFINED)
    is_unit_test: bool = field(init=False, repr=False, default=UNDEFINED)

    def __post_init__(self):
        self.is_unit_test = to_bool(os.environ.get("RWF_IS_UNIT_TEST", False))
        if self.is_unit_test:
            print("--- RWF_IS_UNIT_TEST mode")
        self.is_production = to_bool(os.environ.get("RWF_IS_PRODUCTION", False))
        assert not (self.is_unit_test and self.is_production)

GLOBAL_SETTINGS = GlobalSettings()