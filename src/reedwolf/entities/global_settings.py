import os
from dataclasses import dataclass, field

from .exceptions import EntityConfigError
from .utils import UNDEFINED, to_bool

@dataclass
class _GlobalSettings:
    is_production: bool = field(init=False, repr=False, default=UNDEFINED)
    is_unit_test: bool = field(init=False, repr=False, default=UNDEFINED)
    is_development: bool = field(init=False, repr=False, default=UNDEFINED)

    def __post_init__(self):
        self.is_unit_test = to_bool(os.environ.get("RWF_IS_UNIT_TEST", False))
        if self.is_unit_test:
            print("--- RWF_IS_UNIT_TEST mode")
            self.is_development = True
        else:
            self.is_development = to_bool(os.environ.get("RWF_IS_DEVELOPMENT", False))
        self.is_production = to_bool(os.environ.get("RWF_IS_PRODUCTION", False))

        if self.is_unit_test and self.is_production:
            raise EntityConfigError(owner=self, msg="Can not combine RWF_IS_PRODUCTION and RWF_IS_UNIT_TEST, remove one.")
        if self.is_development and self.is_production:
            raise EntityConfigError(owner=self, msg="Can not combine RWF_IS_PRODUCTION and RWF_IS_DEVELOPMENT, remove one.")

GlobalSettings = _GlobalSettings()