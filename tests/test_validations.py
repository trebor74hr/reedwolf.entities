# unit tests for reeedwolf.rules module
import unittest

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

from reedwolf.rules import ( 
    DP,
    BooleanField,
    BoundModel,
    BoundModelHandler,
    BoundModelWithHandlers,
    Cardinality,
    ChoiceField,
    ChoiceOption,
    Ctx,
    DataVar,
    EnumField,
    Extension,
    F,
    Field,
    FieldTypeEnum,
    M,
    Rules,
    RulesHandlerFunction,
    Section,
    This,
    Unique,
    Utils,
    Validation,
    msg,
)
from reedwolf.rules.exceptions import (
        RuleSetupNameNotFoundError,
        RuleSetupError,
        )

from reedwolf.rules.types import TransMessageType


@dataclass
class Company:
    name: str
    vat_number: str

# ------------------------------------------------------------

class TestExtended(unittest.TestCase):
    """ Extended bound models """

    def test_missing(self):
        rules = Rules(
            name="company_rules", label="Company rules",
            bound_model=BoundModel(name="company", model=Company),
            contains=[
                Field(bind=M.company.name, label="Name"),
                Field(bind=M.company.vat_number, label="VAT number"),
            ])
        rules.setup()

