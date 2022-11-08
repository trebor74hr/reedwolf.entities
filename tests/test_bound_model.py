# unit tests for reeedwolf.rules module
import unittest

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Any

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
class CompanyAccess:
    can_user_access: bool
    can_superuser_change: bool

@dataclass
class BusinessType:
    type: str
    id: int = field(default_factory=int)
    alive: bool = field(default_factory=lambda: True)

@dataclass
class Company:
    name: str
    vat_number: str
    access: CompanyAccess
    business_types: List[BusinessType] = field(default_factory=list)


class CompanyManager:

    def read_business_types(
        self,
        company: Company,
        ) -> Any: # List[BusinessType]:
        raise NotImplementedError()


    def save_business_types(
        self, company: Company, business_types: List[BusinessType]
    ) -> None:
        raise NotImplementedError()

# ------------------------------------------------------------

class TestExtended(unittest.TestCase):
    """ Extended bound models """

    def test_missing(self):
        @dataclass
        class CompanyMissingAccess:
            name: str
            vat_number: str

        rules = Rules(
            name="company_rules", label="Company rules",
            bound_model=BoundModel(name="company", model=CompanyMissingAccess,
                contains=[
                    BoundModel(name="company_access", model=M.company.access),
                ]),
            contains=[
                Field(bind=M.company.name, label="Name"),
            ])
        with self.assertRaisesRegex(RuleSetupNameNotFoundError, "Variable name 'access' not found in fields"):
            rules.setup()

    def test_not_dataclass(self):

        class CompanyAccess:
            can_user_access: bool
            can_superuser_change: bool

        @dataclass
        class Company:
            name: str
            vat_number: str
            access: CompanyAccess

        rules = Rules(
            name="company_rules", label="Company rules",
            bound_model=BoundModel(name="company", model=Company,
                contains=[
                    BoundModel(name="company_access", model=M.company.access),
                ]),
            contains=[
                Field(bind=M.company.name, label="Name"),
            ])
        with self.assertRaisesRegex(RuleSetupError, "Managed model company_access needs to be a @dataclass"):
            rules.setup()

    def test_simple(self):
        rules = Rules(
            name="company_rules", label="Company rules",
            bound_model=BoundModel(name="company", model=Company,
                contains=[
                    BoundModel(name="company_access", model=M.company.access),
                ]),
            contains=[
                Field(bind=M.company.name, label="Name"),
            ])
        rules.setup()


    def test_handlers(self):
        """
        Complex models - custom read/save methods
        """
        rules = Rules(
            name="company_rules", label="Company rules",
            bound_model=BoundModel(name="company", model=Company,
                contains=[
                    BoundModelWithHandlers(
                        name="business_types",
                        label="Company's assigned business types",
                        read_handler=BoundModelHandler(
                            CompanyManager.read_business_types,
                            inject_params=dict(company=M.company),  # noqa: C408
                        ),
                        save_handler=BoundModelHandler(
                            CompanyManager.save_business_types,
                            inject_params=dict(company=M.company),  # noqa: C408
                            model_param_name="business_types",
                        ),
                    )
                ]),
            contains=[
                Field(bind=M.company.name, label="Name"),
            ])
        rules.setup()

    # def test_bound_model_ext_external(self):
    #     rules = Rules(
    #         name="company_rules", label="Company rules",
    #         bound_model=BoundModel(name="company", model=Company,
    #             contains=[
    #                 # ------------------------------------------------------------
    #                 # Interface to API calls
    #                 # TODO: do it with BoundModelWithHandlers - custom read/save methods
    #                 # ------------------------------------------------------------
    #                 BoundModel(name="wallet", model=CompanyWallet),  # api
    #                 BoundModel(name="company_certificate", model=CompanyCertificate),  # api
    #             ]),
    #         contains=[
    #             Field(bind=M.company.name, label="Name"),
    #         ])
    #     rules.setup()

    # dataproviders
    #   funkcije i potpis
    #
    # mismatch columns
    # test various valueexpressions
    # change this:
    #       choices=get_telecom_operators,
    #       choice_value=This.id,
    #       choice_label=This.caption,
    #   ->
    #       choices=Choices(get_telecom_operators,
    #           value=This.id,
    #           label=This.caption,)
            # TODO: choices=ChoiceOptions(
            # TODO:     options=get_telecom_operators,
            # TODO:     value=This.id,
            # TODO:     label=This.caption,
            # TODO:     ),

    # Section()
    # Extension() 
    # extensions
    # enums, choices, bool, etc 

    # --------
    # validations
    # required -> validations=Required()
    # maknuti: autocomplete=False,
    # with unbound models - can and should work, could dump @dataclass and similar ...


    # TODO: 
    # def test_dump_pydantic_models(self):
    #   rules.dump_pydantic_models()


