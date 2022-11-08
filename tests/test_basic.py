# unit tests for reeedwolf.rules module
import unittest

from dataclasses import dataclass
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
from reedwolf.rules.types import TransMessageType



class TestBasic(unittest.TestCase):

    @dataclass
    class Company:
        name: str
        vat_number: str


    def test_late_bind(self):
        rules = Rules(
            name="company_rules", label="Company rules",
            contains=[
                Field(bind=M.company.name, label="Name"),
            ])
        rules.bind_to(BoundModel(name="company", model=self.Company))
        self._test_minimal(rules)


    def test_minimal_example(self):
        rules = Rules(
            name="company_rules", label="Company rules",
            bound_model=BoundModel(name="company", model=self.Company),
            contains=[
                Field(bind=M.company.name, label="Name"),
            ])
        rules.setup()
        self._test_minimal(rules)

    def _test_minimal(self, rules):
        self.assertNotEqual(rules.bound_model, None)
        # rules.print_components()
        self.assertNotEqual(rules.as_str(), "")
        # print(rules.as_str(), "")
        self.assertNotEqual(len(rules.to_strlist()), 0)

        self.assertEqual(rules.owner, None)
        self.assertSetEqual(set(rules.components.keys()), set(['company_rules', 'company', 'name']))

        self.assertEqual(list(rules.models.keys()), ['company'])
        self.assertEqual(list(rules.models.keys()), ['company'])
        self.assertEqual(rules.get_owner_container(), rules)
        self.assertEqual(rules.is_extension(), False)
        self.assertEqual(rules.validations, [])
        self.assertEqual(rules.dataproviders, [])
        self.assertNotEqual(rules.heap, None)
        self.assertEqual(rules.heap.finished, True)
        # heap: 'add', 'finish', 'get_var', 'get_var_by_bound_model', 'get_var_by_vexp', 'getset_attribute_var', 'is_empty', 'name', 'owner', 'variables', 'variables_count']

        name_component = rules.get_component("name")
        self.assertNotEqual(name_component, None)
        # component: 
        # 'as_str', 'autocomplete', 'available', 'bind', 'bound_variable', 'default', 'description', 'dump_pydantic_models', 'editable', 'enables', 'fill_components', 
        # 'get_bound_variable', 'get_children', 'get_name_from_bind'
        # 'hint', 'label', 'name', 'owner', 'owner_name', 'required', 'set_owner', 'setup', 'to_strlist', 'type', 'validations', 'variable']
        self.assertEqual(name_component.is_finished(), True)
        self.assertEqual(name_component.name, "name")
        self.assertEqual(name_component.owner, rules)
        self.assertEqual(name_component.get_owner_container(), rules)
        
        self.assertEqual(rules.get_children(), [name_component]) 



if __name__ == '__main__':
    unittest.main()

# company_rules = Rules(
#     name="company_rules",
#     label="Company rules",
#     bound_model=BoundModel(
#         name="company",
#         model=Company,
#         contains=[
#             BoundModel(name="company_extra", model=CompanyExtra),
#         ],
#     ),
#     dataproviders=[
#         DataVar(
#             name="get_active_count",
#             label=_("How many companies are active"),
#             datatype=int,
#             value=DP.Company.get_active_count,
#         ),
#     ],
#     validations=[
#         Validation(
#             name="validate_is_active_not_expired",
#             label=_("Validate that active company should not have expired date field filled"),
#             ensure=(~(F.is_active & F.expired)),
#             error=_(
#                 "Cannot have active company with expired date filled!"
#             ),
#         ),
#     ],
#     contains=[
#         Section(
#             name="general",
#             label="General information",
#             contains=[
#                 BooleanField(
#                     bind=M.company.is_active,
#                     label=_("Is company active"),
#                 ),
#                 Field(
#                     bind=M.company.address,
#                     label=_("Address"),
#                     required=True,
#                 ),
#             ],
#         ),
#     ]
# )

