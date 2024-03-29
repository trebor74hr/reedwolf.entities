from .utils import (
    UNDEFINED,
    )
from .namespaces import (  # noqa: F401
    FieldsNS,
    F,
    ModelsNS,
    M,
    ThisNS,
    This,
    ContextNS,
    Ctx,
    )
from .exceptions import (
    EntityError,
    EntitySetupError,
    EntityApplyError,
    EntityValidationError,
    EntityNameNotFoundError,
    )
from .expressions import (
    Just,
    )
from .custom_attributes import (
    Attribute,
    AttributeByMethod,
)
from .meta import (
    CustomCtxAttributeList, 
    _, 
    msg,
)
from .value_accessors import (
    AttributeValueAccessor,
    DictValueAccessor,
    AutodetectValueAccessor,
)
from .data_models import (
    DataModel,
    DataModelWithHandlers,
    )
from .base import (  # noqa: F401
    # Not for __all__ case (from <> import *), just for direct import:
    #   from <> import _, msg
    IApplyResult,
    )
from .fields import (
    AutocomputedEnum,
    StringField,
    SizedStringField,
    BooleanField,
    IntegerField,
    PositiveIntegerField,
    IdField,
    FloatField,
    DecimalField,
    ChoiceField,
    EnumField,
    DateField,
    DateTimeField,
    TimeField,
    DurationField,
    EmailField,
    ChoiceOption,
    FieldGroup,
    )
from .valid_field import (
    FieldValidation,
    Required,
    Readonly,
    MaxLength,
    ExactLength,
    MinLength,
    HourFieldValidation,
    RangeLength,
    )
from .valid_items import (
    ItemsValidation,
    Cardinality,
    Unique,
    SingleValidation,
    )
from .valid_children import (
    ChildrenValidation,
    )
from .eval_field import (
    FieldEvaluation,
    Default,
    )
from .eval_items import (
    ItemsEvaluation,
    )
from .eval_children import (
    ChildrenEvaluation,
    )
from .settings import (
    Settings,
    ApplySettings,
    CustomFunctionFactoryList,
)
from .containers import (  # noqa: F401
    KeyFields,
    SubEntityItems,
    SubEntity,
    Entity,
    )
from .functions import (
    Function,
    FunctionByMethod,
    )
from . import functions

from .func_builtin import (
    EnumMembers,
    )

from .apply import ApplyResult
from .load import load


# what to import in case "from reedwolf import *"
__all__ = [
    # utils
    UNDEFINED,
    # namespaces - no aliases
    # "FunctionsNS",
    "ModelsNS",
    "FieldsNS",
    "ThisNS",
    "ContextNS",

    # NOTE: namespace aliases not in "import *" case
    #   "F",
    #   "S",
    #   "M",
    "This",
    "Ctx",

    # exceptions
    "EntityError",
    "EntitySetupError",
    "EntityApplyError",
    "EntityValidationError",
    "EntityNameNotFoundError",

    # meta
    "CustomCtxAttributeList",

    # expressions
    "Just",

    # custom_attributes
    "Attribute",
    "AttributeByMethod",

    # models
    "DataModel",
    "DataModelWithHandlers",

    # value_accessors
    "AttributeValueAccessor",
    "DictValueAccessor",
    "AutodetectValueAccessor",

    # fields
    "AutocomputedEnum",
    "StringField",
    "SizedStringField",
    "BooleanField",
    "IntegerField",
    "PositiveIntegerField",
    "IdField",
    "FloatField",
    "DecimalField",
    "ChoiceField",
    "EnumField",
    "DateField",
    "DateTimeField",
    "TimeField",
    "DurationField",
    "EmailField",

    "ChoiceOption",

    "FieldGroup",

    # validations
    "FieldValidation",
    "HourFieldValidation",
    "Required",
    "MinLength",
    "MaxLength",
    "ExactLength",
    "RangeLength",
    "Readonly",

    "ChildrenValidation",
    "ItemsValidation",
    "Cardinality",
    "Unique",
    "SingleValidation",

    # evaluations
    "FieldEvaluation",
    "Default",

    "ChildrenEvaluation",
    "ItemsEvaluation",

    # containers
    "KeyFields",

    # containers
    "SubEntity",
    "SubEntityItems",
    # top container
    "Entity",

    # functions
    "functions",
    "Function",
    "FunctionByMethod",
    # "_",

    # func_builtin
    "EnumMembers",

    # settings
    "Settings",
    "ApplySettings",
    "CustomFunctionFactoryList",

    # apply
    "IApplyResult",  # from base.py
    "ApplyResult",

    # load
    "load",

    ]
