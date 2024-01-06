from collections import OrderedDict
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, List, Type, Dict

from .exceptions import EntityInternalError
from .expressions import IFunctionFactory
from .meta import (
    ExpressionsAttributesMap,
    Self,
    MethodName,
    ModelType,
    AttrName,
    ModelField,
    get_model_fields,
)
from .utils import UndefinedType, UNDEFINED
from .values_accessor import IValueAccessor

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
@dataclass
class SettingsBase:
    """
    The Settings instances contain general predefined Entity configuration parameters (settings).
    One can add custom settings params.
    Settings will be available in ContextNS namespace (Ctx..).
    - but only attributes from get_contextns_attributes() method.
    For values only literal / plain callables (python functions) are accepted,
    no DotExpression or Function() instances allowed.
    and belonging.
    This is plain class, no setup()/Setup() process.

    For a same Ctx.AttributeName - this is order of preferences - which will win:
        1. custom attributes in apply settings
        2. custom attributes in setup settings
        3. common attributes in apply settings (usually not overridden)
        4. common attributes in setup settings (usually not overridden)

    See ContextRegistry.
    """
    # if not set will use default ValueExpress
    value_accessor: Union[IValueAccessor, UndefinedType] = field(default=UNDEFINED, metadata={"dexp_exposed": False})

    # -----------------------------------------------------------
    # Tracing, analytics, debug ...
    # -----------------------------------------------------------
    # None indicates that argument/param is not passed in constructor
    debug: Union[bool, UndefinedType] = UNDEFINED

    # Should all changes of values are collected in
    # ApplyResult.value_history_dict and ValueNode.value_history
    # None indicates that argument/param is not passed in constructor
    trace: Union[bool, UndefinedType] = UNDEFINED

    @classmethod
    # TODO: CustomFunctionFactory
    def get_custom_functions(cls) -> List[IFunctionFactory]:
        """
        For override.
        Defined on class level.
        For (setup) Settings, can be overridden by instance attribute custom_functions.
        TODO: put example
        """
        return []

    @classmethod
    def get_custom_ctx_attributes(cls) -> ExpressionsAttributesMap:
        """
        For override.
        Defined on class level.
        Values are fetched from this instance only. Usually overridden.
        Defined on class level.
        For (setup) Settings, can be overridden by instance attribute custom_ctx_attributes.

        Example:
            {
            "Now": MethodName("get_now"),
            "UserId": FieldName("user_id"),
            }
        """
        return {}


@dataclass
class ApplySettings(SettingsBase):
    ...


# ------------------------------------------------------------

class SettingsType(str, Enum):
    SETUP_SETTINGS = "SETUP_SETTINGS"
    APPLY_SETTINGS = "APPLY_SETTINGS"


@dataclass
class SettingsSource:
    settings_type: SettingsType
    klass: ModelType
    fields: Dict[AttrName, ModelField] = field(init=False, repr=False)

    def __post_init__(self):
        self.fields = get_model_fields(self.klass)



@dataclass
class Settings(SettingsBase):
    """
    Setup settings can implement Ctx. (ContextNS) variables:
    - in class method get_custom_ctx_attributes() and/or
    - in instance attribute custom_ctx_attributes
    If both defined then last wins.
    """
    # TODO: CustomFunctionFactory
    custom_functions: Optional[List[IFunctionFactory]] = field(repr=False, default_factory=list, metadata={"skip_dump": True})
    custom_ctx_attributes: ExpressionsAttributesMap = field(repr=False, default_factory=dict, metadata={"skip_dump": True})

    apply_settings_class: Optional[Type[ApplySettings]] = field(repr=False, default=None)

    # set and reset back in apply phase
    _apply_settings: Union[Self, None, UndefinedType] = field(init=False, repr=False, compare=False, default=UNDEFINED)

    # TODO: ...
    # def set_value_accessor(self, value_accessor: IValueAccessor) -> None:
    #     assert isinstance(value_accessor, IValueAccessor)
    #     assert self.value_accessor is None
    #     self.value_accessor = value_accessor

    def _ensure_in_with_block(self, method_name: str):
        if self._apply_settings is UNDEFINED:
            raise EntityInternalError(owner=self, msg=f"Method '{method_name}' must be callwed in 'with .use_apply_settingsm()' block")

    def is_trace(self) -> bool:
        self._ensure_in_with_block(method_name="is_tracee")
        out = (self._apply_settings.debug or self._apply_settings.trace) if self._apply_settings is not None \
               else (self.debug or self.trace)
        return out if out is not UNDEFINED else False

    def is_debug(self) -> bool:
        self._ensure_in_with_block(method_name="is_tracee")
        out = self._apply_settings.debug if self._apply_settings is not None else self.debug
        return out if out is not UNDEFINED else False

    def _get_all_custom_functions(self, apply_settings_class: Optional[Type[ApplySettings]]) -> List[IFunctionFactory]:
        # make a copy and merge with instance defined functions
        custom_functions = OrderedDict([(fn.name, fn)  for fn in self.get_custom_functions()])
        custom_functions.update(OrderedDict([(fn.name, fn)  for fn in  self.custom_functions]))
        if apply_settings_class:
            # this one wins. Last wins.
            custom_functions.update(OrderedDict([(fn.name, fn) for fn in apply_settings_class.get_custom_functions()]))
        return custom_functions.values()

    def _get_attribute_settings_source_list_pairs(self, apply_settings_class: Optional[Type[ApplySettings]]):
        """
        TODO: very ugly function name
        First param apply_settings_class is used instead of self.apply_settings_class
        since Entity can have its own apply_settings_class attribute which wins.

        For the same attribute name - this is order of preferences - which will win:
            1. custom attributes in apply settings
            2. custom attributes in setup settings
            3. common attributes in apply settings (usually not overridden)
            4. common attributes in setup settings (usually not overridden)
        """
        setup_settings_source = SettingsSource(SettingsType.SETUP_SETTINGS, self.__class__)
        common_dict = self._get_common_contextns_attributes()

        # make a copy and merge with instance defined attributes
        setup_custom_dict = self.get_custom_ctx_attributes().copy()
        setup_custom_dict.update(self.custom_ctx_attributes)

        if apply_settings_class:
            apply_settings_source = SettingsSource(SettingsType.APPLY_SETTINGS, apply_settings_class)
            apply_custom_dict = apply_settings_class.get_custom_ctx_attributes()
            # last wins
            settings_source_list_pairs = [
                (common_dict, [setup_settings_source, apply_settings_source]),
                (setup_custom_dict, [setup_settings_source]),
                (apply_custom_dict, [apply_settings_source]),
            ]
        else:
            settings_source_list_pairs = [
                (common_dict, [setup_settings_source]),
                (setup_custom_dict, [setup_settings_source]),
            ]
        return settings_source_list_pairs

    @classmethod
    def _get_common_contextns_attributes(cls) -> ExpressionsAttributesMap:
        """
        Values are fetched from apply_settings and setup_settings - fetched from first available. Usually not overridden.
        recommendation on override is to merge dict with super(), e.g.:
            out super().common_contextns_attributes()
            out.update({...})
            return out
        """
        return {
            "Debug": MethodName("is_debug"),
        }

    def _use_apply_settings(self, apply_settings: Optional[Self]) -> "UseApplySettingsCtxManager":
        return UseApplySettingsCtxManager(setup_settings=self, apply_settings=apply_settings)

    def _set_apply_settings(self, apply_settings: Union[Self, None, UndefinedType]):
        if apply_settings is not UNDEFINED:
            if self._apply_settings is not UNDEFINED:
                raise EntityInternalError(owner=self, msg=f"Can not set new settings to {apply_settings}, already set to: {self._apply_settings}")
        else:
            if self._apply_settings is UNDEFINED:
                raise EntityInternalError(owner=self, msg=f"Can not set new settings back to {apply_settings}, it is already reset to: {self._apply_settings}")
        self._apply_settings = apply_settings




@dataclass()
class UseApplySettingsCtxManager(AbstractContextManager):
    setup_settings: Settings
    apply_settings: Optional[ApplySettings]

    def __enter__(self):
        self.setup_settings._set_apply_settings(self.apply_settings)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.setup_settings._set_apply_settings(apply_settings=UNDEFINED)


