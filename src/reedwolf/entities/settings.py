from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Optional, Union, List

from .exceptions import EntityInternalError
from .expressions import IFunctionFactory
from .meta import ExpressionsAttributesMap, Self, MethodName
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
    # TODO: CustomFunctionFactory
    custom_functions: Optional[List[IFunctionFactory]] = field(repr=False, default_factory=list, metadata={"skip_dump": True})

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
    def get_custom_contextns_attributes(cls) -> ExpressionsAttributesMap:
        """
        Values are fetched from this instance only. Usually overridden.
        Example:
            {
            "Now": MethodName("get_now"),
            "UserId": FieldName("user_id"),
            }
        """
        return {}


@dataclass
class Settings(SettingsBase):

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


@dataclass
class ApplySettings(SettingsBase):
    ...



@dataclass()
class UseApplySettingsCtxManager(AbstractContextManager):
    setup_settings: Settings
    apply_settings: Optional[ApplySettings]

    def __enter__(self):
        self.setup_settings._set_apply_settings(self.apply_settings)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.setup_settings._set_apply_settings(apply_settings=UNDEFINED)


