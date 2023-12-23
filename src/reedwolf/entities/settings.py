from contextlib import AbstractContextManager
from dataclasses import dataclass, field, fields as dc_fields
from typing import Optional, Union, ClassVar, Type

from .exceptions import EntityInternalError
from .meta import ExpressionsAttributesMap, FieldName, Self, MethodName
from .utils import UndefinedType, UNDEFINED
from .values_accessor import IValueAccessor

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

@dataclass
class Settings:
    """
    The Settings instances contain general predefined Entity configuration parameters (settings).
    One can add custom settings params.
    Settings will be available in ContextNS namespace (Ctx..).
    - but only attributes from get_contextns_attributes() method.
    For values only literal / plain callables (python functions) are accepted,
    no DotExpression or Function() instances allowed.
    and belonging.
    This is plain class, no setup()/Setup() process.
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

    # # collect
    # contextns_attributes : ClassVar[Union[ExpressionsAttributesMap, UndefinedType]] = UNDEFINED

    # set and reset back in apply phase
    _apply_settings: Union[Self, None, UndefinedType] = field(init=False, repr=False, compare=False, default=UNDEFINED)

    # TODO: ...
    # def set_value_accessor(self, value_accessor: IValueAccessor) -> None:
    #     assert isinstance(value_accessor, IValueAccessor)
    #     assert self.value_accessor is None
    #     self.value_accessor = value_accessor

    def is_trace(self) -> bool:
        if self._apply_settings is UNDEFINED:
            raise EntityInternalError(owner=self, msg="Method must be callwed in 'with .use_apply_settingsm()' block")
        return (self._apply_settings.debug or self._apply_settings.trace) if self._apply_settings is not None \
                else (self.debug or self.trace)

    def is_debug(self) -> bool:
        if not self._apply_settings is UNDEFINED:
            raise EntityInternalError(owner=self, msg="Method must be callwed in 'with .use_apply_settingsm()' block")
        return self._apply_settings.debug if self._apply_settings is not None else self.debug

    @classmethod
    def custom_contextns_attributes(cls) -> ExpressionsAttributesMap:
        """
        can be overridden
        """
        return {}

    @classmethod
    def common_contextns_attributes(cls) -> ExpressionsAttributesMap:
        return {
            # "Trace": MethodName("is_trace"),
            "Debug": MethodName("is_debug"),
        }

    def use_apply_settings(self, apply_settings: Optional[Self]) -> "UseApplySettingsCtxManager":
        return UseApplySettingsCtxManager(setup_settings=self, apply_settings=apply_settings)

    def set_apply_settings(self, apply_settings: Union[Self, None, UndefinedType]):
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
    apply_settings: Optional[Settings]

    def __enter__(self):
        self.setup_settings.set_apply_settings(self.apply_settings)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.setup_settings.set_apply_settings(apply_settings=UNDEFINED)


