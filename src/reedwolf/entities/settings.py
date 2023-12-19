from contextlib import AbstractContextManager
from dataclasses import dataclass, field, fields as dc_fields
from typing import Optional, Union, ClassVar

from .exceptions import EntityInternalError
from .meta import ExpressionsAttributesMap, FieldName, Self
from .utils import UndefinedType, UNDEFINED
from .values_accessor import IValueAccessor

# ------------------------------------------------------------
# IConfig
# ------------------------------------------------------------


@dataclass
class Settings:
    """
    The Settings instances contain general predefined Entity configuration parameters (settings).
    One can add custom settings params.
    Settings will be available in ConfigNS namespace (Cfg.).
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

    # collect
    contextns_attributes : ClassVar[Union[ExpressionsAttributesMap, UndefinedType]] = UNDEFINED

    # # set and reset back in apply phase
    # _current_context: Union[Settings, None, UndefinedType] = field(init=False, repr=False, compare=False, default=UNDEFINED)

    # _trace: Union[bool, UndefinedType] = field(init=False, repr=False, compare=False, default=UNDEFINED)
    # _debug: Union[bool, UndefinedType] = field(init=False, repr=False, compare=False, default=UNDEFINED)

    @classmethod
    def create_merged_settings(cls, setup_settings: Self, apply_settings: Self) -> Self:
        """
        fetches all dataclass init=True arguments from both settings
        and chooses first non-undefined, apply settings is preferred.
        if both are undefined, arg is skipped.
        Creates new settings instance based on collected arguments.
        """
        kwargs = {}
        for dc_field in dc_fields(cls):
            if not dc_field.init:
                continue
            setup_attr_value = getattr(setup_settings, dc_field.name, UNDEFINED)
            apply_attr_value = getattr(apply_settings, dc_field.name, UNDEFINED)
            attr_value = apply_attr_value if apply_attr_value is not UNDEFINED else setup_attr_value
            if attr_value is not UNDEFINED:
                kwargs[dc_field.name] = attr_value

        return cls(**kwargs)

    # TODO: ...
    # def set_value_accessor(self, value_accessor: IValueAccessor) -> None:
    #     assert isinstance(value_accessor, IValueAccessor)
    #     assert self.value_accessor is None
    #     self.value_accessor = value_accessor

    # def is_trace(self) -> bool:
    #     if isinstance(self._current_context, ConfigOverrideMixin) \
    #       and (ctx_trace := self._current_context.is_trace()) is not None:
    #         return ctx_trace
    #     return self.debug or self.trace

    # def is_debug(self) -> bool:
    #     if isinstance(self._current_context, ConfigOverrideMixin) \
    #             and (ctx_debug := self._current_context.is_debug()) is not None:
    #         return ctx_debug
    #     return self.debug

    @classmethod
    def get_contextns_attributes(cls) -> ExpressionsAttributesMap:
        return {
            "Debug": FieldName("debug"),
            # "Debug": MethodName("is_debug"),
            # "Trace": MethodName("is_trace"),
        }

    # def use_context(self, settings: Optional[IContext]) -> "ConfigSetContextCtxManager":
    #     return ConfigSetContextCtxManager(settings=self, settings=settings)

    # def set_context(self, settings: Union[IContext, None, UndefinedType]):
    #     if settings is not UNDEFINED:
    #         if self._current_context is not UNDEFINED:
    #             raise EntityInternalError(owner=self, msg=f"Can not set new settings to {settings}, already set to: {self._current_context}")
    #     else:
    #         if self._current_context is UNDEFINED:
    #             raise EntityInternalError(owner=self, msg=f"Can not set new settings back to {settings}, it is already reset to: {self._current_context}")
    #     self._current_context = settings


# @dataclass()
# class ConfigSetContextCtxManager(AbstractContextManager):
#     setup_settings: Settings
#     apply_settings: Optional[Settings]
#
#     def __enter__(self):
#         self.settings.set_context(self.settings)
#
#     def __exit__(self, exc_type, exc_value, exc_tb):
#         self.settings.set_context(UNDEFINED)


