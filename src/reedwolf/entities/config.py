from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Optional, Union

from .exceptions import EntityInternalError
from .meta import ExpressionsAttributesDict, FieldName
from .contexts import IContext, ConfigOverrideMixin
from .utils import UndefinedType, UNDEFINED
from .values_accessor import IValueAccessor

# ------------------------------------------------------------
# IConfig
# ------------------------------------------------------------


@dataclass
class Config(IContext):
    """
    The Config instances contain general predefined Entity configuration parameters (settings).
    One can add custom config params.
    Config will be available in ConfigNS namespace (Cfg.).
    For values only literal / plain callables (python functions) are accepted,
    no DotExpression or Function() instances allowed.
    and belonging.
    This is plain class, no setup()/Setup() process.
    """
    # if not set will use default ValueExpress
    value_accessor: Optional[IValueAccessor] = field(default=None, metadata={"dexp_exposed": False})

    # -----------------------------------------------------------
    # Tracing, analytics, debug ...
    # -----------------------------------------------------------
    debug: bool = False

    # Should all changes of values are collected in
    # ApplyResult.value_history_dict and ValueNode.value_history
    trace: bool = False

    # set and reset back in apply phase
    _current_context: Union[IContext, None, UndefinedType] = field(init=False, repr=False, compare=False, default=UNDEFINED)

    _trace: Union[bool, UndefinedType] = field(init=False, repr=False, compare=False, default=UNDEFINED)
    _debug: Union[bool, UndefinedType] = field(init=False, repr=False, compare=False, default=UNDEFINED)

    # def set_value_accessor(self, value_accessor: IValueAccessor) -> None:
    #     assert isinstance(value_accessor, IValueAccessor)
    #     assert self.value_accessor is None
    #     self.value_accessor = value_accessor

    def is_trace(self) -> bool:
        if isinstance(self._current_context, ConfigOverrideMixin) \
          and (ctx_trace := self._current_context.is_trace()) is not None:
            return ctx_trace
        return self.debug or self.trace

    def is_debug(self) -> bool:
        if isinstance(self._current_context, ConfigOverrideMixin) \
                and (ctx_debug := self._current_context.is_debug()) is not None:
            return ctx_debug
        return self.debug

    @classmethod
    def get_expressions_attributes(cls) -> ExpressionsAttributesDict:
        return {
            # "Debug": FieldName("debug"),
            "Debug": cls.is_debug,
            # "Trace": cls.is_trace,
        }

    def use_context(self, context: Optional[IContext]) -> "ConfigSetContextCtxManager":
        return ConfigSetContextCtxManager(config=self, context=context)

    def set_context(self, context: Union[IContext, None, UndefinedType]):
        if context is not UNDEFINED:
            if self._current_context is not UNDEFINED:
                raise EntityInternalError(owner=self, msg=f"Can not set new context to {context}, already set to: {self._current_context}")
        else:
            if self._current_context is UNDEFINED:
                raise EntityInternalError(owner=self, msg=f"Can not set new context back to {context}, it is already reset to: {self._current_context}")
        self._current_context = context


@dataclass()
class ConfigSetContextCtxManager(AbstractContextManager):
    config: Config
    context: Optional[IContext]

    def __enter__(self):
        self.config.set_context(self.context)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.config.set_context(UNDEFINED)


