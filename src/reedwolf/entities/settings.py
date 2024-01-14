from collections import OrderedDict
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from inspect import isclass, getmro
from typing import Optional, Union, List, Type, Dict, Tuple

from .exceptions import (
    EntityInternalError,
    EntitySetupNameError,
    EntityTypeError,
    EntityNameNotFoundError, EntityInstatiateError,
)
from .meta import (
    Self,
    AttrName,
    IAttribute,
    SettingsType,
    SettingsSource,
    CustomCtxAttributeList,
)
from .expressions import IFunctionFactory
from .custom_attributes import AttributeByMethod
from .utils import (
    UndefinedType,
    UNDEFINED,
    get_available_names_example,
)
from .value_accessors import (
    IValueAccessor,
    AttributeValueAccessor,
    get_standard_accessor_class_registry,
    ValueAccessorCode,
    ValueAccessorInputType,
)

CustomFunctionFactoryList = List[IFunctionFactory]

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
    # default_factory must be IValueAccessor class with no required arguments
    accessor: Optional[ValueAccessorInputType] = field(repr=False, default=None)

    # -----------------------------------------------------------
    # Tracing, analytics, debug ...
    # -----------------------------------------------------------
    # None indicates that argument/param is not passed in constructor
    debug: Union[bool, UndefinedType] = UNDEFINED

    # Should all changes of values are collected in
    # ApplyResult.value_history_dict and ValueNode.value_history
    # None indicates that argument/param is not passed in constructor
    trace: Union[bool, UndefinedType] = UNDEFINED

    _accessor: IValueAccessor = field(init=False, repr=False, default=UNDEFINED)

    @classmethod
    # TODO: CustomFunctionFactory
    def get_custom_functions(cls) -> CustomFunctionFactoryList:
        """
        For override.
        Defined on class level.
        For (setup) Settings, can be overridden by instance attribute custom_functions.
        TODO: put example
        """
        return []

    @classmethod
    def get_custom_ctx_attributes(cls) -> CustomCtxAttributeList:
        """
        For override.
        Defined on class level.
        Values are fetched from this instance only. Usually overridden.
        Defined on class level.
        For (setup) Settings, can be overridden by instance attribute custom_ctx_attributes.

        Example:
            {
            "Now": AttributeByMethod("get_now"),
            "UserId": Attribute("user_id"),
            }
        """
        return {}


@dataclass
class ApplySettings(SettingsBase):
    """
    Setup settings can implement Ctx. (ContextNS) attributes and ApplySettings too.
    If both defined then last wins.
    Setup settings can implement Ctx. (ContextNS) functions and ApplySettings too.
    If both defined then last wins.
    """
    ...


_BUILTIN_FUNCTION_FACTORIES_DICT = None


def get_builtin_function_factories_dict() -> Dict[str, IFunctionFactory]:
    """
    will return same instance every time
    CACHED
    """
    global _BUILTIN_FUNCTION_FACTORIES_DICT
    if _BUILTIN_FUNCTION_FACTORIES_DICT is None:
        # TODO: resolve this properly
        from . import func_builtin

        out: Dict[str, IFunctionFactory] = {}
        for func_name, global_var in vars(func_builtin).items():
            if not (global_var and isinstance(global_var, IFunctionFactory)):
                continue
            assert func_name
            out[func_name] = global_var
        _BUILTIN_FUNCTION_FACTORIES_DICT = out
    return _BUILTIN_FUNCTION_FACTORIES_DICT


# ------------------------------------------------------------

@dataclass
class Settings(SettingsBase):
    """
    Setup settings can implement Ctx. (ContextNS) variables:
    - in class method get_custom_ctx_attributes() and/or
    - in instance attribute custom_ctx_attributes
    If both defined then last wins.

    Setup settings can implement Ctx. (ContextNS) functions:
    - in class method get_custom_functions() and/or
    - in instance attribute custom_functions
    If both defined then last wins.
    """
    # TODO: CustomFunctionFactory
    custom_functions: CustomFunctionFactoryList = field(repr=False, default_factory=list, metadata={"skip_dump": True})
    custom_ctx_attributes: CustomCtxAttributeList = field(repr=False, default_factory=list)
    custom_accessor_class_registry : Dict[str, Type[IValueAccessor]] = field(repr=False, default_factory=dict)
    apply_settings_class: Optional[Type[ApplySettings]] = field(repr=False, default=None)

    # set and reset back in apply phase
    _apply_settings: Union[Self, None, UndefinedType] = field(init=False, repr=False, compare=False, default=UNDEFINED)
    _all_custom_functions: Optional[CustomFunctionFactoryList] = field(init=False, repr=False, compare=False, default=None)
    # _all_builtin_functions_dict: Optional[Dict[AttrName, IFunctionFactory]] = field(init=False, repr=False, compare=False, default=None)
    _accessor_class_registry : Optional[Dict[str, Type[IValueAccessor]]] = field(init=False, repr=False, default=None)

    # ------------------------------------------------------------
    def _init(self):
        # multiple calls won't matter - will just skip
        self._accessor = self._get_accessor(self.accessor if self.accessor else AttributeValueAccessor)

    def _get_accessor(self, accessor: ValueAccessorInputType) -> IValueAccessor:

        # 1st case
        if isinstance(accessor, IValueAccessor):
            return accessor

        # 2nd case
        if isclass(accessor) and IValueAccessor in getmro(accessor):
            try:
                accessor = accessor()
            except Exception as ex:
                raise EntityInstatiateError(owner=self, msg=f"Can not create instance from: {accessor}") from ex
            return accessor

        if not isinstance(accessor, ValueAccessorCode):
            raise EntityTypeError(owner=self, msg=f"Invalid type of accessor: {accessor}. Expecting IValueAccessor type or instance, or string code of some of a registered accessor.")

        # 3rd case:
        accessor_class_registry = self._get_standard_accessor_class_registry()
        if accessor not in accessor_class_registry:
            avail_names = get_available_names_example(accessor, list(accessor_class_registry.keys()), max_display=15)
            raise EntityNameNotFoundError(owner=self, msg=f"Value accessor with code '{accessor}' is not recognized. Valid codes: {avail_names}")

        accessor_type = accessor_class_registry[accessor]
        try:
            accessor = accessor_type()
        except Exception as ex:
            raise EntityInstatiateError(owner=self, msg=f"Can not create instance from: {accessor_type}") from ex

        return accessor


    def _get_standard_accessor_class_registry(self) -> Dict[str, Type[IValueAccessor]]:
        """
        copies standard registry and updates with custom
        TODO: currently not used
        """
        if self._accessor_class_registry is None:
            self._accessor_class_registry = get_standard_accessor_class_registry().copy()
            self._accessor_class_registry.update(self.custom_accessor_class_registry)
        return self._accessor_class_registry

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

    def _custom_function_list_to_dict(self, functions: CustomFunctionFactoryList) -> Dict[AttrName, IFunctionFactory]:
        function_dict = OrderedDict()
        for func in functions:
            if func.name in function_dict:
                raise EntitySetupNameError(owner=self,
                                           msg=f"Found duplicate function name: {func.name}. Pass unique 'name' attribute.")
            function_dict[func.name] = func
        return function_dict

    def get_all_custom_functions(self) -> CustomFunctionFactoryList:
        if self._all_custom_functions is None:
            raise EntityInternalError(owner=self, msg="Call _get_all_custom_functions() first")
        return self._all_custom_functions

    def _setup_all_custom_functions(self, apply_settings_class: Optional[Type[ApplySettings]]):
        # make a copy and merge with instance defined functions
        if self._all_custom_functions is not None:
            raise EntityInternalError(owner=self, msg=f"Attribute ._all_custom_functions already set.")

        custom_functions = self._custom_function_list_to_dict(self.get_custom_functions())
        custom_functions.update(self._custom_function_list_to_dict(self.custom_functions))

        apply_custom_functions = self._custom_function_list_to_dict(apply_settings_class.get_custom_functions()) \
                                 if apply_settings_class else {}
        # this one wins. Last wins.
        custom_functions.update(apply_custom_functions)

        # TODO: resolve this dependency properly
        from .functions import FunctionByMethod
        custom_functions = list(custom_functions.values())
        for function in custom_functions:
            if not isinstance(function, IFunctionFactory):
                raise EntityTypeError(owner=self,
                                      msg=f"Function '{function.name}' needs to be class of IFunctionFactory, got: {function}. Have you used Function() or FunctionByMethod()?")

            if isinstance(function, FunctionByMethod):
                settings_type, settings_class = (SettingsType.APPLY_SETTINGS, apply_settings_class) \
                                                if function.name in apply_custom_functions else \
                                                (SettingsType.SETUP_SETTINGS, self.__class__)
                function.set_settings_class(settings_type=settings_type, settings_class=settings_class)
        self._all_custom_functions = custom_functions


    def get_all_builtin_functions_dict(self) -> Dict[AttrName, IFunctionFactory]:
        """
        can be overridden in order to setup own standard functions
        TODO: not so nice - but good enough for the start
        """
        # if self._all_builtin_functions_dict is None:
        #     self._all_builtin_functions_dict = get_builtin_function_factories_dict()
        # return self._all_builtin_functions_dict
        return get_builtin_function_factories_dict()

    def _custom_ctx_attribute_list_to_dict(self, attributes: CustomCtxAttributeList) -> Dict[AttrName, IAttribute]:
        attribute_dict = OrderedDict()
        for attr in attributes:
            if attr.dexp_attr_name in attribute_dict:
                raise EntitySetupNameError(owner=self,
                                           msg=f"Found duplicate attribute name: {attr.dexp_attr_name}. Pass unique 'dexp_attr_name' attribute.")
            attribute_dict[attr.dexp_attr_name] = attr
        return attribute_dict


    def _get_attribute_settings_source_list_pairs(self, apply_settings_class: Optional[Type[ApplySettings]]) \
        -> List[Tuple[Dict[AttrName, IAttribute], List[SettingsSource]]]:
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
        common_dict = self._custom_ctx_attribute_list_to_dict(self._get_common_contextns_attributes())

        # make a copy and merge with instance defined attributes
        setup_custom_dict = self._custom_ctx_attribute_list_to_dict(self.get_custom_ctx_attributes())
        setup_custom_dict.update(self._custom_ctx_attribute_list_to_dict(self.custom_ctx_attributes))

        if apply_settings_class:
            apply_settings_source = SettingsSource(SettingsType.APPLY_SETTINGS, apply_settings_class)
            apply_custom_dict = self._custom_ctx_attribute_list_to_dict(apply_settings_class.get_custom_ctx_attributes())
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
    def _get_common_contextns_attributes(cls) -> CustomCtxAttributeList:
        """
        Values are fetched from apply_settings and setup_settings - fetched from first available. Usually not overridden.
        recommendation on override is to merge dict with super(), e.g.:
            out super().common_contextns_attributes()
            out.update({...})
            return out
        """
        return [
            AttributeByMethod("is_debug", "Debug"),
        ]

    def _use_apply_settings(self, apply_settings: Optional[Self]) -> "UseApplySettingsCtxManager":
        return UseApplySettingsCtxManager(setup_settings=self, apply_settings=apply_settings)

    def _set_apply_settings(self, apply_settings: Union[Self, None, UndefinedType]):
        is_reset = (apply_settings is UNDEFINED)
        if not is_reset:
            if self._apply_settings is not UNDEFINED:
                raise EntityInternalError(owner=self, msg=f"Can not set new settings to {apply_settings}, already set to: {self._apply_settings}")
        else:
            if self._apply_settings is UNDEFINED:
                raise EntityInternalError(owner=self, msg=f"Can not set new settings back to {apply_settings}, it is already reset to: {self._apply_settings}")
        self._apply_settings = apply_settings

        # TODO: resolve this dependency properly
        from .functions import FunctionByMethod
        for function in self.get_all_custom_functions():
            if isinstance(function, FunctionByMethod):
                settings = (apply_settings if function.settings_type == SettingsType.APPLY_SETTINGS else self) \
                           if not is_reset else UNDEFINED
                function.set_settings_instance(settings)



@dataclass()
class UseApplySettingsCtxManager(AbstractContextManager):
    setup_settings: Settings
    apply_settings: Optional[ApplySettings]

    def __enter__(self):
        self.setup_settings._set_apply_settings(self.apply_settings)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.setup_settings._set_apply_settings(apply_settings=UNDEFINED)

