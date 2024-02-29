import inspect
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Callable, Tuple

from .exceptions import EntitySetupNameError
from .meta import (
    UNDEFINED,
    IAttribute,
    is_instancemethod_by_name,
    FunctionNoArgs,
    TypeInfo,
    SettingsSource,
    ModelField, SELF_ARG_NAME, get_function_non_empty_arguments,
)
from .utils import get_available_names_example


@dataclass
class Attribute(IAttribute):
    """
    Attribute from dataclass Field's name.
    """
    attr_field: ModelField = field(repr=False, init=False, default=None)

    def setup_dexp_attr_source(self, settings_source_list: List[SettingsSource]) -> SettingsSource:
        attr_field = settings_source = UNDEFINED
        for settings_source in settings_source_list:
            attr_field = settings_source.fields.get(self.name, UNDEFINED)
            if attr_field is not UNDEFINED:
                break

        if attr_field is UNDEFINED:
            all_keys = set()
            for settings_source in settings_source_list:
                all_keys.union(set(settings_source.fields.keys()))
            models = [settings_source.klass for settings_source in settings_source_list]
            aval_names = get_available_names_example(attr_field, list(all_keys))
            raise EntitySetupNameError(owner=self,
                                       msg=f"Attribute {self.name} must be field name of class(es) '{models}', available: {aval_names}")

        # NOTE: attr_field is not used later
        self.output_type_info = TypeInfo.get_or_create_by_type(py_type_hint=attr_field, caller=settings_source.klass)
        self.settings_source = settings_source
        self.attr_field = attr_field

        return self.settings_source


@dataclass
class AttributeByMethod(IAttribute):
    """
    Attribute from to method's by its name.
    """
    # TODO: try to reuse IFunction() logic to enable more advanced method configuration by using: args/kwargs
    #    # NOTE: make list of input fields/args in sync with functions.py :: FunctionFactoryBase
    #    args: Optional[List[Union[LiteralType, DotExpression]]] = UNDEFINED
    #    kwargs: Optional[Dict[str, Union[LiteralType, DotExpression]]] = UNDEFINED
    #    arg_validators: Optional[ValueArgValidatorPyFuncDictType] = None
    # TODO: if previous task is done, then maybe make AttributeByFunction()

    # filled later
    output_type_info: TypeInfo = field(repr=False, init=False, default=None)
    py_function: Callable = field(repr=False, init=False, default=None)
    settings_source: SettingsSource = field(repr=False, init=False, default=None)

    def setup_dexp_attr_source(self, settings_source_list: List[SettingsSource]) -> SettingsSource:
        # NOTE: similar logic in functions.py :: FunctionByMethod.set_settings_class()
        py_function = settings_source = UNDEFINED
        for settings_source in settings_source_list:
            py_function: FunctionNoArgs = getattr(settings_source.klass, self.name, UNDEFINED)
            if py_function is not UNDEFINED:
                break

        if py_function is UNDEFINED:
            # TODO: could I get all methods with no args?
            models = [settings_source.klass for settings_source in settings_source_list]
            raise EntitySetupNameError(owner=self,
                                       msg=f"Method name '{self.name}' is not found within class(es): {models}")

        klass = settings_source.klass
        function_name = py_function.__name__
        if not is_instancemethod_by_name(klass, function_name):
            raise EntitySetupNameError(owner=self, msg=f"Function '{self.name}' is not instance method of class '{klass.__name__}'.")

        # Check that function receives only single param if method(self), or no param if function()
        non_empty_params = get_function_non_empty_arguments(py_function)
        if len(non_empty_params) != 0:
            raise EntitySetupNameError(owner=self,
                                       msg=f"Method '{klass.__name__}.{self.name}()' must not have arguments without defaults. Found unfilled arguments: {', '.join(non_empty_params)} ")

        # NOTE: py_function is not used later
        self.output_type_info = TypeInfo.extract_function_return_type_info(py_function, allow_nonetype=True)
        self.py_function = py_function
        self.settings_source = settings_source

        return settings_source



