# Copied and adapted from Reedwolf project (project by robert.lujo@gmail.com - git@bitbucket.org:trebor74hr/reedwolf.git)
"""
Functions can be registered and used / refereneced in 2 ways:
1. registration and usage of DynanicData:

    data  = [DynanicData(name="Countries", function=Function(py_function, parrams))[

In S case the function will be called with passed params
and reference in attribute way- e.g. 'D.Countries' with no arguments

2. Registered functions:
       functions = [Function("CountAll", py_function, params...)]
   In 'functions' case - function is registereed and can be used in
       Fn.CountAll() 
       M.addressses.CountAll()
   extra args could be passed and must be called as function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from abc import ABC

from typing import (
        Dict,
        Optional,
        ClassVar,
        Any,
        List,
        Callable,
        Union,
        Type,
        Tuple,
        )
from .utils import (
        UNDEFINED,
        get_available_names_example,
        )
from .exceptions import (
        RuleSetupValueError,
        RuleSetupNameError,
        RuleSetupNameNotFoundError,
        RuleInternalError,
        RuleSetupTypeError,
        RuleSetupError,
        RuleApplyError,
        )
from .meta import (
        FunctionArgumentsType,
        TypeInfo,
        is_function,
        EmptyFunctionArguments,
        NoneType,
        )
from .expressions import (
        ValueExpression,
        IValueExpressionNode,
        IFunctionVexpNode,
        ExecResult,
        execute_vexp_or_node,
        )
from .func_args import (
        FunctionArguments,
        create_function_arguments,
        PreparedArguments,
        )


ValueArgValidatorPyFuncType = Callable[..., NoneType]
ValueArgValidatorPyFuncDictType = Dict[str, Union[ValueArgValidatorPyFuncType, List[ValueArgValidatorPyFuncType]]] 

# 4 types
class DatatypeBasicEnum(Enum):
    STANDARD     = 301
    OBJECT       = 302
    # PRESERVED    = 303

class DatatypeCardinalityEnum(Enum):
    SINGLE  = 301
    LIST    = 302

class FunctionEngineBase:
    pass

class PythonFunctionEngine(FunctionEngineBase):
    pass


DEFAULT_ENGINE = PythonFunctionEngine()


# ------------------------------------------------------------
# IFunction - name used to emphasize that this is a function instance
# ------------------------------------------------------------

@dataclass
class IFunction(IFunctionVexpNode):
    """
    single left parameter (value from parent - previous dot chain, parent ) ,
    i.e. function wrapped should not have 2+ required params left
    """
    INPUT_CARDINALITY: ClassVar[DatatypeCardinalityEnum] = DatatypeCardinalityEnum.SINGLE

    # 1. python pure function that will be called
    py_function : Callable[..., Any] 

    # 2. arguments in value expression usage. This is the only required (can be empty ([], {}))
    #   e.g. in chain .my_custom_function(a=1, b=2) # in this case: {"a": 1, "b": 2}
    func_args           : FunctionArgumentsType

    # 3. Registries are required for validation and type *data* of function
    #     arguments, e.g. creating ThisNS, getting vars from ContextNS, DataNS etc.
    registries         : "Registries" = field(repr=False)  # noqa: F821

    # 4. in usage when in chain (value)
    #   e.g. some_struct_str_attr.lower() #  in this case: some_struct_str_attr
    value_arg_type_info : Optional[TypeInfo] = field(default=None)

    # 5. if value_arg_type_info is not supplied then it will be passed to first argument 
    #   e.g. in chain: some-func-returns 100 -> Fn.my_custom_function(d=3) () # in this case: value_arg_name = "d", value = 3
    value_arg_name      : Optional[str] = field(default=None)

    # 6. fixed arguments - when declared.
    #  e.g. Function(my_py_custom_function, c=3)
    #  e.g. Function(my_py_custom_function, fixed_args = ([1, 2], {"a": 3, "b": 4}))
    fixed_args          : FunctionArgumentsType = EmptyFunctionArguments

    # 7. if not provided, then autocomputed
    name                : Optional[str] = field(default=None)

    # 8. if not provided, then autocomputed from py_function type_hints
    function_arguments  : Optional[FunctionArguments] = field(repr=False, default=None)

    # 9. caller - value expression node which calls function, can be None
    caller              : Optional[IValueExpressionNode] = field(default=None)

    # 10. extra validations of input arguments, list of python functions, 
    #    e.g. Length() can operate on objects that have __len__ function (btw.
    #    collection.abc.Sized is used in this case).
    #    Functions are plain python functions, lists of validation functions
    #    are stored in a dict by by argument name.  validation function
    #    receives argument name and argument type. 
    #    For validation error cases should:
    #       * return string error message, or
    #       * raise RuleSetupError based error
    arg_validators      : Optional[ValueArgValidatorPyFuncDictType] = field(repr=False, default=None)

    # 11. Registries are required for validation and type *data* of function
    #     arguments, e.g. creating ThisNS, getting vars from ContextNS, DataNS etc.
    # registries         : Optional["Registries"] = field(repr=False, default=None)  # noqa: F821


    # --- Autocomputed
    # required for IValueExpressionNode
    _output_type_info    : TypeInfo = field(init=False, repr=False)
    # required for IValueExpressionNode
    func_name           : str = field(init=False, repr=False)

    # computed from function_arguments(fixed_args, func_args)
    prepared_args       : PreparedArguments = field(init=False, repr=False)

    is_finished: bool = field(init=False, repr=False, default=False)


    def __post_init__(self):
        if not self.py_function:
            raise RuleInternalError(owner=self, msg=f"py_function input parameter is obligatory")
        if not is_function(self.py_function):
            raise RuleSetupValueError(owenr=self, msg=f"py_function is not a function, got: {self.py_function}")

        self.func_name = self.py_function.__name__

        if not self.name:
            self.name = self.func_name

        if not getattr(self, "full_name", None):
            self.full_name = f"Function.{self.name}" \
                    + (self.func_name if self.func_name!=self.name else "")

        self.engine: FunctionEngineBase = DEFAULT_ENGINE
        # chain_arg_type_info
        # self.value_arg_type_info = self.get_value_arg_type_info()

        self._output_type_info = TypeInfo.extract_function_return_type_info(self.py_function)

        if not self.function_arguments:
            self.function_arguments = create_function_arguments(self.py_function)

        # else: Can be cloned object (.clone()) - reuse same object or not?

        # put this in self.parsed_arguments
        # try:
        self.prepared_args = self.function_arguments.parse_func_args(
                registries=self.registries,
                caller=self.caller,
                owner_name=f"{self.as_str()}",
                func_args=self.func_args,
                fixed_args=self.fixed_args,
                value_arg_type_info=self.value_arg_type_info,
                value_arg_name=self.value_arg_name)
        # except RuleSetupTypeError as ex:
        #     ex.set_msg(f"{self.as_str()}: {ex.msg}")
        #     # raise ex.__class__(f"{self.as_str()}: {ex.msg}")
        #     raise

        # first validate value type matches
        # if self.prepared_args.value_arg_implicit==True:
        # self._validate_value_arg_type(vexp_node=self.caller)
        if self.arg_validators:
            self._call_arg_validators()

        # self.registries.register_vexp_node(self)


    def get_type_info(self) -> TypeInfo:
        return self._output_type_info


    def _call_arg_validators(self):
        """
        validate value from chain / stream - previous dot-node. e.g. 
            M.name.Lower() # value is passed from .name
        """
        assert self.arg_validators

        if not isinstance(self.arg_validators, dict):
            raise RuleSetupValueError(f"Parementer 'arg_validators' should be dictionary, got: {type(self.arg_validators)} / {self.arg_validators}")

        exception_list : List[RuleSetupTypeError] = []

        for arg_name, validators in self.arg_validators.items():
            if arg_name not in self.function_arguments:
                raise RuleSetupValueError(owner=self, msg=f"Argument 'arg_validators' has invalid argument name '{arg_name}'")

            prep_arg = self.prepared_args.get(arg_name)
            if not prep_arg:
                continue

            if not isinstance(validators, (list, tuple)):
                validators = [validators]

            for validator_func in validators:
                if not is_function(validator_func):
                    raise RuleSetupValueError(f"Parementer 'arg_validators[{arg_name}]' is not function, got: {type(validator_func)} / {validator_func}")
                try:
                    err_msg = validator_func(arg_name, prep_arg.type_info)
                    if err_msg:
                        exception_list.append(RuleSetupValueError(f"[{arg_name}] -> {err_msg}"))
                except RuleSetupError as ex:
                    exception_list.append(ex.__class__(f"[{arg_name}] -> {ex.msg}"))

        if exception_list:
            err_msgs = ', '.join([ex.msg for ex in exception_list])
            raise RuleSetupValueError(owner=self, msg=f"{self.as_str()}: Validation(s) failed ({len(exception_list)}): {err_msgs}")

    def as_str(self):
        args_str = "()"
        if self.caller:
            return f"{self.caller.full_name} -> {self.name}{args_str}"
        return f"{self.name}{args_str}"


    def _set_type_info(self, type_info:TypeInfo):
        if self.INPUT_CARDINALITY==DatatypeCardinalityEnum.LIST and not type_info.is_list:
            raise RuleSetupValueError(owner=self, msg=f"Expected 'List' type, got {type_info}.")
        elif self.INPUT_CARDINALITY==DatatypeCardinalityEnum.SINGLE and type_info.is_list:
            raise RuleSetupValueError(owner=self, msg=f"'List' type not valid: {type_info}.")


    # def get_output_type_info(self) -> TypeInfo: return TypeInfo.extract_function_return_type_info(self.py_function)

    # TODO: execute_arg - i.e. from ValueExpression to standard pythoh types
    #       this will need new input parameter: contexts/attr_node/...
    @staticmethod
    def execute_arg(apply_session: "IApplySession", arg_value: Any) -> Any:
        if isinstance(arg_value, (ValueExpression, IValueExpressionNode)):
            # arg_value._evaluator.execute(apply_session=apply_session)
            vexp_result = execute_vexp_or_node(
                            arg_value,
                            arg_value,
                            vexp_result = None,
                            apply_session=apply_session)
            arg_value = vexp_result.value

        return arg_value

    # TODO: IApplySession is in base.py which imports .functions just for one case ...
    def execute_node(self, apply_session: "IApplySession", vexp_result: ExecResult, is_last:bool) -> Any:
        """
        will be called when actual function logic needs to be executed. Input
        is/are function argument(s).

        # TODO: check all input arguments and output type match:
        #       check first / vexp_result argument that matches self.value_arg_type_info
        #       check output type that matches output_type_info
        """
        if is_last and not self.is_finished:
            raise RuleInternalError(owner=self, msg=f"Last vexp-node is not finished")  # , {id(self)} / {type(self)}

        args = []
        kwargs = {}

        if vexp_result is UNDEFINED:
            # namespace toplevel call, e.g. Fn.Length()
            vexp_result = ExecResult()
            top_level_call = True
        else:
            if not isinstance(vexp_result, ExecResult):
                raise RuleInternalError(owner=self, msg=f"vexp_result is not ExecResult, got: {vexp_result}") 
            top_level_call = False
            input_value = vexp_result.value
            if self.value_arg_name:
                kwargs[self.value_arg_name] = input_value
            else:
                args.insert(0, input_value)

        if self.func_args:
            # TODO: copy all arguments or not?
            if self.func_args.args:
                args.extend(self.func_args.args)
            if self.func_args.kwargs:
                kwargs.update(self.func_args.kwargs)

        if self.fixed_args:
            # TODO: copy or not?
            if self.fixed_args.args:
                args.extend(self.fixed_args.args)
            if self.fixed_args.kwargs:
                kwargs.update(self.fixed_args.kwargs)


        args   = [self.execute_arg(apply_session, arg_value) for arg_value in args]
        kwargs = {arg_name: self.execute_arg(apply_session, arg_value) for arg_name, arg_value in kwargs.items()}

        try:
            ouptut_value = self.py_function(*args, **kwargs)
        except Exception as ex:
            raise RuleApplyError(owner=self, msg=f"failed in calling '{self.name}({args}, {kwargs})' => {ex}")

        vexp_result.set_value(ouptut_value, attr_name="", changer_name=f"{self.name}")

        return vexp_result


# ------------------------------------------------------------
# CustomFunctions - need 2 steps and 3 layers:
#   Function -> CustomFunctionFactory -> CustomFunction
# ------------------------------------------------------------

@dataclass
class CustomFunction(IFunction):
    pass


@dataclass
class BuiltinFunction(IFunction):
    pass

# ------------------------------------------------------------

@dataclass
class IFunctionFactory(ABC):

    """
    Function wrapper for arbitrary regular python function.
    that will reside in Global namespace and can not be 
    e.g. can be used in DynanicData-s and similar in/out bindings.

    inject_args should be used as functools.partial()
    later reference will fill func_args
    """
    # regular python function - should be pure (do not change input args,
    # rather copy and return modified copy
    py_function : Callable[..., Any]

    # fixed arguments - when declared - see IFunctionVexpNode.fixed_args
    fixed_args     : FunctionArgumentsType = field(default=EmptyFunctionArguments)

    # can be evaluated later
    name           : Optional[str] = field(default=None) 

    # for dot-chained - when kwarg to use, target parameter
    value_arg_name : Optional[str] = field(default=None) 

    # # validator function - see details in IFunction.arg_validators
    arg_validators : Optional[ValueArgValidatorPyFuncDictType] = field(default=None, repr=False)

    # autocomputed
    _output_type_info: TypeInfo = field(init=False, repr=False)

    def __post_init__(self):
        if not is_function(self.py_function):
            raise RuleSetupValueError(owner=self, msg=f"py_function is not a function, got: {type(self.py_function)} / {self.py_function}")

        if not self.name:
            self.name = self.py_function.__name__
        if self.fixed_args is None:
            self.fixed_args = ()
        self._output_type_info = TypeInfo.extract_function_return_type_info(self.py_function)

    # def get_type_info(self) -> TypeInfo: return self._output_type_info

    def create_function(self, 
                func_args:FunctionArgumentsType, 
                registries         : Registries,
                value_arg_type_info: Optional[TypeInfo] = None,
                name               : Optional[str] = None,
                caller             : Optional[IValueExpressionNode] = None,
                ) -> IFunction:
        custom_function = self.FUNCTION_CLASS(
                py_function         = self.py_function,     # noqa: E251
                fixed_args          = self.fixed_args,      # noqa: E251
                func_args           = func_args,            # noqa: E251
                value_arg_type_info = value_arg_type_info,  # noqa: E251
                value_arg_name      = self.value_arg_name,  # noqa: E251
                name                = name if name else self.name, # noqa: E251
                caller              = caller,               # noqa: E251
                registries          = registries,           # noqa: E251
                arg_validators      = self.arg_validators,  # noqa: E251
                )
        return custom_function

    def get_type_info(self) -> TypeInfo:
        return self._output_type_info


# ------------------------------------------------------------

def Function(py_function : Callable[..., Any], 
             name: Optional[str] = None,
             value_arg_name:Optional[str]=None,
             # TODO: not only Vexp, can be literal too, e.g. 1, 2.3, "a"
             args   : Optional[List[ValueExpression]] = UNDEFINED,
             kwargs : Optional[Dict[str, ValueExpression]] = UNDEFINED,
             arg_validators : Optional[ValueArgValidatorPyFuncDictType] = None,
             ) -> CustomFunctionFactory:
    """
    goes to Global namespace (Fn.)
    can accept predefined params (like partial) 
    and in use (later) can be called with rest params.
    can not be used in ValueExpression chains (e.g. D.Country.SomeFunction()

    Usage: 
        def add(a, b, c=0):
            return a+b+c

        # get function wrapper, 
        Add = Function(add, kwargs=dict(a=Fn.some_function))

        # reference and store in wrapped function
        add_function = Fn.Add(b=This.value, c=3)

        # call / execute referenced function wrapper
        print (add_function.execute(ctx=this_ctx, registries)) 
    """
    # value_arg_type_info = None # TODO: TypeInfo
    return CustomFunctionFactory(
                py_function=py_function,
                name=name,
                value_arg_name=value_arg_name,
                fixed_args=FunctionArgumentsType(
                    args if args is not UNDEFINED else (), 
                    kwargs if kwargs is not UNDEFINED else {}
                    ),
                arg_validators=arg_validators,
                )

# ------------------------------------------------------------

def create_builtin_function_factory(
            py_function : Callable[..., Any], 
            name: Optional[str] = None,
            value_arg_name:Optional[str]=None,
            args   : Optional[List[ValueExpression]] = None,
            kwargs: Optional[Dict[str, ValueExpression]] = None,
            arg_validators : Optional[ValueArgValidatorPyFuncDictType] = None,
            ) -> CustomFunctionFactory:
    """
    wrapper around BuiltinFunctionFactory - better api look
    TODO: consider creating decorator
    """
    return BuiltinFunctionFactory(
                py_function=py_function,
                name=name,
                value_arg_name=value_arg_name,
                fixed_args=FunctionArgumentsType(
                                args if args is not None else (), 
                                kwargs if kwargs is not None else {}),
                arg_validators=arg_validators,
                )

# ------------------------------------------------------------

@dataclass
class CustomFunctionFactory(IFunctionFactory):
    FUNCTION_CLASS : ClassVar[Type[IFunction]] = CustomFunction

# ------------------------------------------------------------

@dataclass
class BuiltinFunctionFactory(IFunctionFactory):
    FUNCTION_CLASS : ClassVar[Type[IFunction]] = BuiltinFunction


# =====================================================================
# FunctionsFactoryRegistry
#    classes or functions when called create instances of IFunction
#    Currently IFunction class and CustomFunctionFactory function
# =====================================================================

class FunctionsFactoryRegistry:

    def __init__(self, functions: Optional[List[CustomFunctionFactory]] = None,
                 include_standard:bool = True, 
                 ):
        """
        functions are custom_function_factories
        """
        # TODO: typing is not good
        #       Union[type(IFunction), CustomFunctionFactory]
        self.store: Dict[str, Type(IFunction)]= {}
        self.include_standard:bool = include_standard

        if self.include_standard:
            self.register_standard_functions()

        if functions:
            for function_factory in functions:
                if not isinstance(function_factory, CustomFunctionFactory):
                    raise RuleSetupNameError(owner=self, msg=f"Function '{function_factory}' should be CustomFunctionFactory instance. Maybe you need to wrap it with Function()?")
                self.add(function_factory)

    def __str__(self):
        return f"{self.__class__.__name__}(functions={', '.join(self.store.keys())}, include_standard={self.include_standard})"
    __repr__ = __str__

    def items(self) -> List[Tuple[str, Type(IFunction)]]:
        return self.store.items()

    # @staticmethod
    # def is_function_factory(function_factory: Any, exclude_custom: bool = False):
    #     out = isinstance(function_factory, IFunctionFactory)
    #     # out = (inspect.isclass(function_factory) 
    #     #         and IFunction in inspect.getmro(function_factory) 
    #     #         and not inspect.isabstract(function_factory)
    #     #         and function_factory not in (CustomFunction, IFunction)
    #     #         )
    #     if not out and not exclude_custom:
    #         out = out or isinstance(function_factory, (CustomFunctionFactory))
    #     return out


    def add(self, function_factory: IFunctionFactory, func_name: Optional[str]=None):
        # IFunction
        if not isinstance(function_factory, IFunctionFactory):
            raise RuleInternalError(f"Exepected function factory, got: {type(function_factory)} -> {function_factory}")

        if not func_name:
            func_name = function_factory.name

        assert isinstance(func_name, str) and func_name, func_name

        if func_name in self.store:
            raise RuleSetupNameError(owner=self, msg=f"Function '{func_name}' already in registry: {self.store[func_name]}.")
        self.store[func_name] = function_factory


    def get(self, name:str, strict:bool=False) -> Optional[IFunction]:
        if strict and name not in self.store:
            funcs_avail = get_available_names_example(name, self.store.keys())
            raise RuleSetupNameNotFoundError(owner=self, msg=f"Function '{name}' is not valid. Valid are: {funcs_avail}")
        function = self.store.get(name, None)
        return function

    def get_all_names(self, first_custom:bool = False) -> List[str]:
        if first_custom:
            out = [fnname for fnname in self.store.keys() if fnname.islower()]
            out += [fnname for fnname in self.store.keys() if not fnname.islower()]
            return out
        return list(self.store.keys())

    def register_standard_functions(self) -> int:
        # init_functions_factory_registry
        """
        Traverse all classes that inherit IFunction and register them
        This function needs to be inside of this module or change to: import .functions; for ... vars(functions)
        """
        from .func_standard import get_builtin_function_factories_dict

        items = get_builtin_function_factories_dict().items()
        for func_name, builtin_function in items:
            self.add(builtin_function, func_name=func_name)

        assert len(self.store)>=3, self.store
        return len(self.store)

    def dump_all(self):
        print("Functions:")
        for func_name, function in self.store.items():
            print(f"  {func_name} -> {function}")


# ------------------------------------------------------------

# ------------------------------------------------------------

def try_create_function(
        registries: "Registries",  # noqa: F821
        caller: IValueExpressionNode,
        # functions_factory_registry: IFunctionFactory, 
        attr_node_name:str, 
        func_args: FunctionArgumentsType,
        value_arg_type_info: TypeInfo,
        ) -> IFunction:

    if value_arg_type_info and not isinstance(value_arg_type_info, TypeInfo):
        raise RuleInternalError(f"{value_arg_type_info} should be TypeInfo, got '{type(value_arg_type_info)}'")

    functions_factory_registry: IFunctionFactory = registries.functions_factory_registry
    if not functions_factory_registry:
        raise RuleSetupNameNotFoundError(item=registries, msg=f"Functions not available, '{attr_node_name}' function could not be found.")

    function_factory = None
    registries_current = registries

    names_avail_all = []
    registry_ids = set()

    while True:
        if id(registries_current) in registry_ids:
            # prevent infinitive loop
            raise RuleInternalError("Registry already processed")
        registry_ids.add(id(registries_current))

        function_factory = registries_current.functions_factory_registry.get(attr_node_name)
        if function_factory:
            break

        # TODO: filter functions based on type: is_list or similar ...
        first_custom = (attr_node_name and attr_node_name[0].islower())
        names_avail = get_available_names_example(attr_node_name,
                            functions_factory_registry.get_all_names(
                                first_custom=first_custom))
        names_avail_all.append(names_avail)

        if not registries.owner.owner:
            break
        registries_current = registries.owner.owner.registries

    func_node = None
    if function_factory:
        # ===== Create IFunction instance ===== 
        func_node = function_factory.create_function(
                        registries=registries,
                        caller=caller,
                        func_args=func_args,
                        value_arg_type_info=value_arg_type_info)

    if not func_node:
        names_avail_all = " ; ".join(names_avail_all)
        raise RuleSetupNameNotFoundError(f"Function name '{attr_node_name}' not found. Valid are: {names_avail_all}")

    return func_node
