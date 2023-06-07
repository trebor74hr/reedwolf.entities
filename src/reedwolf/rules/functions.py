"""
Functions can be registered and used only if registered within container attribute:

    functions=[Function(),...]

e.g.:

   functions = [Function("CountAll", py_function, params...)]

function is registereed and can be used in

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
from .namespaces import (
        FieldsNS,
        )
from .meta import (
        FunctionArgumentsType,
        TypeInfo,
        is_function,
        EmptyFunctionArguments,
        NoneType,
        )
from .expressions import (
        DotExpression,
        IDotExpressionNode,
        IFunctionDexpNode,
        ExecResult,
        execute_dexp_or_node,
        ISetupSession,
        )
from .func_args import (
        FunctionArguments,
        create_function_arguments,
        PreparedArguments,
        )
from .base import (
        AttrDexpNodeTypeEnum,
        ReservedArgumentNames,
        IFieldBase,
        )


ValueArgValidatorPyFuncType = Callable[..., NoneType]
ValueArgValidatorPyFuncDictType = Dict[str, Union[ValueArgValidatorPyFuncType, List[ValueArgValidatorPyFuncType]]] 

ValueOrDexp = Union[DotExpression, IDotExpressionNode, Any]

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
class IFunction(IFunctionDexpNode):
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

    # 3. SetupSession are required for validation and type *data* of function
    #     arguments, e.g. creating ThisNS, getting vars from ContextNS, etc.
    setup_session         : ISetupSession = field(repr=False)  # noqa: F821

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
    caller              : Optional[IDotExpressionNode] = field(default=None)

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

    # 11. SetupSession are required for validation and type *data* of function
    #     arguments, e.g. creating ThisNS, getting vars from ContextNS etc.
    # setup_session         : Optional[ISetupSession] = field(repr=False, default=None)  # noqa: F821

    # misc data, used in EnumMembers
    data: Optional[Any] = field(default=None, repr=False)

    # --- Autocomputed
    # required for IDotExpressionNode
    _output_type_info    : TypeInfo = field(init=False, repr=False)
    # required for IDotExpressionNode
    func_name           : str = field(init=False, repr=False)

    # computed from function_arguments(fixed_args, func_args)
    prepared_args       : PreparedArguments = field(init=False, repr=False)

    is_finished: bool = field(init=False, repr=False, default=False)


    def __post_init__(self):
        if not self.py_function:
            raise RuleInternalError(owner=self, msg="py_function input parameter is obligatory")
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
        self.prepared_args = self.function_arguments.parse_func_args(
                setup_session=self.setup_session,
                caller=self.caller,
                owner_name=f"{self.as_str()}",
                func_args=self.func_args,
                fixed_args=self.fixed_args,
                value_arg_type_info=self.value_arg_type_info,
                value_arg_name=self.value_arg_name)

        # first validate value type matches
        # if self.prepared_args.value_arg_implicit==True:
        # self._validate_value_arg_type(dexp_node=self.caller)
        if self.arg_validators:
            self._call_arg_validators()

        # self.setup_session.register_dexp_node(self)


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

    # TODO: execute_arg - i.e. from DotExpression to standard pythoh types
    #       this will need new input parameter: contexts/attr_node/...
    @staticmethod
    def execute_arg(
            apply_session: "IApplySession", # noqa: F821
            arg_value: ValueOrDexp,
            prev_node_type_info:TypeInfo, 
            ) -> Any:
        if isinstance(arg_value, (DotExpression, IDotExpressionNode)):
            # arg_value._evaluator.execute(apply_session=apply_session)
            dexp_result = execute_dexp_or_node(
                            arg_value,
                            arg_value,
                            dexp_result = UNDEFINED,
                            prev_node_type_info=prev_node_type_info,
                            apply_session=apply_session)
            arg_value = dexp_result.value

        return arg_value


    # TODO: IApplySession is in base.py which imports .functions just for one case ...
    def execute_node(self, 
            apply_session: "IApplySession", # noqa: F821
            dexp_result: ExecResult, 
            is_last:bool,
            prev_node_type_info: TypeInfo,
            ) -> Any:
        """
        will be called when actual function logic needs to be executed. Input
        is/are function argument(s).

        # TODO: check all input arguments and output type match:
        #       check first / dexp_result argument that matches self.value_arg_type_info
        #       check output type that matches output_type_info
        """
        if is_last and not self.is_finished:
            raise RuleInternalError(owner=self, msg="Last dexp-node is not finished")  # , {id(self)} / {type(self)}

        args = []
        kwargs = {}

        if dexp_result is UNDEFINED:
            # top_level_call = True
            # namespace toplevel call, e.g. Fn.Length()
            dexp_result = ExecResult()
        else:
            # top_level_call = False
            if not isinstance(dexp_result, ExecResult):
                raise RuleInternalError(owner=self, msg=f"dexp_result is not ExecResult, got: {dexp_result}") 

            if dexp_result.value is not UNDEFINED:
                input_value = dexp_result.value
                if self.value_arg_name:
                    kwargs[self.value_arg_name] = input_value
                else:
                    args.insert(0, input_value)

            self._process_inject_pargs(apply_session=apply_session, kwargs=kwargs)


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

        args   = [self.execute_arg(apply_session, arg_value, prev_node_type_info=prev_node_type_info) 
                  for arg_value in args]
        kwargs = {arg_name: self.execute_arg(apply_session, arg_value,prev_node_type_info=prev_node_type_info) 
                  for arg_name, arg_value in kwargs.items()}

        try:
            ouptut_value = self.py_function(*args, **kwargs)
        except Exception as ex:
            raise RuleApplyError(owner=self, msg=f"failed in calling '{self.name}({args}, {kwargs})' => {ex}")

        dexp_result.set_value(ouptut_value, attr_name="", changer_name=f"{self.name}")

        return dexp_result

    # ------------------------------------------------------------

    def _process_inject_pargs(self, apply_session: IApplySession, kwargs: Dict[str, ValueOrDexp]):

        prep_arg = self.prepared_args.get(ReservedArgumentNames.INJECT_COMPONENT_TREE)
        if not prep_arg:
            return 

        if not isinstance(prep_arg.caller, IDotExpressionNode):
            raise RuleInternalError(owner=self, msg=f"Expected IDotExpressionNode, got: {type(prep_arg.caller)} / {prep_arg.caller}") 

        dexp_node: IDotExpressionNode = prep_arg.caller

        if not (dexp_node.attr_node_type == AttrDexpNodeTypeEnum.FIELD
            and dexp_node.namespace == FieldsNS
            and isinstance(dexp_node.data, IFieldBase)):

            raise RuleInternalError(owner=self, msg=f"INJECT_COMPONENT_TREE :: PrepArg '{prep_arg.name}' expected DExp(F.<field>) -> Field(),  got: '{dexp_node}' -> '{dexp_node.data}' ") 

        component: ComponentBase = dexp_node.data
        assert component == apply_session.current_frame.component

        key_string = apply_session.get_key_string(component)
        # get complete tree with values
        output = apply_session.get_values_tree(key_string=key_string)

        assert ReservedArgumentNames.INJECT_COMPONENT_TREE not in kwargs
        kwargs[ReservedArgumentNames.INJECT_COMPONENT_TREE] = output

        # prep_arg.caller.namespace / field_name = prep_arg.caller.name / prep_arg.caller.get_type_info()
        #   TypeInfo(py_type_hint=<class 'bool'>, types=[<class 'bool'>])
        # prep_arg.caller.data
        #   BooleanField(owner_name='company_rules', bind=DExpr(Models.can_be_accessed), name='can_be_accessed')

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

    # fixed arguments - when declared - see IFunctionDexpNode.fixed_args
    fixed_args     : FunctionArgumentsType = field(default=EmptyFunctionArguments)

    # can be evaluated later
    name           : Optional[str] = field(default=None) 

    # for dot-chained - when kwarg to use, target parameter
    value_arg_name : Optional[str] = field(default=None) 

    # validator function - see details in IFunction.arg_validators
    arg_validators : Optional[ValueArgValidatorPyFuncDictType] = field(default=None, repr=False)

    # misc data, used in EnumMembers
    data: Optional[Any] = field(default=None, repr=False)

    # autocomputed
    _output_type_info: TypeInfo = field(init=False, repr=False)

    # TODO: 'instantiate' class mode - get result as instance and cache result -
    #       py_function is Class, result is instance


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
                setup_session         : ISetupSession, # noqa: F821
                value_arg_type_info: Optional[TypeInfo] = None,
                name               : Optional[str] = None,
                caller             : Optional[IDotExpressionNode] = None,
                ) -> IFunction:
        custom_function = self.FUNCTION_CLASS(
                py_function         = self.py_function,     # noqa: E251
                fixed_args          = self.fixed_args,      # noqa: E251
                func_args           = func_args,            # noqa: E251
                value_arg_type_info = value_arg_type_info,  # noqa: E251
                value_arg_name      = self.value_arg_name,  # noqa: E251
                name                = name if name else self.name, # noqa: E251
                caller              = caller,               # noqa: E251
                setup_session          = setup_session,           # noqa: E251
                arg_validators      = self.arg_validators,  # noqa: E251
                data                = self.data,
                )
        return custom_function

    def get_type_info(self) -> TypeInfo:
        return self._output_type_info


# ------------------------------------------------------------

def Function(py_function : Callable[..., Any], 
             name: Optional[str] = None,
             value_arg_name:Optional[str]=None,
             # TODO: not only Dexp, can be literal too, e.g. 1, 2.3, "a"
             args   : Optional[List[DotExpression]] = UNDEFINED,
             kwargs : Optional[Dict[str, DotExpression]] = UNDEFINED,
             arg_validators : Optional[ValueArgValidatorPyFuncDictType] = None,
             ) -> CustomFunctionFactory:
    """
    goes to Global namespace (Fn.)
    can accept predefined params (like partial) 
    and in use (later) can be called with rest params.

    Usage: 
        def add(a, b, c=0):
            return a+b+c

        # get function wrapper, 
        Add = Function(add, kwargs=dict(a=Fn.some_function))

        # reference and store in wrapped function
        add_function = Fn.Add(b=This.value, c=3)

        # call / execute referenced function wrapper
        print (add_function.execute(ctx=this_ctx, setup_session)) 
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
            args   : Optional[List[DotExpression]] = None,
            kwargs: Optional[Dict[str, DotExpression]] = None,
            arg_validators : Optional[ValueArgValidatorPyFuncDictType] = None,
            ) -> BuiltinFunctionFactory:
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
                 include_builtin:bool = True, 
                 ):
        """
        functions are custom_function_factories
        """
        # TODO: typing is not good
        #       Union[type(IFunction), CustomFunctionFactory]
        self.store: Dict[str, Type(IFunction)]= {}
        self.include_builtin:bool = include_builtin

        if self.include_builtin:
            self.register_builtin_functions()

        if functions:
            for function_factory in functions:
                if not isinstance(function_factory, CustomFunctionFactory):
                    raise RuleSetupNameError(owner=self, msg=f"Function '{function_factory}' should be CustomFunctionFactory instance. Maybe you need to wrap it with Function()?")
                self.add(function_factory)

    def __str__(self):
        return f"{self.__class__.__name__}(functions={', '.join(self.store.keys())}, include_builtin={self.include_builtin})"
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

    def register_builtin_functions(self) -> int:
        # init_functions_factory_registry
        """
        Traverse all classes that inherit IFunction and register them
        This function needs to be inside of this module or change to: import .functions; for ... vars(functions)
        """
        from .func_builtin import get_builtin_function_factories_dict

        items = get_builtin_function_factories_dict().items()
        for func_name, builtin_function in items:
            self.add(builtin_function, func_name=func_name)

        assert len(self.store)>=3, self.store
        return len(self.store)

    def pprint(self):
        print("Functions:")
        for func_name, function in self.store.items():
            print(f"  {func_name} -> {function}")


# ------------------------------------------------------------

# ------------------------------------------------------------

def try_create_function(
        setup_session: ISetupSession,  # noqa: F821
        caller: IDotExpressionNode,
        # functions_factory_registry: IFunctionFactory, 
        attr_node_name:str, 
        func_args: FunctionArgumentsType,
        value_arg_type_info: TypeInfo,
        ) -> IFunction:

    if value_arg_type_info \
            and not isinstance(value_arg_type_info, TypeInfo) \
            and not is_function(value_arg_type_info):
        raise RuleInternalError(f"{value_arg_type_info} should be TypeInfo or function, got '{type(value_arg_type_info)}'")

    functions_factory_registry: IFunctionFactory = setup_session.functions_factory_registry
    if not functions_factory_registry:
        raise RuleSetupNameNotFoundError(item=setup_session, msg=f"Functions not available, '{attr_node_name}' function could not be found.")

    function_factory = None
    setup_session_current = setup_session

    names_avail_all = []
    registry_ids = set()

    while True:
        if id(setup_session_current) in registry_ids:
            # prevent infinitive loop
            raise RuleInternalError("Registry already processed")
        registry_ids.add(id(setup_session_current))

        function_factory = setup_session_current.functions_factory_registry.get(attr_node_name)
        if function_factory:
            break

        # TODO: filter functions based on type: is_list or similar ...
        first_custom = (attr_node_name and attr_node_name[0].islower())
        names_avail = get_available_names_example(attr_node_name,
                            functions_factory_registry.get_all_names(
                                first_custom=first_custom))
        names_avail_all.append(names_avail)

        if not setup_session.owner.owner:
            break
        setup_session_current = setup_session.owner.owner.setup_session

    func_node = None
    if function_factory:
        # ===== Create IFunction instance ===== 
        func_node = function_factory.create_function(
                        setup_session=setup_session,
                        caller=caller,
                        func_args=func_args,
                        value_arg_type_info=value_arg_type_info)

    if not func_node:
        names_avail_all = " ; ".join(names_avail_all)
        raise RuleSetupNameNotFoundError(f"Function name '{attr_node_name}' not found. Valid are: {names_avail_all}")

    return func_node
