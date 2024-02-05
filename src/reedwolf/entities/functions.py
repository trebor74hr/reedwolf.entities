"""
Functions can be registered and used only if registered within container attribute:

    functions=[Function(),...]

e.g.:

   functions = [Function("CountAll", py_function, params...)]

function is registereed and can be used in

   Ctx.CountAll()
   M.addressses.CountAll()

extra args could be passed and must be called as function.
"""
import inspect
from dataclasses import (
    dataclass,
    field,
)
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

from .expr_attr_nodes import AttrDexpNode
from .utils import (
    UNDEFINED,
    get_available_names_example, UndefinedType,
)
from .exceptions import (
    EntitySetupValueError,
    EntitySetupNameError,
    EntitySetupNameNotFoundError,
    EntityInternalError,
    EntitySetupTypeError,
    EntitySetupError,
    EntityApplyError, EntityApplyTypeError, EntityTypeError,
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
    STANDARD_TYPE_LIST,
    is_model_klass,
    ItemType,
    ModelKlassType,
    ComponentTreeWValuesType,
    IInjectFuncArgHint,
    AttrValue,
    IExecuteFuncArgHint,
    LiteralType,
    ValueArgValidatorPyFuncDictType,
    AttrName,
    is_instancemethod_by_name, SELF_ARG_NAME, get_function_non_empty_arguments, SettingsType,
)
from .expressions import (
    DotExpression,
    IDotExpressionNode,
    IFunctionDexpNode,
    ExecResult,
    ISetupSession,
    IThisRegistry,
    JustDotexprFuncArgHint,
    DotexprFuncArgHint,
    IFunctionFactory, IDexpValueSource,
)
from .func_args import (
    FunctionArguments,
    create_function_arguments,
    PreparedArguments,
    FuncArg,
    PrepArg,
)
from .settings import (
    Settings,
    SettingsBase,
    ApplySettings,
    CustomFunctionFactoryList,
)
from .base import (
    AttrDexpNodeTypeEnum,
    IField,
    IApplyResult,
    IComponent,
    SetupStackFrame,
    ApplyStackFrame,
)

ValueOrDexp = Union[DotExpression, IDotExpressionNode, Any]

# 4 types
class DatatypeBasicEnum(Enum):
    STANDARD     = 301
    OBJECT       = 302
    # PRESERVED    = 303

class DatatypeCardinalityEnum(Enum):
    SINGLE  = 301
    ITEMS   = 302

class FunctionEngineBase:
    pass

class PythonFunctionEngine(FunctionEngineBase):
    pass


DEFAULT_ENGINE = PythonFunctionEngine()


# ------------------------------------------------------------
# IFunction - name used to emphasize that this is a function instance
# ------------------------------------------------------------

@dataclass
class IFunction(IFunctionDexpNode, ABC):
    """
    single left parameter (value from parent - previous dot chain, parent ) ,
    i.e. function wrapped should not have 2+ required params left
    """
    IS_ITEMS_FOR_EACH: ClassVar[bool] = False

    INPUT_CARDINALITY: ClassVar[DatatypeCardinalityEnum] = DatatypeCardinalityEnum.SINGLE

    # 1. python pure function that will be called
    py_function: Callable[..., Any]

    # 2. arguments in value expression usage. This is the only required (can be empty ([], {}))
    #   e.g. in chain .my_custom_function(a=1, b=2) # in this case: {"a": 1, "b": 2}
    func_args: FunctionArgumentsType

    # 3. SetupSession are required for validation and type *data* of function
    #     arguments, e.g. creating ThisNS, getting vars from ContextNS, etc.
    setup_session: ISetupSession = field(repr=False)  # noqa: F821

    # 4. in usage when in chain (value)
    #   e.g. some_struct_str_attr.lower() #  in this case: some_struct_str_attr
    value_arg_type_info: Optional[TypeInfo] = field(default=None)

    # 5. if value_arg_type_info is not supplied then it will be passed to first argument 
    #   e.g. in chain: some-func-returns 100 -> Ctx.my_custom_function(d=3) () # in this case: value_arg_name = "d", value = 3
    value_arg_name: Optional[str] = field(default=None)

    # 6. fixed arguments - when declared.
    #  e.g. Function(my_py_custom_function, c=3)
    #  e.g. Function(my_py_custom_function, fixed_args = ([1, 2], {"a": 3, "b": 4}))
    fixed_args: FunctionArgumentsType = EmptyFunctionArguments

    # 7. if not provided, then autocomputed
    name: Optional[str] = field(default=None)

    # 8. if not provided, then autocomputed from py_function type_hints
    function_arguments: Optional[FunctionArguments] = field(repr=False, default=None)

    # 9. caller - value expression node which calls function, can be None
    caller: Optional[IDotExpressionNode] = field(default=None)

    # 10. extra validations of input arguments, list of python functions, 
    #    e.g. Length() can operate on objects that have __len__ function (btw.
    #    collection.abc.Sized is used in this case).
    #    Functions are plain python functions, lists of validation functions
    #    are stored in a dict by by argument name.  validation function
    #    receives argument name and argument type. 
    #    For validation error cases should:
    #       * return string error message, or
    #       * raise EntitySetupError based error
    arg_validators: Optional[ValueArgValidatorPyFuncDictType] = field(repr=False, default=None)

    # 11. SetupSession are required for validation and type *data* of function
    #     arguments, e.g. creating ThisNS, getting vars from ContextNS etc.
    # setup_session: Optional[ISetupSession] = field(repr=False, default=None)  # noqa: F821

    # misc data, used in EnumMembers
    data: Optional[Any] = field(default=None, repr=False)

    # --- Autocomputed
    # required for IDotExpressionNode
    _output_type_info: TypeInfo = field(init=False, repr=False)
    # required for IDotExpressionNode
    func_name: str = field(init=False, repr=False)

    # computed from function_arguments(fixed_args, func_args)
    prepared_args: PreparedArguments = field(init=False, repr=False)

    is_finished: bool = field(init=False, repr=False, default=False)

    # all DotExpressions prepared arguments of this function will have access to this_registry within stack_frame
    this_registry: Union[IThisRegistry, NoneType, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)

    # set only when IS_ITEMS_FOR_EACH
    items_func_arg: Optional[FuncArg] = field(init=False, repr=True, default=None)

    def __post_init__(self):
        if not (self.py_function and is_function(self.py_function)):
            raise EntityInternalError(owner=self, msg="py_function input parameter is obligatory and must be a function, got: {self.py_function}")
        self.func_name = self.py_function.__name__

        if not self.name:
            self.name = self.func_name

        if not getattr(self, "full_name", None):
            self.full_name = f"Function.{self.name}" \
                    + (self.func_name if self.func_name!=self.name else "")

        self.engine: FunctionEngineBase = DEFAULT_ENGINE
        # chain_arg_type_info
        # self.value_arg_type_info = self.get_value_arg_type_info()

        # TODO: dry this - same cade in FunctionFactoryBase
        self._output_type_info = TypeInfo.extract_function_return_type_info(self.py_function)

        if not self.function_arguments:
            self.function_arguments = create_function_arguments(self.py_function)

        if self.IS_ITEMS_FOR_EACH:
            if not self.value_arg_name:
                raise EntityInternalError(owner=self, msg="value_arg_name is not set")
            self.items_func_arg = self.function_arguments.get(self.value_arg_name)
            if not self.items_func_arg:
                args_avail = get_available_names_example(self.value_arg_name, self.function_arguments.names())
                raise EntitySetupNameError(owner=self, msg=f"value_arg_name '{self.value_arg_name}' is unknown, available function's argument names: {args_avail}")
            if not self.items_func_arg.type_info.is_list:
                raise EntitySetupNameError(owner=self, msg=f"For 'Items' functions value_arg_name '{self.value_arg_name}' must have List type, got: {self.items_func_arg.type_info.py_type_hint}")

        prep_args_kwargs = dict(
            setup_session = self.setup_session,
            caller = self.caller,
            parent_name = f"{self.as_str()}",
            func_args = self.func_args,
            fixed_args = self.fixed_args,
            value_arg_type_info = self.value_arg_type_info,
            value_arg_name = self.value_arg_name)

        # NOTE: Stack and creation of this_registry should not be required
        #       if there is no DotExpression in argument directly or indirectly.
        #       Won't do this now, too much effort for no big benefit.
        if self.setup_session.current_frame:
            # Make available This. namespace to all prepared arguments DotExpression
            #       and all nested/inner expressions too.
            with self.setup_session.use_stack_frame(
                    SetupStackFrame(
                        container = self.setup_session.current_frame.container,
                        component = self.setup_session.current_frame.component,
                        this_registry = self.setup_this_registry(),
              )):
                self.prepared_args = self.function_arguments.parse_func_args( **prep_args_kwargs)
        else:
            # container is None only in direct functino creation - then This NS is not available.
            # Currently only in unit tests this case occurs.
            self.prepared_args = self.function_arguments.parse_func_args( **prep_args_kwargs)

        # first validate value type matches
        # if self.prepared_args.value_arg_implicit==True:
        # self._validate_value_arg_type(dexp_node=self.caller)
        if self.arg_validators:
            self._call_arg_validators()

        # self.setup_session.register_dexp_node(self)

    def setup_this_registry(self) -> Optional[IThisRegistry]:
        # TODO: resolve this circular dependency
        from .registries import ThisRegistry

        # TODO: maybe DRY is needed - similar logic found in base:: get_or_create_this_registry
        if self.this_registry is not UNDEFINED:
            raise EntityInternalError(owner=self, msg=f"this_registry already set: {self.this_registry}")
        assert self.setup_session

        # model_klass: ModelKlassType = None
        type_info: Optional[TypeInfo] = None  # not used

        if not self.caller:
            # NOTE: Namespace top level like: Ctx.Length(This.name)
            #       DataModelWithHandlers with read_handlers case

            # TODO: test case self.items_func_arg: - top level function call, e.g. '<Namespace>.{self.name}(...)'
            if self.setup_session.current_frame is None:
                # direct function creation - currently only unit test uses it
                pass
            else:
                model_klass = self.setup_session.current_frame.data_model.model_klass
                if isinstance(model_klass, DotExpression):
                    if not model_klass.IsFinished():
                        # container = self.get_first_parent_container(consider_self=True)
                        # model_dexp_node: IDotExpressionNode = model.Setup(setup_session=setup_session, owner=container)
                        raise EntityInternalError(owner=self, msg=f"{model_klass} dot-expression is not finished")
                    model_dexp_node: IDotExpressionNode = model_klass._dexp_node
                    type_info = model_dexp_node.get_type_info()
                elif is_model_klass(model_klass):
                    # model_klass = model
                    type_info = TypeInfo.get_or_create_by_type(model_klass)
                else:
                    raise EntityInternalError(owner=self, msg=f"expecting model class or dot expression, got: {model_klass}")
        else:
            if isinstance(self.caller, IFunction):
                function: IFunction = self.caller
                func_output_type_info = function.get_type_info()
                if func_output_type_info.type_ == ItemType:
                    # TODO: I don't like this solution, make it more clean and robust
                    type_info_or_callable = function.value_arg_type_info
                    type_info = type_info_or_callable() if callable(type_info_or_callable) else type_info_or_callable
                    if type_info is None:
                        # Fallback?: type_info = func_output_type_info
                        raise EntityInternalError(owner=self, msg=f"Type info can not be extracted in this moment, check function output type again ...")
                    elif not (func_output_type_info.is_list==type_info.is_list):
                        raise EntityInternalError(owner=self, msg=f"function type info <> value arg type info: {func_output_type_info.is_list} != {type_info.is_list if type_info else '<None>'}")
                else:
                    type_info = func_output_type_info
            elif isinstance(self.caller, IDotExpressionNode):
                # TODO: to drop this case or not? To change to 'setup_session.current_frame' case?
                dexp_node: IDotExpressionNode = self.caller
                type_info = dexp_node.get_type_info()
            else:
                raise EntityInternalError(owner=self, msg=f"Unsupported caller type, expected Function/DotExpressionNode, got: {self.caller}")

        model_class = type_info.type_ if type_info else None

        if model_class is None:
            # direct function creation - currently only unit test uses it
            this_registry = None

        elif self.items_func_arg or is_model_klass(model_class) or type_info.is_list:
            # complex structs: pydantic / dataclasses / List[Any] / ...
            if self.items_func_arg and not type_info.is_list:
                raise EntitySetupTypeError(owner=self, msg=f"For 'Items' functions only list types are supported, got: {type_info.py_type_hint}")

            this_registry = ThisRegistry.create_for_model_klass(
                is_items_for_each_mode = (self.items_func_arg is not None),
                setup_session=self.setup_session,
                model_klass=type_info.py_type_hint)

            if self.items_func_arg and not this_registry.is_items_for_each_mode:
                raise EntityInternalError(owner=self, msg=f"Failed to setup this_registry for Items in 'for-each' mode: {this_registry}")

        elif model_class in STANDARD_TYPE_LIST:
            this_registry = None
        # elif type_info and type_info.is_list:
        #     # NOTE: u ovaj slučaj nikad se ne ulazi
        #     model_klass = type_info.type_
        #     this_registry = ThisRegistry.create_for_model_class(
        #         setup_session=self.setup_session,
        #         model_klass=model_klass)
        #     # this_registry = self.setup_session.container.create_this_registry_for_model_class(
        #     #     setup_session=self.setup_session,
        #     #     model_klass=model_klass,
        #     # )
        else:
            raise EntitySetupValueError(owner=self, msg=f"Can not set registry for This. namespace, unsupported type: {model_class} / {type_info}. Caller: {self.caller}")

        self.this_registry = this_registry
        return self.this_registry


    def get_type_info(self) -> TypeInfo:
        return self._output_type_info


    def _call_arg_validators(self):
        """
        validate value from chain / stream - previous dot-node. e.g. 
            M.name.Lower() # value is passed from .name
        """
        assert self.arg_validators

        if not isinstance(self.arg_validators, dict):
            raise EntitySetupValueError(f"Parementer 'arg_validators' should be dictionary, got: {type(self.arg_validators)} / {self.arg_validators}")

        exception_list: List[EntitySetupTypeError] = []

        for arg_name, validators in self.arg_validators.items():
            if arg_name not in self.function_arguments:
                raise EntitySetupValueError(owner=self, msg=f"Argument 'arg_validators' has invalid argument name '{arg_name}'")

            prep_arg = self.prepared_args.get(arg_name)
            if not prep_arg:
                continue

            if not isinstance(validators, (list, tuple)):
                validators = [validators]

            for validator_func in validators:
                if not is_function(validator_func):
                    raise EntitySetupValueError(f"Parementer 'arg_validators[{arg_name}]' is not function, got: {type(validator_func)} / {validator_func}")
                try:
                    err_msg = validator_func(arg_name, prep_arg.type_info)
                    if err_msg:
                        exception_list.append(EntitySetupValueError(f"[{arg_name}] -> {err_msg}"))
                except EntitySetupError as ex:
                    exception_list.append(ex.__class__(f"[{arg_name}] -> {ex.msg}"))

        if exception_list:
            err_msgs = ', '.join([ex.msg for ex in exception_list])
            raise EntitySetupValueError(owner=self, msg=f"{self.as_str()}: Validation(s) failed ({len(exception_list)}): {err_msgs}")

    def as_str(self):
        args_str = "()"
        if self.caller:
            return f"{self.caller.full_name} -> {self.name}{args_str}"
        return f"{self.name}{args_str}"


    def _set_type_info(self, type_info:TypeInfo):
        if self.INPUT_CARDINALITY==DatatypeCardinalityEnum.ITEMS and not type_info.is_list:
            raise EntitySetupValueError(owner=self, msg=f"Expected 'List' type, got {type_info}.")
        elif self.INPUT_CARDINALITY==DatatypeCardinalityEnum.SINGLE and type_info.is_list:
            raise EntitySetupValueError(owner=self, msg=f"'List' type not valid: {type_info}.")


    # def get_output_type_info(self) -> TypeInfo: return TypeInfo.extract_function_return_type_info(self.py_function)

    # TODO: execute_arg - i.e. from DotExpression to standard pythoh types
    #       this will need new input parameter: contexts/attr_node/...
    @staticmethod
    def execute_arg(
            apply_result: "IApplyResult", # noqa: F821
            exp_arg: FuncArg,
            arg_value: ValueOrDexp,
            prev_node_type_info:TypeInfo,
            ) -> Any:

        execute_func_arg_hint: Optional[IExecuteFuncArgHint] = None
        if isinstance(exp_arg.type_info.py_type_hint, JustDotexprFuncArgHint):
            execute_func_arg_hint = None
        elif isinstance(exp_arg.type_info.py_type_hint, IExecuteFuncArgHint):
            execute_func_arg_hint: DotexprFuncArgHint = exp_arg.type_info.py_type_hint
        elif isinstance(arg_value, IDexpValueSource):
            arg_value = arg_value.get_value(strict=False)
        elif isinstance(arg_value, (DotExpression, IDotExpressionNode)):
            # TODO: Explain when this happens!!
            # NOTE: currently no inner type is passed
            execute_func_arg_hint = DotexprFuncArgHint()
        elif isinstance(arg_value, InjectFuncArgValueByCallable):
            arg_value = arg_value.get_apply_value()

        if execute_func_arg_hint:
            arg_value = execute_func_arg_hint.get_apply_value(
                exp_arg=exp_arg,
                arg_value=arg_value,
                prev_node_type_info=prev_node_type_info,
                apply_result=apply_result)

        return arg_value

    def execute_node(self,
                     apply_result: "IApplyResult",  # noqa: F821
                     dexp_result: ExecResult,
                     is_last:bool,
                     prev_node_type_info: Optional[TypeInfo],
                     ) -> Any:
        instance = apply_result.current_frame.instance

        with apply_result.use_stack_frame(
                ApplyStackFrame(
                    container=apply_result.current_frame.container,
                    component=apply_result.current_frame.component,
                    instance = instance,
                    this_registry=self.this_registry,
                )):
            result = self._execute_node(
                      apply_result=apply_result,
                      dexp_result=dexp_result,
                      is_last=is_last,
                      prev_node_type_info=prev_node_type_info)
        return result

    def _execute_node(self,
                     apply_result: "IApplyResult",  # noqa: F821
                     dexp_result: ExecResult,
                     is_last:bool,
                     prev_node_type_info: Optional[TypeInfo],
                     ) -> Any:
        """
        will be called when actual function logic needs to be executed. Input
        is/are function argument(s).

        # TODO: IApplyResult is in dropthis which imports .functions just for one case ... Explain or remove !!!
        # TODO: check all input arguments and output type match:
        #       check first / dexp_result argument that matches self.value_arg_type_info
        #       check output type that matches output_type_info
        """
        if is_last and not self.is_finished:
            raise EntityInternalError(owner=self, msg="Last dexp-node is not finished")  # , {id(self)} / {type(self)}

        args = []
        kwargs = {}

        if dexp_result is UNDEFINED:
            # top_level_call = True
            # namespace toplevel call, e.g. Ctx.Length()
            dexp_result = ExecResult()
        else:
            # top_level_call = False
            if not isinstance(dexp_result, ExecResult):
                raise EntityInternalError(owner=self, msg=f"dexp_result is not ExecResult, got: {dexp_result}") 

            if dexp_result.value is not UNDEFINED:
                input_value = dexp_result.value
                if self.value_arg_name:
                    kwargs[self.value_arg_name] = input_value
                else:
                    args.insert(0, input_value)

            for exp_arg in self.function_arguments.func_arg_list:
                # if inspect.isclass(prep_arg.type_info.type_) and issubclass(prep_arg.type_info.type_, IInjectFuncArgHint):
                if exp_arg.type_info.is_inject_func_arg:
                    prep_arg = self.prepared_args.get(exp_arg.name)
                    assert prep_arg
                    # TODO: check that prep_arg is not filled
                    self._process_inject_prepared_args(prep_arg=prep_arg,
                                                       exp_arg=exp_arg,
                                                       apply_result=apply_result,
                                                       kwargs=kwargs)

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

        args_unfilled_by_kwargs = [exp_arg for exp_arg in self.function_arguments.func_arg_list if exp_arg.name not in kwargs]
        # will need name to each arg and
        kwargs_from_args = {}
        for arg_value, exp_arg in zip(args, args_unfilled_by_kwargs):
            kwargs_from_args[exp_arg.name] = self.execute_arg(apply_result=apply_result,
                                                arg_value = arg_value,
                                                exp_arg = exp_arg,
                                                prev_node_type_info = prev_node_type_info)

        kwargs_from_kwargs = {arg_name: self.execute_arg(apply_result=apply_result,
                                             arg_value=arg_value,
                                             exp_arg=self.function_arguments.func_arg_dict[arg_name],
                                             prev_node_type_info=prev_node_type_info)
                  for arg_name, arg_value in kwargs.items()}

        overlaps = list(set(kwargs_from_kwargs.keys()).intersection(set(kwargs_from_args.keys())))
        if overlaps:
            # TODO: this should have been checked before - in setup / check type phase. reuse that code - DRY.
            raise EntityApplyError(owner=self, msg=f"Overlapping positional and keyword arguments: {overlaps}")

        kwargs_all =  kwargs_from_kwargs.copy()
        kwargs_all.update(kwargs_from_args)

        try:
            # Finally call the function with all arguments prepared
            output_value = self.py_function(**kwargs_all)
        except Exception as ex:
            # REMOVE_THIS: raise
            raise EntityApplyError(owner=self, msg=f"failed in calling '{self.name}({args}, {kwargs})' => {ex}")

        dexp_result.set_value(output_value, attr_name="", changer_name=f"{self.name}")

        return dexp_result

    # ------------------------------------------------------------

    def _process_inject_prepared_args(self,
                                      exp_arg: PrepArg,
                                      prep_arg: PrepArg,
                                      apply_result: IApplyResult,
                                      kwargs: Dict[str, ValueOrDexp]):
        # exp_arg has good type-hint, prep_arg has caller
        if not isinstance(exp_arg.type_info.py_type_hint, IInjectFuncArgHint):
            raise EntityApplyTypeError(owner=self, msg=f"Expecting IInjectFuncArgHint for {exp_arg} argument type, got: {exp_arg.type_info}")

        func_arg_hint: IInjectFuncArgHint = exp_arg.type_info.py_type_hint

        output = func_arg_hint.get_apply_inject_value(apply_result=apply_result, prep_arg=prep_arg)

        assert prep_arg.name not in kwargs
        kwargs[prep_arg.name] = output


# ------------------------------------------------------------
# CustomFunctions - need 2 steps and 3 layers:
#   Function -> CustomFunctionFactory -> CustomFunction
# ------------------------------------------------------------

@dataclass
class CustomFunction(IFunction):
    pass

@dataclass
class CustomItemsFunction(IFunction):
    IS_ITEMS_FOR_EACH: ClassVar[bool] = True

@dataclass
class BuiltinFunction(IFunction):
    pass

@dataclass
class BuiltinItemsFunction(IFunction):
    IS_ITEMS_FOR_EACH: ClassVar[bool] = True

# ------------------------------------------------------------

@dataclass
class FunctionFactoryBase(IFunctionFactory):
    """
    Function wrapper for arbitrary regular python function.
    that will reside in Global namespace and can not be 
    e.g. can be used in DynanicData-s and similar in/out bindings.

    inject_args should be used as functools.partial()
    later reference will fill func_args

    NOTE: when adding new field(s), add to custom_attributes.py :: AttributeByMethod
    """
    FUNCTION_CLASS: ClassVar[Union[UndefinedType, Type[IFunction]]] = UNDEFINED

    # regular python function - should be pure (do not change input args,
    # rather copy and return modified copy
    py_function: Callable[..., Any]

    # fixed arguments - when declared - see IFunctionDexpNode.fixed_args
    fixed_args: FunctionArgumentsType = field(default=EmptyFunctionArguments)

    # can be evaluated later
    name: Optional[str] = field(default=None)

    # for dot-chained - when kwarg to use, target parameter
    value_arg_name: Optional[str] = field(default=None)

    # validator function - see details in IFunction.arg_validators
    arg_validators: Optional[ValueArgValidatorPyFuncDictType] = field(default=None, repr=False)

    # misc data, used in EnumMembers
    data: Optional[Any] = field(default=None, repr=False)

    # used when custom functions needs to override builtin functions - e.g. Lower()
    force: bool = False

    # autocomputed
    _output_type_info: TypeInfo = field(init=False, repr=False)

    # TODO: 'instantiate' class mode - get result as instance and cache result -
    #       py_function is Class, result is instance

    def __post_init__(self):
        if not is_function(self.py_function):
            raise EntitySetupValueError(owner=self, msg=f"py_function is not a function, got: {type(self.py_function)} / {self.py_function}")

        if not self.name:
            self.name = self.py_function.__name__
        if self.fixed_args is None:
            self.fixed_args = FunctionArgumentsType((), {})
        # TODO: dry this - same cade in IFunction
        self._output_type_info = TypeInfo.extract_function_return_type_info(self.py_function)

        if self.FUNCTION_CLASS is UNDEFINED:
            raise EntityInternalError(owner=self, msg="FUNCTION_CLASS is not set on class level")

        # TODO: dry this - same cade in IFunction
        if self.FUNCTION_CLASS.IS_ITEMS_FOR_EACH:
            if not self.value_arg_name:
                raise EntityInternalError(owner=self, msg="value_arg_name is not set")

    # def get_type_info(self) -> TypeInfo: return self._output_type_info

    def create_function(self, 
                func_args: FunctionArgumentsType,
                setup_session: ISetupSession, # noqa: F821
                value_arg_type_info: Optional[TypeInfo] = None,
                name: Optional[str] = None,
                caller: Optional[IDotExpressionNode] = None,
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

@dataclass
class CustomFunctionFactory(FunctionFactoryBase):
    FUNCTION_CLASS: ClassVar[Type[IFunction]] = CustomFunction

@dataclass
class CustomItemsFunctionFactory(CustomFunctionFactory):
    FUNCTION_CLASS: ClassVar[Type[IFunction]] = CustomItemsFunction

# ------------------------------------------------------------

@dataclass
class BuiltinFunctionFactory(FunctionFactoryBase):
    FUNCTION_CLASS: ClassVar[Type[IFunction]] = BuiltinFunction

@dataclass
class BuiltinItemsFunctionFactory(BuiltinFunctionFactory):
    FUNCTION_CLASS: ClassVar[Type[IFunction]] = BuiltinItemsFunction


@dataclass
class FunctionByMethod(IFunctionFactory):
    """
    Proxy class to Function() / FunctionFactory, needs extra step - set_instance(settings)
    """
    method_name: AttrName
    name: Optional[str] = None
    value_arg_name:Optional[str]=None
    args: Optional[List[Union[LiteralType, DotExpression]]] = UNDEFINED
    kwargs: Optional[Dict[str, Union[LiteralType, DotExpression]]] = UNDEFINED
    arg_validators: Optional[ValueArgValidatorPyFuncDictType] = None

    # setup later
    settings_class: Optional[Type[SettingsBase]] = field(init=False, repr=False, default=None)
    settings_type: Optional[SettingsType] = field(init=False, repr=False, default=None)
    function_factory: Optional[CustomFunctionFactory] = field(init=False, repr=False, default=None)

    # setup after later
    settings: Union[SettingsBase, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)

    def set_settings_class(self, settings_type: SettingsType, settings_class: Type[SettingsBase]):
        """
        Called in setup phase - no apply-settings instance available.
        """
        if self.settings_class:
            raise EntityInternalError(owner=self, msg=f"Settings class already set to: {self.settings_class}")
        if settings_type == SettingsType.SETUP_SETTINGS:
            if Settings not in inspect.getmro(settings_class):
                raise EntityInternalError(owner=self, msg=f"Argument 'settings_class' must inherit Settings class, got: {settings_class}")
        else:
            assert settings_type == SettingsType.APPLY_SETTINGS, settings_type
            if ApplySettings not in inspect.getmro(settings_class):
                raise EntityInternalError(owner=self, msg=f"Argument 'settings_class' must inherit ApplySettings class, got: {settings_class}")

        # NOTE: similar logic in custom_attributes.py :: AttributeByMethod.setup_dexp_attr_source()

        # Method used for custom function must be member of settings class that defines this function entry - what
        # is different from custom_ctx_attributes logic.
        py_function: Callable = getattr(settings_class, self.method_name, UNDEFINED)
        if py_function is UNDEFINED:
            raise EntitySetupNameError(owner=self, msg=f"Method name '{self.method_name}' is not found within class: {settings_class}")

        self.settings_class = settings_class
        self.settings_type = settings_type

        if is_instancemethod_by_name(settings_class, self.method_name):
            kwargs = self.kwargs.copy()
            if SELF_ARG_NAME in kwargs:
                raise EntitySetupNameError(owner=self,
                                           msg=f"FunctionByMethod('{self.method_name}') has already 'self' argument set, so settings instance could not be set. Use Function() instead or unset 'self' argument value.")
            kwargs[SELF_ARG_NAME] = InjectFuncArgValueByCallable(self.get_settings_instance)
        else:
            kwargs = self.kwargs

        self.function_factory = Function(
            py_function=py_function,
            name=self.name,
            value_arg_name=self.value_arg_name,
            args=self.args,
            kwargs=kwargs,
            arg_validators=self.arg_validators,
        )

    def set_settings_instance(self, settings: Union[SettingsBase, UndefinedType]):
        """
        Called in early step of apply phase when apply_settings instance is available. No user caused error expected.
        """
        if settings is UNDEFINED and self.settings is UNDEFINED:
            raise EntityInternalError(owner=self, msg=f"Settings already reset to: {self.settings}")
        if settings is not UNDEFINED:
            if self.settings is not UNDEFINED:
                raise EntityInternalError(owner=self, msg=f"Settings already set to: {self.settings}")
            if not isinstance(settings, self.settings_class):
                raise EntityInternalError(owner=self, msg=f"Argument 'settings' must be instance of {self.settings_class} class, got: {settings}")
        self.settings = settings

    def get_settings_instance(self) -> SettingsBase:
        if self.settings is UNDEFINED:
            raise EntityInternalError(owner=self, msg="Settings not yet set. Call set_settings_instance() first.")
        return self.settings

    def create_function(self,
                        func_args:FunctionArgumentsType,
                        setup_session: ISetupSession, # noqa: F821
                        value_arg_type_info: Optional[TypeInfo] = None,
                        name: Optional[str] = None,
                        caller: Optional[IDotExpressionNode] = None,
                        ) -> IFunctionDexpNode:  # IFunction
        if self.function_factory is None:
            raise EntityInternalError(owner=self, msg="Attribute 'function_factory' not set, call 'set_instance()' method first.")
        return self.function_factory.create_function(
                    func_args=func_args,
                    setup_session=setup_session,
                    value_arg_type_info=value_arg_type_info,
                    name=name,
                    caller=caller,
                    )

    def get_type_info(self) -> TypeInfo:
        if self.function_factory is None:
            raise EntityInternalError(owner=self, msg="Attribute 'function_factory' not set, call 'set_instance()' method first.")
        return self.function_factory.get_type_info()

# ------------------------------------------------------------

def Function(py_function: Callable[..., Any],
             name: Optional[str] = None,
             value_arg_name:Optional[str]=None,
             args: Optional[List[Union[LiteralType, DotExpression]]] = UNDEFINED,
             kwargs: Optional[Dict[str, Union[LiteralType, DotExpression]]] = UNDEFINED,
             arg_validators: Optional[ValueArgValidatorPyFuncDictType] = None,
             force: bool = False,
             ) -> CustomFunctionFactory:
    """
    goes to Global namespace (Ctx.)
    can accept predefined params (like partial) 
    and in use (later) can be called with rest params.

    Usage: 
        def add(a, b, c=0):
            return a+b+c

        # get function wrapper, 
        Add = Function(add, kwargs=dict(a=Ctx.some_function))

        # reference and store in wrapped function
        add_function = Ctx.Add(b=This.value, c=3)

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
                force=force,
                )


def ItemsFunction(py_function: Callable[..., Any],
                  # NOTE: this parameter is required
                  items_value_arg_name: str,
                  name: Optional[str] = None,
                  args: Optional[List[DotExpression]] = UNDEFINED,
                  kwargs: Optional[Dict[str, DotExpression]] = UNDEFINED,
                  arg_validators: Optional[ValueArgValidatorPyFuncDictType] = None,
                  ) -> CustomItemsFunctionFactory:
    return CustomItemsFunctionFactory(
        py_function=py_function,
        name=name,
        value_arg_name=items_value_arg_name,
        fixed_args=FunctionArgumentsType(
            args if args is not UNDEFINED else (),
            kwargs if kwargs is not UNDEFINED else {}
        ),
        arg_validators=arg_validators,
    )


def create_builtin_function_factory(
            py_function: Callable[..., Any],
            name: Optional[str] = None,
            value_arg_name:Optional[str]=None,
            args: Optional[List[DotExpression]] = None,
            kwargs: Optional[Dict[str, DotExpression]] = None,
            arg_validators: Optional[ValueArgValidatorPyFuncDictType] = None,
            ) -> BuiltinFunctionFactory:
    """
    wrapper around BuiltinFunctionFactory - to have better api look
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

def create_builtin_items_function_factory(
        py_function: Callable[..., Any],
        # NOTE: this parameter is required
        items_value_arg_name:str,
        name: Optional[str] = None,
        args: Optional[List[DotExpression]] = None,
        kwargs: Optional[Dict[str, DotExpression]] = None,
        arg_validators: Optional[ValueArgValidatorPyFuncDictType] = None,
) -> BuiltinItemsFunctionFactory:
    """
    wrapper around BuiltinFunctionFactory - better api look
    TODO: consider creating decorator
    """
    return BuiltinItemsFunctionFactory(
        py_function=py_function,
        name=name,
        value_arg_name=items_value_arg_name,
        fixed_args=FunctionArgumentsType(
            args if args is not None else (),
            kwargs if kwargs is not None else {}),
        arg_validators=arg_validators,
    )

# =====================================================================
# FunctionsFactoryRegistry
#    classes or functions when called create instances of IFunction
#    Currently IFunction class and CustomFunctionFactory function
# =====================================================================

@dataclass(repr=False)
class FunctionsFactoryRegistry:

    functions: CustomFunctionFactoryList
    builtin_functions_dict: Dict[AttrName, IFunctionFactory]

    # later evaluated
    store: Dict[str, FunctionFactoryBase] = field(repr=False, init=False, default_factory=dict)

    def __post_init__(self):
        """
        functions are custom_function_factories
        """
        # TODO: typing is not good
        #       Union[type(IFunction), CustomFunctionFactory]
        #       ex. Type[IFunction]

        if self.builtin_functions_dict:
            # NOTE: not called for subentities
            self.register_builtin_functions()

        if self.functions:
            for function_factory in self.functions:
                if not isinstance(function_factory, (CustomFunctionFactory, FunctionByMethod)):
                    raise EntitySetupNameError(owner=self, msg=f"Function '{function_factory}' should be CustomFunctionFactory instance. Maybe you need to wrap it with Function() or FunctionByMethod()?")
                self.add(function_factory)


    def register_builtin_functions(self) -> int:
        # init_functions_factory_registry
        """
        Traverse all classes that inherit IFunction and register them
        This function needs to be inside of this module or change to: import .functions; for ... vars(functions)
        """
        for func_name, builtin_function in self.builtin_functions_dict.items():
            self.add(builtin_function, func_name=func_name)

        # assert len(self.store)>=3, self.store
        return len(self.store)


    def add(self, function_factory: FunctionFactoryBase, func_name: Optional[str]=None):
        # IFunction
        if not isinstance(function_factory, (FunctionFactoryBase, FunctionByMethod)):
            raise EntityInternalError(f"Exepected function factory, got: {type(function_factory)} -> {function_factory}")

        if not func_name:
            func_name = function_factory.name

        assert isinstance(func_name, str) and func_name, func_name

        if func_name in self.store:
            if not function_factory.force:
                raise EntitySetupNameError(owner=self, msg=f"Function '{func_name}' is already defined: {self.store[func_name]}. You may choose another name.")
        self.store[func_name] = function_factory



    def as_str(self):
        return f"{self.__class__.__name__}(functions={', '.join(self.store.keys())}, include_builtin={self.builtin_functions_dict})"

    def __str__(self):
        return f"{self.__class__.__name__}({len(self.store)})"
    __repr__ = __str__

    def items(self) -> List[Tuple[str, FunctionFactoryBase]]:
        return list(self.store.items())

    def get(self, name:str, strict:bool=False) -> Optional[IFunction]:
        if strict and name not in self.store:
            funcs_avail = get_available_names_example(name, self.store.keys())
            raise EntitySetupNameNotFoundError(owner=self, msg=f"Function '{name}' is not valid. Valid are: {funcs_avail}")
        function = self.store.get(name, None)
        return function

    def get_all_names(self, first_custom:bool = False) -> List[str]:
        if first_custom:
            out = [fnname for fnname in self.store.keys() if fnname.islower()]
            out += [fnname for fnname in self.store.keys() if not fnname.islower()]
            return out
        return list(self.store.keys())


    def pprint(self):
        print("Functions:")
        for func_name, function in self.store.items():
            print(f"  {func_name} -> {function}")


# ------------------------------------------------------------

def try_create_function(
        setup_session: ISetupSession,  # noqa: F821
        caller: IDotExpressionNode,
        # functions_factory_registry: FunctionFactoryBase,
        attr_node_name:str, 
        func_args: FunctionArgumentsType,
        value_arg_type_info: TypeInfo,
        ) -> IFunction:

    if value_arg_type_info \
            and not isinstance(value_arg_type_info, TypeInfo) \
            and not is_function(value_arg_type_info):
        raise EntityInternalError(f"{value_arg_type_info} should be TypeInfo or function, got '{type(value_arg_type_info)}'")

    functions_factory_registry: FunctionFactoryBase = setup_session.functions_factory_registry
    if not functions_factory_registry:
        raise EntitySetupNameNotFoundError(item=setup_session, msg=f"Functions not available, '{attr_node_name}' function could not be found.")

    function_factory = None
    setup_session_current = setup_session

    names_avail_all = []
    registry_ids = set()

    while True:
        if id(setup_session_current) in registry_ids:
            # prevent infinitive loop
            raise EntityInternalError("Registry already processed")
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

        if not setup_session.container.parent:
            break
        setup_session_current = setup_session.container.parent.setup_session

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
        raise EntitySetupNameNotFoundError(f"Function name '{attr_node_name}' not found. Valid are: {names_avail_all}")

    return func_node

# ---------------------------------------------------------------------
# Special function arguments helpers - custom type hints or wrappers
# ---------------------------------------------------------------------
# TODO: other are in expressions.py - move all to the same place:
#       class AttrnameFuncArgHint(IExecuteFuncArgHint):
#       class DotexprFuncArgHint(IExecuteFuncArgHint):
#       class JustDotexprFuncArgHint(IFuncArgHint):

@dataclass
class InjectComponentTreeValuesFuncArgHint(IInjectFuncArgHint):

    def setup_check(self, setup_session: ISetupSession, caller: Optional[IDotExpressionNode], func_arg: FuncArg):
        if not (isinstance(caller, AttrDexpNode)
                and caller.namespace == FieldsNS):
            raise EntityInternalError(owner=self, msg=f"Expected F.<fieldname>, got: {caller}")

    def get_type(self) -> Type:
        return self.__class__

    def get_inner_type(self) -> Optional[Type]:
        return ComponentTreeWValuesType

    def get_apply_inject_value(self, apply_result: IApplyResult, prep_arg: PrepArg) -> AttrValue:
        # maybe belongs to implementation
        if not isinstance(prep_arg.caller, IDotExpressionNode):
            raise EntityInternalError(owner=self, msg=f"Expected IDotExpressionNode, got: {type(prep_arg.caller)} / {prep_arg.caller}")

        dexp_node: IDotExpressionNode = prep_arg.caller

        if not (dexp_node.attr_node_type == AttrDexpNodeTypeEnum.FIELD
                and dexp_node.namespace == FieldsNS
                and isinstance(dexp_node.data, IField)):
            raise EntityInternalError(owner=self, msg=f"Inject function argument value:: PrepArg '{prep_arg.name}' expected DExp(F.<field>) -> Field(),  got: '{dexp_node}' -> '{dexp_node.data}' ")

        component: IComponent = dexp_node.data
        assert component == apply_result.current_frame.component

        key_string = apply_result.get_key_string(component)

        # get complete tree with values
        output = apply_result.get_values_tree(key_string=key_string)
        return output

    def __hash__(self):
        return hash((self.__class__.__name__))

# ------------------------------------------------------------

@dataclass
class DotexprExecuteOnItemFactoryFuncArgHint(IInjectFuncArgHint):
    """
    TODO: name is too long and unreadable.
    """
    inner_type: Optional[Type] = field(repr=True, default=Any)
    type: Type = field(init=False, default=DotExpression)

    def setup_check(self, setup_session: "ISetupSession", caller: Optional["IDotExpressionNode"], func_arg: "FuncArg"):
        exp_type_info = TypeInfo.get_or_create_by_type(self.type)
        err_msg = exp_type_info.check_compatible(func_arg.type_info)
        if err_msg:
            raise EntityTypeError(owner=self, msg=f"Function argument {func_arg} type not compatible: {err_msg}")


    def get_type(self) -> Type:
        return self.type

    def get_inner_type(self) -> Optional[Type]:
        return self.inner_type

    def __hash__(self):
        return hash((self.__class__.__name__, self.type, self.inner_type))

    def get_apply_inject_value(self, apply_result: "IApplyResult", prep_arg: "PrepArg"
                               ) -> Callable[[DotExpression, ModelKlassType], AttrValue]:
        """
        create this registry function that retriieves settings processor
        which will crearte this factory for providded item
        """
        # TODO: 2nd) resolve again ThisRegistry dependency
        from .registries import ThisRegistry

        def execute_dot_expr_w_this_registry_of_item(
                dot_expr:  DotExpression,
                item: ModelKlassType,
        ) -> AttrValue:
            # TODO: if this becommes heavy - instead of new frame, reuse existing and change instance only
            #       this_registry should be the same for same session (same type items)
            setup_session = apply_result.current_frame.component.setup_session
            this_registry = ThisRegistry.create_for_model_klass(
                setup_session=setup_session,
                model_klass=type(item),
            )
            with apply_result.use_stack_frame(
                    ApplyStackFrame(
                        container = apply_result.current_frame.container,
                        component = apply_result.current_frame.component,
                        instance=item,
                        this_registry = this_registry,
                    )):
                dexp_result = dot_expr._evaluator.execute_dexp(
                    apply_result=apply_result,
                )
            return dexp_result.value

        return execute_dot_expr_w_this_registry_of_item

# ------------------------------------------------------------

@dataclass
class InjectFuncArgValueByCallable:
    py_function: Callable

    def __post_init__(self):
        non_empty_params = get_function_non_empty_arguments(self.py_function)
        if len(non_empty_params) != 0:
            raise EntitySetupNameError(owner=self,
                                       msg=f"Function '{self.py_function}' must not have arguments without defaults. Found unfilled arguments: {', '.join(non_empty_params)} ")

    def get_apply_value(self) -> AttrValue:
        return self.py_function()

