# Copied and adapted from Reedwolf project (project by robert.lujo@gmail.com - git@bitbucket.org:trebor74hr/reedwolf.git)
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from collections import OrderedDict
from functools import partial

from typing import (
        Dict,
        Optional,
        Any,
        List,
        Callable,
        Union,
        )
from .utils import (
        UNDEFINED,
        format_arg_name_list,
        be_conjugate,
        plural_suffix,
        )
from .namespaces import ( 
        ThisNS,
        Namespace,
        )
from .exceptions import (
        RuleSetupValueError,
        RuleSetupTypeError,
        RuleInternalError,
        )
from .meta import (
        FunctionArgumentsType,
        TypeInfo,
        extract_function_arguments_default_dict,
        is_model_class,
        STANDARD_TYPE_LIST,
        )
from .expressions import (
        ValueExpression,
        IValueExpressionNode,
        )


TypeInfoCallable = Callable[[], TypeInfo]

# In Python 3.7, dictionaries are ordered.
# Dict[arg_name: str, type_info: TypeInfo]:
@dataclass
class PrepArg:
    name: str

    owner_name: str = field(repr=False)

    # can be None on creation, but filled later after complete finish is done
    # Example: F. namespace (FieldsNS.)
    type_info_or_callable: Union[TypeInfo, TypeInfoCallable]

    # TODO: Union[NoneType, StdPyTypes (Literal[]), IValueExpressionNode
    value_or_vexp: Any = field(repr=True)

    def __post_init__(self):
        if self.type_info_or_callable is None:
            raise RuleInternalError(owner=self, msg="type_info / 'xallable() -> type_info' not supplied")

    @property
    def type_info(self) -> Optional[TypeInfo]:
        " in some cases type_info is not available, then callable is provided which will start returning non-None value after finish() is completed "
        return self.type_info_or_callable() \
                if callable(self.type_info_or_callable) \
                else self.type_info_or_callable


@dataclass
class PreparedArguments:
    prep_arg_list: List[PrepArg]

    owner_name: str = field(repr=False)

    # TODO: is this field really used/required?
    # How value_arg (dot-chain mode) is filled:
    #   None - no dot-chain mode, call within namespace. E.g.
    #          Fn.my_custom_function(1,2) 
    #   True - dot-chain mode -> value is set to first empty positional or
    #          kwarg when value_arg_name is predefined. E.g.
    #          M.address_set.Count()
    #          M.name.Length()
    #   False- dot-chain mode -> value is not set. Allowed only when there is
    #          any This ValueExpression within arguments, e.g.
    #          M.Length(This.name)
    # TOOD: maybe change to True and adapt unit-tests?
    value_arg_implicit: Optional[bool] = field(repr=False)

    # autocomputed
    prep_arg_dict: Dict[str, PrepArg] = field(init=False, repr=False)

    # any_prep_arg_lack_type_info: bool = field(init=False, repr=False)

    def __post_init__(self):
        names_unfilled = [prep_arg.name for prep_arg in self.prep_arg_list if not prep_arg]
        if names_unfilled:
            raise RuleInternalError(f"{self.owner_name}: Following prepared arguments are left unfilled: {', '.join(names_unfilled)}")

        self.prep_arg_dict = {
                prep_arg.name: prep_arg 
                for prep_arg in self.prep_arg_list
                }
        self.any_prep_arg_lack_type_info()

    def any_prep_arg_lack_type_info(self):
        return any((
                not bool(pa.type_info) 
                for pa in self.prep_arg_dict.values() 
                ))

    def get(self, attr_name:str) -> Optional[PrepArg]:
        return self.prep_arg_dict.get(attr_name, None)

    def __iter__(self):
        for prep_arg in self.prep_arg_list:
            yield prep_arg


@dataclass
class FuncArg:
    name : str
    type_info: TypeInfo = field(repr=False)
    default: Any = field(repr=False, default=UNDEFINED)


@dataclass
class FunctionArguments:

    func_arg_list: List[FuncArg] = field(init=False, default_factory=list)
    func_arg_dict: Dict[str, FuncArg] = field(init=False, repr=False, default_factory=dict)
    finished: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        pass

    def items(self):
        assert self.finished
        return self.func_arg_list

    def names(self):
        assert self.finished
        return self.func_arg_dict.keys()

    def arguments(self):
        assert self.finished
        return self.func_arg_dict.values()

    def get(self, name, default=None):
        assert self.finished
        return self.func_arg_dict.get(name, default)

    def __contains__(self, name):
        assert self.finished
        return name in self.func_arg_dict

    def __getitem__(self, name):
        assert self.finished
        return self.func_arg_dict[name]

    def add(self, func_arg: FuncArg): 
        assert not self.finished 
        assert func_arg.name not in self.func_arg_dict, func_arg
        self.func_arg_dict[func_arg.name] = func_arg
        self.func_arg_list.append(func_arg)

    def finish(self):
        assert not self.finished
        self.finished = True

    # ------------------------------------------------------------

    def _process_func_args_raw(self, func_args_raw: FunctionArgumentsType):
        args, kwargs = (), {}
        if func_args_raw:
            if not isinstance(func_args_raw, FunctionArgumentsType):
                raise RuleInternalError(owner=self, msg=f"Bad type {type(func_args_raw)} : {func_args_raw}")
            args, kwargs = func_args_raw.get_args_kwargs()
        return args, kwargs


    def _create_prep_arg(self, 
                        caller: Union[Namespace, IValueExpressionNode],
                        owner_name: str,
                        registries: Optional["Registries"],  # noqa: F821
                        arg_name: str, 
                        value_object: Any
                        ) -> PrepArg:
        if isinstance(value_object, TypeInfo):
            type_info_or_callable = value_object
            # TODO: should be ValueExpression or IValueExpressionNode
            value_or_vexp = UNDEFINED 
        elif isinstance(value_object, ValueExpression):
            vexp: ValueExpression = value_object

            if vexp.IsFinished():
                raise RuleInternalError(owner=self, msg=f"{owner_name}: ValueExpression is already setup {vexp}")

            if not caller: 
                # NOTE: Namespace top level like: Fn.Length(This.name) 
                #       This could be ambigous, but in this case container_owner is used.
                container = registries.owner.get_container_owner()
                container.bound_model.type_info
                caller = container.bound_model
                # raise RuleSetupValueError(owner=self, msg="{owner_name}: ValueExpression cwn be used in dot-chain mode, e g. M.name.Lenght().")

            model_class = caller.type_info.type_
            assert model_class

            if is_model_class(model_class):
                # pydantic / dataclasses
                assert registries
                local_registries = registries.create_local_registries(this_ns_model_class=model_class)
            elif model_class in STANDARD_TYPE_LIST:
                local_registries = None 
            else:
                raise RuleSetupValueError(owner=self, msg=f"{owner_name}: Unsupported type: {caller} / {model_class}")

            if not registries:
                raise RuleInternalError(owner=self, msg=f"{owner_name}: Registries is required for ValueExpression() function argument case") 

            vexp_node = vexp.Setup(
                                registries=registries, 
                                # TODO: is this good?
                                owner=self, 
                                local_registries=local_registries)

            # NOTE: pass callable since type_info for some Vexp-s are not avaialble (e.g. FieldsNS, F.name)
            type_info_or_callable = vexp_node.get_type_info()
            if not type_info_or_callable:
                # if type_info is not provided then set callable which will start to return values after finish is completed
                type_info_or_callable = vexp_node.get_type_info
            value_or_vexp = vexp

        elif inspect.isclass(value_object):
            type_info_or_callable = TypeInfo.get_or_create_by_type(value_object)
            value_or_vexp = value_object

        else:
            # check that these are standard classes
            type_info_or_callable = TypeInfo.get_or_create_by_type(type(value_object))
            # TODO: Literal()
            value_or_vexp = value_object


        prep_arg = PrepArg(
                        name=arg_name, 
                        type_info_or_callable=type_info_or_callable, 
                        value_or_vexp=value_or_vexp,
                        owner_name=owner_name)

        return prep_arg

    # ------------------------------------------------------------

    def _try_fill_given_args(self, 
                        caller: Union[Namespace, IValueExpressionNode],
                        owner_name: str,
                        registries: Optional["Registries"],  # noqa: F821
                        args_title: str, 
                        expected_args: OrderedDict, 
                        given_args:List, 
                        given_kwargs:Dict
                        ):
        # == POSITIONAL ARGUMENTS

        expected_args_unfilled_names = [arg_name for index, (arg_name, prep_arg) in enumerate(expected_args.items()) if prep_arg is None]
        expected_args_unfilled_count = len(expected_args_unfilled_names)
        given_args_count = len(given_args)

        if given_args_count > expected_args_unfilled_count:
            raise RuleSetupValueError(owner=self, msg=f"{owner_name}: Function takes at most {expected_args_unfilled_count} unfilled argument{plural_suffix(expected_args_unfilled_count)} but {given_args_count} positional {be_conjugate(given_args_count)} given ({args_title})")

        # --- Fill unfilled positional arguments in correct order - needs 2 jumps
        for unfill_index, value_object in enumerate(given_args):
            arg_name = expected_args_unfilled_names[unfill_index]
            # expected_args[arg_name] = value_object
            expected_args[arg_name] = self._create_prep_arg(
                                                owner_name=owner_name,
                                                registries=registries, 
                                                caller=caller,
                                                arg_name=arg_name, 
                                                value_object=value_object)

        # == NAMED ARGUMENTS

        # Any unknown named arguments?
        expected_arg_names = list(expected_args.keys())
        unknown_arg_names = [arg_name for arg_name in given_kwargs.keys() if arg_name not in expected_arg_names]
        if unknown_arg_names:
            raise RuleSetupTypeError(owner=self, msg=f"{owner_name}: Function got an unexpected keyword argument{plural_suffix(len(unknown_arg_names))}: {format_arg_name_list(unknown_arg_names)} ({args_title})")

        # any multiple kwargs / named arguments? filled by positional and kwargs
        multiple_arg_names = [arg_name for arg_name in given_kwargs.keys() if expected_args[arg_name]]
        if multiple_arg_names:
            raise RuleSetupTypeError(owner=self, msg=f"{owner_name}: Function got multiple values for argument{plural_suffix(len(multiple_arg_names))}: {format_arg_name_list(multiple_arg_names)} ({args_title})")

        # --- Finally fill named arguments
        for arg_name, value_object in given_kwargs.items():
            # expected_args[arg_name] = value_object
            expected_args[arg_name] = self._create_prep_arg(
                                                caller=caller,
                                                owner_name=owner_name,
                                                registries=registries, 
                                                arg_name=arg_name, 
                                                value_object=value_object)


    # ------------------------------------------------------------


    def parse_func_args(self, 
                 caller              : Union[Namespace, IValueExpressionNode],
                 owner_name          : str,
                 func_args           : FunctionArgumentsType,
                 registries          : "Registries" = field(repr=False),  # noqa: F821
                 fixed_args          : Optional[FunctionArgumentsType] = field(default=None),
                 value_arg_type_info : Optional[TypeInfo] = field(default=None),
                 value_arg_name      : Optional[str] = field(default=None),
                 ) -> PreparedArguments:
        """
        Although default argument values are processed, they are not filled.
        """
        # ==== Placeholders
        # check if all used or extra
        # NOTE: In Python 3.7, dictionaries are ordered (order is preserved),
        #       so plain Dict could be used.
        expected_args: Dict[str, Optional[PrepArg]] = OrderedDict([(arg.name, None) for arg in self.func_arg_list])

        if not registries:
            raise RuleInternalError(owner=self, msg=f"{owner_name}: registries is empty") 

        # ==== 1/3 : FIX_ARGS - by registration e.g. Function(my_py_function, args=(1,), kwargs={"b":2})

        args, kwargs = self._process_func_args_raw(fixed_args)
        self._try_fill_given_args(
                caller=caller,
                owner_name=owner_name,
                registries=registries,
                args_title="fixed arguments", 
                expected_args=expected_args, 
                given_args=args, 
                given_kwargs=kwargs)


        # ==== 2/3: VALUE ARGUMENTS - are filled at the end

        value_args = []
        if caller is None:
            # direct from namespace, e.g. Fn.get_my_country()
            value_arg_implicit = None
        else:
            # value from chain / stream - previous dot-node. e.g. 
            #   M.name.Lower() # value is passed from .name
            if not isinstance(caller, IValueExpressionNode):
                raise RuleInternalError(f"{owner_name}: Caller is not IValueExpressionNode, got: {type(caller)} / {caller}")

            value_arg_implicit = UNDEFINED

            if not self.func_arg_list:
                # no arguments could be accepted at all, nor fixed nor value_arg nor func_args
                # will raise error later - wrong nr. of arguments
                raise RuleSetupTypeError(owner=self, msg=f"{owner_name}: Filling dot-chain argument value from '{caller.full_name}' failed, function accepts no arguments.")
            elif value_arg_name:
                # value to kwarg 'value_arg_name' value argument
                # value_object = expected_args.get(value_arg_name, UNDEFINED)
                prep_arg = expected_args.get(value_arg_name, UNDEFINED)
                if prep_arg:
                    value_or_vexp = prep_arg.value_or_vexp
                    if not (isinstance(value_or_vexp, ValueExpression) 
                            and value_or_vexp._namespace == ThisNS):
                        raise RuleSetupTypeError(owner=self, msg=f"{owner_name}: Function can not fill argument '{value_arg_name}' from '{caller.full_name}', argument is already filled with value '{value_or_vexp}'. Change arguments' setup or use 'This.' value expression.")
                    value_arg_implicit = False
            else:
                # value to first unfilled positional argument
                any_expected_args_unfilled = any([arg_name for arg_name, type_info in expected_args.items() if type_info is None])
                # when everything is filled - allow only when there is any ValueExpression with reference to ThisNS
                if not any_expected_args_unfilled:
                    prep_args_within_thisns = [prep_arg for prep_arg in expected_args.values() 
                                                   if prep_arg 
                                                   and isinstance(prep_arg.value_or_vexp, ValueExpression) 
                                                   and prep_arg.value_or_vexp._namespace == ThisNS]
                    if not prep_args_within_thisns:
                        raise RuleSetupTypeError(owner=self, msg=f"{owner_name}: Function can not take additional argument from '{caller.full_name}'. Remove at least one predefined argument or use value expression argument within 'This.' namespace.")
                    value_arg_implicit = False

            if value_arg_implicit is UNDEFINED:
                # otherwise - fill value to first positional or kwarg (case when value_arg_name is predefined)
                if value_arg_name:
                    value_args = []
                    value_kwargs = {value_arg_name: value_arg_type_info}
                else:
                    value_args = [value_arg_type_info]
                    value_kwargs = {}

                # TODO: validates vexp_node.get_type_info() matches value_arg_type_info

                args, kwargs = self._process_func_args_raw((FunctionArgumentsType(value_args, value_kwargs)))
                # vexp_node : IValueExpressionNode = caller
                self._try_fill_given_args(
                        registries=registries,
                        caller=caller,
                        owner_name=owner_name,
                        args_title="dot-chain argument", 
                        expected_args=expected_args, 
                        given_args=args, 
                        given_kwargs=kwargs)
                value_arg_implicit = True

        # ==== 3 / 3. FUNC_ARGS - by invocation e.g. M.function(1, b=2)
        args, kwargs = self._process_func_args_raw(func_args)
        self._try_fill_given_args(
                caller=caller,
                owner_name=owner_name,
                registries=registries,
                args_title="invoke arguments", 
                expected_args=expected_args, 
                given_args=args, 
                given_kwargs=kwargs)


        # ==== VALIDATIONS

        # ---- Any required unfilled?
        unfilled = [arg_name for arg_name, type_info in expected_args.items() 
                    if not type_info and self.func_arg_dict[arg_name].default is UNDEFINED]
        if unfilled:
            raise RuleSetupTypeError(owner=self, msg=f"{owner_name}: Function missing {len(unfilled)} required argument{plural_suffix(len(unfilled))}: {format_arg_name_list(unfilled)}")

        # ---- Convert to common instance - will be used for later invocation
        prepared_args = PreparedArguments(
                    owner_name=owner_name,
                    value_arg_implicit=value_arg_implicit,
                    # NOTE: currently value_or_vexp could not be fetched here - need more info
                    # PrepArg(name=arg_name, type_info=type_info, value_or_vexp=UNDEFINED) 
                    #  for arg_name, type_info in expected_args.items() 
                    prep_arg_list=[
                        prep_arg
                        for arg_name, prep_arg in expected_args.items() 
                        if prep_arg is not None
                        ]
                    )

        kwargs = dict(func_arg_dict=self.func_arg_dict, prepared_args=prepared_args)

        if not prepared_args.any_prep_arg_lack_type_info():
            check_prepared_arguments(**kwargs)
        else:
            registries.add_hook_on_finished_all(
                partial(check_prepared_arguments, **kwargs)
                )

        return prepared_args


# ------------------------------------------------------------

def check_prepared_arguments(
        func_arg_dict: Dict[str, FuncArg],
        prepared_args: PreparedArguments):
    """ 
    check if all types match 
    set to be pure function since to make it a bit easier to call later
    """
    # assert not prepared_args.any_prep_arg_lack_type_info()
    err_messages = []
    for prep_arg in prepared_args:
        exp_arg : FuncArg = func_arg_dict[prep_arg.name]
        if prep_arg.type_info is None:
            raise RuleInternalError(f"{prepared_args.owner_name} -> Argument '{prep_arg}' type_info is not set")

        err_msg = exp_arg.type_info.check_compatible(prep_arg.type_info)
        if err_msg:
            msg = f"[{prep_arg.name}]: {err_msg}"
            err_messages.append(msg)

    if err_messages:
        msg = ', '.join(err_messages)
        raise RuleSetupTypeError(f"{prepared_args.owner_name}: {len(err_messages)} data type issue(s) => {msg}")



# ------------------------------------------------------------


def create_function_arguments(
        py_function: Callable[[...], Any]
        ) -> FunctionArguments:

    type_info_dict = TypeInfo.extract_function_arguments_type_info_dict(py_function)

    arguments_default_dict = extract_function_arguments_default_dict(py_function)

    args_default = set(arguments_default_dict.keys())
    args_types   = set(type_info_dict.keys())
    if not args_default==args_types:
        # TODO: when unbound method is attached, then first set has "self" and other does not
        #       in this case report ValueError - unbound method not supported
        raise RuleInternalError(owner=py_function, 
                msg=f"Function's default arguments '{args_default}' not same as all arguments:  {args_types}. Args extraction issue...")

    func_args_specs = FunctionArguments()

    for arg_name, type_info in type_info_dict.items():
        default = arguments_default_dict[arg_name]
        func_arg = FuncArg(name=arg_name, type_info=type_info, default=default)

        func_args_specs.add(func_arg)

    func_args_specs.finish()

    return func_args_specs

