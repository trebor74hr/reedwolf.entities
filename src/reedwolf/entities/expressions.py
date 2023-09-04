from abc import (
        ABC, 
        abstractmethod,
        )
import operator
import ast
from dataclasses import (
        dataclass, 
        field,
        replace as dataclasses_replace,
        )
from enum import (
        Enum
        )
from typing import (
    List,
    Optional,
    Union,
    Any,
    Callable,
    Tuple, Type,
)

from .utils import (
        UNDEFINED,
        UndefinedType,
        )
from .exceptions import (
        EntitySetupValueError,
        EntitySetupError,
        EntitySetupNameError,
        EntityInternalError,
        EntityApplyError,
        )
from .namespaces import (
    DynamicAttrsBase,
    FunctionsNS,
    OperationsNS,
    Namespace, ThisNS,
)
from .meta import (
    TypeInfo,
    FunctionArgumentsType,
    FunctionArgumentsTupleType,
    STANDARD_TYPE_W_NONE_LIST,
    HookOnFinishedAllCallable,
    LiteralType,
    AttrName,
    Self,
    ModelType, FuncArgTypeHint, AttrValue,
)
# ------------------------------------------------------------
# interfaces / base classes / internal structs
# ------------------------------------------------------------

# ------------------------------------------------------------

@dataclass
class RawAttrValue:
    value: Any
    attr_name: str
    changer_name: str

# ------------------------------------------------------------

@dataclass
class ExecResult:
    # last value, mutable
    value: Any  = field(init=False, default=UNDEFINED)

    # TODO: set compoenent (owner) that triggerred value change
    #       value evaluation - can have attr_node.name
    value_history : List[Tuple[str, RawAttrValue]] = field(repr=False, init=False, default_factory=list)

    @classmethod
    def create(cls, value: Any, attr_name: str="", changer_name: str=""):
        instance = cls()
        instance.set_value(value, attr_name=attr_name, changer_name=changer_name)
        return instance

    def set_value(self, value: Any, attr_name: str, changer_name: str):
        self.value_history.append(
                RawAttrValue(value=value, attr_name=attr_name, changer_name=changer_name)
                )
        self.value = value

    def is_not_available(self) -> bool:
        return isinstance(self, NotAvailableExecResult)


# ------------------------------------------------------------
# IDotExpressionNode
# ------------------------------------------------------------

@dataclass
class IDotExpressionNode(ABC):
    """ wrapper around one element in DotExpression e.g. M.name.Count()
    .company, .name, .Count() are nodes
    """
    def clone(self):
        # If already setup then copy it and reuse
        return dataclasses_replace(self)

    @abstractmethod
    def execute_node(self, 
                 apply_result: "IApplyResult", # noqa: F821
                 # previous - can be undefined too
                 dexp_result: Union[ExecResult, UndefinedType],
                 is_last: bool,
                 prev_node_type_info: Optional[TypeInfo],
                 ) -> ExecResult:
        ...

    @abstractmethod
    def get_type_info(self) -> TypeInfo:
        ...

    def finish(self):
        if self.is_finished:
            raise EntityInternalError(owner=self, msg="already finished") 
        self.is_finished = True


# ------------------------------------------------------------

@dataclass
class RootValue:
    value_root: AttrValue
    attr_name_new : Optional[AttrName]
    do_fetch_by_name: Union[bool, UndefinedType] = UNDEFINED

# ------------------------------------------------------------


class IRegistry:

    @abstractmethod
    def setup(self, setup_session: "ISetupSession") -> None:
        ...

    @abstractmethod
    def finish(self) -> None:
        ...

    @abstractmethod
    def apply_to_get_root_value(self, apply_result: "IApplyResult", attr_name: AttrName) -> RootValue: # noqa: F821
        """
        Used in apply phase. returns instance or instance attribute value + a
        different attribute name when different attribute needs to be retrieved
        from instance.
        """
        ...

# ------------------------------------------------------------



# ------------------------------------------------------------

class ISetupSession(ABC):

    # @abstractmethod
    # def create_local_setup_session(self, this_registry: IThisRegistry) -> Self:
    #     ...

    # @abstractmethod
    # def create_local_setup_session_for_this_instance(self,
    #                                                  model_class: ModelType,
    #                                                  ) -> Self:
    #     " creates Session with ThisRegistryForInstance "
    #     # TODO: ugly name :(
    #     ...

    @abstractmethod
    def get_registry(self, namespace: Namespace, strict: bool = True, is_internal_use: bool = False) -> IRegistry:
        ...

    @abstractmethod
    def __getitem__(self, namespace: Namespace) -> IRegistry:
        ...

    @abstractmethod
    def register_dexp_node(self, dexp_node: IDotExpressionNode):
        " TODO: can be done here "
        ...

    @abstractmethod
    def add_hook_on_finished_all(self, hook_function: HookOnFinishedAllCallable):
        " TODO: can be done here "
        ...

# ------------------------------------------------------------

@dataclass
class IThisRegistry(IRegistry):
    ...

# ------------------------------------------------------------

class IAttributeAccessorBase(ABC):
    " used in registry "

    @abstractmethod
    def get_attribute(self, apply_result: 'IApplyResult', attr_name:str, is_last:bool) -> Self: # noqa: F821
        """ 
        is_last -> True - need to get final literal value from object
        (usually primitive type like int/str/date ...) 
        """
        ...

# ------------------------------------------------------------
# DotExpression
# ------------------------------------------------------------


class DotExpression(DynamicAttrsBase):
    """
    Could be called DotExpression too, but Value expresses the final goal, what
    is more important than syntax or structure.

    shortcut used in the library for var names: dexp
    """
    # NOTE: not a @dataclass for now, reason: overriding too many __dunder__ methods so ...

    # NOTE: each item in this list should be implemented as attribute or method in this class
    # "GetAttrDexpNode",
    # "Read", 
    # "GetNamespace", 
    RESERVED_ATTR_NAMES = {"Clone", "Path", "Setup", "Equals", 
                           "_evaluator",  # allwyays filled, contains all nodes
                           "_dexp_node",  # is last node (redundant), but None if case of error
                           "_node", "_namespace", "_name", "_func_args", "_is_top", "_status",
                           "_is_literal", "_is_internal_use",
                           "_EnsureFinished", "IsFinished",
                           # "_is_reserved_function",
                           } # "_read_functions",  "_dexp_node_name"
    # RESERVED_FUNCTION_NAMES = ("Value",)
    # "First", "Second",

    def __init__(
        self,
        node: Union[str, IDotExpressionNode],
        namespace: Namespace,
        Path: Optional[List[Self]] = None,
        is_literal: bool = False,
        is_internal_use: bool = False,
    ):
        " NOTE: when adding new params, add to Clone() too "
        # SAFE OPERATIONS
        self._status : DExpStatusEnum = DExpStatusEnum.INITIALIZED
        self._namespace = namespace
        self._node = node
        self._is_top = Path is None
        self._name = str(self._node)
        self._is_literal = is_literal
        self._is_internal_use = is_internal_use

        # init Path => to make __str__ works
        self.Path = None 

        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.register(self)

        # -- CHECK PRECONDITIONS
        # risky -> can cause failure in: str(self), repr(self)
        if not isinstance(self._namespace, Namespace):
            raise EntitySetupValueError(owner=self, msg=f"Namespace parameter '{self._namespace}' needs to be instance of Namespace inherited class.")

        if self._is_literal:
            if not isinstance(self._node, STANDARD_TYPE_W_NONE_LIST):
                raise EntitySetupValueError(owner=self, msg=f"Literal type '{self._node}' needs to some standard type: {STANDARD_TYPE_W_NONE_LIST}, got: {type(self._node)}")

        elif isinstance(self._node, str):
            if self._node in self.RESERVED_ATTR_NAMES:
                raise EntitySetupValueError(owner=self, msg=f"Value expression's attribute '{self._node}' is a reserved name, choose another.")
        else:
            if not isinstance(self._node, OperationDexpNode):
                raise EntitySetupValueError(owner=self, msg=f"Value expression's attribute '{self._node}' needs to be string or OperationDexpNode, got: {type(self._node)}")

        # -- COMPUTED VALUES
        # copy list of owner/previous
        self.Path = [] if self._is_top else Path[:]
        self.Path.append(self)

        # FunctionArgumentsType
        self._func_args: Optional[FunctionArgumentsTupleType] = None
        # self._is_reserved_function:bool = self._name in self.RESERVED_FUNCTION_NAMES

        # -- can calculate all, contains all attr_node-s, and the last one is in 
        # not defined in is_predefined mode
        self._evaluator = UNDEFINED
        # the last one attr_node
        self._dexp_node = UNDEFINED


    def Clone(self) -> Self:
        # NOTE: currently not used
        return self.__class__(node=self._node, namespace=self._namespace, Path=self.Path)

    # def GetNamespace(self) -> Namespace:
    #     return self._namespace

    def Equals(self, other: Any) -> bool:
        return (isinstance(other, self.__class__)
                and other._namespace == self._namespace
                and other._func_args == self._func_args
                and other.Path == self.Path)

    def IsFinished(self):
        return self._status!=DExpStatusEnum.INITIALIZED

    def _EnsureFinished(self):
        if self._status!=DExpStatusEnum.INITIALIZED:
            raise EntitySetupError(owner=self, msg=f"Method Setup() already called, further DotExpression building/operator-building is not possible (status={self._status}).")

    def Setup(self, setup_session:ISetupSession, owner:Any) -> Optional['IDotExpressionNode']:
        """
        Owner used just for reference count.
        """
        # TODO: Owner is "ComponentBase" - define some common
        #       protocol/interface and use it
        # TODO: consider dropping owner parameter and use
        #       setup_session.current_frame.component or owner instead?
        # TODO: circular dependency - maybe to pass eval class to this method

        from .expression_evaluators import DotExpressionEvaluator

        self._EnsureFinished()

        # this_registry = setup_session.current_frame.this_registry if setup_session.current_frame else None
        if self._namespace == ThisNS:
            # TODO: DRY this - identical logic in expressions.py :: Setup()
            if not setup_session.current_frame.this_registry:
                raise EntitySetupNameError(owner=self, msg=f"Namespace 'This.' is not available in this context, got: {self._name}")
            registry = setup_session.current_frame.this_registry
        else:
            # if local repo not available or ns not found in it, find in container repo
            registry = setup_session.get_registry(self._namespace, is_internal_use=self._is_internal_use)

        # registry = None
        # local_setup_session = setup_session.current_frame.local_setup_session if setup_session.current_frame else None
        # if local_setup_session:
        #     # try to find in local repo
        #     registry = local_setup_session.get_registry(self._namespace, strict=False, is_internal_use=self._is_internal_use)
        # if not registry:
        #     # if local repo not available or ns not found in it, find in container repo
        #     registry = setup_session.get_registry(self._namespace, is_internal_use=self._is_internal_use)

        if not registry:
            raise EntityInternalError(owner=self, msg=f"Registry not created for: {self._namespace}")

        if self._namespace != registry.NAMESPACE:
            raise EntityInternalError(owner=self, msg=f"Registry has diff namespace from variable: {self._namespace} != {registry.NAMESPACE}")

        current_dexp_node = None
        last_dexp_node = None
        dexp_node_name = None
        dexp_evaluator = DotExpressionEvaluator()

        for bnr, bit in enumerate(self.Path, 1):
            # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.setup_called(bit)

            assert bit._namespace==self._namespace
            # TODO: if self._func_args:
            # operator.attrgetter("last_name")

            # last_dexp_node = (current_dexp_node
            #                if current_dexp_node is not None
            #                else owner)

            # TODO: what with: 
            #       if bit._is_literal:

            # ========================================
            # Operations 
            # ========================================
            if isinstance(bit._node, OperationDexpNode):
                op_node = bit._node
                # one level deeper
                # owner not required, but in this case should be 
                assert bnr == 1
                current_dexp_node = op_node.Setup(setup_session=setup_session, owner=owner)
                dexp_node_name = bit._node
                # _read_functions.append(op_node.apply)

            # ========================================
            # Functions - IFunctionDexpNode
            # ========================================
            elif bit._func_args:
                if bnr==1:
                    if self._namespace != FunctionsNS:
                        raise EntitySetupNameError(owner=self, msg=f"Only FunctionsNS (Fn.) namespace accepts direct function calls. Got '{bit}' on '{bit._namespace}) namespace.")

                func_args = FunctionArgumentsType(*bit._func_args)

                dexp_node_name = bit._node
                if last_dexp_node:
                    owner_arg_type_info = last_dexp_node.get_type_info()
                    if owner_arg_type_info is None:
                        # Postponed ... e.g. F. / FieldsNS  - will be added to hooks later
                        owner_arg_type_info = last_dexp_node.get_type_info
                else:
                    owner_arg_type_info = None

                # : IFunctionDexpNode
                current_dexp_node = registry.create_func_node(
                        setup_session=setup_session,
                        caller=last_dexp_node,
                        attr_node_name=dexp_node_name,
                        func_args=func_args,
                        value_arg_type_info=owner_arg_type_info
                        )
            else:
                # ========================================
                # Attributes
                # ========================================
                # ----------------------------------------
                # Check if Path goes to correct attr_node
                # ----------------------------------------
                # when copy_to_setup_session defined:
                #   read from copy_to_setup_session.setup_session_bind_from and store in both setup_sessions in the
                #   same namespace (usually ModelsNS)
                # setup_session_read_from = copy_to_setup_session.setup_session_bind_from if copy_to_setup_session else setup_session
                # setup_session_read_from = setup_session

                dexp_node_name = bit._node
                current_dexp_node = registry.create_node(
                            dexp_node_name=dexp_node_name,
                            owner_dexp_node=last_dexp_node,
                            owner=owner,
                            )

            # add node to evaluator
            dexp_evaluator.add(current_dexp_node)

            # if is_last and copy_to_setup_session:
            #     current_dexp_node.add_bound_dexp_node(BoundVar(setup_session.name, copy_to_setup_session.dexp_node_name))
            #     setup_session.add(current_dexp_node, alt_dexp_node_name=copy_to_setup_session.dexp_node_name)

            # except NotImplementedError:
            #     raise 
            #     # if strict:
            #     #     raise
            #     # self._status = DExpStatusEnum.ERR_TO_IMPLEMENT
            #     # dexp_evaluator.failed(str(ex))
            #     # break

            last_dexp_node = current_dexp_node

            # can be Component or can be managed Model dataclass Field - when .denied is not appliable
            if hasattr(current_dexp_node, "denied") and current_dexp_node.denied:
                # '{dexp_node_name}' 
                raise EntitySetupValueError(owner=self, msg=f"DexpNode (owner={owner.name}) references '{current_dexp_node.name}' what is not allowed due: {current_dexp_node.deny_reason}.")

        dexp_evaluator.finish()


        if dexp_evaluator.is_all_ok():
            self._status = DExpStatusEnum.BUILT
            self._all_ok = True
            self._evaluator = dexp_evaluator
            self._dexp_node = dexp_evaluator.last_node()
            setup_session.register_dexp_node(self._dexp_node)
        else:
            # TODO: raise EntitySetupError()
            self._all_ok = False
            self._evaluator = dexp_evaluator
            self._dexp_node = None

        return self._dexp_node

    # def __getitem__(self, ind):
    #     # list [0] or dict ["test"]
    #     return DotExpression( Path=self.Path + "." + str(ind))

    def __getattr__(self, aname):
        self._EnsureFinished()

        if aname in self.RESERVED_ATTR_NAMES: # , "%r -> %s" % (self._node, aname):
            raise EntitySetupNameError(owner=self, msg=f"DotExpression's attribute '{aname}' is reserved name, choose another.")
        if aname.startswith("__") and aname.endswith("__"):
            raise AttributeError(f"Attribute '{type(self)}' object has no attribute '{aname}'")
        if aname.startswith("_"):
            raise AttributeError(f"Attribute '{type(self)}' object has no attribute (2) '{aname}'")
        return DotExpression(node=aname, namespace=self._namespace, Path=self.Path)

    def __call__(self, *args, **kwargs):
        if self._func_args is not None:
            raise EntityInternalError(owner=self, msg="Node already a function, duplicate call?") 
        self._func_args = [args, kwargs]
        return self

    def as_str(self):
        " NOTE: this is important for dumping and later loading - must match same string as code "
        if self._is_literal:
            out=f"Just({repr(self._node)})"
        else:
            out = ""
            if self._is_top and self._namespace != OperationsNS:
                out += f"{self._namespace}."
            out += f"{self._node}"
            if self._func_args:
                out += "("
                args, kwargs = self._func_args
                if args:
                    out += ", ".join([f"{a}" for a in args])
                if kwargs:
                    out += ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                out += ")"
        return out

    def __str__(self):
        return ".".join([ve.as_str() for ve in self.Path]) if self.Path else f"{self.__class__.__name__}(... {self._node})"

    def __repr__(self):
        return f"DExpr({self})"

    # --------------------------------
    # ------- Reserved methods -------
    # --------------------------------
    # NOTE: each method should be listed in RESERVED_ATTR_NAMES

    # ----------------------------------
    # ------- Internal methods ---------
    # https://realpython.com/python-bitwise-operators/#custom-data-types
    # comparison operators <, <=, ==, !=, >=, >

    # NOTE: Operations are put in internal OperationsNS

    def __eq__(self, other):        self._EnsureFinished(); return DotExpression(OperationDexpNode("==", self, other), namespace=OperationsNS, is_internal_use=True)  # noqa: E702
    def __ne__(self, other):        self._EnsureFinished(); return DotExpression(OperationDexpNode("!=", self, other), namespace=OperationsNS, is_internal_use=True)  # noqa: E702
    def __gt__(self, other):        self._EnsureFinished(); return DotExpression(OperationDexpNode(">" , self, other), namespace=OperationsNS, is_internal_use=True)  # noqa: E702
    def __ge__(self, other):        self._EnsureFinished(); return DotExpression(OperationDexpNode(">=", self, other), namespace=OperationsNS, is_internal_use=True)  # noqa: E702
    def __lt__(self, other):        self._EnsureFinished(); return DotExpression(OperationDexpNode("<" , self, other), namespace=OperationsNS, is_internal_use=True)  # noqa: E702
    def __le__(self, other):        self._EnsureFinished(); return DotExpression(OperationDexpNode("<=", self, other), namespace=OperationsNS, is_internal_use=True)  # noqa: E702

    # +, -, *, /
    def __add__(self, other):       self._EnsureFinished(); return DotExpression(OperationDexpNode("+" , self, other), namespace=OperationsNS, is_internal_use=True)  # noqa: E702
    def __sub__(self, other):       self._EnsureFinished(); return DotExpression(OperationDexpNode("-" , self, other), namespace=OperationsNS, is_internal_use=True)  # noqa: E702
    def __mul__(self, other):       self._EnsureFinished(); return DotExpression(OperationDexpNode("*" , self, other), namespace=OperationsNS, is_internal_use=True)  # noqa: E702
    def __truediv__(self, other):   self._EnsureFinished(); return DotExpression(OperationDexpNode("/" , self, other), namespace=OperationsNS, is_internal_use=True)  # noqa: E702

    # what is this??
    def __floordiv__(self, other):  self._EnsureFinished(); return DotExpression(OperationDexpNode("//", self, other), namespace=OperationsNS, is_internal_use=True)  # noqa: E702

    # in
    def __contains__(self, other):  self._EnsureFinished(); return DotExpression(OperationDexpNode("in", self, other), namespace=OperationsNS, is_internal_use=True)  # noqa: E702

    # Bool operators, NOT, AND, OR
    def __invert__(self):           self._EnsureFinished(); return DotExpression(OperationDexpNode("not", self       ),namespace=OperationsNS, is_internal_use=True)  # ~  # noqa: E702
    def __and__(self, other):       self._EnsureFinished(); return DotExpression(OperationDexpNode("and", self, other),namespace=OperationsNS, is_internal_use=True)  # &  # noqa: E702
    def __or__(self, other):        self._EnsureFinished(); return DotExpression(OperationDexpNode("or" , self, other),namespace=OperationsNS, is_internal_use=True)  # |  # noqa: E702

    # ------------------------------------------------------------

    # NOTE: good candidates for future operator's implementations: 
    #   [i], [i:j]          Slicing 	seq[i:j] 	getitem(seq, slice(i, j))
    #   ^   __xor__         Return the bitwise exclusive or of a and b.
    #   <<  __lshift__      Return a shifted left by b.
    #   >>  __rshift__      Return a shifted right by b.
    #   ** 	__pow__ 	    Exponentiation
    #   @   __matmul__      Matrix Multiplication 	a @ b 	matmul(a, b)
    #   %   __mod__         String Formatting / Modulus	/ mod(s, obj)

    # https://docs.python.org/3/library/operator.html

    # other:
    #   __abs__ - abs()
    #   Positive 	+ a 	pos(a)
    #   Slice Assignment 	seq[i:j] = values 	setitem(seq, slice(i, j), values)
    #   Slice Deletion 	del seq[i:j] 	delitem(seq, slice(i, j))
    #   Truth Test 	obj 	truth(obj)
    #   setitem(seq, slice(i, j), values) - seq[i:j] = values
    #       When[ a > b : 25 ], When[ a > b : 25 ]
    #   operator.__ixor__(a, b) - a ^= b.
    #   operator.__irshift__(a, b) - a >>= b.
    #         When(a > b) >>= 25


    # def __rshift__(self, other):
    #     """
    #     Streaming:
    #         List[Instance] >> Function() # receives Qs/List, like Map(Function, List) -> List
    #         Instance >> Function()       # receives Instance, returns
    #     """
    #     return DotExpression(StreamOperation(">>", self, other), namespace=OperationsNS)  # >>


class Just(DotExpression):
    """ simple wrapper to enable literal as first argument in operations - e.g. 1 + M.id => Just(1) + M.id """

    def __init__(self, value: LiteralType):
        if not isinstance(value, STANDARD_TYPE_W_NONE_LIST):
            raise NotImplementedError(f"{value}.type unhandled: '{type(value)}'")
        super().__init__(
            node = value,
            namespace=OperationsNS,
            is_literal=True,
            )

# ------------------------------------------------------------


@dataclass
class NotAvailableExecResult(ExecResult):
    " used when available yields False - value contains "
    reason: str = field(default=None)

    @classmethod
    def create(cls, 
            available_dexp_result: Union[ExecResult, UndefinedType]=UNDEFINED, 
            reason: Optional[str] = None) -> Self:
        if not reason:
            if available_dexp_result:
                assert not isinstance(ExecResult, NotAvailableExecResult)
                reason=f"Not available since expression yields: {available_dexp_result.value}", 
            else:
                if available_dexp_result not in (None, UNDEFINED):
                    raise EntityInternalError(msg=f"Expected None/Undefined, got: {available_dexp_result}") 
                reason="Value not available"
        instance = cls(reason=reason)
        instance.value = available_dexp_result
        return instance




# ------------------------------------------------------------
# IFunctionDexpNode - name used in-between just to emphasize that this is Dexp
#                     Node
# ------------------------------------------------------------

class IFunctionDexpNode(IDotExpressionNode):
    """
    a bit an overhead -> just to have better naming for base class
    """
    pass


@dataclass
class LiteralDexpNode(IDotExpressionNode):

    value : Any
    dexp_result: ExecResult = field(repr=False, init=False)
    type_info: TypeInfo = field(repr=False, init=False)

    # later evaluated
    is_finished: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        self.dexp_result = ExecResult()
        self.dexp_result.set_value(self.value, attr_name="", changer_name="")
        self.type_info = TypeInfo.get_or_create_by_type(type(self.value), caller=self)

    def get_type_info(self) -> TypeInfo:
        return self.type_info

    def execute_node(self, 
                 apply_result: "IApplyResult", # noqa: F821
                 dexp_result: ExecResult,
                 is_last: bool,
                 prev_node_type_info: Optional[TypeInfo],
                 ) -> ExecResult:
        assert not dexp_result
        return self.dexp_result

# ------------------------------------------------------------

class DExpStatusEnum(str, Enum):
    INITIALIZED      = "INIT"
    ERR_NOT_FOUND    = "ERR_NOT_FOUND"
    ERR_TO_IMPLEMENT = "ERR_TO_IMPLEMENT"
    BUILT            = "BUILT" # OK

# ------------------------------------------------------------
# Operations
# ------------------------------------------------------------

# Custom implementation - no operator function, needs custom
# logic

def _op_apply_and(first, second):
    return bool(first) and bool(second)

def _op_apply_or(first, second):
    return bool(first) or bool(second)

class OutputTypeInfoTypeEnum(str, Enum):
    FROM_FIRST_DEXP = "FROM_1ST"
    BOOL_TYPE = "BOOL"

@dataclass
class Operation:
    code: str
    dexp_code: str
    ast_node_type : ast.AST
    apply_function: Callable
    load_function : Callable
    output_type_info_type: OutputTypeInfoTypeEnum \
        = OutputTypeInfoTypeEnum.FROM_FIRST_DEXP



# https://florian-dahlitz.de/articles/introduction-to-pythons-operator-module
# https://docs.python.org/3/library/operator.html#mapping-operators-to-functions
OPCODE_TO_FUNCTION = {
    # NOTE: no need to check unary/binary vs have 1 or 2 params
    #       python parser/interpretor will ensure this
    # binary operatorsy - buultin
      "=="  : Operation(code="==" , dexp_code="==" , ast_node_type= ast.Eq      , apply_function= operator.eq      , load_function= operator.eq,  # noqa: E131
                        output_type_info_type=OutputTypeInfoTypeEnum.BOOL_TYPE)
    , "!="  : Operation(code="!=" , dexp_code="!=" , ast_node_type= ast.NotEq   , apply_function= operator.ne      , load_function= operator.ne,  # noqa: E131
                        output_type_info_type=OutputTypeInfoTypeEnum.BOOL_TYPE)
    , ">"   : Operation(code=">"  , dexp_code=">"  , ast_node_type= ast.Gt      , apply_function= operator.gt      , load_function= operator.gt,
                        output_type_info_type=OutputTypeInfoTypeEnum.BOOL_TYPE)
    , ">="  : Operation(code=">=" , dexp_code=">=" , ast_node_type= ast.GtE     , apply_function= operator.ge      , load_function= operator.ge,
                        output_type_info_type=OutputTypeInfoTypeEnum.BOOL_TYPE)
    , "<"   : Operation(code="<"  , dexp_code="<"  , ast_node_type= ast.Lt      , apply_function= operator.lt      , load_function= operator.lt,
                        output_type_info_type=OutputTypeInfoTypeEnum.BOOL_TYPE)
    , "<="  : Operation(code="<=" , dexp_code="<=" , ast_node_type= ast.LtE     , apply_function= operator.le      , load_function= operator.le,
                        output_type_info_type=OutputTypeInfoTypeEnum.BOOL_TYPE)

    , "+"   : Operation(code="+"  , dexp_code="+"  , ast_node_type= ast.Add     , apply_function= operator.add     , load_function= operator.add)
    , "-"   : Operation(code="-"  , dexp_code="-"  , ast_node_type= ast.Sub     , apply_function= operator.sub     , load_function= operator.sub)
    , "*"   : Operation(code="*"  , dexp_code="*"  , ast_node_type= ast.Mult    , apply_function= operator.mul     , load_function= operator.mul)
    , "/"   : Operation(code="/"  , dexp_code="/"  , ast_node_type= ast.Div     , apply_function= operator.truediv , load_function= operator.truediv)
    , "//"  : Operation(code="//" , dexp_code="//" , ast_node_type= ast.FloorDiv, apply_function= operator.floordiv, load_function= operator.floordiv)

    , "in"  : Operation(code="in" , dexp_code="in" , ast_node_type= ast.In      , apply_function= operator.contains, load_function= operator.contains,
                        output_type_info_type = OutputTypeInfoTypeEnum.BOOL_TYPE)

    # TODO: logicaal operators - work as python OR -> returns non-bool or not?
    , "and" : Operation(code="and", dexp_code="&"  , ast_node_type= ast.BitAnd  , apply_function= _op_apply_and    , load_function= operator.iand)
    , "or"  : Operation(code="or" , dexp_code="|"  , ast_node_type= ast.BitOr   , apply_function= _op_apply_or     , load_function= operator.ior)

    # unary operators                              
    , "not" : Operation(code="not", dexp_code="~"  , ast_node_type= ast.Invert  , apply_function= operator.not_    , load_function= operator.invert,
                        output_type_info_type = OutputTypeInfoTypeEnum.BOOL_TYPE)
}


# Other:
#   ast.Dict ast.DictComp ast.List ast.ListComp
#   ast.Pow ast.MatMult ast.Mod
#   ast.BitXor ast.RShift ast.LShift
#   ast.BoolOp ast.Compare ast.Constant ast.ExtSlice ast.Index ast.Is ast.IsNot ast.Slice

AST_NODE_TYPE_TO_FUNCTION = {
    operation.ast_node_type: operation
    for operation in OPCODE_TO_FUNCTION.values()
}


#       ast.Eq       : operator.eq  # noqa: E131
#     , ast.NotEq    : operator.ne   
#     , ast.Gt       : operator.gt
#     , ast.GtE      : operator.ge
#     , ast.Lt       : operator.lt
#     , ast.LtE      : operator.le
# 
#     , ast.Add      : operator.add
#     , ast.Sub      : operator.sub
#     , ast.Mult     : operator.mul
#     , ast.Div      : operator.truediv
#     , ast.FloorDiv : operator.floordiv
# 
#     , ast.In       : operator.contains
# 
#     # binary ops
#     , ast.BitAnd   : operator.iand # &
#     , ast.BitOr    : operator.ior  #  |
# 
#     # unary operators:
#     , ast.Invert   : operator.invert  # ~
# 
# 
# }


@dataclass
class OperationDexpNode(IDotExpressionNode):
    """
    Binary or Unary operators that handles
    basically will do following:

        op_function(first[, second]) -> value

    """
    op: str  # OPCODE_TO_FUNCTION.keys()
    first: Any
    second: Union[Any, UndefinedType] = UNDEFINED

    # later evaluated
    op_function: Callable = field(repr=False, init=False)
    operation: Operation = field(repr=False, init=False)
    _status : DExpStatusEnum = field(repr=False, init=False, default=DExpStatusEnum.INITIALIZED)
    _all_ok : Optional[bool] = field(repr=False, init=False, default=None)

    _first_dexp_node : Optional[IDotExpressionNode] = field(repr=False, init=False, default=None)
    _second_dexp_node : Optional[IDotExpressionNode] = field(repr=False, init=False, default=None)

    is_finished: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        self.operation = self._get_operation(self.op)
        self.op_function = self.operation.apply_function

        self._status : DExpStatusEnum = DExpStatusEnum.INITIALIZED
        self._all_ok : Optional[bool] = None
        self._output_type_info: Union[TypeInfo, UNDEFINED] = UNDEFINED 
        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.register(self)


    def _get_operation(self, op: str) -> Operation:
        operation = OPCODE_TO_FUNCTION.get(op, None)
        if operation is None:
            raise EntitySetupValueError(owner=self, msg="Invalid operation code, {self.op} not one of: {', '.join(self.OP_TO_CODE.keys())}")
        return operation

    def get_type_info(self) -> TypeInfo:
        " type_info from first node "
        # Assumption - for all operations type_info of first operation
        #       argumnent will persist to result
        # TODO: consider also last node if available
        if self._output_type_info is UNDEFINED:
            # TODO: treba biti bool
            if self.operation.output_type_info_type == OutputTypeInfoTypeEnum.BOOL_TYPE:
                self._output_type_info = TypeInfo.get_or_create_by_type(bool, caller=self)
            elif self.operation.output_type_info_type == OutputTypeInfoTypeEnum.FROM_FIRST_DEXP:
                self._output_type_info = self._first_dexp_node.get_type_info()
            else:
                raise EntityInternalError(owner=self, msg=f"Unsupported operation.output_type_info_type={self.operation.output_type_info_type}")
        return self._output_type_info

    @staticmethod
    def create_dexp_node(
                   dexp_or_other: Union[DotExpression, Any], 
                   title: str,
                   setup_session: ISetupSession, # noqa: F821
                   owner: Any) -> IDotExpressionNode:
        if isinstance(dexp_or_other, DotExpression):
            dexp_node = dexp_or_other.Setup(setup_session, owner=owner)
        elif isinstance(dexp_or_other, IDotExpressionNode):
            raise NotImplementedError(f"{title}.type unhandled: '{type(dexp_or_other)}' => '{dexp_or_other}'")
            # dexp_node = dexp_or_other
        else:
            # TODO: check other types - maybe some unappliable
            if not isinstance(dexp_or_other, STANDARD_TYPE_W_NONE_LIST):
                raise NotImplementedError(f"{title}.type unhandled: '{type(dexp_or_other)}' => '{dexp_or_other}'")
            dexp_node = LiteralDexpNode(value=dexp_or_other)

        return dexp_node


    def Setup(self, setup_session: ISetupSession, owner: Any) -> "DotExpressionEvaluator":  # noqa: F821
        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.setup_called(self)

        if not self._status==DExpStatusEnum.INITIALIZED:
            raise EntitySetupError(owner=setup_session, item=self, msg=f"AttrDexpNode not in INIT state, got {self._status}")

        # just to check if all ok
        self._first_dexp_node = self.create_dexp_node(self.first, title="First", setup_session=setup_session, owner=owner)
        setup_session.register_dexp_node(self._first_dexp_node)

        if self.second is not UNDEFINED:
            self._second_dexp_node = self.create_dexp_node(self.second, title="second", setup_session=setup_session, owner=owner)
            setup_session.register_dexp_node(self._second_dexp_node)

        self._status=DExpStatusEnum.BUILT

        return self

    def __str__(self):
        if self.second is not UNDEFINED:
            return f"({self.first} {self.operation.dexp_code} {self.second})"
        else:
            return f"({self.operation.dexp_code} {self.first})"

    def __repr__(self):
        return f"Op{self}"


    def execute_node(self, 
                 apply_result: "IApplyResult", # noqa: F821
                 dexp_result: ExecResult,
                 is_last: bool,
                 prev_node_type_info: Optional[TypeInfo],
                 ) -> ExecResult:

        if is_last and not self.is_finished:
            raise EntityInternalError(owner=self, msg="Last dexp node is not finished.") 

        if dexp_result:
            raise NotImplementedError("TODO:")

        first_dexp_result = execute_dexp_or_node(
                                self.first, 
                                self._first_dexp_node, 
                                prev_node_type_info=prev_node_type_info,
                                dexp_result=dexp_result,
                                apply_result=apply_result)

        # if self.second is not in (UNDEFINED, None):
        if self._second_dexp_node is not None:
            second_dexp_result = execute_dexp_or_node(
                                    self.second, 
                                    self._second_dexp_node, 
                                    prev_node_type_info=prev_node_type_info,
                                    dexp_result=dexp_result,
                                    apply_result=apply_result)
            # binary operation second argument adaption?
            #   string + number -> string + str(int)
            first_value = first_dexp_result.value
            second_value = second_dexp_result.value
            type_adapter = apply_result\
                    .binary_operations_type_adapters\
                    .get((type(first_value), type(second_value)), None)

            if type_adapter:
                second_value = type_adapter(second_value)

            try:
                new_value = self.op_function(first_value, second_value)
            except Exception as ex:
                raise EntityApplyError(owner=apply_result, msg=f"{self} := {self.op_function}({first_dexp_result.value}, {second_dexp_result.value}) raised error: {ex}")
        else:
            # unary operation
            try:
                new_value = self.op_function(first_dexp_result.value)
            except Exception as ex:
                # raise
                raise EntityApplyError(owner=apply_result, msg=f"{self} := {self.op_function}({first_dexp_result.value}) raised error: {ex}")

        op_dexp_result = ExecResult()

        # TODO: we are loosing: first_dexp_result / second_dexp_result
        #       to store it in result somehow?
        op_dexp_result.set_value(new_value, "", f"Op[{self.op}]")

        return op_dexp_result



# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def execute_available_dexp(
        available_dexp: Optional[Union[bool, DotExpression]], 
        apply_result: "IApplyResult") \
                -> Optional[NotAvailableExecResult]: # noqa: F821
    " returns NotAvailableExecResult when not available with details in instance, if all ok -> returns None "
    if isinstance(available_dexp, DotExpression):
        available_dexp_result = available_dexp._evaluator.execute_dexp(apply_result=apply_result)
        if not bool(available_dexp_result.value):
            return NotAvailableExecResult.create(available_dexp_result=available_dexp_result)
    elif isinstance(available_dexp, bool):
        if available_dexp is False:
            return NotAvailableExecResult.create(available_dexp_result=None)
    # all ok
    return None


# ------------------------------------------------------------


def execute_dexp_or_node(
        dexp_or_value: Union[DotExpression, Any],
        # Union[OperationDexpNode, IFunctionDexpNode, LiteralDexpNode]
        dexp_node: Union[IDotExpressionNode, Any], 
        prev_node_type_info: TypeInfo,
        dexp_result: ExecResult,
        apply_result: "IApplyResult" # noqa: F821
        ) -> ExecResult:

    # TODO: this function has ugly interface - solve this better

    if isinstance(dexp_or_value, DotExpression):
        dexp_result = dexp_or_value._evaluator.execute_dexp(
                            apply_result=apply_result,
                            )
    # AttrDexpNode, OperationDexpNode, IFunctionDexpNode, LiteralDexpNode,
    elif isinstance(dexp_node, (
            IDotExpressionNode,
            )):
        dexp_result = dexp_node.execute_node(
                            apply_result=apply_result, 
                            dexp_result=dexp_result,
                            prev_node_type_info=prev_node_type_info,
                            is_last=True,
                            )
    else:
        raise EntityInternalError(
                f"Expected Dexp, OpDexp or FuncDexp, got: {type(dexp_or_value)} -> {dexp_or_value} / {type(dexp_node)} -> {dexp_node}")

    return dexp_result


@dataclass
class FuncArgAttrnameTypeHint(FuncArgTypeHint):
    inner_type: Optional[Type] = field(repr=True, default=Any)
    type: Type = field(init=False, default=DotExpression)

    def __hash__(self):
        return hash((self.__class__.__name__, self.type, self.inner_type))

@dataclass
class FuncArgDotexprTypeHint(FuncArgTypeHint):
    inner_type: Optional[Type] = field(repr=True, default=Any)
    type: Type = field(init=False, default=DotExpression)

    def __hash__(self):
        return hash((self.__class__.__name__, self.type, self.inner_type))

