# Copied and adapted from Reedwolf project (project by robert.lujo@gmail.com - git@bitbucket.org:trebor74hr/reedwolf.git)
from __future__ import annotations

from abc import (
        ABC, 
        abstractmethod,
        )
import operator
from dataclasses import (
        dataclass, 
        field,
        replace as dataclasses_replace,
        )
from enum import Enum

from typing import (
        List,
        Optional,
        Union,
        Any,
        Callable,
        )
from .utils import (
        UNDEFINED,
        )
from .exceptions import (
        RuleSetupValueError,
        RuleSetupError,
        RuleSetupNameError,
        RuleInternalError,
        RuleApplyError,
        )
from .namespaces import (
        DynamicAttrsBase,
        FunctionsNS,
        OperationsNS,
        Namespace,
        )
from .meta import (
        FunctionArgumentsType,
        FunctionArgumentsTupleType,
        STANDARD_TYPE_W_NONE_LIST,
        )


# ------------------------------------------------------------
# internal structs
# ------------------------------------------------------------

@dataclass
class RawAttrValue:
    value: Any
    attr_name: str
    changer_name: str

@dataclass
class VexpResult:
    # last value, mutable
    value: Any  = field(init=False, default=UNDEFINED)
    # TODO: set compoenent (owner) that triggerred value change
    #       value evaluation - can have attr_node.name
    value_history : List[Tuple[str, RawAttrValue]] = field(repr=False, init=False, default_factory=list)

    def set_value(self, value: Any, attr_name: str, changer_name: str, ):
        self.value_history.append(
                RawAttrValue(value=value, attr_name=attr_name, changer_name=changer_name)
                )
        self.value = value


# ------------------------------------------------------------
# evaluate_vexp_or_node
# ------------------------------------------------------------

def evaluate_vexp_or_node(
        vexp_or_value: Union[ValueExpression, Any],
        # Union[OperationVexpNode, IFunctionVexpNode, LiteralVexpNode]
        vexp_node: Union[IValueExpressionNode, Any], 
        vexp_result: VexpResult,
        apply_session: "IApplySession"
        ) -> VexpResult:

    # TODO: this function has ugly interface - solve this better

    if isinstance(vexp_or_value, ValueExpression):
        vexp_result = vexp_or_value._evaluator.evaluate(
                            apply_session=apply_session,
                            )
    # AttrVexpNode, OperationVexpNode, IFunctionVexpNode, LiteralVexpNode,
    elif isinstance(vexp_node, (
            IValueExpressionNode,
            )):
        vexp_result = vexp_node.evaluate(
                            apply_session=apply_session, 
                            vexp_result=vexp_result,
                            )
    else:
        raise RuleInternalError(
                f"Expected Vexp, OpVexp or FuncVexp, got: {type(vexp_or_value)} -> {vexp_or_value} / {type(vexp_node)} -> {vexp_node}")

    return vexp_result

# ------------------------------------------------------------
# get_type_info_from_vexp_or_node
# ------------------------------------------------------------
def get_type_info_from_vexp_or_node(
        # IValueExpressionNode == Union[OperationVexpNode, IFunctionVexpNode, AttrVexpNode, LiteralVexpNode]
        vexp_node: IValueExpressionNode,
        ) -> TypeInfo:
    # TODO: remove the function and expose same function (abstract) to all vexpnode types: get_output_type_info()
    if isinstance(vexp_node, IFunctionVexpNode):
        type_info = vexp_node.get_output_type_info()
    elif isinstance(vexp_node, OperationVexpNode):
        # Assumption - for all operations type_info of first operation
        # argumnent will persist to result
        type_info = vexp_node.get_first_type_info()
    elif isinstance(vexp_node, IValueExpressionNode):
        # TODO: AttrVexpNode?
        type_info = vexp_node.type_info
    else:
        # TODO: what with LiteralVexpNode
        raise RuleSetupTypeError(msg=f"Unsupported type: {type(vexp_node)} / {vexp_node}")
    return type_info

# ------------------------------------------------------------
# IValueExpressionNode
# ------------------------------------------------------------

@dataclass
class IValueExpressionNode(ABC):
    """ wrapper around one element in ValueExpression e.g. M.company.name.Count()
    .company, .name, .Count() are nodes
    """
    def clone(self):
        # If already setup then copy it and reuse
        return dataclasses_replace(self)

    @abstractmethod
    def evaluate(self, 
                 apply_session: "IApplySession", 
                 # previous - can be undefined too
                 vexp_result: Union[VexpResult, UndefinedType],
                 ) -> VexpResult:
        raise NotImplementedError()



# ------------------------------------------------------------
# IFunctionVexpNode - name used in-between just to emphasize that this is Vexp
#                     Node
# ------------------------------------------------------------

class IFunctionVexpNode(IValueExpressionNode):
    """
    a bit an overhead -> just to have better naming for base class
    """
    pass


@dataclass
class LiteralVexpNode(IValueExpressionNode):

    value : Any
    vexp_result: VexpResult = field(repr=False, init=False)

    def __post_init__(self):
        self.vexp_result = VexpResult()
        self.vexp_result.set_value(self.value, attr_name="", changer_name="")

    def evaluate(self, 
                 apply_session: "IApplySession", 
                 vexp_result: VexpResult,
                 ) -> VexpResult:
        assert not vexp_result
        return self.vexp_result

# ------------------------------------------------------------

class VExpStatusEnum(str, Enum):
    INITIALIZED      = "INIT"
    ERR_NOT_FOUND    = "ERR_NOT_FOUND"
    ERR_TO_IMPLEMENT = "ERR_TO_IMPLEMENT"
    BUILT            = "BUILT" # OK

# ------------------------------------------------------------
# Operations
# ------------------------------------------------------------

# Custom implementation - no operator function, needs custom
# logic

def _op_apply_and(self, first, second):
    return bool(first) and bool(second)

def _op_apply_or(self, first, second):
    return bool(first) or bool(second)


# https://florian-dahlitz.de/articles/introduction-to-pythons-operator-module
# https://docs.python.org/3/library/operator.html#mapping-operators-to-functions
OPCODE_TO_FUNCTION = {
    # binary operatorsy - buultin
      "=="  : operator.eq  # noqa: E131
    , "!="  : operator.ne   
    , ">"   : operator.gt
    , ">="  : operator.ge
    , "<"   : operator.lt
    , "<="  : operator.le

    , "+"   : operator.add
    , "-"   : operator.sub
    , "*"   : operator.mul
    , "/"   : operator.truediv
    , "//"  : operator.floordiv

    , "in"  : operator.contains

    # binary operatorsy - custom implementation 
    , "and" : _op_apply_and   # orig: &
    , "or"  : _op_apply_or    # orig: |

    # unary operators:
    , "not" : operator.not_   # orig: ~

}


@dataclass
class OperationVexpNode(IValueExpressionNode):
    """
    Binary or Unary operators that handles
    basically will do following:

        op_function(first[, second]) -> value

    """
    op: str  # OPCODE_TO_FUNCTION.keys()
    first: Any
    second: Union[Any, UNDEFINED] = UNDEFINED

    # later evaluated
    op_function: Callable[[...], Any] = field(repr=False, init=False)
    _status : VExpStatusEnum = field(repr=False, init=False, default=VExpStatusEnum.INITIALIZED)
    _all_ok : Optional[bool] = field(repr=False, init=False, default=None)

    _first_vexp_node : Optional[IValueExpressionNode] = field(repr=False, init=False, default=None)
    _second_vexp_node : Optional[IValueExpressionNode] = field(repr=False, init=False, default=None)

    def __post_init__(self):
        self.op_function = self.get_op_function(self.op)
        self._status : VExpStatusEnum = VExpStatusEnum.INITIALIZED
        self._all_ok : Optional[bool] = None

        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.register(self)

    def get_first_type_info(self):
        return self._first_vexp_node.type_info

    def get_op_function(self, op: str) -> Callable[..., Any]:
        op_function = OPCODE_TO_FUNCTION.get(op, None)
        if op_function is None:
            raise RuleSetupValueError(owner=self, msg="Invalid operation code, {self.op} not one of: {', '.join(self.OP_TO_CODE.keys())}")
        return op_function


    @staticmethod
    def create_vexp_node(
                   vexp_or_other: Union[ValueExpression, Any], 
                   label: str,
                   registries: "Registries", 
                   owner: Any) -> IValueExpressionNode:
        if isinstance(vexp_or_other, ValueExpression):
            vexp_node = vexp_or_other.Setup(registries, owner=owner)
        elif isinstance(vexp_or_other, IValueExpressionNode):
            raise NotImplementedError(f"{label}.type unhandled: '{type(vexp_or_other)}' => '{vexp_or_other}'")
            # vexp_node = vexp_or_other
        else:
            # TODO: check other types - maybe some unappliable
            if not isinstance(vexp_or_other, STANDARD_TYPE_W_NONE_LIST):
                raise NotImplementedError(f"{label}.type unhandled: '{type(vexp_or_other)}' => '{vexp_or_other}'")
            vexp_node = LiteralVexpNode(value=vexp_or_other)

        return vexp_node


    def Setup(self, registries: "Registries", owner: Any) -> ValueExpressionEvaluator:  # noqa: F821
        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.setup_called(self)

        if not self._status==VExpStatusEnum.INITIALIZED:
            raise RuleSetupError(owner=registries, item=self, msg=f"AttrVexpNode not in INIT state, got {self._status}")

        # just to check if all ok
        self._first_vexp_node = self.create_vexp_node(self.first, label="First", registries=registries, owner=owner)

        if self.second is not UNDEFINED:
            self._second_vexp_node = self.create_vexp_node(self.second, label="second", registries=registries, owner=owner)

        self._status=VExpStatusEnum.BUILT

        return self

    def __str__(self):
        if self.second is not UNDEFINED:
            return f"({self.first} {self.op} {self.second})"
        else:
            return f"({self.op} {self.first})"

    def __repr__(self):
        return f"Op{self}"


    # vexp_result = node.evaluate(apply_session, vexp_result)
    # def evaluate(self, registries: "Registries", input_value: Any, context:Any) -> Any:  # noqa: F821
    def evaluate(self, 
                 apply_session: "IApplySession", 
                 vexp_result: VexpResult,
                 ) -> VexpResult:
        # this should be included: context.this_registry: ThisRegistry

        if vexp_result:
            raise NotImplementedError("TODO:")

        first_vexp_result = evaluate_vexp_or_node(
                                self.first, self._first_vexp_node, 
                                vexp_result=vexp_result,
                                apply_session=apply_session)

        # TODO: use self._second_vexp_node
        if self.second is not None:
            second_vexp_result = evaluate_vexp_or_node(
                                    self.second, self._second_vexp_node, 
                                    vexp_result=vexp_result,
                                    apply_session=apply_session)
            # binary operation second argument adaption?
            #   string + number -> string + str(int)
            first_value = first_vexp_result.value
            second_value = second_vexp_result.value
            type_adapter = apply_session\
                    .binary_operations_type_adapters\
                    .get((type(first_value), type(second_value)), None)

            if type_adapter:
                second_value = type_adapter(second_value)

            try:
                new_value = self.op_function(first_value, second_value)
            except Exception as ex:
                # raise
                raise RuleApplyError(owner=apply_session, msg=f"{self} := {self.op_function}({first_vexp_result.value}, {second_vexp_result.value}) raised error: {ex}")
        else:
            # unary operation
            try:
                new_value = self.op_function(first_vexp_result.value)
            except Exception as ex:
                # raise
                raise RuleApplyError(owner=apply_session, msg=f"{self} := {self.op_function}({first_vexp_result.value}) raised error: {ex}")

        op_vexp_result = VexpResult()

        # TODO: we are loosing: first_vexp_result / second_vexp_result
        #       to store it in result somehow?
        op_vexp_result.set_value(new_value, "", f"Op[{self.op}]")

        return op_vexp_result



# ------------------------------------------------------------
# ValueExpression
# ------------------------------------------------------------


class ValueExpression(DynamicAttrsBase):
    """
    Could be called DotExpression too, but Value expresses the final goal, what
    is more important than syntax or structure.

    shortcut used in the library for var names: vexp
    """
    # NOTE: not a @dataclass for now, reason: overriding too many __dunder__ methods so ...

    # NOTE: each item in this list should be implemented as attribute or method in this class
    # "GetAttrVexpNode",
    # "Read", 
    RESERVED_ATTR_NAMES = {"Path", "Setup", "GetNamespace",
                           "_evaluator",  # allwyays filled, contains all nodes
                           "_vexp_node",  # is last node (redundant), but None if case of error
                           "_node", "_namespace", "_name", "_func_args", "_is_top", "_status",
                           "_EnsureFinished", "IsFinished",
                           } # "_read_functions",  "_vexp_node_name"
    RESERVED_FUNCTION_NAMES = ("Value",)
    # "First", "Second",

    def __init__(
        self,
        node: Union[str, OperationVexpNode],
        namespace: Namespace,
        Path: Optional[List[ValueExpression]] = None,
    ):
        # SAFE OPERATIONS
        self._status : VExpStatusEnum = VExpStatusEnum.INITIALIZED
        self._namespace = namespace
        self._node = node
        self._is_top = Path is None
        self._name = str(self._node)
        # init Path => to make __str__ works
        self.Path = None 

        # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.register(self)

        # -- CHECK PRECONDITIONS
        # risky -> can cause failure in: str(self), repr(self)
        if not isinstance(self._namespace, Namespace):
            raise RuleSetupValueError(owner=self, msg=f"Namespace parameter '{self._namespace}' needs to be instance of Namespace inherited class.")
        if isinstance(self._node, str):
            if self._node in self.RESERVED_ATTR_NAMES:
                raise RuleSetupValueError(owner=self, msg=f"Value expression's attribute '{self._node}' is a reserved name, choose another.")
        else:
            if not isinstance(self._node, OperationVexpNode):
                raise RuleSetupValueError(owner=self, msg=f"Value expression's attribute '{self._node}' needs to be string or OperationVexpNode, got: {type(self._node)}")

        # -- COMPUTED VALUES
        # copy list of parent/previous
        self.Path = [] if self._is_top else Path[:]
        self.Path.append(self)

        # FunctionArgumentsType
        self._func_args: Optional[FunctionArgumentsTupleType] = None
        self._is_reserved_function:bool = self._name in self.RESERVED_FUNCTION_NAMES

        # -- can calculate all, contains all attr_node-s, and the last one is in 
        # not defined in is_predefined mode
        self._evaluator = UNDEFINED
        # the last one attr_node
        self._vexp_node = UNDEFINED


    def GetNamespace(self) -> Namespace:
        return self._namespace

    def IsFinished(self):
        return self._status!=VExpStatusEnum.INITIALIZED

    def _EnsureFinished(self):
        if self._status!=VExpStatusEnum.INITIALIZED:
            raise RuleInternalError(msg=f"{self}: Setup() already called, further ValueExpression building is not possible (status={self._status}).")


    def Setup(self, 
            registries:"IRegistries",  # noqa: F821
            owner:"Component",  # noqa: F821
            # local registry, not default by namespace
            registry: Optional["IRegistry"]=None,  # noqa: F821
            strict:bool = False,
            ) -> Optional['IValueExpressionNode']:
        """
        Owner used just for reference count.
        """
        # TODO: circular dependency - maybe to pass eval class to this method
        from .expression_evaluators import ValueExpressionEvaluator

        self._EnsureFinished()

        if registry:
            if not self._namespace._manual_setup:
                raise RuleInternalError(f"{self}: registry should be passed only for namespace._manual_setup cases, got: {registry}")
        else:
            if self._namespace._manual_setup:
                raise RuleInternalError(f"{self}: registry should be passed for namespace._manual_setup cases (usually manually created ThisRegistry()).")
            registry = registries.get_registry(self._namespace)

        if self._namespace != registry.NAMESPACE:
            raise RuleInternalError(f"{self}: registry has diff namespace from variable: {self._namespace} != {registry.NAMESPACE}")

        current_vexp_node = None
        last_vexp_node = None
        vexp_node_name = None
        vexp_evaluator = ValueExpressionEvaluator()

        # if str(self) == "G.(G.varname1 + G.varname2)": 
        for bnr, bit in enumerate(self.Path, 1):
            # if SETUP_CALLS_CHECKS.can_use(): SETUP_CALLS_CHECKS.setup_called(bit)

            assert bit._namespace==self._namespace
            # TODO: if self._func_args:
            # operator.attrgetter("last_name")
            try:
                # last_vexp_node = (current_vexp_node
                #                if current_vexp_node is not None
                #                else parent)

                # ========================================
                # Operations 
                # ========================================
                if isinstance(bit._node, OperationVexpNode):
                    op_node = bit._node
                    # one level deeper
                    # parent not required, but in this case should be 
                    assert bnr == 1
                    current_vexp_node = op_node.Setup(registries=registries, owner=owner)
                    vexp_node_name = bit._node
                    # _read_functions.append(op_node.apply)

                # ========================================
                # Functions - IFunctionVexpNode
                # ========================================
                elif bit._func_args:
                    if bnr==1:
                        if self._namespace != FunctionsNS:
                            raise RuleSetupNameError(f"{self}: Only FunctionsNS (G.) namespace accepts direct function calls. Got '{bit}' on '{bit._namespace}) namespace.")

                    func_args = FunctionArgumentsType(*bit._func_args)

                    vexp_node_name = bit._node
                    if last_vexp_node:
                        parent_arg_type_info = get_type_info_from_vexp_or_node(last_vexp_node)
                    else:
                        parent_arg_type_info = None

                    # : IFunctionVexpNode
                    current_vexp_node = registry.create_func_node(
                            registries=registries,
                            caller=last_vexp_node,
                            attr_node_name=vexp_node_name,
                            func_args=func_args,
                            value_arg_type_info=parent_arg_type_info
                            )
                else:
                    # ========================================
                    # Attributes
                    # ========================================
                    # ----------------------------------------
                    # Check if Path goes to correct attr_node
                    # ----------------------------------------
                    # when copy_to_registries defined:
                    #   read from copy_to_registries.registries_bind_from and store in both registriess in the
                    #   same namespace (usually ModelsNS)
                    # registries_read_from = copy_to_registries.registries_bind_from if copy_to_registries else registries
                    # registries_read_from = registries

                    vexp_node_name = bit._node
                    current_vexp_node = registry.create_node(
                                vexp_node_name=vexp_node_name,
                                parent_vexp_node=last_vexp_node,
                                # func_args=bit._func_args,
                                )

                # add node to evaluator
                vexp_evaluator.add(current_vexp_node)

                # if is_last and copy_to_registries:
                #     current_vexp_node.add_bound_vexp_node(BoundVar(registries.name, copy_to_registries.vexp_node_name))
                #     registries.add(current_vexp_node, alt_vexp_node_name=copy_to_registries.vexp_node_name)

            except NotImplementedError as ex:
                # print(f"XX {self} -> {ex}")
                if strict:
                    raise
                self._status = VExpStatusEnum.ERR_TO_IMPLEMENT
                vexp_evaluator.failed(str(ex))
                break

            last_vexp_node = current_vexp_node

            # except (RuleError) as ex:
            # NOTE: RL 221104 strict mode - do not tolerate attr_node not found any more
            # except (RuleSetupNameNotFoundError) as ex:
            #     if False:
            #         self._status = VExpStatusEnum.ERR_NOT_FOUND
            #         # current_vexp_node = registries.get(namespace=bit._namespace, vexp_node_name=vexp_node_name, owner=owner, parent=current_vexp_node)
            #         print(f"== TODO: RuleSetupError - {self} -> Registries error {bit}: {ex}")
            #         break
            #     else:
            #         raise RuleSetupError(owner=self, msg=f"Registries {registries!r} attribute {vexp_node_name} not found: {ex}")

            # if not isinstance(current_vexp_node, AttrVexpNode):
            #     raise RuleInternalError(owner=self, msg=f"Type of found object is not AttrVexpNode, got: {type(current_vexp_node)}.")

            # can be Component/IData or can be managed Model dataclass Field - when .denied is not appliable
            if hasattr(current_vexp_node, "denied") and current_vexp_node.denied:
                raise RuleSetupValueError(owner=self, msg=f"VexpNode '{vexp_node_name}' (owner={owner.name}) references '{current_vexp_node.name}' is not allowed in ValueExpression due: {current_vexp_node.deny_reason}.")

        vexp_evaluator.finish()


        if vexp_evaluator.is_all_ok():
            self._status = VExpStatusEnum.BUILT
            self._all_ok = True
            self._evaluator = vexp_evaluator
            self._vexp_node = vexp_evaluator.last_node()

        else:
            # TODO: raise RuleSetupError()
            self._all_ok = False
            self._evaluator = vexp_evaluator
            self._vexp_node = None

        return self._vexp_node

    # def __getitem__(self, ind):
    #     # list [0] or dict ["test"]
    #     return ValueExpression( Path=self.Path + "." + str(ind))

    def __getattr__(self, aname):
        self._EnsureFinished()

        if aname in self.RESERVED_ATTR_NAMES: # , "%r -> %s" % (self._node, aname):
            raise RuleSetupNameError(owner=self, msg=f"ValueExpression's attribute '{aname}' is reserved name, choose another.")
        if aname.startswith("__") and aname.endswith("__"):
            raise AttributeError(f"Attribute '{type(self)}' object has no attribute '{aname}'")
        return ValueExpression(node=aname, namespace=self._namespace, Path=self.Path)

    def __call__(self, *args, **kwargs):
        assert self._func_args is None
        self._func_args = [args, kwargs]
        return self

    def as_str(self):
        out = ""
        if self._is_top:
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
        return f"VExpr({self})"

    # --------------------------------
    # ------- Reserved methods -------
    # --------------------------------

    # NOTE: each method should be listed in RESERVED_ATTR_NAMES

    # --------------------------------
    # ------- Terminate methods ------
    #           return plain python objects

    # ----------------------------------
    # ------- Internal methods ---------

    # https://realpython.com/python-bitwise-operators/#custom-data-types
    # comparison operators <, <=, ==, !=, >=, >

    # NOTE: Operations are in internal OperationsNS

    def __eq__(self, other):        self._EnsureFinished(); return ValueExpression(OperationVexpNode("==", self, other), namespace=OperationsNS)  # noqa: E702
    def __ne__(self, other):        self._EnsureFinished(); return ValueExpression(OperationVexpNode("!=", self, other), namespace=OperationsNS)  # noqa: E702
    def __gt__(self, other):        self._EnsureFinished(); return ValueExpression(OperationVexpNode(">" , self, other), namespace=OperationsNS)  # noqa: E702
    def __ge__(self, other):        self._EnsureFinished(); return ValueExpression(OperationVexpNode(">=", self, other), namespace=OperationsNS)  # noqa: E702
    def __lt__(self, other):        self._EnsureFinished(); return ValueExpression(OperationVexpNode("<" , self, other), namespace=OperationsNS)  # noqa: E702
    def __le__(self, other):        self._EnsureFinished(); return ValueExpression(OperationVexpNode("<=", self, other), namespace=OperationsNS)  # noqa: E702

    # +, -, *, /
    def __add__(self, other):       self._EnsureFinished(); return ValueExpression(OperationVexpNode("+" , self, other), namespace=OperationsNS)  # noqa: E702
    def __sub__(self, other):       self._EnsureFinished(); return ValueExpression(OperationVexpNode("-" , self, other), namespace=OperationsNS)  # noqa: E702
    def __mul__(self, other):       self._EnsureFinished(); return ValueExpression(OperationVexpNode("*" , self, other), namespace=OperationsNS)  # noqa: E702
    def __truediv__(self, other):   self._EnsureFinished(); return ValueExpression(OperationVexpNode("/" , self, other), namespace=OperationsNS)  # noqa: E702

    # what is this??
    def __floordiv__(self, other):  self._EnsureFinished(); return ValueExpression(OperationVexpNode("//", self, other), namespace=OperationsNS)  # noqa: E702

    # in
    def __contains__(self, other):  self._EnsureFinished(); return ValueExpression(OperationVexpNode("in", self, other), namespace=OperationsNS)  # noqa: E702

    # Bool operators, NOT, AND, OR
    def __invert__(self):           self._EnsureFinished(); return ValueExpression(OperationVexpNode("not", self), namespace=OperationsNS)  # ~  # noqa: E702
    def __and__(self, other):       self._EnsureFinished(); return ValueExpression(OperationVexpNode("and", self, other), namespace=OperationsNS)  # &  # noqa: E702
    def __or__(self, other):        self._EnsureFinished(); return ValueExpression(OperationVexpNode("or" , self, other), namespace=OperationsNS)  # |  # noqa: E702

    # ------------------------------------------------------------

    # def __rshift__(self, other):
    #     """
    #     Streaming:
    #         List[Instance] >> Function() # receives Qs/List, like Map(Function, List) -> List
    #         Instance >> Function()       # receives Instance, returns
    #     """
    #     return ValueExpression(StreamOperation(">>", self, other), namespace=OperationsNS)  # >>



    # __abs__ - abs()
    # __xor__ ==> ^
    # <<, >>
    # ** 	__pow__(self, object) 	Exponentiation
    # Matrix Multiplication 	a @ b 	matmul(a, b)
    # Positive 	+ a 	pos(a)
    # Slice Assignment 	seq[i:j] = values 	setitem(seq, slice(i, j), values)
    # Slice Deletion 	del seq[i:j] 	delitem(seq, slice(i, j))
    # Slicing 	seq[i:j] 	getitem(seq, slice(i, j))
    # String Formatting 	s % obj 	mod(s, obj)
    #       % 	__mod__(self, object) 	Modulus
    # Truth Test 	obj 	truth(obj)

    # https://docs.python.org/3/library/operator.html
    #   operator.__lshift__(a, b) - a << b - Return a shifted left by b.
    #   operator.__rshift__(a, b) - a >> b - Return a shifted right by b.
    #   operator.__mod__(a, b) - Return a % b.
    #   operator.__matmul__(a, b) -  Return a @ b.
    #   operator.__pow__(a, b) - Return a ** b, for a and b numbers.
    #   operator.__xor__(a, b) -  a ^ b Return the bitwise exclusive or of a and b.
    #   setitem(seq, slice(i, j), values) - seq[i:j] = values
    #       When[ a > b : 25 ], When[ a > b : 25 ]
    #   operator.__ixor__(a, b) - a ^= b.
    #   operator.__irshift__(a, b) - a >>= b.
    #         When(a > b) >>= 25


