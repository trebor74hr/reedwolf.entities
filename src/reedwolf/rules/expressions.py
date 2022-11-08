# Copied and adapted from Reedwolf project (project by robert.lujo@gmail.com - git@bitbucket.org:trebor74hr/reedwolf.git)
from __future__ import annotations

import operator

from enum import Enum
from functools import partial
from dataclasses import dataclass

from typing import (
        List, 
        Optional, 
        Union, 
        Any, 
        )
from .exceptions import (
        RuleSetupValueError, 
        RuleSetupError, 
        RuleSetupNameError, 
        RuleError, 
        RuleInternalError,
        RuleSetupNameNotFoundError,
        )
from .utils import (
        composite_functions, 
        UNDEFINED,
        )
from .namespaces import RubberObjectBase, GlobalNS, Namespace, ThisNS, UtilsNS

# ------------------------------------------------------------

class VExpStatusEnum(str, Enum):
    INITIALIZED      = "INIT"
    OK               = "OK"
    ERR_NOT_FOUND    = "ERR_NOT_FOUND"
    ERR_TO_IMPLEMENT = "ERR_TO_IMPLEMENT"

# ------------------------------------------------------------

class Operation:

    def __init__(self, op: str, first: Any, second: Optional[Any] = None):
        self.op, self.first, self.second = op, first, second
        self.op_function = self.OPCODE_TO_FUNCTION.get(self.op, None)
        if self.op_function is None:
            raise RuleSetupValueError(owner=self, msg="Invalid operation code, {self.op} not one of: {', '.join(self.OP_TO_CODE.keys())}")
        self._status : VExpStatusEnum = VExpStatusEnum.INITIALIZED
        self._all_ok : Optional[bool] = None

    # no operator, needs custom logic
    def apply_and(self, first, second): return bool(first) and bool(second)
    def apply_or (self, first, second): return bool(first) or  bool(second)

    # https://florian-dahlitz.de/articles/introduction-to-pythons-operator-module
    # https://docs.python.org/3/library/operator.html#mapping-operators-to-functions
    OPCODE_TO_FUNCTION = {
          "=="  : operator.eq
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

        , "not" : operator.not_   # orig: ~
        # no operator, needs custom logic
        , "and" : apply_and       # orig: &
        , "or"  : apply_or        # orig: |
    }

    def Setup(self, heap: "VariableHeap", owner: Any):
        # parent:"Variable"
        assert self._status==VExpStatusEnum.INITIALIZED, self

        if isinstance(self.first, (ValueExpression, Operation)):
            self.first.Setup(heap, owner=owner, parent=None)
        if self.second!=None and isinstance(self.second, (ValueExpression, Operation)):
            self.second.Setup(heap, owner=owner, parent=None)
        self._status=VExpStatusEnum.OK

    def apply(self, heap):
        first  = self.first.Read(ctx)
        if self.second!=None:
            # binary operator
            second = self.second.Read(ctx)
            try:
                res = self.op_function(first, second)
            except Exception as ex:
                raise RuleSetupError(owner=heap, item=self, msg=f"Apply {self.first} {self.op} {self.second} => {first} {self.op} {second} raised error: {ex}")
        else:
            # unary operator
            try:
                res = self.op_function(first, second)
            except Exception as ex:
                raise RuleSetupError(owner=heap, item=self, msg=f"Apply {self.op} {self.first} => {self.op} {first} raised error: {ex}")
        return res


    def __str__(self):
        if self.second:
            return f"({self.first} {self.op} {self.second})"
        else:
            return f"({self.op} {self.first})"

    def __repr__(self):
        return f"Op{self}"




class ValueExpression(RubberObjectBase):

    # NOTE: each item in this list should be implemented as attribute or method in this class
    # "GetVariable", 
    RESERVED_ATTR_NAMES = {"Path", "Read", "Setup", "GetNamespace",  
                           "_var_name", "_node", "_namespace", "_name", "_func_args", "_is_top", "_read_functions", "_status"}
    RESERVED_FUNCTION_NAMES = ("Value",)
    # "First", "Second", 

    def __init__(
        self,
        node: Union[str, Operation],
        namespace: Namespace,
        Path: Optional[List[ValueExpression]] = None,
    ):
        self._status : VExpStatusEnum = VExpStatusEnum.INITIALIZED

        self._namespace = namespace
        if not isinstance(self._namespace, Namespace):
            raise RuleSetupValueError(owner=self, msg=f"Namespace parameter '{self._namespace}' needs to be instance of Namespace inherited class.")
        self._node = node
        if isinstance(self._node, str):
            if self._node in self.RESERVED_ATTR_NAMES:
                raise RuleSetupValueError(owner=self, msg=f"Value expression's attribute '{self._node}' is a reserved name, choose another.")
        else:
            if not isinstance(self._node, Operation):
                raise RuleSetupValueError(owner=self, msg=f"Value expression's attribute '{self._node}' needs to be string or Operation, got: {type(self._node)}")
            self._node = node

        self._is_top = Path is None
        self._name = str(self._node)

        self.Path = [] if self._is_top else Path[:]
        self.Path.append(self)

        self._func_args = None

        self._read_functions = UNDEFINED
        self._var_name = UNDEFINED

        self._reserved_function = self._name in self.RESERVED_FUNCTION_NAMES


    def GetNamespace(self) -> Namespace:
        return self._namespace

    # NOTE: replaced with VariableHeap.get_var_by_vexp(vexp ...)
    # def GetVariable(self, heap:'VariableHeap', strict=True) -> 'Variable':
    #     if self._var_name==UNDEFINED:
    #         if strict:
    #             raise RuleSetupNameError(owner=self, msg=f"Variable not processed/Setup yet")
    #         return UNDEFINED
    #     return heap.get_var(self._namespace, self._var_name)

    def Setup(self, heap:"VariableHeap", owner:"Component", parent:"Variable") -> Optional['Variable']:
        """
        owner used just for reference count.
        """
        # , copy_to_heap:Optional[CopyToHeap]=None
        # TODO: create single function - which is composed of functions 
        #           see: https://florian-dahlitz.de/articles/introduction-to-pythons-operator-module
        #             callable = operator.attrgetter("last_name")           -> .last_name
        #             callable = operator.itemgetter(1)                     -> [1]
        #             callable = operator.methodcaller("run", "foo", bar=1) -> .run("foot", bar=1)
        #           and this:
        #             functools.partial(function, x=1, y=2)
        #       https://www.geeksforgeeks.org/function-composition-in-python/

        from .variables import Variable

        if self._status!=VExpStatusEnum.INITIALIZED:
            raise RuleInternalError(owner=self, msg=f"Setup() already called (status={self._status}).")

        if self._read_functions!=UNDEFINED:
            raise RuleSetupError(owner=self, msg=f"Setup() already called (found _read_functions).")

        _read_functions = []

        current_variable = None
        last_parent = parent
        var_name = None

        all_ok = True
        bit_length = len(self.Path)
        # if "status" in str(self.Path): import pdb;pdb.set_trace() 
        for bnr, bit in enumerate(self.Path, 1):
            is_last = (bnr==bit_length)
            assert bit._namespace==self._namespace
            # TODO: if self._func_args:
            # operator.attrgetter("last_name")
            if isinstance(bit._node, Operation):
                operation = bit._node
                # one level deeper
                operation.Setup(heap=heap, owner=owner) 
                _read_functions.append(operation.apply)
            else:
                # ----------------------------------------
                # Check if Path goes to correct variable 
                # ----------------------------------------
                var_name = bit._node
                try:
                    last_parent = (current_variable 
                                   if current_variable is not None 
                                   else parent)
                    # when copy_to_heap defined:
                    #   read from copy_to_heap.heap_bind_from and store in both heaps in the
                    #   same namespace (usually ModelsNS)
                    # heap_read_from = copy_to_heap.heap_bind_from if copy_to_heap else heap
                    heap_read_from = heap

                    current_variable = heap_read_from.getset_attribute_var(
                                            namespace=self._namespace,
                                            var_name=var_name, 
                                            # owner=owner, 
                                            parent_var=last_parent)

                    # if is_last and copy_to_heap:
                    #     current_variable.add_bound_var(BoundVar(heap.name, copy_to_heap.var_name))
                    #     heap.add(current_variable, alt_var_name=copy_to_heap.var_name)

                except NotImplementedError as ex:
                    self._status = VExpStatusEnum.ERR_TO_IMPLEMENT
                    all_ok = False
                    break
                # except (RuleError) as ex:
                # NOTE: RL 221104 strict mode - do not tolerate variable not found any more
                # except (RuleSetupNameNotFoundError) as ex:
                #     if False:
                #         self._status = VExpStatusEnum.ERR_NOT_FOUND
                #         all_ok = False
                #         # current_variable = heap.get(namespace=bit._namespace, var_name=var_name, owner=owner, parent=current_variable)
                #         print(f"== TODO: RuleSetupError - {self} -> Heap error {bit}: {ex}")
                #         break
                #     else:
                #         raise RuleSetupError(owner=self, msg=f"Heap {heap!r} attribute {var_name} not found: {ex}")

                if not isinstance(current_variable, Variable):
                    raise RuleInternalError(owner=self, msg=f"Type of found object is not Variable, got: {type(current_variable)}.")

                # can be Component/DataVar or can be managed Model dataclass Field - when .denied is not appliable
                if hasattr(current_variable, "denied") and current_variable.denied:
                    raise RuleSetupValueError(owner=self, msg=f"Variable '{var_name}' (owner={owner.name}) references '{current_variable.name}' is not allowed in ValueExpression due: {current_variable.deny_reason}.")

                # print(f"OK: {self} -> {bit}")
                if bit._func_args is not None:
                    args, kwargs = bit._func_args
                    # getter = operator.attrgetter(var_name)
                    # def func_call(obj):
                    #     return getter(obj)(*args, **kwargs)
                    # -> .<var_name>(*args, **kwargs)
                    func_call  = operator.methodcaller(var_name, *args, **kwargs)
                    _read_functions.append(func_call)
                    # raise NotImplementedError(f"Call to functions {bit} in {self} not implemented yet!")
                else:
                    getter = operator.attrgetter(var_name)
                    _read_functions.append(getter)

        variable = None

        # if "select_id_of_default_device" in repr(self):
        #     import pdb;pdb.set_trace() 

        if all_ok:
            self._status = VExpStatusEnum.OK
            self._all_ok = True
            self._read_functions = _read_functions
            variable = current_variable
            if not variable:
                if self._namespace not in (GlobalNS, ThisNS, UtilsNS):
                    raise RuleSetupValueError(owner=self, msg=f"Variable not found.")
                # self._all_ok = False?
                self._var_name = None
            else:
                # self._all_ok = False?
                variable.add_reference(owner.name)
                self._var_name = variable.name

        else:
            self._all_ok = False
            self._var_name = None
            self._read_functions = None

        return variable


    def Read(self, heap:'VariablesHeap', model_name:Optional[str]):
        if not "_read_functions" in dir(self):
            raise RuleSetupInternalError(owner=self, msg=f"Setup not done.")
        val = UNDEFINED
        # TODO: if self._var_name
        for func in rself._read_functions:
            if val is UNDEFINED:
                val = func(heap)
            else:
                val = func(val)
        return val

    # def __getitem__(self, ind):
    #     # list [0] or dict ["test"]
    #     return ValueExpression(
    #         Path=self.Path
    #         + "."
    #         + str(ind)
    #     )

    def __getattr__(self, aname):
        # if aname.startswith("_"):
        #     raise RuleSetupNameError(owner=self, msg=f"VariableExpression name {aname} starts with _ what is reserved, choose another name.")
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
        return ".".join([ve.as_str() for ve in self.Path])

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
    def __eq__(self, other):        return ValueExpression(Operation("==", self, other), namespace=GlobalNS)
    def __ne__(self, other):        return ValueExpression(Operation("!=", self, other), namespace=GlobalNS)
    def __gt__(self, other):        return ValueExpression(Operation(">", self, other), namespace=GlobalNS)
    def __ge__(self, other):        return ValueExpression(Operation(">=", self, other), namespace=GlobalNS)
    def __lt__(self, other):        return ValueExpression(Operation("<", self, other), namespace=GlobalNS)
    def __le__(self, other):        return ValueExpression(Operation("<=", self, other), namespace=GlobalNS)
    def __add__(self, other):       return ValueExpression(Operation("+", self, other), namespace=GlobalNS)
    def __sub__(self, other):       return ValueExpression(Operation("-", self, other), namespace=GlobalNS)
    def __mul__(self, other):       return ValueExpression(Operation("*", self, other), namespace=GlobalNS)
    def __truediv__(self, other):   return ValueExpression(Operation("/", self, other), namespace=GlobalNS)
    def __floordiv__(self, other):  return ValueExpression(Operation("//", self, other), namespace=GlobalNS)
    def __contains__(self, other):  return ValueExpression(Operation("in", self, other), namespace=GlobalNS)
    def __invert__(self):           return ValueExpression(Operation("not", self), namespace=GlobalNS)  # ~
    def __and__(self, other):       return ValueExpression(Operation("and", self, other), namespace=GlobalNS)  # &
    def __or__(self, other):        return ValueExpression(Operation("or", self, other), namespace=GlobalNS)  # |


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
