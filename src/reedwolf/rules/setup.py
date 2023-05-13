from __future__ import annotations

import inspect
from abc import abstractmethod
from dataclasses import dataclass, field
from contextlib import AbstractContextManager

from typing import (
        Any,
        List,
        Union,
        Optional,
        ClassVar,
        Tuple,
        Dict,
        Type,
        )
from .utils import (
        get_available_names_example,
        UNDEFINED,
        UndefinedType,
        check_identificator_name,
        )
from .exceptions import (
        RuleError,
        RuleSetupError,
        RuleSetupNameError,
        RuleSetupNameNotFoundError,
        RuleSetupValueError,
        RuleInternalError,
        RuleApplyTypeError,
        RuleApplyNameError,
        )
from .namespaces import (
        Namespace,
        ModelsNS,
        FieldsNS,
        ThisNS,
        FunctionsNS,
        ContextNS,
        ConfigNS,
        OperationsNS,
        )
from .expressions import (
        ValueExpression,
        IValueExpressionNode,
        IFunctionVexpNode,
        IRegistry,
        ISetupSession,
        IAttributeAccessorBase,
        )
from .meta import (
        FunctionArgumentsType,
        FunctionArgumentsTupleType,
        is_model_class,
        ModelType,
        get_model_fields,
        TypeInfo,
        HookOnFinishedAllCallable,
        )
from .base import (
        ComponentBase,
        IFieldBase, 
        IContainerBase,
        extract_type_info,
        IApplySession,
        BoundModelBase,
        ReservedAttributeNames,
        SetupStackFrame,
        )
from .functions import (
        FunctionsFactoryRegistry,
        CustomFunctionFactory,
        try_create_function,
        )
from .attr_nodes import (
        AttrVexpNode,
        )
from .components import (
        FieldGroup,
        ValidationBase,
        EvaluationBase,
        )
from .contexts import (
        IContext,
        )
from .config import (
        Config,
        )

# ------------------------------------------------------------


def get_vexp_node_name(parent_name: Optional[str], 
                       vexp_node_name: str, 
                       func_args: Optional[Union[FunctionArgumentsTupleType, FunctionArgumentsType]]
                       ) -> str:
    # vexp_node_name = 
    check_identificator_name(vexp_node_name)

    if func_args:
        if isinstance(func_args, (list, tuple)) and len(func_args)==2:
            func_args = FunctionArgumentsType(*func_args)
        assert isinstance(func_args, FunctionArgumentsType), func_args
        args, kwargs = func_args.get_args_kwargs()
        all_args_str = []
        if args:
            all_args_str.append(", ".join(
                [f"'{a}':{type(a)}" for a in args]))
        if kwargs:
            all_args_str.append(", ".join(
                [f"{k}='{v}':{type(v)}" 
                    for k,v in kwargs.items()]))
        all_args_str = ", ".join(all_args_str)
        vexp_node_name = f"{vexp_node_name}({all_args_str})"
    else:
        vexp_node_name = f"{vexp_node_name}"
    if parent_name is not None:
        # not validated at all
        vexp_node_name = f"{parent_name}.{vexp_node_name}"

    return vexp_node_name


# ------------------------------------------------------------


class RegistryBase(IRegistry):
    """
    Namespaces are DynamicAttrsBase with flexible attributes, so it is not
    convenient to have some advanced logic within. Thus Registry logic 
    is put in specialized classes - SetupSession.
    """
    # TODO: with 3.11 - Protocol
    NAMESPACE : ClassVar[Namespace] = None
    ROOT_VALUE_NEEDS_FETCH_BY_NAME: ClassVar[bool] = True


    def __init__(self):
        self.store : Dict[str, AttrVexpNode] = {}
        self.setup_session = UNDEFINED
        self.finished: bool = False

    def set_setup_session(self, setup_session: ISetupSession):
        if self.setup_session:
            raise RuleInternalError(owner=self, msg=f"setup_session already set: {self.setup_session}") 
        self.setup_session = setup_session

    def finish(self):
        if self.finished:
            raise RuleInternalError(owner=self, msg="already finished") 
        if not self.setup_session:
            raise RuleInternalError(owner=self, msg="setup_session not set, function set_setup_session() not called") 
        self.finished = True

    def count(self) -> int:
        return len(self.store)

    def get(self, key:str, default:Any=None) -> Optional[AttrVexpNode]:
        return self.store.get(key, default)

    def items(self) -> List[Tuple[str, AttrVexpNode]]:
        return self.store.items()

    @classmethod
    def _create_instance_attr_node(cls, model_class: ModelType) -> AttrVexpNode:
        " used for This.Instance to return instance itself "
        th_field = None
        type_info = TypeInfo.get_or_create_by_type(
                        py_type_hint=model_class,
                        )
        attr_node = AttrVexpNode(
                        name=ReservedAttributeNames.INSTANCE_ATTR_NAME,
                        data=type_info,
                        namespace=cls.NAMESPACE,
                        type_info=type_info, 
                        th_field=th_field,
                        )
        return attr_node

    @classmethod
    def _create_attr_node_for_model_attr(cls, model_class: ModelType, attr_name:str) -> AttrVexpNode:
        # NOTE: will go again and again into get_model_fields()
        #       but shortcut like this didn't worked: 
        #           type_info: TypeInfo = TypeInfo.get_or_create_by_type(th_field)

        if attr_name in (ReservedAttributeNames.INSTANCE_ATTR_NAME,):
            raise RuleSetupNameError(msg=f"Sorry but model attribute name {attr_name} is reserved. Rename it and try again (model={model_class.__name__}).")

        # This one should not fail
        # , func_node
        type_info, th_field = \
                extract_type_info(
                    attr_node_name=attr_name,
                    inspect_object=model_class)

        attr_node = AttrVexpNode(
                        name=attr_name,
                        data=type_info,
                        namespace=cls.NAMESPACE,
                        type_info=type_info, 
                        th_field=th_field,
                        )
        return attr_node


    def register_vexp_node(self, vexp_node:IValueExpressionNode, alt_vexp_node_name=None):
        """
        Data can register IFunctionVexpNode-s instances since the
        output will be used directly as data and not as a function call.
            function=[Function(name="Countries", label="Countries", 
                              py_function=CatalogManager.get_countries)],
            ...
            available=(S.Countries.name != "test"),
        """
        if self.finished:
            raise RuleInternalError(owner=self, msg=f"Register({vexp_node}) - already finished, adding not possible.")

        if not isinstance(vexp_node, IValueExpressionNode):
            raise RuleInternalError(f"{type(vexp_node)}->{vexp_node}")
        vexp_node_name = alt_vexp_node_name if alt_vexp_node_name else vexp_node.name

        if not vexp_node_name.count(".") == 0:
            raise RuleInternalError(owner=self, msg=f"Node {vexp_node_name} should not contain . - only first level vars allowed")

        if vexp_node_name in self.store:
            raise RuleSetupNameError(owner=self, msg=f"AttrVexpNode {vexp_node} does not have unique name within this registry, found: {self.store[vexp_node_name]}")
        self.store[vexp_node_name] = vexp_node


    def register_attr_node(self, attr_node:AttrVexpNode, alt_attr_node_name=None):
        if not isinstance(attr_node, AttrVexpNode):
            raise RuleInternalError(f"{type(attr_node)}->{attr_node}")
        if not self.NAMESPACE == attr_node.namespace:
            raise RuleInternalError(owner=self, msg=f"Method register({attr_node}) - namespace mismatch: {self.NAMESPACE} != {attr_node.namespace}")
        return self.register_vexp_node(vexp_node=attr_node, alt_vexp_node_name=alt_attr_node_name)

    def pp(self):
        print(f"  Namespace {self.NAMESPACE._name}:")
        for vexp_node_name, vexp_node in self.store.items():
            print(f"    {vexp_node_name} = {vexp_node.pp()}")

    def __str__(self):
        return f"{self.__class__.__name__}(cnt={len(self.store)})"

    __repr__ = __str__

    # ------------------------------------------------------------
    # create_func_node - only Functions i.e. IFunctionVexpNode
    # ------------------------------------------------------------
    def create_func_node(self, 
            setup_session: ISetupSession,
            caller: IValueExpressionNode,
            attr_node_name:str,
            func_args:FunctionArgumentsType,
            value_arg_type_info:TypeInfo,
            ) -> IFunctionVexpNode:

        func_node : IFunctionVexpNode = \
                try_create_function(
                    setup_session=setup_session,
                    caller=caller,
                    # functions_factory_registry=self.functions_factory_registry,
                    attr_node_name=attr_node_name,
                    func_args=func_args,
                    value_arg_type_info=value_arg_type_info,
                    )
        return func_node

    # ------------------------------------------------------------
    # create_node -> AttrVexpNode, Operation or IFunctionVexpNode
    # ------------------------------------------------------------
    def create_node(self, 
                    vexp_node_name: str, 
                    parent_vexp_node: IValueExpressionNode, 
                    # func_args: FunctionArgumentsType
                    owner: ComponentBase,
                    ) -> IValueExpressionNode:
        """
        Will create a new attr_node when missing, even in the case when the var
        is just "on the path" to final attr_node needed.

        Namespace usually is: ModelsNS, FunctionsNS

        """
        # can return ValueExpression / class member / object member
        # TODO: circ dep?
        assert isinstance(vexp_node_name, str), vexp_node_name

        if vexp_node_name.startswith("_"):
            raise RuleSetupNameError(owner=owner, msg=f"Namespace '{self.NAMESPACE}': AttrVexpNode '{vexp_node_name}' is invalid, should not start with _")

        type_info = None

        if parent_vexp_node:
            assert parent_vexp_node.name!=vexp_node_name
            if not isinstance(parent_vexp_node, IValueExpressionNode):
                raise RuleSetupNameError(owner=owner, msg=f"Namespace '{self.NAMESPACE}': '{vexp_node_name}' -> parent_vexp_node={parent_vexp_node} :{type(parent_vexp_node)} is not IValueExpressionNode")

        # TODO: za Sum(This.name) this is not good. Should be unique, since 
        #       This is ambigous - can evaluate to different contexts. 
        #       should have some context @ID (parent)
        full_vexp_node_name = get_vexp_node_name(parent_name=parent_vexp_node.name if parent_vexp_node else None, 
                                          vexp_node_name=vexp_node_name,
                                          func_args=None # func_args
                                          )

        if parent_vexp_node is None:
            # --------------------------------------------------
            # get() -> TOP LEVEL - only level that is stored
            # --------------------------------------------------
            # e.g. M.company Predefined before in Rules.setup() function.
            if full_vexp_node_name not in self.store:
                names_avail = get_available_names_example(full_vexp_node_name, self.store.keys())
                valid_names = f"Valid attributes: {names_avail}" if self.store.keys() else "Namespace has no attributes at all."
                raise RuleSetupNameNotFoundError(owner=owner, msg=f"Namespace '{self.NAMESPACE}': Invalid attribute name '{full_vexp_node_name}'. {valid_names}")

            attr_node_template = self.store.get(full_vexp_node_name)

            # clone it - each attr_node must be unique - cached are used only as templates
            vexp_node = attr_node_template.clone()
            assert id(vexp_node) != attr_node_template

            # TODO: needed?
            #       if isinstance(parent_vexp_node.data, IFunctionVexpNode):
            #           inspect_object = parent_vexp_node.data.get_type_info()
        else:
            # --------------------------------------------------------------
            # IN-DEPTH LEVEL 2+, e.g. M.company.<this-node-and-next>
            # ---------------------------------------------------------------

            if isinstance(parent_vexp_node, IFunctionVexpNode):
                inspect_object = parent_vexp_node.get_type_info()
            else:
                assert isinstance(parent_vexp_node, AttrVexpNode)

                if False:
                    ...
                # elif isinstance(parent_vexp_node.data, StaticData):
                #     inspect_object = parent_vexp_node.data.value
                # elif isinstance(parent_vexp_node.data, DynamicData):
                #     raise NotImplementedError("TODO")
                elif isinstance(parent_vexp_node.data, BoundModelBase):
                    inspect_object = parent_vexp_node.data.model
                else:
                    # can be TypeInfo, @dataclass, Pydantic etc.
                    inspect_object = parent_vexp_node.data

            try:
                # , func_node
                type_info, th_field = \
                        extract_type_info(
                            attr_node_name=vexp_node_name,
                            inspect_object=inspect_object, 
                            # func_args=func_args,
                            # functions_factory_registry=self.functions_factory_registry
                            )
            except RuleSetupNameNotFoundError as ex:
                ex.set_msg(f"{owner} / {self.NAMESPACE}-NS: {ex.msg}")
                raise 
            except RuleError as ex:
                ex.set_msg(f"'{owner} / {self.NAMESPACE}-NS: {parent_vexp_node.full_name} -> '.{vexp_node_name}' metadata / type-hints read problem: {ex}")
                raise 

            # if func_node:
            #     assert func_args
            #     assert isinstance(func_node, IFunctionVexpNode)
            #     vexp_node = func_node
            # else:

            assert type_info

            # --------------------------------------------------
            # Create()
            # --------------------------------------------------
            # attributes / functions / operations
            vexp_node = AttrVexpNode(
                            name=full_vexp_node_name,
                            data=type_info,
                            namespace=self.NAMESPACE,
                            type_info=type_info, 
                            th_field=th_field, 
                            ) # function=function

        if not isinstance(vexp_node, IValueExpressionNode):
            raise RuleInternalError(owner=owner, msg=f"Namespace {self.NAMESPACE}: Type of found object is not IValueExpressionNode, got: {type(vexp_node)}.")

        # NOTE: vexp_node.type_info can be None, will be filled later in finish()

        return vexp_node

    @abstractmethod
    def get_root_value(self, apply_session: IApplySession, attr_name: str) -> Any:
        """ 
        Apply phase - Namespace.<attr_name> - 
        function returns the instance (root value) from which attr_name will be read from 

        attr_name is not used for fetching the value, but for rare cases when 
        different source / instance is used based on different attr_name. See
        ThisNS.Instance.
        """
        ...


# ------------------------------------------------------------


class RegistryUseDenied(RegistryBase):

    def get_root_value(self, apply_session: IApplySession, attr_name: str) -> Any:
        raise RuleInternalError(owner=self, msg="Registry should not be used to get root value.")


# ------------------------------------------------------------


@dataclass
class ComponentAttributeAccessor(IAttributeAccessorBase):
    " used in FieldsNS "
    component: ComponentBase
    instance: ModelType

    def get_attribute(self, apply_session:IApplySession, attr_name: str, is_last:bool) -> ComponentAttributeAccessor:
        children_dict = apply_session.get_upward_components_dict(self.component)
        if attr_name not in children_dict:
            avail_names = get_available_names_example(attr_name, children_dict.keys())
            raise RuleApplyNameError(
                    owner=self.component, 
                    msg=f"Attribute '{attr_name}' not found in '{self.component.name}' ({type(self.component.name)}). Available: {avail_names}")

        component = children_dict[attr_name]

        if is_last:
            if not hasattr(component, "bind"):
                raise RuleApplyNameError(
                        owner=self.component,
                        msg=f"Attribute '{attr_name}' is '{type(component)}' type which has no binding, therefore can not extract value. Use standard *Field components instead.")
            # TODO: needs some class wrapper and caching ...
            vexp_result = component.bind._evaluator.execute_vexp(apply_session)
            out = vexp_result.value
        else:
            out = ComponentAttributeAccessor(component)
        return out


# ------------------------------------------------------------


class UseSetupStackFrame(AbstractContextManager):
    " with() ... custom context manager. Very similar to UseApplyStackFrame "
    
    # ALT: from contextlib import contextmanager
    def __init__(self, setup_session: ISetupSession, frame: SetupStackFrame):
        self.setup_session = setup_session
        self.frame = frame

    def __enter__(self):
        self.setup_session.push_frame_to_stack(self.frame)
        return self.frame

    def __exit__(self, exc_type, exc_value, exc_tb):
        frame_popped = self.setup_session.pop_frame_from_stack()
        if not exc_type and frame_popped != self.frame:
            raise RuleInternalError(owner=self, msg=f"Something wrong with frame stack, got {frame_popped}, expected {self.frame}")


# ------------------------------------------------------------

@dataclass
class SetupSessionBase(ISetupSession):

    owner                       : Optional[Any]
    owner_setup_session         : Optional[ISetupSession]

    # custom_function_factories store 
    functions                   : Optional[List[CustomFunctionFactory]] = field(repr=False, default=None)
    functions_factory_registry  : Optional[FunctionsFactoryRegistry] = field(repr=False, default=None)
    include_builtin_functions   : bool = field(repr=False, default=True)

    # autocomputed and internals
    is_top_setup_session        : bool = field(init=False, repr=False)
    top_owner_setup_session     : ISetupSession = field(init=False, repr=False)

    _registry_dict              : Dict[str, IRegistry] = field(init=False, repr=False, default_factory=dict)
    name                        : str = field(init=False, repr=False)
    vexp_node_dict              : Dict[str, IValueExpressionNode] = field(init=False, repr=False, default_factory=dict)
    finished                    : bool = field(init=False, repr=False, default=False)
    hook_on_finished_all_list   : Optional[List[HookOnFinishedAllCallable]] = field(init=False, repr=False)

    # stack of frames - first frame is current. On the end of the process the
    # stack must be empty
    frames_stack: List[StackyStackFrame] = field(repr=False, init=False, default_factory=list)


    # def __init__(self, 
    #         # usually 'ContainerBase'
    #         owner:  Optional[Any],  # noqa: F821
    #         owner_setup_session: Optional[ISetupSession], 
    #         functions: Optional[List[CustomFunctionFactory]] = None, 
    #         functions_factory_registry: Optional[FunctionsFactoryRegistry] = None,
    #         include_builtin_functions: bool = True,
    #         ):
    #     # owner is usually: 'ContainerBase'
    #     self.owner: Optional[Any] = owner  # noqa: F821
    #     self.owner_setup_session: Optional[ISetupSession] = owner_setup_session

    def __post_init__(self):
        self.is_top_setup_session: bool = (self.owner_setup_session is None)

        if self.is_top_setup_session:
            self.top_owner_setup_session = self
        else:
            self.top_owner_setup_session = self.owner_setup_session
            while self.top_owner_setup_session.owner_setup_session:
                self.top_owner_setup_session = self.top_owner_setup_session.owner_setup_session

        assert self.top_owner_setup_session

        # compputed
        # self._registry_dict : Dict[str, IRegistry] = {}
        # self.vexp_node_dict: Dict[str, IValueExpressionNode] = {}
        # self.finished: bool = False

        self.name: str = self.owner.name if self.owner else "no-owner"

        if self.functions_factory_registry:
            assert not self.functions
            # self.functions_factory_registry = functions_factory_registry
        else:
            self.functions_factory_registry: FunctionsFactoryRegistry = \
                    FunctionsFactoryRegistry(functions=self.functions, 
                                            include_builtin=self.include_builtin_functions)

        self.hook_on_finished_all_list: Optional[List[HookOnFinishedAllCallable]] =  \
            [] if self.is_top_setup_session else None


    # ------------------------------------------------------------

    def __str__(self):
        counts = ", ".join([f"{k}={v.count()}" for k, v in self._registry_dict.items() if v])
        # name={self.name},
        # cnt={self.entries_count}, 
        return f"SetupSession(owner={self.owner}, {counts})"

    def __repr__(self):
        return str(self)


    # ------------------------------------------------------------

    def add_registry(self, registry: IRegistry):
        if self.finished:
            raise RuleInternalError(owner=self, msg=f"Registry already in finished satte, adding '{registry}' not possible.")
        ns_name = registry.NAMESPACE._name
        if ns_name in self._registry_dict:
            raise RuleInternalError(owner=self, msg=f"Registry {registry} already in registry")
        self._registry_dict[ns_name] = registry
        registry.set_setup_session(self)


    # ------------------------------------------------------------

    def use_stack_frame(self, frame: SetupStackFrame) -> UseSetupStackFrame:
        if not isinstance(frame, SetupStackFrame):
            raise RuleInternalError(owner=self, msg=f"Expected SetupStackFrame, got frame: {frame}") 

        return UseSetupStackFrame(setup_session=self, frame=frame)

    # ------------------------------------------------------------

    def push_frame_to_stack(self, frame: ApplyStackFrame):
        self.frames_stack.insert(0, frame)

    def pop_frame_from_stack(self) -> ApplyStackFrame:
        assert self.frames_stack
        return self.frames_stack.pop(0)

    @property
    def current_frame(self) -> ApplyStackFrame:
        assert self.frames_stack
        return self.frames_stack[0]

    # ------------------------------------------------------------


    def get_registry(self, namespace: Namespace, strict:bool= True) -> IRegistry:
        if namespace._name not in self._registry_dict:
            if strict:
                raise RuleInternalError(owner=self, msg=f"Registry '{namespace._name}' not found, available are: {self._registry_dict.keys()}")
            return UNDEFINED
        return self._registry_dict[namespace._name]

    def __getitem__(self, namespace: Namespace) -> IRegistry:
        return self._registry_dict[namespace._name]


    # ------------------------------------------------------------

    def add_hook_on_finished_all(self, hook_function: HookOnFinishedAllCallable):
        self.top_owner_setup_session.hook_on_finished_all_list.append(hook_function)

    def call_hooks_on_finished_all(self):
        if not self.is_top_setup_session:
            raise RuleInternalError(owner=self, msg="call_hooks_on_finished_all() can be called on top setup_session") 
        for hook_function in self.hook_on_finished_all_list:
            hook_function()

    # ------------------------------------------------------------

    def dump_all(self, with_functions:bool = False) -> None:
        # recursive: bool = False, depth: int = 0, 
        # has {self.entries_count} attr_node(s), 
        print(f"{self.owner}: SetupSession '{self.name}', finished={self.finished}. List:")
        for ns_name, store in self._registry_dict.items():
            store.pp()

        if with_functions:
            self.functions_factory_registry.dump_all()

    # ------------------------------------------------------------

    def _register_attr_node(self, attr_node:AttrVexpNode, alt_attr_node_name=None):
        """
        !!!!! NOTE helper method - USED ONLY IN UNIT TESTING !!!!!
        TODO: replace with register_vexp_node
        """
        assert not self.finished
        ns_name = attr_node.namespace._name
        if ns_name not in self._registry_dict:
            raise RuleInternalError(f"{ns_name} not in .setup_session, available: {self._registry_dict.keys()}")
        self._registry_dict[ns_name].register_attr_node(attr_node, alt_attr_node_name=alt_attr_node_name)

    # ------------------------------------------------------------

    def get_vexp_node_by_vexp(self,
                        vexp: ValueExpression,
                        default:[None, UndefinedType]=UNDEFINED,
                        strict:bool=False
                        ) -> Union[IValueExpressionNode, None, UndefinedType]:
        if not isinstance(vexp,  ValueExpression):
            raise RuleInternalError(owner=self, msg=f"Vexp not ValueExpression, got {vexp} / {type(vexp)}")
        return vexp._vexp_node

    # ------------------------------------------------------------

    def register_vexp_node(self, vexp_node: IValueExpressionNode):
        " used to validate if all value expressions are completely setup (called finish() and similar) in setup phase " 
        # TODO: this could be cache - to reuse M.name, F.id, etc. (not for functions and operations)
        #       key = vexp_node.name

        key = id(vexp_node)
        if key in self.vexp_node_dict:
            if self.vexp_node_dict[key] != vexp_node:
                raise RuleInternalError(owner=self, msg=f"vexp key {key}: vexp_node already registered with different object: \n  == {self.vexp_node_dict[key]} / {id(self.vexp_node_dict[key])}\n  got:\n  == {vexp_node} / {id(vexp_node)}")  
        else:
            self.vexp_node_dict[key] = vexp_node

    # ------------------------------------------------------------

    def finish(self):
        if self.finished:
            raise RuleSetupError(owner=self, msg="Method finish() already called.")

        if not len(self.frames_stack) == 0:
            raise RuleInternalError(owner=self, msg=f"Stack frames not released: {self.frames_stack}") 

        for ns, registry in self._registry_dict.items():
            for vname, vexp_node in registry.items():
                assert isinstance(vexp_node, IValueExpressionNode)
                # do some basic validate
                vexp_node.finish()

            registry.finish()

        for vexp_node in self.vexp_node_dict.values():
            if not vexp_node.is_finished:
                vexp_node.finish()
            if not vexp_node.is_finished:
                raise RuleInternalError(owner=self, msg=f"VexpNode {vexp_node} still not finished, finish() moethod did not set is_finished")  

        self.finished = True





# ------------------------------------------------------------
# OBSOLETE
# ------------------------------------------------------------
# if vname!=vexp_node.name:
#     found = [vexp_node_name for setup_session_name, ns, vexp_node_name in vexp_node.bound_list if vname==vexp_node_name]
#     if not found:
#         raise RuleInternalError(owner=self, msg=f"Attribute name not the same as stored in setup_session {vexp_node.name}!={vname} or bound list: {vexp_node.bound_list}")



# # ------------------------------------------------------------
# 
# # obsolete - using FunctionNS() , functions, EnumMembers instead
# 
# class DataRegistry(RegistryBase):
# 
#     NAMESPACE = DataNS
# 
#     def create_vexp_node(self, data_var:IData) -> IValueExpressionNode:
#         # ------------------------------------------------------------
#         # A.2. DATAPROVIDERS - Collect all vexp_nodes from dataproviders fieldgroup
#         # ------------------------------------------------------------
#         if False:
#             ...
#         # elif isinstance(data_var, StaticData):
#         #     vexp_node = AttrVexpNode(
#         #                             name=data_var.name,
#         #                             data=data_var,
#         #                             namespace=DataNS,
#         #                             type_info=data_var.type_info,
#         #                             )
#         # elif isinstance(data_var, DynamicData):
#         #     assert isinstance(data_var.function, CustomFunctionFactory)
#         #     # TODO: consider storing CustomFactoryFunction instead of CustomFunction instances
#         #     #       to allow extra arguments when referenced
#         #     vexp_node = data_var.function.create_function(
#         #                     setup_session=self.setup_session,
#         #                     caller=None,
#         #                     func_args=EmptyFunctionArguments, 
#         #                     name=data_var.name) 
#         else:
#             # TODO: does Operation needs special handling?
#             # if not isinstance(data_var, IData:
#             raise RuleSetupError(owner=self, msg=f"Register expexted IData, got {data_var} / {type(data_var)}.")
# 
#         return vexp_node
# 
#     def register(self, data_var:IData):
#         vexp_node = self.create_vexp_node(data_var)
#         # can be AttrVexpNode or FunctionVexpNode
#         # alt_vexp_node_name=data_var.name
#         self.register_vexp_node(vexp_node)
#         return vexp_node
# 
#     def get_root_value(self, apply_session: IApplySession, attr_name: str) -> Any:
#         raise NotImplementedError()
