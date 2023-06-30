from abc import abstractmethod
from dataclasses import dataclass, field
from typing import (
        Any,
        List,
        Union,
        Optional,
        ClassVar,
        Tuple,
        Dict,
        )

from .utils import (
        get_available_names_example,
        UNDEFINED,
        UndefinedType,
        check_identificator_name,
        )
from .exceptions import (
        EntityError,
        EntitySetupError,
        EntitySetupNameError,
        EntitySetupNameNotFoundError,
        EntityInternalError,
        EntityApplyNameError,
        EntitySetupValueError,
        )
from .namespaces import (
        Namespace,
        )
from .expressions import (
        DotExpression,
        IDotExpressionNode,
        IFunctionDexpNode,
        IRegistry,
        ISetupSession,
        IAttributeAccessorBase,
        )
from .meta import (
        Self,
        is_model_class,
        FunctionArgumentsType,
        FunctionArgumentsTupleType,
        ModelType,
        TypeInfo,
        HookOnFinishedAllCallable,
        )
from .base import (
        IStackOwnerSession,
        ComponentBase,
        IContainerBase,
        extract_type_info,
        IApplySession,
        BoundModelBase,
        ReservedAttributeNames,
        SetupStackFrame,
        UseStackFrameCtxManagerBase,
        )
from .functions import (
        FunctionsFactoryRegistry,
        CustomFunctionFactory,
        try_create_function,
        )
from .attr_nodes import (
        AttrDexpNode,
        )

# ------------------------------------------------------------


def get_dexp_node_name(owner_name: Optional[str], 
                       dexp_node_name: str, 
                       func_args: Optional[Union[FunctionArgumentsTupleType, FunctionArgumentsType]]
                       ) -> str:
    # dexp_node_name = 
    check_identificator_name(dexp_node_name)

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
        dexp_node_name = f"{dexp_node_name}({all_args_str})"
    else:
        dexp_node_name = f"{dexp_node_name}"
    if owner_name is not None:
        # not validated at all
        dexp_node_name = f"{owner_name}.{dexp_node_name}"

    return dexp_node_name


# ------------------------------------------------------------


@dataclass
class RegistryBase(IRegistry):
    """
    Namespaces are DynamicAttrsBase with flexible attributes, so it is not
    convenient to have some advanced logic within. Thus Registry logic 
    is put in specialized classes - SetupSession.
    """
    store : Dict[str, AttrDexpNode] = field(repr=False, init=False, default_factory=dict)
    setup_session: Union[ISetupSession, UndefinedType] = field(repr=False, init=False, default=UNDEFINED)
    finished: bool                  = field(init=False, repr=False, default=False)

    # TODO: with 3.11 - Protocol
    NAMESPACE : ClassVar[Namespace] = None
    ROOT_VALUE_NEEDS_FETCH_BY_NAME: ClassVar[bool] = True

    def __post_init__(self):
        ...

    def set_setup_session(self, setup_session: ISetupSession):
        if self.setup_session:
            raise EntityInternalError(owner=self, msg=f"setup_session already set: {self.setup_session}") 
        self.setup_session = setup_session

    def finish(self):
        if self.finished:
            raise EntityInternalError(owner=self, msg="already finished") 
        if not self.setup_session:
            raise EntityInternalError(owner=self, msg="setup_session not set, function set_setup_session() not called") 
        self.finished = True

    def count(self) -> int:
        return len(self.store)

    def get(self, key:str, default:Any=None) -> Optional[AttrDexpNode]:
        return self.store.get(key, default)

    def items(self) -> List[Tuple[str, AttrDexpNode]]:
        return self.store.items()


    def register_instance_attr_node(self, model_class: ModelType, attr_name_prefix: Optional[str]=None) -> AttrDexpNode:
        " used for This.Instance / M.Instance to return instance itself "
        if not is_model_class(model_class):
            raise EntitySetupValueError(owner=self, msg=f"Expected model class (DC/PYD), got: {type(model_class)} / {model_class} ")

        th_field = None
        type_info = TypeInfo.get_or_create_by_type(
                        py_type_hint=model_class,
                        caller=None,
                        )
        attr_name = ReservedAttributeNames.INSTANCE_ATTR_NAME.value
        if attr_name_prefix:
            attr_name = f"{attr_name_prefix}{attr_name}"

        attr_node = AttrDexpNode(
                        name=attr_name,
                        data=type_info,
                        namespace=self.NAMESPACE,
                        type_info=type_info, 
                        th_field=th_field,
                        )
        self.register_attr_node(attr_node)
        return attr_node


    def register_value_attr_node(self, attr_node: AttrDexpNode) -> AttrDexpNode:
        " used for This.Value to return attribute value "
        if not isinstance(attr_node, AttrDexpNode):
            raise EntitySetupValueError(owner=self, msg=f"Expected AttrDexpNode, got: {type(attr_node)} / {attr_node} ")

        type_info = attr_node.get_type_info()
        # NOTE: original name is: attr_node.name
        attr_name = ReservedAttributeNames.VALUE_ATTR_NAME.value
        attr_node = AttrDexpNode(
                        name=attr_name,
                        data=type_info,
                        namespace=self.NAMESPACE,
                        type_info=type_info, 
                        th_field=None,
                        )
        self.register_attr_node(attr_node)
        return attr_node


    @classmethod
    def _create_attr_node_for_model_attr(cls, model_class: ModelType, attr_name:str) -> AttrDexpNode:
        # NOTE: will go again and again into get_model_fields()
        #       but shortcut like this didn't worked: 
        #           type_info: TypeInfo = TypeInfo.get_or_create_by_type(th_field)

        if attr_name in (ReservedAttributeNames.INSTANCE_ATTR_NAME,):
            raise EntitySetupNameError(msg=f"Model attribute name '{attr_name} is reserved. Rename it and try again (model={model_class.__name__}).")

        # This one should not fail
        # , func_node
        type_info, th_field = \
                extract_type_info(
                    attr_node_name=attr_name,
                    inspect_object=model_class)

        attr_node = AttrDexpNode(
                        name=attr_name,
                        data=type_info,
                        namespace=cls.NAMESPACE,
                        type_info=type_info, 
                        th_field=th_field,
                        )
        return attr_node


    def register_dexp_node(self, dexp_node:IDotExpressionNode, alt_dexp_node_name=None):
        """
        Data can register IFunctionDexpNode-s instances since the
        output will be used directly as data and not as a function call.
            function=[Function(name="Countries", title="Countries", 
                              py_function=CatalogManager.get_countries)],
            ...
            available=(S.Countries.name != "test"),
        """
        if self.finished:
            raise EntityInternalError(owner=self, msg=f"Register({dexp_node}) - already finished, adding not possible.")

        if not isinstance(dexp_node, IDotExpressionNode):
            raise EntityInternalError(f"{type(dexp_node)}->{dexp_node}")
        dexp_node_name = alt_dexp_node_name if alt_dexp_node_name else dexp_node.name

        if not dexp_node_name.count(".") == 0:
            raise EntityInternalError(owner=self, msg=f"Node {dexp_node_name} should not contain . - only first level vars allowed")

        if dexp_node_name in self.store:
            raise EntitySetupNameError(owner=self, msg=f"AttrDexpNode {dexp_node} does not have unique name within this registry, found: {self.store[dexp_node_name]}")
        self.store[dexp_node_name] = dexp_node


    def register_attr_node(self, attr_node:AttrDexpNode, alt_attr_node_name=None):
        if not isinstance(attr_node, AttrDexpNode):
            raise EntityInternalError(f"{type(attr_node)}->{attr_node}")
        if not self.NAMESPACE == attr_node.namespace:
            raise EntityInternalError(owner=self, msg=f"Method register({attr_node}) - namespace mismatch: {self.NAMESPACE} != {attr_node.namespace}")
        return self.register_dexp_node(dexp_node=attr_node, alt_dexp_node_name=alt_attr_node_name)

    def pprint(self):
        print(f"  Namespace {self.NAMESPACE._name}:")
        for dexp_node_name, dexp_node in self.store.items():
            print(f"    {dexp_node_name} = {dexp_node.as_str()}")

    def __str__(self):
        return f"{self.__class__.__name__}(cnt={len(self.store)})"

    __repr__ = __str__

    # ------------------------------------------------------------
    # create_func_node - only Functions i.e. IFunctionDexpNode
    # ------------------------------------------------------------
    def create_func_node(self, 
            setup_session: ISetupSession,
            caller: IDotExpressionNode,
            attr_node_name:str,
            func_args:FunctionArgumentsType,
            value_arg_type_info:TypeInfo,
            ) -> IFunctionDexpNode:

        func_node : IFunctionDexpNode = \
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
    # create_node -> AttrDexpNode, Operation or IFunctionDexpNode
    # ------------------------------------------------------------
    def create_node(self, 
                    dexp_node_name: str, 
                    owner_dexp_node: IDotExpressionNode, 
                    # func_args: FunctionArgumentsType
                    owner: ComponentBase,
                    ) -> IDotExpressionNode:
        """
        Will create a new attr_node when missing, even in the case when the var
        is just "on the path" to final attr_node needed.

        Namespace usually is: ModelsNS, FunctionsNS

        """
        # can return DotExpression / class member / object member
        # TODO: circ dep?
        assert isinstance(dexp_node_name, str), dexp_node_name

        if dexp_node_name.startswith("_"):
            raise EntitySetupNameError(owner=owner, msg=f"Namespace '{self.NAMESPACE}': AttrDexpNode '{dexp_node_name}' is invalid, should not start with _")

        type_info = None
        if owner_dexp_node:
            assert owner_dexp_node.name!=dexp_node_name
            if not isinstance(owner_dexp_node, IDotExpressionNode):
                raise EntitySetupNameError(owner=owner, msg=f"Namespace '{self.NAMESPACE}': '{dexp_node_name}' -> owner_dexp_node={owner_dexp_node} :{type(owner_dexp_node)} is not IDotExpressionNode")

        # TODO: za Sum(This.name) this is not good. Should be unique, since 
        #       This is ambigous - can evaluate to different contexts. 
        #       should have some context @ID (owner)
        full_dexp_node_name = get_dexp_node_name(
                                    owner_name=owner_dexp_node.name if owner_dexp_node else None, 
                                    dexp_node_name=dexp_node_name,
                                    func_args=None # func_args
                                    )

        if owner_dexp_node is None:
            # --------------------------------------------------
            # get() -> TOP LEVEL - only level that is stored
            # --------------------------------------------------
            # e.g. M.company Predefined before in Entity.setup() function.
            if full_dexp_node_name not in self.store:
                names_avail = get_available_names_example(full_dexp_node_name, self.store.keys())
                valid_names = f"Valid attributes: {names_avail}" if self.store.keys() else "Namespace has no attributes at all."
                raise EntitySetupNameNotFoundError(owner=owner, msg=f"Namespace '{self.NAMESPACE}': Invalid attribute name '{full_dexp_node_name}'. {valid_names}")

            attr_node_template = self.store.get(full_dexp_node_name)

            # NOTE: RL 230525 do not clone, .finish() is not called for cloned,
            #       they were not registered anywhere. OLD:
            #           dexp_node = attr_node_template.clone()
            #           assert id(dexp_node) != attr_node_template
            dexp_node = attr_node_template
        else:
            # --------------------------------------------------------------
            # IN-DEPTH LEVEL 2+, e.g. M.company.<this-node-and-next>
            # ---------------------------------------------------------------

            if isinstance(owner_dexp_node, IFunctionDexpNode):
                inspect_object = owner_dexp_node.get_type_info()
            else:
                assert isinstance(owner_dexp_node, AttrDexpNode)

                if False:
                    ...
                # elif isinstance(owner_dexp_node.data, StaticData):
                #     inspect_object = owner_dexp_node.data.value
                # elif isinstance(owner_dexp_node.data, DynamicData):
                #     raise NotImplementedError("TODO")
                elif isinstance(owner_dexp_node.data, BoundModelBase):
                    inspect_object = owner_dexp_node.data.model
                else:
                    # can be TypeInfo, @dataclass, Pydantic etc.
                    inspect_object = owner_dexp_node.data

            try:
                # , func_node
                type_info, th_field = \
                        extract_type_info(
                            attr_node_name=dexp_node_name,
                            inspect_object=inspect_object, 
                            # func_args=func_args,
                            # functions_factory_registry=self.functions_factory_registry
                            )
            except EntitySetupNameNotFoundError as ex:
                ex.set_msg(f"{owner} / {self.NAMESPACE}-NS: {ex.msg}")
                raise 
            except EntityError as ex:
                ex.set_msg(f"'{owner} / {self.NAMESPACE}-NS: {owner_dexp_node.full_name} -> '.{dexp_node_name}' metadata / type-hints read problem: {ex}")
                raise 

            # if func_node:
            #     assert func_args
            #     assert isinstance(func_node, IFunctionDexpNode)
            #     dexp_node = func_node
            # else:

            assert type_info

            # --------------------------------------------------
            # Create()
            # --------------------------------------------------
            # attributes / functions / operations
            dexp_node = AttrDexpNode(
                            name=full_dexp_node_name,
                            data=type_info,
                            namespace=self.NAMESPACE,
                            type_info=type_info, 
                            th_field=th_field, 
                            ) # function=function

        if not isinstance(dexp_node, IDotExpressionNode):
            raise EntityInternalError(owner=owner, msg=f"Namespace {self.NAMESPACE}: Type of found object is not IDotExpressionNode, got: {type(dexp_node)}.")

        # NOTE: dexp_node.type_info can be None, will be filled later in finish()

        return dexp_node

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
        raise EntityInternalError(owner=self, msg="Registry should not be used to get root value.")


# ------------------------------------------------------------


@dataclass
class ComponentAttributeAccessor(IAttributeAccessorBase):
    " used in FieldsNS "
    component: ComponentBase
    instance: ModelType

    def __post_init__(self):
        ...

    def get_attribute(self, apply_session:IApplySession, attr_name: str, is_last:bool) -> Self:
        children_dict = apply_session.get_upward_components_dict(self.component)
        if attr_name not in children_dict:
            avail_names = get_available_names_example(attr_name, children_dict.keys())
            raise EntityApplyNameError(owner=self.component, 
                    msg=f"Attribute '{attr_name}' not found in '{self.component.name}' ({type(self.component.name)}). Available: {avail_names}")

        component = children_dict[attr_name]

        # OLD: 
        #   if is_last:

        if not hasattr(component, "bind"):
            raise EntityApplyNameError(owner=self.component,
                    msg=f"Attribute '{attr_name}' is '{type(component)}' type which has no binding, therefore can not extract value. Use standard *Field components instead.")
        # TODO: needs some class wrapper and caching ...
        dexp_result = component.bind._evaluator.execute_dexp(apply_session)
        out = dexp_result.value

        return out


# ------------------------------------------------------------


class UseSetupStackFrameCtxManager(UseStackFrameCtxManagerBase):
    " with() ... custom context manager. "

    def copy_from_previous_frame(self):
        if not self.owner_session.stack_frames:
            return

        previous_frame = self.owner_session.stack_frames[0]

        self._copy_attr_from_previous_frame(previous_frame, "local_setup_session", 
                                            if_set_must_be_same=False)


# ------------------------------------------------------------

@dataclass
class SetupSessionBase(IStackOwnerSession, ISetupSession):

    container                    : Optional[IContainerBase]
    parent_setup_session         : Optional[ISetupSession]

    # custom_function_factories store 
    functions                   : Optional[List[CustomFunctionFactory]] = field(repr=False, default=None)
    functions_factory_registry  : Optional[FunctionsFactoryRegistry] = field(repr=False, default=None)
    include_builtin_functions   : bool = field(repr=False, default=True)

    # autocomputed and internals
    is_top_setup_session        : bool = field(init=False, repr=False)
    top_parent_setup_session     : ISetupSession = field(init=False, repr=False)

    _registry_dict              : Dict[str, IRegistry] = field(init=False, repr=False, default_factory=dict)
    name                        : str = field(init=False, repr=False)
    dexp_node_dict              : Dict[str, IDotExpressionNode] = field(init=False, repr=False, default_factory=dict)
    finished                    : bool = field(init=False, repr=False, default=False)
    hook_on_finished_all_list   : Optional[List[HookOnFinishedAllCallable]] = field(init=False, repr=False)

    # stack of frames - first frame is current. On the end of the process the stack must be empty
    stack_frames: List[SetupStackFrame] = field(repr=False, init=False, default_factory=list)

    # autocomputed
    current_frame: Optional[SetupStackFrame] = field(repr=False, init=False, default=None)

    STACK_FRAME_CLASS: ClassVar[type] = SetupStackFrame
    STACK_FRAME_CTX_MANAGER_CLASS: ClassVar[type] = UseSetupStackFrameCtxManager

    def __post_init__(self):
        if self.container is not None and not isinstance(self.container, IContainerBase):
            raise EntityInternalError(owner=self, msg=f"Expecting container for parent, got: {type(self.container)} / {self.container}") 

        self.is_top_setup_session: bool = (self.parent_setup_session is None)

        if self.is_top_setup_session:
            self.top_parent_setup_session = self
        else:
            self.top_parent_setup_session = self.parent_setup_session
            while self.top_parent_setup_session.parent_setup_session:
                self.top_parent_setup_session = self.top_parent_setup_session.parent_setup_session

        assert self.top_parent_setup_session

        # compputed
        # self._registry_dict : Dict[str, IRegistry] = {}
        # self.dexp_node_dict: Dict[str, IDotExpressionNode] = {}
        # self.finished: bool = False

        self.name: str = self.container.name if self.container else "no-container"

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
        return f"SetupSession(parent={self.container}, {counts})"

    def __repr__(self):
        return str(self)


    # ------------------------------------------------------------

    def add_registry(self, registry: IRegistry):
        if self.finished:
            raise EntityInternalError(owner=self, msg=f"Registry already in finished satte, adding '{registry}' not possible.")
        ns_name = registry.NAMESPACE._name
        if ns_name in self._registry_dict:
            raise EntityInternalError(owner=self, msg=f"Registry {registry} already in registry")
        self._registry_dict[ns_name] = registry
        registry.set_setup_session(self)


    # ------------------------------------------------------------

    # def use_stack_frame(self, frame: SetupStackFrame) -> UseSetupStackFrameCtxManager:
    #     if not isinstance(frame, SetupStackFrame):
    #         raise EntityInternalError(owner=self, msg=f"Expected SetupStackFrame, got frame: {frame}") 

    #     return UseSetupStackFrameCtxManager(owner_session=self, frame=frame)

    # ------------------------------------------------------------

    def _get_avail_registry_names(self, namespace: Namespace, is_internal_use: bool) -> List[str]:
        if not is_internal_use:
            registry_names = [reg.NAMESPACE._GetNameWithAlias() 
                              for reg in self._registry_dict.values() 
                              if not reg.NAMESPACE._is_for_internal_use_only]
        else:
            registry_names = [reg.NAMESPACE._GetNameWithAlias() 
                              for reg in self._registry_dict.values()]

        avail_names = get_available_names_example(namespace._name, registry_names)
        return avail_names


    def get_registry(self, namespace: Namespace, strict: bool = True, is_internal_use: bool = False) -> IRegistry:
        registry = self._registry_dict.get(namespace._name, UNDEFINED)
        if registry is UNDEFINED:
            if strict:
                avail_names = self._get_avail_registry_names(namespace, is_internal_use)
                raise EntitySetupNameError(owner=self, msg=f"Registry '{namespace._name}' not found. Available: {avail_names}")
            return UNDEFINED

        if not is_internal_use and namespace._is_for_internal_use_only:
            avail_names = self._get_avail_registry_names(namespace, is_internal_use)
            raise EntitySetupNameError(owner=self, msg=f"Registry '{namespace._name}' is for internal use only. Available registries: {avail_names}")

        return self._registry_dict[namespace._name]


    def __getitem__(self, namespace: Namespace) -> IRegistry:
        return self._registry_dict[namespace._name]


    # ------------------------------------------------------------

    def add_hook_on_finished_all(self, hook_function: HookOnFinishedAllCallable):
        self.top_parent_setup_session.hook_on_finished_all_list.append(hook_function)

    def call_hooks_on_finished_all(self):
        if not self.is_top_setup_session:
            raise EntityInternalError(owner=self, msg="call_hooks_on_finished_all() can be called on top setup_session") 
        for hook_function in self.hook_on_finished_all_list:
            hook_function()

    # ------------------------------------------------------------

    def pprint(self, with_functions:bool = False) -> None:
        " ppprint == pp == pretty print "
        # recursive: bool = False, depth: int = 0, 
        # has {self.entries_count} attr_node(s), 
        print(f"{self.container}: SetupSession '{self.name}', finished={self.finished}. List:")
        for ns_name, store in self._registry_dict.items():
            store.pprint()

        if with_functions:
            self.functions_factory_registry.dump_all()

    # ------------------------------------------------------------

    def _register_attr_node(self, attr_node:AttrDexpNode, alt_attr_node_name=None):
        """
        !!!!! NOTE helper method - USED ONLY IN UNIT TESTING !!!!!
        TODO: replace with register_dexp_node
        """
        assert not self.finished
        ns_name = attr_node.namespace._name
        if ns_name not in self._registry_dict:
            raise EntityInternalError(f"{ns_name} not in .setup_session, available: {self._registry_dict.keys()}")
        self._registry_dict[ns_name].register_attr_node(attr_node, alt_attr_node_name=alt_attr_node_name)

    # ------------------------------------------------------------

    def get_dexp_node_by_dexp(self,
                        dexp: DotExpression,
                        default:[None, UndefinedType]=UNDEFINED,
                        strict:bool=False
                        ) -> Union[IDotExpressionNode, None, UndefinedType]:
        if not isinstance(dexp,  DotExpression):
            raise EntityInternalError(owner=self, msg=f"Dexp not DotExpression, got {dexp} / {type(dexp)}")
        return dexp._dexp_node

    # ------------------------------------------------------------

    def register_dexp_node(self, dexp_node: IDotExpressionNode):
        " used to validate if all value expressions are completely setup (called finish() and similar) in setup phase " 
        # TODO: this could be cache - to reuse M.name, F.id, etc. (not for functions and operations)
        #       key = dexp_node.name

        key = id(dexp_node)
        if key in self.dexp_node_dict:
            if self.dexp_node_dict[key] != dexp_node:
                raise EntityInternalError(owner=self, msg=f"dexp key {key}: dexp_node already registered with different object: \n  == {self.dexp_node_dict[key]} / {id(self.dexp_node_dict[key])}\n  got:\n  == {dexp_node} / {id(dexp_node)}")  
        else:
            self.dexp_node_dict[key] = dexp_node

    # ------------------------------------------------------------

    def finish(self):
        if self.finished:
            raise EntitySetupError(owner=self, msg="Method finish() already called.")

        if not len(self.stack_frames) == 0:
            raise EntityInternalError(owner=self, msg=f"Stack frames not released: {self.stack_frames}") 

        for ns, registry in self._registry_dict.items():
            for vname, dexp_node in registry.items():
                assert isinstance(dexp_node, IDotExpressionNode)
                # do some basic validate
                dexp_node.finish()

            registry.finish()

        for dexp_node in self.dexp_node_dict.values():
            if not dexp_node.is_finished:
                dexp_node.finish()
            if not dexp_node.is_finished:
                raise EntityInternalError(owner=self, msg=f"DexpNode {dexp_node} still not finished, finish() moethod did not set is_finished")  

        self.finished = True

