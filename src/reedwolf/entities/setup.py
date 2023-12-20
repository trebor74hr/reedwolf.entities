from abc import abstractmethod
from dataclasses import dataclass, field
import inspect
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
    to_repr,
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
    Namespace, ModelsNS,
)
from .expressions import (
    DotExpression,
    IDotExpressionNode,
    IFunctionDexpNode,
    IRegistry,
    ISetupSession,
    IAttributeAccessorBase,
    RegistryRootValue,
)
from .meta import (
    Self,
    is_model_class,
    FunctionArgumentsType,
    FunctionArgumentsTupleType,
    ModelType,
    TypeInfo,
    HookOnFinishedAllCallable,
    get_model_fields, STANDARD_TYPE_LIST,
    AttrName,
    ExpressionsAttributesMap,
    FunctionNoArgs,
    FieldName,
    MethodName,
    is_function,
    is_instancemethod_by_name,
)
from .base import (
    IComponentFields,
    ChildField,
    IField,
    IStackOwnerSession,
    IComponent,
    IContainer,
    extract_type_info,
    IApplyResult,
    IBoundModel,
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

    @staticmethod
    def is_unbound_models_registry() -> bool:
        """ default implementation """
        return False

    def apply_to_get_root_value(self, apply_result: IApplyResult, attr_name: AttrName, caller: Optional[str] = None) -> RegistryRootValue: # noqa: F821
        # TODO: same method declared in IRegistry
        """ 
        Apply phase - Namespace.<attr_name> - 
        function returns the instance (root value) from which attr_name will be read from 

        attr_name is not used for fetching the value, but for rare cases when 
        different source / instance is used based on different attr_name. See
        ThisNS.Instance.
        """

        if attr_name not in self.store:
            # NOTE: Should not hhappen if DotExpression.setup() has done its job properly
            names_avail = get_available_names_example(attr_name, list(self.store.keys()))
            raise EntityApplyNameError(owner=self, msg=f"Unknown attribute '{attr_name}, available attribute(s): {names_avail}."
                                                       + (f" Caller: {caller}" if caller else ""))

        attr_dexp_node: AttrDexpNode = self.store[attr_name]
        root_value: RegistryRootValue = self._apply_to_get_root_value(apply_result=apply_result,
                                                                      attr_name=attr_name)
        root_value.set_attr_dexp_node(attr_dexp_node)
        return root_value

    def setup(self, setup_session: ISetupSession):
        if self.setup_session:
            raise EntityInternalError(owner=self, msg=f"setup_session already set: {self.setup_session}") 
        self.setup_session = setup_session

    def finish(self):
        if self.finished:
            raise EntityInternalError(owner=self, msg="already finished") 
        if not self.setup_session:
            raise EntityInternalError(owner=self, msg="setup_session not set, function .setup() not called")
        self.finished = True

    def count(self) -> int:
        return len(self.store)

    def get(self, key:str, default:Any=None) -> Optional[AttrDexpNode]:
        return self.store.get(key, default)

    def items(self) -> List[Tuple[str, AttrDexpNode]]:
        return self.store.items()

    # ------------------------------------------------------------

    # def _register_value_attr_node(self, attr_node: AttrDexpNode) -> AttrDexpNode:
    #     " used for This.Value to return attribute value "
    #     if not isinstance(attr_node, AttrDexpNode):
    #         raise EntitySetupValueError(owner=self, msg=f"Expected AttrDexpNode, got: {type(attr_node)} / {attr_node} ")

    #     type_info = attr_node.get_type_info()
    #     # NOTE: original name is: attr_node.name
    #     attr_name = ReservedAttributeNames.VALUE_ATTR_NAME.value
    #     attr_node = AttrDexpNode(
    #                     name=attr_name,
    #                     data=type_info,
    #                     namespace=self.NAMESPACE,
    #                     type_info=type_info,
    #                     th_field=None,
    #                     )
    #     self.register_attr_node(attr_node)
    #     return attr_node

    # --------------------

    def _register_all_children(self,
                               setup_session: ISetupSession,
                               attr_name: ReservedAttributeNames,
                               owner: IComponent,
                               attr_name_prefix: str = None,
                               ) -> AttrDexpNode:
        """
        This.Children to return instance itself "
        """
        if not isinstance(owner, IComponent):
            raise EntitySetupValueError(owner=self, msg=f"Expected ComponentBase, got: {type(owner)} / {to_repr(owner)}")

        component_fields_dataclass, child_field_list = owner.get_component_fields_dataclass(setup_session=setup_session)
        assert child_field_list
        for nr, child_field in enumerate(child_field_list, 1):
            attr_node = AttrDexpNode(
                            name=child_field.Name,
                            data=child_field._type_info,
                            namespace=self.NAMESPACE,
                            type_info=child_field._type_info,
                            th_field=None,
                            )
            self.register_attr_node(attr_node)

        owner_model_class = List[ChildField]
        type_info = TypeInfo.get_or_create_by_type(owner_model_class)

        # owner_model_class = List[component_fields_dataclass]
        children_attr_node = self._register_special_attr_node(
                        # model_class = owner_model_class,
                        type_info=type_info,
                        attr_name=ReservedAttributeNames.CHILDREN_ATTR_NAME.value,
                        attr_name_prefix = attr_name_prefix,
                        # TODO: missusing
                        th_field = component_fields_dataclass,
                        )
        return children_attr_node

    # --------------------

    def _register_special_attr_node(self,
                                    # model_class: ModelType,
                                    type_info: TypeInfo,
                                    attr_name = ReservedAttributeNames,
                                    attr_name_prefix: Optional[str]=None,
                                    th_field: Optional[Any] = None,
                                    ) -> AttrDexpNode:
        # NOTE: removed restriction - was too strict
        # if not (is_model_class(model_class) or model_class in STANDARD_TYPE_LIST or is_enum(model_class)):
        #     raise EntitySetupValueError(owner=self, msg=f"Expected model class (DC/PYD), got: {type(model_class)} / {model_class} ")

        if th_field and not (inspect.isclass(th_field) and issubclass(th_field, IComponentFields)):
            raise EntitySetupValueError(owner=self, msg=f"Expected th_field is IComponentFields, got: {type(th_field)} / {th_field} ")

        # type_info = TypeInfo.get_or_create_by_type(py_type_hint=model_class)
        # model_class: ModelType = type_info.type_

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

    # --------------------

    # def register_items_attr_node(self,
    #                              owner_component: IContainer
    #                              ) -> AttrDexpNode:
    #     " used for This.Items to return list items - each having children "
    #     # for nr, child in enumerate(children, 1):
    #     #     if not isinstance(child, IComponent):
    #     #         raise EntitySetupValueError(owner=self, msg=f"Child {nr}: Expected IComponent, got: {type(child)} / {to_repr(child)} ")
    #     assert self.is_items_mode
    #     type_info = owner_component.bound_model.get_type_info()
    #     attr_name = ReservedAttributeNames.ITEMS_ATTR_NAME.value
    #     attr_node = AttrDexpNode(
    #                     name=attr_name,
    #                     data=type_info,
    #                     namespace=self.NAMESPACE,
    #                     type_info=type_info,
    #                     th_field=None,
    #                     )
    #     self.register_attr_node(attr_node)
    #     return attr_node

    # --------------------

    def _register_model_nodes(self, model_class: ModelType):
        if not is_model_class(model_class):
            raise EntitySetupValueError(owner=self, msg=f"Expected model class (DC/PYD), got: {type(model_class)} / {to_repr(model_class)} ")

        for attr_name in get_model_fields(model_class):
            # th_field: ModelField in .values()
            attr_node = self._create_attr_node_for_model_attr(model_class, attr_name)
            self.register_attr_node(attr_node)


    def _register_from_attributes_dict(self,
                                       model_class: ModelType,
                                       attributes_dict: ExpressionsAttributesMap):
        # map User, Session, Now and similar Attribute -> function calls
        fields = get_model_fields(model_class)

        for attr_name, attr_getter in attributes_dict.items():
            if isinstance(attr_getter, MethodName):
                # or is_function(attr_getter):
                attr_getter: MethodName = attr_getter
                py_function: FunctionNoArgs = getattr(model_class, attr_getter, UNDEFINED)
                if py_function is UNDEFINED:
                    # could get all methods with no args
                    raise EntitySetupNameError(owner=self, msg=f"Attribute {attr_name} must be name of method with no arguments from class '{model_class}', got: {attr_getter}")
                # else: py_function: FunctionNoArgs = attr_getter

                function_name = py_function.__name__
                if not is_instancemethod_by_name(model_class,  function_name):
                    raise EntitySetupNameError(owner=self,
                                               msg=f"Attribute {attr_name} must be name of method with no arguments of class '{model_class}', function {attr_getter} is not instance method of this class.")

                # Check that function receives only single param if method(self), or no param if function()
                py_fun_signature = inspect.signature(py_function)
                # TODO: resolve properly first arg name as 'self' convention
                non_empty_params = [param.name for param in py_fun_signature.parameters.values() if param.empty and param.name != 'self']
                if len(non_empty_params)!=0:
                    raise EntitySetupNameError(owner=self,
                           msg=f"{attr_name}: Method '{model_class.__name__}.{attr_getter}()' must not have arguments without defaults. Found: {', '.join(non_empty_params)} ")

                type_info = TypeInfo.extract_function_return_type_info(
                    py_function,
                    allow_nonetype=True)

                data = attr_getter
                # TODO: the type or name of th_field is not ok
                th_field = py_function

            elif isinstance(attr_getter, FieldName):
                attr_getter: FieldName = attr_getter
                attr_field = fields.get(attr_getter, None)
                if not attr_field:
                    aval_names = get_available_names_example(attr_field, fields.keys())
                    raise EntitySetupNameError(owner=self, msg=f"Attribute {attr_name} must be field name of class '{model_class}', got: {attr_getter}, available: {aval_names}")
                type_info = TypeInfo.get_or_create_by_type(
                    py_type_hint=attr_field,
                    caller=model_class,
                )
                data = attr_getter
                th_field = attr_field
            else:
                raise EntitySetupValueError(owner=self, msg=f"Attribute {attr_name} expected FieldName or MethodName instance, got: {attr_getter} / {type(attr_getter)}")

            attr_node = AttrDexpNode(
                name=attr_name,
                data=data,
                namespace=self.NAMESPACE,
                type_info=type_info,
                th_field=th_field,
            )
            if attr_name in self.store:
                raise EntitySetupNameError(f"Attribute name '{attr_name}' is reserved. Rename class attribute in '{self.apply_settings_class}'")

            self.register_attr_node(attr_node, attr_name)


    def _apply_to_get_root_value_by_attributes_dict(self,
                                                    # attributes_dict: ExpressionsAttributesMap,
                                                    attr_name: AttrName,
                                                    klass_attr_name: AttrName,
                                                    klass: type) -> RegistryRootValue:
        if klass in (UNDEFINED, None):
            # component = apply_result.current_frame.component
            raise EntityApplyNameError(owner=self, msg=f"Attribute '{attr_name}' can not be fetched since '{klass_attr_name}' is not set ({klass}).")

        if attr_name not in self.store:
            avail_names = get_available_names_example(attr_name, list(self.store.keys()))
            raise EntityApplyNameError(owner=self, msg=f"Invalid attribute name '{attr_name}', available: {avail_names}.")

        attr_dexp_node = self.store[attr_name]
        attr_getter = attr_dexp_node.data

        # if attr_name not in attributes_dict:
        #     avail_names = get_available_names_example(attr_name, attributes_dict.keys())
        #     raise EntityApplyNameError(owner=self, msg=f"Invalid attribute name '{attr_name}', available: {avail_names}.")
        # attr_getter = attributes_dict[attr_name]

        if isinstance(attr_getter, MethodName):
            # NOTE: convert unbound method to its name to be able to fetch bound method later.
            method = attr_dexp_node.th_field
            assert callable(method), method
            attr_name_new = method.__name__
        elif isinstance(attr_getter, FieldName):
            # must preserve FieldName type
            attr_name_new = attr_getter
        else:
            raise EntityInternalError(owner=self, msg=f"Expected FieldName or MethodName instance, got: {attr_getter}")

        # else: attr_name_new = None
        return RegistryRootValue(klass, attr_name_new)


    # ------------------------------------------------------------

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

    # ------------------------------------------------------------

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
            raise EntitySetupNameError(owner=self, msg=f"AttrDexpNode '{dexp_node}' does not have unique name '{dexp_node_name}' within this registry, found: {self.store[dexp_node_name]}")
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
                    owner: IComponent,
                    ) -> IDotExpressionNode:
        """
        Will create a new attr_node when missing, even in the case when the var
        is just "on the path" to final attr_node needed.

        Namespace usually is: ModelsNS, FunctionsNS

        """
        # can return DotExpression / class member / object member
        # TODO: circ dep?
        if not isinstance(dexp_node_name, str):
            raise EntityInternalError(owner=self, msg=f"dexp_node_name is not a string, got: {dexp_node_name}")

        if dexp_node_name.startswith("_"):
            raise EntitySetupNameError(owner=owner, msg=f"Namespace '{self.NAMESPACE}': AttrDexpNode '{dexp_node_name}' is invalid, should not start with _")

        type_info = None
        if owner_dexp_node:
            assert owner_dexp_node.name!=dexp_node_name
            if not isinstance(owner_dexp_node, IDotExpressionNode):
                raise EntitySetupNameError(owner=owner, msg=f"Namespace '{self.NAMESPACE}': '{dexp_node_name}' -> owner_dexp_node={owner_dexp_node} :{type(owner_dexp_node)} is not IDotExpressionNode")

        # TODO: za Sum(This.name) this is not good. Should be unique, since 
        #       This is ambigous - can evaluate to different contexts. 
        #       should have some settings @ID (owner)
        full_dexp_node_name = get_dexp_node_name(
                                    owner_name=owner_dexp_node.name if owner_dexp_node else None, 
                                    dexp_node_name=dexp_node_name,
                                    func_args=None
                                    )

        if owner_dexp_node is None:
            # --------------------------------------------------
            # get() -> TOP LEVEL - only level that is stored
            # --------------------------------------------------
            # e.g. M.company Predefined before in Entity.setup() function.
            if owner.is_unbound() and self.NAMESPACE == ModelsNS:
                if full_dexp_node_name in self.store:
                    raise EntitySetupNameError(owner=self, msg=f"full_dexp_node_name={full_dexp_node_name} already in store: {self.store[full_dexp_node_name]}")
                self.register_unbound_attr_node(component=owner, full_dexp_node_name=full_dexp_node_name)
            else:
                if full_dexp_node_name not in self.store:
                    names_avail = get_available_names_example(full_dexp_node_name, list(self.store.keys()), max_display=7)
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
            if owner.is_unbound() and self.NAMESPACE == ModelsNS:
                # search base.py :: if len(self.bind.Path) > 1:
                raise EntitySetupError(owner=self, msg=f"Should not happen, more than 1 level DotExpressions are not supported, got: {owner_dexp_node}")

            if isinstance(owner_dexp_node, IFunctionDexpNode):
                inspect_object = owner_dexp_node.get_type_info()
            else:
                assert isinstance(owner_dexp_node, AttrDexpNode)

                if isinstance(owner_dexp_node.data, IBoundModel):
                    inspect_object = owner_dexp_node.data.model
                elif is_model_class(owner_dexp_node.data):
                    # @dataclass, Pydantic etc.
                    inspect_object = owner_dexp_node.data
                else:
                    # take TypeInfo
                    inspect_object = owner_dexp_node.type_info

            try:
                # , func_node
                type_info, th_field = extract_type_info(
                            attr_node_name=dexp_node_name,
                            inspect_object=inspect_object, 
                            )
            except EntitySetupNameNotFoundError as ex:
                ex.set_msg(f"{owner} / {self.NAMESPACE}-NS: {ex.msg}")
                raise 
            except EntityError as ex:
                ex.set_msg(f"'{owner} / {self.NAMESPACE}-NS: {owner_dexp_node.full_name} -> '.{dexp_node_name}' metadata / type-hints read problem: {ex}")
                raise 

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


# ------------------------------------------------------------



class RegistryUseDenied(RegistryBase):

    def apply_to_get_root_value(self, apply_result: "IApplyResult", attr_name: AttrName, caller: Optional[str] = None) -> RegistryRootValue: # noqa: F821
        raise EntityInternalError(owner=self, msg="Registry should not be used to get root value.")


# ------------------------------------------------------------


@dataclass
class ComponentAttributeAccessor(IAttributeAccessorBase):
    " used in FieldsNS "
    component: IComponent
    instance: ModelType

    def get_attribute(self, apply_result:IApplyResult, attr_name: str) -> Self:
        children_dict = apply_result.get_upward_components_dict(self.component)
        if attr_name not in children_dict:
            avail_names = get_available_names_example(attr_name, children_dict.keys())
            raise EntityApplyNameError(owner=self.component, 
                    msg=f"Attribute '{attr_name}' not found in '{self.component.name}' ({type(self.component.name)}). Available: {avail_names}")

        component = children_dict[attr_name]

        # OLD: if is_last:
        if not isinstance(component, IField):
            # ALT: not hasattr(component, "bind")
            raise EntityApplyNameError(owner=self.component,
                    msg=f"Attribute '{attr_name}' is '{type(component)}' type which has no binding, therefore can not extract value. Use standard *Field components instead.")
        # TODO: needs some class wrapper and caching ...
        dexp_result = component.bind._evaluator.execute_dexp(apply_result)
        out = dexp_result.value

        return out


# ------------------------------------------------------------


class UseSetupStackFrameCtxManager(UseStackFrameCtxManagerBase):
    " with() ... custom settings manager. "
    owner_session: "SetupSessionBase"
    frame: SetupStackFrame


# ------------------------------------------------------------

@dataclass
class SetupSessionBase(IStackOwnerSession, ISetupSession):

    container                    : Optional[IContainer]
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
        if self.container is not None and not isinstance(self.container, IContainer):
            raise EntityInternalError(owner=self, msg=f"Expecting container for parent, got: {type(self.container)} / {self.container}")

        self.is_top_setup_session: bool = self.parent_setup_session is None

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

    def add_registry(self, registry: IRegistry, replace: bool = False):
        if self.finished:
            raise EntityInternalError(owner=self, msg=f"Registry already in finished satte, adding '{registry}' not possible.")
        ns_name = registry.NAMESPACE._name
        if ns_name in self._registry_dict:
            if not replace:
                raise EntityInternalError(owner=self, msg=f"Registry {registry} already in registry")
        else:
            if replace:
                raise EntityInternalError(owner=self, msg=f"Registry {registry} to replace should be in registry")
        self._registry_dict[ns_name] = registry
        registry.setup(setup_session=self)

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

    # def __getitem__(self, namespace: Namespace) -> IRegistry:
    #     return self._registry_dict[namespace._name]


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
            if registry.is_unbound_models_registry():
                raise EntityInternalError(owner=self, msg=f"Found unbound models registry for ns={ns} and it should have been replaced. Got: {registry}")
            for vname, dexp_node in registry.items():
                assert isinstance(dexp_node, IDotExpressionNode)
                # do some basic validate
                dexp_node.finish()
                if dexp_node.dexp_validate_type_info_func:
                    dexp_node.dexp_validate_type_info_func(dexp_node)

            registry.finish()

        for dexp_node in self.dexp_node_dict.values():
            if not dexp_node.is_finished:
                dexp_node.finish()
            if not dexp_node.is_finished:
                raise EntityInternalError(owner=self, msg=f"DexpNode {dexp_node} still not finished, finish() moethod did not set is_finished")  

        self.finished = True

    # ------------------------------------------------------------

    # def create_local_setup_session_for_this_instance(self,
    #                                                  model_class: ModelType,
    #                                                  ) -> IThisRegistry:
    #     # NOTE: must be placed here since:
    #     #   expressions
    #     this_registry = self.container.create_this_registry_for_instance(model_class=model_class)
    #     local_setup_session = self.create_local_setup_session(this_registry)
    #     return local_setup_session



