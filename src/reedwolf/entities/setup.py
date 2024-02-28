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

from .global_settings import (
    GlobalSettings,
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
    RegistryRootValue,
)
from .meta import (
    Self,
    is_model_klass,
    FunctionArgumentsType,
    FunctionArgumentsTupleType,
    ModelKlassType,
    TypeInfo,
    HookOnFinishedAllCallable,
    get_model_fields,
    AttrName,
    ComponentStatus,
    is_list_instance_or_type, ChildField, ListChildField, AttrDexpNodeTypeEnum,
)
from .base import (
    IComponentFields,
    IStackOwnerSession,
    IComponent,
    IContainer,
    extract_type_info,
    IApplyResult,
    IDataModel,
    ReservedAttributeNames,
    SetupStackFrame,
    UseStackFrameCtxManagerBase, )
from .functions import (
    FunctionsFactoryRegistry,
    try_create_function,
)
from .expr_attr_nodes import (
    IAttrDexpNode, AttrDexpNodeForComponent, AttrDexpNodeForModelKlass,
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


def create_exception_name_not_found_error(namespace: Namespace,
                                          owner: IComponent,
                                          valid_varnames_str: str,
                                          full_dexp_node_name: str,
                                          custom_msg: str = "") -> EntitySetupNameNotFoundError:
    valid_names = f"Valid attributes: {valid_varnames_str}" if valid_varnames_str \
             else "Namespace has no attributes at all."
    if custom_msg:
        valid_names = f"{custom_msg} {valid_names}"
    return EntitySetupNameNotFoundError(owner=owner, msg=f"Namespace '{namespace}': Invalid attribute name '{full_dexp_node_name}'. {valid_names}")

# ------------------------------------------------------------


@dataclass
class RegistryBase(IRegistry):
    """
    Namespaces are DynamicAttrsBase with flexible attributes, so it is not
    convenient to have some advanced logic within. Thus Registry logic 
    is put in specialized classes - SetupSession.
    """
    store: Dict[AttrName, IAttrDexpNode] = field(repr=False, init=False, default_factory=dict)
    setup_session: Union[ISetupSession, UndefinedType] = field(repr=False, init=False, default=UNDEFINED)
    finished: bool                  = field(init=False, repr=False, default=False)

    # TODO: with 3.11 - Protocol
    NAMESPACE: ClassVar[Namespace] = None
    ROOT_VALUE_NEEDS_FETCH_BY_NAME: ClassVar[bool] = True
    CALL_DEXP_NOT_FOUND_FALLBACK: ClassVar[bool] = False

    @staticmethod
    def is_unbound_models_registry() -> bool:
        """ default implementation """
        return False

    def setup(self, setup_session: ISetupSession):
        if self.setup_session:
            raise EntityInternalError(owner=self, msg=f"setup_session already set: {self.setup_session}")
        self.setup_session = setup_session

    def dexp_not_found_fallback(self, owner: IComponent, full_dexp_node_name: AttrName) -> Union[IAttrDexpNode, UndefinedType]:
        """
        method is called when CALL_DEXP_NOT_FOUND_FALLBACK is set to True
        returns:
            - UNDEFINED if this functino does not find attribute
            - otherwise return IAttrDexpNode.
        """
        raise NotImplementedError("When setting CALL_DEXP_NOT_FOUND_FALLBACK to True, you must implement dexp_not_found_fallback() method")

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
            # NOTE: Should not happen if DotExpression.setup() has done its job properly
            _, valid_varnames_str = self.get_valid_varnames(attr_name)
            raise EntityApplyNameError(owner=self, msg=f"Unknown attribute '{attr_name}', available attribute(s): {valid_varnames_str}."
                                                       + (f" Caller: {caller}" if caller else ""))

        attr_dexp_node: IAttrDexpNode = self.store[attr_name]
        # root_value: RegistryRootValue = self._apply_to_get_root_value(apply_result=apply_result, attr_name=attr_name)
        root_value: RegistryRootValue = self._apply_to_get_root_value(apply_result=apply_result, attr_name=attr_name)
        if not root_value:
            # registry did not retrieve value - internal problem
            raise EntityInternalError(owner=self, msg=f"Unknown attribute '{attr_name}' (2)."
                                                       + (f" Caller: {caller}" if caller else ""))

        root_value.set_attr_dexp_node(attr_dexp_node)
        return root_value

    def finish(self):
        if self.finished:
            raise EntityInternalError(owner=self, msg="already finished") 
        if not self.setup_session:
            raise EntityInternalError(owner=self, msg="setup_session not set, function .setup() not called")
        self.finished = True

    def count(self) -> int:
        return len(self.store)

    def get(self, key:str, default:Any=None) -> Optional[IAttrDexpNode]:
        return self.store.get(key, default)

    def items(self) -> List[Tuple[str, IAttrDexpNode]]:
        return self.store.items()

    # ------------------------------------------------------------

    # def _register_value_attr_node(self, attr_node: IAttrDexpNode) -> IAttrDexpNode:
    #     " used for This.Value to return attribute value "
    #     if not isinstance(attr_node, IAttrDexpNode):
    #         raise EntitySetupValueError(owner=self, msg=f"Expected IAttrDexpNode, got: {type(attr_node)} / {attr_node} ")

    #     type_info = attr_node.get_type_info()
    #     # NOTE: original name is: attr_node.name
    #     attr_name = ReservedAttributeNames.VALUE_ATTR_NAME.value
    #     attr_node = AttrDexpNodeForModelKlass(
    #                     name=attr_name,
    #                     data=type_info,
    #                     namespace=self.NAMESPACE,
    #                     type_info=type_info,
    #                     type_object=None,
    #                     )
    #     self.register_attr_node(attr_node)
    #     return attr_node

    # --------------------

    def _register_all_children(self,
                               setup_session: ISetupSession,
                               attr_name: ReservedAttributeNames,
                               component: IComponent,
                               attr_name_prefix: str = None,
                               ) -> IAttrDexpNode:
        """
        This.Children to return instance itself "
        """
        if not isinstance(component, IComponent):
            raise EntitySetupValueError(owner=self, msg=f"Expected ComponentBase, got: {type(component)} / {to_repr(component)}")

        component_fields_dataclass, child_field_list = component.get_component_fields_dataclass(setup_session=setup_session)
        assert child_field_list
        for nr, child_field in enumerate(child_field_list, 1):
            attr_node = AttrDexpNodeForComponent(
                            name=child_field.Name,
                            # data=child_field._type_info,
                            data=child_field._component,
                            namespace=self.NAMESPACE,
                            type_info=child_field._type_info,
                            type_object=None,
                            )
            self.register_attr_node(attr_node)

        type_info = TypeInfo.get_or_create_by_type(ListChildField)
        children_attr_node = self._register_special_attr_node(attr_name=ReservedAttributeNames.CHILDREN_ATTR_NAME.value,
                                                              component=component,
                                                              type_info=type_info,
                                                              attr_name_prefix=attr_name_prefix,
                                                              # TODO: missusing
                                                              th_field=component_fields_dataclass)
        return children_attr_node

    # --------------------

    def _register_special_attr_node(self,
                                    attr_name : ReservedAttributeNames,
                                    # None for model_klass
                                    component: Optional[IComponent],
                                    type_info: TypeInfo,
                                    attr_name_prefix: Optional[str]=None,
                                    th_field: Optional[Any] = None,
                                    ) -> IAttrDexpNode:
        # NOTE: removed restriction - was too strict
        # if not (is_model_klass(model_klass) or model_klass in STANDARD_TYPE_LIST or is_enum(model_klass)):
        #     raise EntitySetupValueError(owner=self, msg=f"Expected model class (DC/PYD), got: {type(model_klass)} / {model_klass} ")

        if th_field and not (inspect.isclass(th_field) and issubclass(th_field, IComponentFields)):
            raise EntitySetupValueError(owner=self, msg=f"Expected th_field is IComponentFields, got: {type(th_field)} / {th_field} ")

        # type_info = TypeInfo.get_or_create_by_type(py_type_hint=model_klass)
        # model_klass: ModelKlassType = type_info.type_

        if attr_name_prefix:
            attr_name = f"{attr_name_prefix}{attr_name}"

        if component:
            attr_node = AttrDexpNodeForComponent(
                            name=attr_name,
                            data=component,
                            namespace=self.NAMESPACE,
                            type_info=type_info,
                            type_object=th_field,
                            )
        else:
            attr_node = AttrDexpNodeForModelKlass(
                name=attr_name,
                data=type_info,
                namespace=self.NAMESPACE,
                type_info=None,
                type_object=th_field,
            )

        self.register_attr_node(attr_node)
        return attr_node

    # --------------------

    # def register_items_attr_node(self,
    #                              owner_component: IContainer
    #                              ) -> IAttrDexpNode:
    #     " used for This.Items to return list items - each having children "
    #     # for nr, child in enumerate(children, 1):
    #     #     if not isinstance(child, IComponent):
    #     #         raise EntitySetupValueError(owner=self, msg=f"Child {nr}: Expected IComponent, got: {type(child)} / {to_repr(child)} ")
    #     assert self.is_items_mode
    #     type_info = owner_component.data_model.get_type_info()
    #     attr_name = ReservedAttributeNames.ITEMS_ATTR_NAME.value
    #     attr_node = AttrDexpNodeForModelKlass(
    #                     name=attr_name,
    #                     data=type_info,
    #                     namespace=self.NAMESPACE,
    #                     type_info=type_info,
    #                     type_object=None,
    #                     )
    #     self.register_attr_node(attr_node)
    #     return attr_node

    # --------------------

    def _register_model_nodes(self, model_klass: ModelKlassType):
        if not is_model_klass(model_klass):
            raise EntitySetupValueError(owner=self, msg=f"Expected model class (DC/PYD), got: {type(model_klass)} / {to_repr(model_klass)} ")

        for attr_name in get_model_fields(model_klass):
            attr_node = self._create_attr_node_for_model_attr(model_klass, attr_name)
            self.register_attr_node(attr_node)



    # ------------------------------------------------------------

    @classmethod
    def _create_attr_node_for_model_attr(cls, model_klass: ModelKlassType, attr_name:str) -> IAttrDexpNode:
        # NOTE: will go again and again into get_model_fields()
        #       but shortcut like this didn't worked: 
        #           type_info: TypeInfo = TypeInfo.get_or_create_by_type(th_field)

        if attr_name in (ReservedAttributeNames.INSTANCE_ATTR_NAME,):
            raise EntitySetupNameError(msg=f"Model attribute name '{attr_name} is reserved. Rename it and try again (model={model_klass.__name__}).")

        # This one should not fail
        # , func_node
        type_info, th_field = \
                extract_type_info(
                    attr_node_name=attr_name,
                    inspect_object=model_klass)

        attr_node = AttrDexpNodeForModelKlass(
                        name=attr_name,
                        data=type_info,
                        namespace=cls.NAMESPACE,
                        type_object=th_field,
                        )
        return attr_node

    # ------------------------------------------------------------

    def register_dexp_node(self, dexp_node:IDotExpressionNode,
                           alt_dexp_node_name=None,
                           replace_when_duplicate:bool = False):
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

        if not replace_when_duplicate and dexp_node_name in self.store:
            raise EntitySetupNameError(owner=self, msg=f"IAttrDexpNode '{dexp_node}' does not have unique name '{dexp_node_name}' within this registry, found: {self.store[dexp_node_name]}")

        self.store[dexp_node_name] = dexp_node


    def register_attr_node(self, attr_node:IAttrDexpNode,
                           alt_attr_node_name=None,
                           replace_when_duplicate:bool = False):
        if not isinstance(attr_node, IAttrDexpNode):
            raise EntityInternalError(f"{type(attr_node)}->{attr_node}")
        if not self.NAMESPACE == attr_node.namespace:
            raise EntityInternalError(owner=self, msg=f"Method register({attr_node}) - namespace mismatch: {self.NAMESPACE} != {attr_node.namespace}")
        return self.register_dexp_node(dexp_node=attr_node,
                                       alt_dexp_node_name=alt_attr_node_name,
                                       replace_when_duplicate=replace_when_duplicate)

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

        func_node: IFunctionDexpNode = \
                try_create_function(
                    setup_session=setup_session,
                    caller=caller,
                    attr_node_name=attr_node_name,
                    func_args=func_args,
                    value_arg_type_info=value_arg_type_info,
                    )
        return func_node

    def get_valid_varnames(self, attr_name: str) -> Tuple[List[str], str]:
        valid_varnames_list = [vn for vn, attr_dexp_node in self.store.items() if not attr_dexp_node.denied]
        valid_varnames_str = get_available_names_example(attr_name, list(valid_varnames_list), max_display=7)
        return valid_varnames_list, valid_varnames_str

    def create_exception_name_not_found_error(self,
                                              owner: IComponent,
                                              full_dexp_node_name: str,
                                              custom_msg: str = "") -> EntitySetupNameNotFoundError:
        valid_varnames_list, valid_varnames_str = self.get_valid_varnames(full_dexp_node_name)
        return create_exception_name_not_found_error(owner=owner,
                                                     namespace=self.NAMESPACE,
                                                     valid_varnames_str=valid_varnames_str if valid_varnames_list else "",
                                                     full_dexp_node_name=full_dexp_node_name,
                                                     custom_msg=custom_msg)

    # ------------------------------------------------------------
    # create_node -> IAttrDexpNode, Operation or IFunctionDexpNode
    # ------------------------------------------------------------
    def create_node(self,
                    dexp_node_name: str,
                    owner_dexp_node: IDotExpressionNode,
                    owner: IComponent,
                    is_1st_node: bool,
                    ) -> IDotExpressionNode:
        """
        Will create a new attr_node when missing, even in the case when the var
        is just "on the path" to final attr_node needed.

        Namespace usually is: ModelsNS, ContextNS

        """
        # can return DotExpression / class member / object member
        # TODO: circ dep?
        if not isinstance(dexp_node_name, str):
            raise EntityInternalError(owner=self, msg=f"dexp_node_name is not a string, got: {dexp_node_name}")

        if dexp_node_name.startswith("_"):
            raise EntitySetupNameError(owner=owner, msg=f"Namespace '{self.NAMESPACE}': IAttrDexpNode '{dexp_node_name}' is invalid, should not start with _")

        parent_type_info = None
        if owner_dexp_node:
            assert owner_dexp_node.name!=dexp_node_name
            if not isinstance(owner_dexp_node, IDotExpressionNode):
                raise EntitySetupNameError(owner=owner, msg=f"Namespace '{self.NAMESPACE}': '{dexp_node_name}' -> owner_dexp_node={owner_dexp_node}: {type(owner_dexp_node)} is not IDotExpressionNode")

        # TODO: za Sum(This.name) this is not good. Should be unique, since 
        #       This is ambiguous - can evaluate to different contexts.
        #       should have some settings @ID (owner)
        full_dexp_node_name = get_dexp_node_name(
                                    owner_name=owner_dexp_node.name if owner_dexp_node else None, 
                                    dexp_node_name=dexp_node_name,
                                    func_args=None
                                    )

        found_component: Optional[IComponent] = None

        if owner_dexp_node is None:
            # --------------------------------------------------
            # get() -> TOP LEVEL - only level that is stored
            # --------------------------------------------------
            # e.g. M.company Predefined before in Entity.setup() function.
            if owner.is_unbound() and self.NAMESPACE == ModelsNS:
                if full_dexp_node_name in self.store:
                    raise EntitySetupNameError(owner=self, msg=f"full_dexp_node_name={full_dexp_node_name} already in store: {self.store[full_dexp_node_name]}")
                # TODO: solve with IUnboundModelsRegistry - method available in UnboundModelsRegistry only
                attr_node_template = self.register_unbound_attr_node(component=owner, full_dexp_node_name=full_dexp_node_name)
            else:
                attr_node_template = self.store.get(full_dexp_node_name, UNDEFINED)
                if attr_node_template is UNDEFINED:
                    if self.CALL_DEXP_NOT_FOUND_FALLBACK:
                        attr_node_template = self.dexp_not_found_fallback(owner=owner, full_dexp_node_name=full_dexp_node_name)

                    if attr_node_template is UNDEFINED:
                        raise self.create_exception_name_not_found_error(owner=owner, full_dexp_node_name=full_dexp_node_name)


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
                # search base.py:: if len(self.bind_to.Path) > 1:
                raise EntitySetupError(owner=self, msg=f"ModelsNS multi level DotExpressions (e.g. M.a.b) currently not supported, got: {owner_dexp_node} . {dexp_node_name}")

            if isinstance(owner_dexp_node, IFunctionDexpNode):
                # if owner_dexp_node.name in ("Map", "First"): print("here34")
                inspect_object = owner_dexp_node.get_type_info()
            else:
                assert isinstance(owner_dexp_node, IAttrDexpNode)
                # if isinstance(owner_dexp_node.data, IDataModel):
                # if False and owner_dexp_node.attr_node_type == AttrDexpNodeTypeEnum.DATA_MODEL:
                #     inspect_object = owner_dexp_node.data.model_klass
                # # elif is_model_klass(owner_dexp_node.data):
                # #     # @dataclass, Pydantic etc.
                # #     inspect_object = owner_dexp_node.data
                # # elif isinstance(owner_dexp_node.data, IContainer):
                # else:
                #     # elif owner_dexp_node.attr_node_type in (AttrDexpNodeTypeEnum.CONTAINER,
                #     #                                         AttrDexpNodeTypeEnum.FIELD,
                #     #                                         AttrDexpNodeTypeEnum.FIELD_GROUP,
                #     #                                         ):
                #     # Container does not have owner_dexp_node.type_info set in the moment of creation
                #     # if not owner_dexp_node.attr_node_type in (AttrDexpNodeTypeEnum.TYPE_INFO,
                #     #                                           AttrDexpNodeTypeEnum.ATTRIBUTE,
                #     #                                           AttrDexpNodeTypeEnum.ATTR_BY_METHOD):
                #     #     raise EntityInternalError(owner=self, msg=f"Expected type_info, got: {owner_dexp_node} / {owner_dexp_node.attr_node_type}")
                parent_component : Optional[IComponent] = owner_dexp_node.get_component()
                parent_type_info = owner_dexp_node.get_type_info()
                if GlobalSettings.is_unit_test and parent_component:
                    assert parent_type_info is owner_dexp_node.data.get_type_info()
                # for component this is custom created dataclass <Component>__Fields
                inspect_object = parent_type_info
                if parent_component:
                    # if this is a component, then try to find child component by name in direct children
                    children_dict = parent_component.get_children_dict()
                    found_component = children_dict.get(dexp_node_name, UNDEFINED)
                    if not found_component:
                        # in this momemnt
                        children_names = list(children_dict.keys())
                        valid_varnames_str = get_available_names_example(dexp_node_name, children_names, max_display=7) if children_names else ""
                        raise create_exception_name_not_found_error(owner=owner,
                                                                    namespace=self.NAMESPACE,
                                                                    valid_varnames_str=valid_varnames_str,
                                                                    full_dexp_node_name=full_dexp_node_name)

                    if not found_component.has_data():
                        raise EntitySetupNameError(owner=parent_component, msg=f"Found component '{dexp_node_name}' contains no data, choose data component. Got: {found_component}")

            err_msg_prefix = f"'{owner.name} -> {owner_dexp_node.full_name} . {dexp_node_name}'"

            if not found_component:
                # type_info = found_component.get_type_info()
                # data = found_component
                # type_object = type_info.type_
                # else:
                # if this is a component, then try to find child component by name in direct children
                try:
                    type_info, th_field = extract_type_info(
                                inspect_object=inspect_object,
                                attr_node_name=dexp_node_name,
                                )
                    # data = type_info
                    # type_object = th_field
                except EntitySetupNameNotFoundError as ex:
                    ex.set_msg(f"{err_msg_prefix}: {ex.msg}")
                    raise
                except EntityError as ex:
                    ex.set_msg(f"{err_msg_prefix}: metadata / type-hints read problem: {ex}")
                    raise

            # if this is a list and not in a direct
            # TODO: is there a better way?
            if not is_1st_node and is_list_instance_or_type(inspect_object):
                # TODO: too complex error message
                raise EntitySetupValueError(owner=owner, msg=f"{err_msg_prefix} can not be read from list, got: {inspect_object}. Use list accessor functions.")


            # --------------------------------------------------
            # Create()
            # --------------------------------------------------
            if found_component:
                dexp_node = AttrDexpNodeForComponent(
                                name=full_dexp_node_name,
                                data=found_component,
                                namespace=self.NAMESPACE,
                                type_object=found_component.get_type_info().type_,
                            )  # function=function
            else:
                dexp_node = AttrDexpNodeForModelKlass(
                                name=full_dexp_node_name,
                                data=type_info,
                                namespace=self.NAMESPACE,
                                type_object=th_field,
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


class UseSetupStackFrameCtxManager(UseStackFrameCtxManagerBase):
    " with() ... custom settings manager. "
    owner_session: "SetupSession"
    frame: SetupStackFrame


# ------------------------------------------------------------

@dataclass
class SetupSessionBase(IStackOwnerSession, ISetupSession):

    container:                  Optional[IContainer]

    # obligatory init arg - used to compute *top_setup_session - None for TopParentSession
    parent_setup_session:       Optional["SetupSessionBase"] = None

    # builtin + custom_function_factories store
    functions_factory_registry: Optional[FunctionsFactoryRegistry] = field(repr=False, default=None)

    # autocomputed and internals
    top_setup_session:          "TopSetupSession" = field(init=False, repr=False)

    _registry_dict:             Dict[str, IRegistry] = field(init=False, repr=False, default_factory=dict)
    name:                       str = field(init=False, repr=False)
    dexp_node_dict:             Dict[str, IDotExpressionNode] = field(init=False, repr=False, default_factory=dict)
    finished:                   bool = field(init=False, repr=False, default=False)

    # stack of frames - first frame is current. On the end of the process the stack must be empty
    stack_frames:               List[SetupStackFrame] = field(repr=False, init=False, default_factory=list)

    # autocomputed
    current_frame:              Optional[SetupStackFrame] = field(repr=False, init=False, default=None)

    STACK_FRAME_CLASS:          ClassVar[type] = SetupStackFrame
    STACK_FRAME_CTX_MANAGER_CLASS: ClassVar[type] = UseSetupStackFrameCtxManager

    def __post_init__(self):
        if self.container is not None and not isinstance(self.container, IContainer):
            raise EntityInternalError(owner=self, msg=f"Expecting container for parent, got: {type(self.container)} / {self.container}")
        self.name: str = self.container.name if self.container else "no-container"

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

    # ------------------------------------------------------------

    def add_hook_on_finished_all(self, hook_function: HookOnFinishedAllCallable):
        self.top_setup_session.hook_on_finished_all_list.append(hook_function)

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

    def _register_attr_node(self, attr_node:IAttrDexpNode, alt_attr_node_name=None):
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
            if not dexp_node._status == ComponentStatus.finished:
                dexp_node.finish()
            # if not dexp_node.is_finished:
            if not dexp_node._status == ComponentStatus.finished:
                raise EntityInternalError(owner=self, msg=f"DexpNode {dexp_node} still not finished, finish() method did not set status to finished")

        self.finished = True

# ------------------------------------------------------------

class SetupSession(SetupSessionBase):

    def __post_init__(self):
        super().__post_init__()
        self.top_setup_session = self.parent_setup_session.top_setup_session
        assert isinstance(self.top_setup_session, TopSetupSession), self.top_setup_session
        # ALT: while self.top_setup_session.parent_setup_session:
        #     self.top_setup_session = self.top_setup_session.parent_setup_session

# ------------------------------------------------------------

@dataclass
class TopSetupSession(SetupSessionBase):
    # TODO: :TopFieldsRegistry
    top_fields_registry: RegistryBase = field(repr=False, default=None)
    hook_on_finished_all_list:  Optional[List[HookOnFinishedAllCallable]] = field(init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if not self.top_fields_registry:
            raise EntityInternalError(owner=self, msg=f"top_fields_registry is obligatory.")
        self.top_setup_session = self
        self.hook_on_finished_all_list: Optional[List[HookOnFinishedAllCallable]] = []

    def call_hooks_on_finished_all(self):
        for hook_function in self.hook_on_finished_all_list:
            hook_function()


# ============================================================
# OBSOLETE
# ============================================================
# @dataclass
# class ComponentAttributeAccessor(IAttributeAccessorBase):
#     """
#     Used in FieldsNS
#     Value will be fetched by evaluating .bind_to of Field component.
#     """
#     component: IComponent
#     instance: ModelInstanceType
#     value_node: IValueNode
#
#     def get_attribute(self, apply_result:IApplyResult, attr_name: str) -> Self:
#         raise Exception("should not be used any more - use ValueNode instead")
#         children_dict = apply_result.get_upward_components_dict(self.component)
#         if attr_name not in children_dict:
#             avail_names = get_available_names_example(attr_name, list(children_dict.keys()))
#             raise EntityApplyNameError(owner=self.component,
#                     msg=f"Attribute '{attr_name}' not found in '{self.component.name}' ({type(self.component.name)}). Available: {avail_names}")
#
#         component = children_dict[attr_name]
#
#         # OLD: if is_last_node:
#         if not isinstance(component, IField):
#             # ALT: not hasattr(component, "bind_to")
#             raise EntityApplyNameError(owner=self.component,
#                     msg=f"Attribute '{attr_name}' is '{type(component)}' type which has no binding, therefore can not extract value. Use standard *Field components instead.")
#
#         # TODO: needs some class wrapper and caching ...
#         dexp_result = component.bind_to._evaluator.execute_dexp(apply_result)
#         out = dexp_result.value
#
#         return out


