from __future__ import annotations

import re
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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
        DataNS,
        FieldsNS,
        ThisNS,
        FunctionsNS,
        ContextNS,
        ConfigNS,
        OperationsNS,
        )
from .meta import (
        FunctionArgumentsType,
        EmptyFunctionArguments,
        FunctionArgumentsTupleType,
        is_model_class,
        ModelType,
        get_model_fields,
        TypeInfo,
        )
from .base import (
        ComponentBase,
        IData,
        extract_type_info,
        IRegistry,
        IRegistries,
        IApplySession,
        IAttributeAccessorBase,
        )
from .models import (
        BoundModelBase,
        )
from .expressions import (
        ValueExpression,
        IValueExpressionNode,
        IFunctionVexpNode,
        )
from .functions import (
        FunctionsFactoryRegistry,
        CustomFunctionFactory,
        try_create_function,
        )
# from .validators import (
#         ValidatorBase,
#         )
# from .evaluators import (
#         EvaluatorBase,
#         )
from .attr_nodes import AttrVexpNode
from .models import (
        BoundModel,
        BoundModelWithHandlers,
        )
from .components import (
        StaticData,
        DynamicData,
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


_RE_ID_NAME_1, _RE_ID_NAME_2 = "a-zA-Z", "a-zA-Z0-9_"
_RE_ID_NAME = re.compile(f"^[{_RE_ID_NAME_1}][{_RE_ID_NAME_2}]*$")

def _clean_identificator_name(vexp_node_name: str) -> str:
    """ When attr_node/paremeter/argument name is valid 
        it returns the same value. """
    if not vexp_node_name:
        raise RuleSetupNameError(owner=None, msg=f"Invalid identificator name '{vexp_node_name}'. Empty name not allowed.")
    if not re.match(_RE_ID_NAME, vexp_node_name):
        raise RuleSetupNameError(owner=None, msg=f"Invalid identificator name '{vexp_node_name}'. Name should begin with: {_RE_ID_NAME_1} and can continue with one or more: {_RE_ID_NAME_2}.")
    return vexp_node_name


def get_vexp_node_name(parent_name: Optional[str], 
                       vexp_node_name: str, 
                       func_args: Optional[Union[FunctionArgumentsTupleType, FunctionArgumentsType]]
                       ) -> str:
    vexp_node_name = _clean_identificator_name(vexp_node_name)

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
                [f"{_clean_identificator_name(k)}='{v}':{type(v)}" 
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
    is put in specialized classes - Registries.
    """
    # TODO: with 3.11 - Protocol
    NAMESPACE : ClassVar[Namespace] = None
    ROOT_VALUE_NEEDS_FETCH_BY_NAME: ClassVar[bool] = True

    def __init__(self):
        # , functions_factory_registry: FunctionsFactoryRegistry):
        # self.functions_factory_registry = functions_factory_registry
        self.store : Dict[str, AttrVexpNode] = {}
        self.finished: bool = False

    def finish(self):
        assert not self.finished
        self.finished = True

    def count(self) -> int:
        return len(self.store)

    def get(self, key:str, default:Any=None) -> Optional[AttrVexpNode]:
        return self.store.get(key, default)

    def items(self) -> List[Tuple[str, AttrVexpNode]]:
        return self.store.items()

    @classmethod
    def _create_attr_node_for_model_attr(cls, model_class: ModelType, attr_name:str) -> AttrVexpNode:
        # NOTE: will go again and again into get_model_fields()
        #       but shortcut like this didn't worked: 
        #           type_info: TypeInfo = TypeInfo.get_or_create_by_type(th_field)

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
            data=[DynamicData(name="Countries", label="Countries", 
                                         function=Function(CatalogManager.get_countries))],
            ...
            available=(S.Countries.name != "test"),
        """
        if self.finished:
            raise RuleInternalError(f"{self}.register({vexp_node}) - already finished, adding not possible.")

        if not isinstance(vexp_node, IValueExpressionNode):
            raise RuleInternalError(f"{type(vexp_node)}->{vexp_node}")
        vexp_node_name = alt_vexp_node_name if alt_vexp_node_name else vexp_node.name

        if not vexp_node_name.count(".") == 0:
            raise RuleInternalError(f"{self} -> {vexp_node_name} should not contain . - only first level vars allowed")

        if vexp_node_name in self.store:
            raise RuleSetupNameError(owner=self, msg=f"{self}: AttrVexpNode {vexp_node} does not have unique name within this registry, found: {self.store[vexp_node_name]}")
        self.store[vexp_node_name] = vexp_node


    def register_attr_node(self, attr_node:AttrVexpNode, alt_attr_node_name=None):
        if not isinstance(attr_node, AttrVexpNode):
            raise RuleInternalError(f"{type(attr_node)}->{attr_node}")
        if not self.NAMESPACE == attr_node.namespace:
            raise RuleInternalError(f"{self}.register({attr_node}) - namespace mismatch: {self.NAMESPACE} != {attr_node.namespace}")
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
            registries: Registries,
            caller: IValueExpressionNode,
            attr_node_name:str,
            func_args:FunctionArgumentsType,
            value_arg_type_info:TypeInfo,
            ) -> IFunctionVexpNode:

        func_node : IFunctionVexpNode = \
                try_create_function(
                    registries=registries,
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

        Namespace usually is: ModelsNS, FunctionsNS, DataNS

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
            # e.g. M.company. Predefined before in Rules.setup() function.
            if full_vexp_node_name not in self.store:
                names_avail = get_available_names_example(full_vexp_node_name, self.store.keys())
                valid_names = f"Valid attributes: {names_avail}" if self.store.keys() else "Namespace has no attributes at all."
                # owner=self, 
                raise RuleSetupNameNotFoundError(owner=owner, msg=f"Namespace '{self.NAMESPACE}': Invalid attribute name '{full_vexp_node_name}'. {valid_names}")

            attr_node_template = self.store.get(full_vexp_node_name)
            # clone it - each attr_node must be unique - cached are used only as templates
            vexp_node = attr_node_template.clone()
            assert id(vexp_node) != attr_node_template

            # TODO: 
            # print(type(vexp_node))
            # if isinstance(parent_vexp_node.data, IFunctionVexpNode):
            #     inspect_object = parent_vexp_node.data.output_type_info
        else:
            # --------------------------------------------------------------
            # IN-DEPTH LEVEL 2+, e.g. M.company.<this-node-and-next>
            # ---------------------------------------------------------------

            if isinstance(parent_vexp_node, IFunctionVexpNode):
                inspect_object = parent_vexp_node.output_type_info
            else:
                assert isinstance(parent_vexp_node, AttrVexpNode)

                if isinstance(parent_vexp_node.data, StaticData):
                    inspect_object = parent_vexp_node.data.value
                elif isinstance(parent_vexp_node.data, DynamicData):
                    raise NotImplementedError("TODO")
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
                ex.msg = f"{owner} / {self.NAMESPACE}-NS: {ex.msg}"
                raise 
            except RuleError as ex:
                ex.msg = f"'{owner} / {self.NAMESPACE}-NS: {parent_vexp_node.full_name} -> '.{vexp_node_name}' metadata / type-hints read problem: {ex}"
                raise 

            # if func_node:
            #     assert func_args
            #     assert isinstance(func_node, IFunctionVexpNode)
            #     vexp_node = func_node
            # else:

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

        return vexp_node

    @abstractmethod
    def get_root_value(self, apply_session: IApplySession) -> Any:
        # NOTE: abstractmethod does not work - its complicated ...
        ...


# ------------------------------------------------------------


class ModelsRegistry(RegistryBase):
    """
    All models have full path from top container (for better readibility)
    e.g. M.company.address_set.city.street so reading begins from root
    instance, instead of apply_session.current_frame.instance.
    See get_root_value() implementation.
    """
    NAMESPACE = ModelsNS

    # Caller should not fetch attribute by name from return value of
    # get_root_value()
    ROOT_VALUE_NEEDS_FETCH_BY_NAME: ClassVar[bool] = False

    # NOTE: no register() method due complex logic - see
    #       ContainerBase._register_bound_model()

    def create_attr_node(self, bound_model:BoundModelBase):
        # standard DTO class attr_node
        # if not bound_model.type_info:
        #     bound_model.set_type_info()
        # assert bound_model.type_info.type_==model
        attr_node = AttrVexpNode(
                        name=bound_model.name,
                        data=bound_model,
                        namespace=self.NAMESPACE,
                        type_info=bound_model.get_type_info())
        return attr_node

    # ------------------------------------------------------------

    def get_attr_node_by_bound_model(self,
                               bound_model:BoundModelBase,
                               # default:[None, UndefinedType]=UNDEFINED,
                               # strict:bool=False
                               ) -> Union[AttrVexpNode, None, UndefinedType]:
        # allways in models
        attr_node_name = bound_model.name
        assert attr_node_name
        # if strict and attr_node_name not in self.store:
        #     raise RuleSetupNameError(owner=self, msg=f"AttrVexpNode not found {ModelsNS}.{attr_node_name}")
        # return self.store.get(attr_node_name, default)
        return self.store[attr_node_name]

    # ------------------------------------------------------------

    def get_root_value(self, apply_session: IApplySession) -> Any:
        # ROOT_VALUE_NEEDS_FETCH_BY_NAME = False
        instance = apply_session.current_frame.instance

        # bound_model = apply_session.current_frame.container.bound_model
        bound_model_root = apply_session.current_frame.bound_model_root
        assert instance and bound_model_root, f"{instance} / {bound_model_root}"

        expected_type = bound_model_root.type_info.type_ \
                        if isinstance(bound_model_root.model, ValueExpression) \
                        else bound_model_root.model
        if bound_model_root.type_info.is_list and isinstance(instance, (list, tuple)):
            # raise RuleApplyTypeError(owner=self, msg=f"Wrong type, expected list/tuple, got '{instance}'")
            assert instance
            # check only first
            instance_to_test = instance[0]
        else:
            instance_to_test = instance

        if not isinstance(instance_to_test, expected_type):
            raise RuleApplyTypeError(owner=self, msg=f"Wrong type, expected '{expected_type}', got '{instance}'")

        # TODO: activate this when it has sense (not partial mode with Extension/FieldGroup, not Extension)
        # this could differ - e.g. name="address", bind=M.company.address_set
        # if name!=bound_model_root.name:
        #     raise RuleApplyNameError(owner=self, msg=f"Wrong name, expected '{bound_model_root.name}', got '{name}'")

        return instance


# ------------------------------------------------------------


class DataRegistry(RegistryBase):
    NAMESPACE = DataNS

    def create_vexp_node(self, data_var:IData) -> IValueExpressionNode:
        # ------------------------------------------------------------
        # A.2. DATAPROVIDERS - Collect all vexp_nodes from dataproviders fieldgroup
        # ------------------------------------------------------------
        if isinstance(data_var, StaticData):
            vexp_node = AttrVexpNode(
                                    name=data_var.name,
                                    data=data_var,
                                    namespace=DataNS,
                                    type_info=data_var.type_info,
                                    ) # function=function
        elif isinstance(data_var, DynamicData):
            assert isinstance(data_var.function, CustomFunctionFactory)
            # TODO: consider storing CustomFactoryFunction instead of CustomFunction instances
            #       to allow extra arguments when referenced
            vexp_node = data_var.function.create_function(
                            registries=None,
                            caller=None,
                            func_args=EmptyFunctionArguments, 
                            name=data_var.name) 
        else:
            # TODO: does Operation needs special handling?
            # if not isinstance(data_var, IData:
            raise RuleSetupError(owner=self, msg=f"Register expexted IData, got {data_var} / {type(data_var)}.")

        return vexp_node

    def register(self, data_var:IData):
        vexp_node = self.create_vexp_node(data_var)
        # can be AttrVexpNode or FunctionVexpNode
        # alt_vexp_node_name=data_var.name
        self.register_vexp_node(vexp_node)
        return vexp_node

    def get_root_value(self, apply_session: IApplySession) -> Any:
        raise NotImplementedError()

# ------------------------------------------------------------

@dataclass
class ComponentAttributeAccessor(IAttributeAccessorBase):
    " used in FieldsNS "
    component: ComponentBase
    instance: ModelType

    # def has_attribute(self, attr_name: str) -> bool:
    #     raise NotImplementedError()

    def get_attribute(self, apply_session:IApplySession, attr_name: str, is_last:bool) -> ComponentAttributeAccessor:
        children_dict = self.component.get_children_dict()
        if attr_name not in children_dict:
            avail_names = get_available_names_example(attr_name, children_dict.keys())
            raise RuleApplyNameError(owner=self.component, msg=f"Attribute '{attr_name}' not found in '{self.component.name}' ({type(self.component.name)}). Available: {avail_names}")

        component = children_dict[attr_name]
        if is_last:
            if not hasattr(component, "bind"):
                raise RuleApplyNameError(owner=self.component, msg=f"Attribute '{attr_name}' is '{type(component)}' type which has no binding, therefore can not extract value. Use standard *Field components instead.")
            vexp_result = component.bind._evaluator.execute_vexp(apply_session)
            out = vexp_result.value
        else:
            out = ComponentAttributeAccessor(component)
        return out


    # # TODO: get final value ...
    # def get_value(self) -> Any:
    #     raise NotImplementedError()

    # CONTAINER_REGISTRY: ClassVar[Dict[int, ContainerBase]] = {}
    # @classmethod
    # def get_or_create(cls, container):
    #     key = id(container)
    #     if key not in cls.CONTAINER_REGISTRY:
    #         cls.CONTAINER_REGISTRY[key] = container
    #     return cls.CONTAINER_REGISTRY[key]


# ------------------------------------------------------------

class FieldsRegistry(RegistryBase):
    NAMESPACE = FieldsNS

    def create_attr_node(self, component:ComponentBase):
        # TODO: put class in container and remove these local imports
        from .fields import FieldBase
        from .containers import Extension, Rules
        # ------------------------------------------------------------
        # A.3. COMPONENTS - collect attr_nodes - previously flattened (recursive function fill_components)
        # ------------------------------------------------------------
        if not isinstance(component, (ComponentBase, )): # ValidatorBase
            raise RuleSetupError(owner=self, msg=f"Register expexted ComponentBase, got {component} / {type(component)}.")

        component_name = component.name

        if isinstance(component, (FieldBase, IData)):
            denied = False
            # not denied
            deny_reason = ""
            # TODO: type_info = component.attr_node.type_info
            type_info = None
        # containers, validations, evaluations, # dropped: validators, evaluators, ValidatorBase
        elif isinstance(component, (BoundModel, BoundModelWithHandlers, ValidationBase, EvaluationBase, FieldGroup, Extension, Rules, )): # 
            # stored - but should not be used
            denied = True
            deny_reason = f"Component of type {component.__class__.__name__} can not be referenced in ValueExpressions"
            if hasattr(component, "type_info"):
                type_info=component.type_info
            else:
                type_info=None
        else:
            # TODO: this should be automatic, a new registry for field types
            from .fields import (StringField, ChoiceField, BooleanField, EnumField)
            valid_types = ', '.join([t.__name__ for t in (
                            StringField, ChoiceField, BooleanField, EnumField,
                            ValidationBase, EvaluationBase, FieldGroup, Extension,
                            Rules, )]) # , EvaluatorBase, ValidatorBase
            raise RuleSetupError(owner=self, msg=f"RuleSetup does not support type {type(component)}: {repr(component)[:100]}. Valid type of objects or objects inherited from are: {valid_types}")

        attr_node = AttrVexpNode(
                        name=component_name,
                        data=component,
                        namespace=FieldsNS,
                        type_info=type_info,
                        denied=denied,
                        deny_reason=deny_reason)
        return attr_node


    def register(self, component:ComponentBase):
        attr_node = self.create_attr_node(component)
        self.register_attr_node(attr_node) # , is_list=False))
        return attr_node

    def get_root_value(self, apply_session: IApplySession) -> Any:
        # container = apply_session.current_frame.component.get_container_owner()
        component = apply_session.current_frame.component
        instance  = apply_session.current_frame.instance
        top_attr_accessor = ComponentAttributeAccessor(component, instance)
        # attr_accessor = top_attr_accessor.get_attribute(
        #                         apply_session=apply_session, 
        #                         attr_name=name, 
        #                         is_last=is_last)
        return top_attr_accessor

# ------------------------------------------------------------


class FunctionsRegistry(RegistryBase):
    NAMESPACE = FunctionsNS

    def __init__(self): # , functions_factory_registry: FunctionsFactoryRegistry):
        super().__init__() # functions_factory_registry=functions_factory_registry)

        # for func_name, function_factory in self.functions_factory_registry.items():
        #     # print("here-3", func_name, function_factory)
        #     self.register_vexp_node(function_factory)

    def get_root_value(self, apply_session: IApplySession) -> Any:
        raise NotImplementedError()


class OperationsRegistry(RegistryBase):
    NAMESPACE = OperationsNS

    def get_root_value(self, apply_session: IApplySession) -> Any:
        raise RuleInternalError(owner=self, msg=f"This registry should not be used to get root value.")

# ------------------------------------------------------------


class ThisRegistry(RegistryBase):
    NAMESPACE = ThisNS

    def __init__(self, model_class: ModelType): # , functions_factory_registry: FunctionsFactoryRegistry
        super().__init__() # functions_factory_registry=functions_factory_registry)
        self.model_class = model_class
        if not is_model_class(self.model_class):
            raise RuleSetupValueError(owner=self, msg=f"Expected model class (DC/PYD), got: {type(self.model_class)} / {self.model_class} ")

        for attr_name in get_model_fields(self.model_class):
            # th_field: ModelField in .values()
            attr_node = self._create_attr_node_for_model_attr(self.model_class, attr_name)
            self.register_attr_node(attr_node)

    def get_root_value(self, apply_session: IApplySession) -> Any:
        raise NotImplementedError()

# ------------------------------------------------------------


class ContextRegistry(RegistryBase):
    NAMESPACE = ContextNS

    def __init__(self, 
                 context_class: Optional[Type[IContext]], 
                 # functions_factory_registry: FunctionsFactoryRegistry, 
                 ):
        super().__init__() # functions_factory_registry=functions_factory_registry)

        self.context_class = context_class
        if self.context_class:
            self.register_all_nodes()


    def register_all_nodes(self):
        if IContext not in inspect.getmro(self.context_class):
            raise RuleSetupValueError(owner=self, msg=f"Context should inherit IContext, got: {self.context_class}")

        for attr_name in get_model_fields(self.context_class):
            attr_node = self._create_attr_node_for_model_attr(self.context_class, attr_name)
            self.register_attr_node(attr_node)

        for attr_name, py_function in self.context_class.get_vexp_attrname_dict().items():
            type_info = TypeInfo.extract_function_return_type_info(
                            py_function,
                            allow_nonetype=True)
            if attr_name in self.store:
                raise RuleSetupNameError(f"Attribute name '{attr_name}' is reserved. Rename class attribute in '{self.context_class}'")
            attr_node = AttrVexpNode(
                            name=attr_name,
                            data=type_info,
                            namespace=self.NAMESPACE,
                            type_info=type_info, 
                            # TOO: the type or name of th_field is not ok
                            th_field=py_function, 
                            )
            self.register_attr_node(attr_node, attr_name)

    def get_root_value(self, apply_session: IApplySession) -> Any:
        context = apply_session.context
        if context in (UNDEFINED, None):
            # avail_names = get_available_names_example(attr_name, children_dict.keys())
            raise RuleApplyNameError(owner=self, msg=f"Attribute '{attr_name}' not found in context '{context}' ({type(context)}).")
        return context

# ------------------------------------------------------------


class ConfigRegistry(RegistryBase):
    NAMESPACE = ConfigNS

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        if not self.config:
            raise RuleInternalError(owner=self, msg="Config is required")

        self.register_all_nodes()

    def register_all_nodes(self):
        if not isinstance(self.config, Config):
            raise RuleInternalError(owner=self, msg=f".config is not Config instance, got: {type(self.config)} / {self.config}")

        config_class = self.config.__class__
        for attr_name in get_model_fields(config_class):
            attr_node = self._create_attr_node_for_model_attr(config_class, attr_name)
            self.register_attr_node(attr_node)

    def get_root_value(self, apply_session: IApplySession) -> Any:
        raise NotImplementedError()

# ------------------------------------------------------------


class Registries(IRegistries):

    def __init__(self, 
            owner:'ContainerBase',  # noqa: F821
            config: Config,
            functions: Optional[List[CustomFunctionFactory]] = None, 
            context_class: Optional[IContext] = None,
            ):
        """
        input param functions are custom_function_factories
        store 
        """
        # self.registries : Dict[str, List[AttrVexpNode]] = {}
        self.owner: 'ContainerBase' = owner  # noqa: F821

        self.name: str = owner.name
        self.config: Config = config
        self.context_class: IContext = context_class

        self.functions_factory_registry: FunctionsFactoryRegistry = \
                FunctionsFactoryRegistry(functions=functions, 
                                         include_standard=self.owner.is_top_owner())

        self.models_registry: IRegistry = ModelsRegistry() # self.functions_factory_registry)
        self.data_registry: IRegistry = DataRegistry() # self.functions_factory_registry)
        self.fields_registry: IRegistry = FieldsRegistry() # self.functions_factory_registry)
        self.functions_registry: IRegistry = FunctionsRegistry() # self.functions_factory_registry)
        # internal
        self.operations_registry: IRegistry = OperationsRegistry()
        # NOTE: could consider having the same registry for Rules and all Extensions?
        #       but maybe is not needed.
        self.context_registry: IRegistry = ContextRegistry(context_class=self.context_class)
        self.config_registry: IRegistry = ConfigRegistry(config=self.config)


        self.registries: Dict[str, IRegistry] = {
            registry.NAMESPACE._name  : registry
            for registry in [
                self.models_registry,
                self.data_registry,
                self.fields_registry,
                self.functions_registry,
                self.context_registry,
                self.config_registry,
                self.operations_registry,
            ]} 

        self.finished: bool = False

    def create_this_registry(self, model_class: ModelType):
        this_registry = ThisRegistry(
                model_class=model_class,
                # functions_factory_registry=self.functions_factory_registry
                )
        return this_registry

    # ------------------------------------------------------------

    def __str__(self):
        counts = ", ".join([f"{k}={v.count()}" for k, v in self.registries.items() if v])
        # name={self.name},
        # cnt={self.entries_count}, 
        return f"Registries(owner={self.owner}, {counts})"

    def __repr__(self):
        return str(self)

    # ------------------------------------------------------------

    def get_registry(self, namespace:Namespace):
        return self.registries[namespace._name]

    # ------------------------------------------------------------

    def dump_all(self, with_functions:bool = False) -> None:
        # recursive: bool = False, depth: int = 0, 
        # has {self.entries_count} attr_node(s), 
        print(f"{self.owner}: Registries '{self.name}', finished={self.finished}. List:")
        for ns_name, store in self.registries.items():
            store.pp()

        if with_functions:
            self.functions_factory_registry.dump_all()

    # ------------------------------------------------------------

    def _register_attr_node(self, attr_node:AttrVexpNode, alt_attr_node_name=None):
        """
        !!!!! NOTE helper method - USED ONLY IN UNIT TESTING !!!!!
        """
        assert not self.finished
        ns_name = attr_node.namespace._name
        if ns_name not in self.registries:
            raise RuleInternalError(f"{ns_name} not in .registries, available: {self.registries.keys()}")
        self.registries[ns_name].register_attr_node(attr_node, alt_attr_node_name=alt_attr_node_name)

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

    def finish(self):
        if self.finished:
            raise RuleSetupError(owner=self, msg="Method finish() already called.")
        for ns, registry in self.registries.items():
            for vname, vexp_node in registry.items():
                # do some basic validate
                if isinstance(vexp_node, AttrVexpNode):
                    vexp_node.finish()
                else:
                    assert not hasattr(vexp_node, "finish"), vexp_node
            registry.finish()

        self.finished = True


# ------------------------------------------------------------
# OBSOLETE
# ------------------------------------------------------------
# if vname!=vexp_node.name:
#     found = [vexp_node_name for registries_name, ns, vexp_node_name in vexp_node.bound_list if vname==vexp_node_name]
#     if not found:
#         raise RuleInternalError(owner=self, msg=f"Attribute name not the same as stored in registries {vexp_node.name}!={vname} or bound list: {vexp_node.bound_list}")
