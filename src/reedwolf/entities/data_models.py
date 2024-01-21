from typing       import (
    Optional,
    List,
    Union,
    Dict,
    ClassVar,
)
from dataclasses  import (
    dataclass,
    field,
)

from .utils import (
    camel_case_to_snake,
    UNDEFINED,
    UndefinedType,
    to_repr,
    get_available_names_example, NOT_APPLIABLE,
)
from .exceptions import (
    EntitySetupValueError,
    EntityInternalError,
)
from .namespaces import (
    ModelsNS,
)
from .meta import (
    Self,
    TypeInfo,
    is_model_klass,
    ModelKlassType,
    # get_model_fields,
    extract_py_type_hints,
    EmptyFunctionArguments, ModelInstanceType,
)
from .expressions import (
    DotExpression,
    ISetupSession,
    ExecResult, IThisRegistry,
)
from .expr_attr_nodes import (
    AttrDexpNode,
)
from .functions import (
    CustomFunctionFactory,
    IFunction,
)
from .base import (
    get_name_from_bind,
    IDataModel,
    SetupStackFrame,
    IApplyResult,
    ApplyStackFrame,
    IUnboundDataModel,
    IBoundDataModel,
)
from .registries import (
    ThisRegistry, UnboundModelsRegistry,
)

# ------------------------------------------------------------

@dataclass
class ModelWithHandlers:
    name: str
    in_model: bool 
    read_handler_dexp: IFunction = field(repr=False)
    type_info: TypeInfo = field(repr=False)


@dataclass
class BoundDataModelBase(IBoundDataModel):

    def create_this_registry(self, setup_session: ISetupSession) -> Optional[IThisRegistry]:
        model_klass = self.model_klass
        if isinstance(self.model_klass, DotExpression):
            if not self.model_klass.IsFinished():
                # container = self.get_first_parent_container(consider_self=True)
                # model_dexp_node: IDotExpressionNode = model.Setup(setup_session=setup_session, owner=container)
                raise EntityInternalError(owner=self, msg=f"{self.model_klass} dot-expression is not finished")

            model_dexp_node = self.model_klass._dexp_node
            model_klass = model_dexp_node.get_type_info().type_

        this_registry = ThisRegistry(model_klass=model_klass)
        return this_registry


    def _register_nested_models(self, setup_session:ISetupSession):
        """
        called only in _setup_phase_one() phase
        """
        # ALT: self.get_children()
        if not self.contains:
            return False

        if self.models_with_handlers_dict:
            # NOTE: call already done in setup_one phase, calls to this function from setup() (phase two) are not allowed
            raise EntityInternalError(owner=self, msg=f"models_with_handlers_dict should be empty, got: {to_repr(self.models_with_handlers_dict)}. Have you called setup for 2nd time?")

        container_parent = self.parent.get_first_parent_container(consider_self=True)
        if not container_parent or not container_parent.is_top_parent():
            raise EntitySetupValueError(owner=self, msg=f"Currently child data models ('contains') supported only for top contaainers owners (i.e. Entity), got: {self.parent} / {container_parent}")

        # model_fields = get_model_fields(self.model_klass )
        parent_py_type_hints = extract_py_type_hints(self.model_klass, f"{self}")

        models_registry = setup_session.get_registry(ModelsNS)

        # TODO: currently validatiojn of function argument types is done only in ApplyStackFrame() in apply(), 
        #       but should be here used for check attrs in setup() phase ... Define here: 
        #           self.this_registry = setup_session.container... create_this_registry(...self.model_klass )
        #       later reuse it and use it here to check func args types.
        for child_data_model in self.contains:

            if not isinstance(child_data_model, DataModelWithHandlers):
                raise EntitySetupValueError(owner=self, msg=f"Child data model should be DataModelWithHandlers, got: {child_data_model.name}: {type(child_data_model)}")

            model_name = child_data_model.name
            if model_name in self.models_with_handlers_dict:
                raise EntitySetupValueError(owner=self, msg=f"Child data model should be unique, got duplicate name: {model_name}")

            # field = model_fields.get(model_name, None)
            field_py_type_hint = parent_py_type_hints.get(model_name, None)
            read_handler_type_info = child_data_model.read_handler.get_type_info()

            if not field_py_type_hint:
                if child_data_model.in_model:
                    raise EntitySetupValueError(owner=self, msg=f"Child data model `{model_name}` not found in model. Choose existing model attribute name or use `in_model=False` property.")
            else:
                if not child_data_model.in_model:
                    raise EntitySetupValueError(owner=self, msg=f"Child data model `{model_name}` is marked with `in_model=True`, but field already exists. Unset property or use another model name.")

                field_type_info = TypeInfo.get_or_create_by_type(field_py_type_hint, caller=f"{self} ==> {model_name}")

                type_err_msg = field_type_info.check_compatible(read_handler_type_info)
                if type_err_msg:
                    raise EntitySetupValueError(owner=self, msg=f"Child data model `{model_name}` is not compatible with underlying field: {type_err_msg}")

            # 1. if it is non-model -> Register new attribute node within M. /
            #    ModelsNS registry
            if not child_data_model.in_model:
                model_attr_dexp_node = AttrDexpNode(
                                            name=child_data_model.name,
                                            data=read_handler_type_info,
                                            namespace=models_registry.NAMESPACE,
                                            type_info=read_handler_type_info, 
                                            type_object=None,
                                            )

                models_registry.register_attr_node(attr_node=model_attr_dexp_node)

            # 2. Create function object and register that read handlers (with
            #    type_info) in local registry - will be called in apply phase
            assert read_handler_type_info

            with setup_session.use_stack_frame(
                    SetupStackFrame(
                        container = setup_session.current_frame.container,
                        component = self, 
                    )):
                read_handler_dexp = child_data_model.read_handler.create_function(
                                        func_args  = EmptyFunctionArguments,
                                        setup_session = setup_session,
                                        name       = f"{child_data_model.name}__{child_data_model.read_handler.name}")

            read_handler_dexp.finish()

            model_with_handlers = ModelWithHandlers(
                        name=model_name,
                        in_model=child_data_model.in_model,
                        read_handler_dexp=read_handler_dexp,
                        type_info=read_handler_type_info,
                        )

            self.models_with_handlers_dict[model_name] = model_with_handlers


    # ------------------------------------------------------------


    def _apply_nested_models(self, apply_result: IApplyResult, instance: ModelInstanceType):
        children_data_models = self.get_children()
        if not children_data_models:
            return 

        if not isinstance(apply_result, IApplyResult):
            raise EntityInternalError(owner=self, msg=f"apply_result is not IApplyResult, got: {apply_result}")

        if not isinstance(instance, self.model_klass):
            raise EntityInternalError(owner=self, msg=f"Type of instance is not '{self.model_klass}', got: {to_repr(instance)}")

        children_data_models_dict = {
                child_data_model.name: child_data_model 
                for child_data_model in children_data_models
                }

        # TODO: this is strange, setup this_registry in DataModel.setup()
        #       and then just use self.this_registry
        this_registry = self.get_this_registry()
        if not this_registry:
            raise EntityInternalError(owner=self, msg=f"this_registry is not set")
        # this_registry = apply_result.setup_session.container \
        #                     .create_this_registry_for_model_class(
        #                         setup_session=apply_result.setup_session,
        #                         model_klass=self.model_klass )

        # local_setup_session = apply_result.setup_session \
        #                         .create_local_setup_session_for_this_instance(
        #                                 model_klass=self.model,
        #                                 )

        container = self.get_first_parent_container(consider_self=False)

        with apply_result.use_stack_frame(
                ApplyStackFrame(
                    container = container, 
                    component = self,
                    value_node=NOT_APPLIABLE,
                    instance = instance,
                    instance_new = None,
                    this_registry=this_registry,
                    )):

            for model_name, model_with_handler in self.models_with_handlers_dict.items():
                model_with_handler: ModelWithHandlers = model_with_handler

                # TODO: warn: 
                #   current_value = getattr(instance, model_with_handler.name, UNDEFINED)
                #   if current_value is UNDEFINED and model_with_handler.in_model:
                #   elif current_value is not UNDEFINED and not model_with_handler.in_model:

                # NOTE: can check 'model_with_handler.type_info'

                rh_dexp_result = model_with_handler.read_handler_dexp.execute_node(
                                    apply_result=apply_result, 
                                    dexp_result=ExecResult(),
                                    prev_node_type_info=None,
                                    is_last=True)

                child_instances = rh_dexp_result.value

                # apply_result.settings.logger.warn(f"set data model read_handler to instance: {to_repr(instance)}.{model_with_handler.name} = {to_repr(rh_dexp_result.value)}")
                setattr(instance, model_with_handler.name, child_instances)

                if child_instances:
                    # TODO: if expected list and result not list then convert to list and vice versa
                    child_data_model = children_data_models_dict.get(model_name, None)

                    if not child_data_model:
                        names_avail = get_available_names_example(model_name, children_data_models_dict.keys())
                        raise EntityInternalError(owner=self, msg=f"Child data model '{model_name}' not found, available: {names_avail}")

                    # ------------------------------------------------------------
                    # RECURSION
                    # ------------------------------------------------------------
                    if not isinstance(child_instances, (list, tuple)):
                        child_instances = [child_instances]

                    for child_instance in child_instances:
                        child_data_model._apply_nested_models(
                                                apply_result=apply_result, 
                                                instance=child_instance)

@dataclass
class UnboundModel(IUnboundDataModel):
    """
    This is a dummy class, just to mark unbound mode
    """
    name: Optional[str] = field(default=None, init=False)

    # model_klass: Union[ModelKlassType, UndefinedType] = field(repr=False, init=False, default=UNDEFINED)
    # parent: Union[IDataModel, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    # parent_name: Union[str, UndefinedType] = field(init=False, default=UNDEFINED)
    # type_info: Optional[TypeInfo] = field(init=False, default=None, repr=False)

    def get_type_info(self):
        raise EntityInternalError(owner=self, msg="Method should not be called")

    def create_this_registry(self, setup_session: ISetupSession) -> Optional[IThisRegistry]:
        """
        all referenced nodes should be added as they are referenced
        """
        return UnboundModelsRegistry()

# ------------------------------------------------------------
# DataModelWithHandlers
# ------------------------------------------------------------

@dataclass
class DataModelWithHandlers(BoundDataModelBase):
    # return type of this function is used as model
    read_handler:   CustomFunctionFactory

    name:           Optional[str] = field(default=None)
    title:          Optional[str] = field(default=None, repr=False)
    in_model:       bool = field(default=True)
    contains:       Optional[List[Self]] = field(repr=False, default_factory=list)

    # --- evaluated later
    # Filled from from .read_hanlder -> (.type_info: TypeInfo).type_
    model_klass:          ModelKlassType = field(init=False, metadata={"skip_setup": True})
    parent:         Union[IDataModel, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    parent_name:    Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

    type_info:      Union[TypeInfo, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    models_with_handlers_dict: Dict[str, ModelWithHandlers] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):

        if not isinstance(self.read_handler, CustomFunctionFactory):
            raise EntitySetupValueError(owner=self, msg=f"read_handler={self.read_handler} should be instance of CustomFunctionFactory. Maybe wrap plain python function with Function(? ")
        # if not isinstance(self.save_handler, CustomFunctionFactory):
        #     raise EntitySetupValueError(owner=self, msg=f"save_handler={self.save_handler} should be instance of CustomFunctionFactory")

        # ------------------------------------------------------------
        # TODO: do it better 
        # ------------------------------------------------------------
        self.type_info = self.read_handler.get_type_info() # factory
        self.model_klass = self.type_info.type_

        if not is_model_klass(self.model_klass):
            raise EntitySetupValueError(f"Model got from read_handler output type - should not be Model class (DC/PYD), got: {self.model_klass}")

        if not self.name:
            self.name = "__".join([
                camel_case_to_snake(self.__class__.__name__),
                get_name_from_bind(self.model_klass)
                ])

        super().__post_init__()

        # self.read_handler

        # TODO: check read handler params (args)

        #       read_type_info_dict = TypeInfo.extract_function_arguments_type_info_dict(function=self.read_handler.function)
        #       read_params_expected = sorted(self.read_handler.inject_params.keys())
        #       read_params_found = sorted(read_type_info_dict.keys())
        #       if read_params_found != read_params_expected:
        #           raise EntitySetupValueError(owner=self, msg=f"read_handler={self.read_handler} has arguments '{read_params_found}' and expected '{read_params_expected}'. Check function declaration or inject_params.")


    def setup(self, setup_session:ISetupSession):
        super().setup(setup_session=setup_session)

        if self.contains:
            container =self.get_first_parent_container(consider_self=False)
            if not container.is_top_parent():
                # NOTE: not allowed in SubEntityItems-s for now
                raise EntitySetupValueError(owner=self, msg=f"DataModel* nesting (attribute 'contains') is not supported for '{type(container)}'")

        # self._register_nested_models(setup_session)
        self._finished = True


    def get_type_info(self) -> TypeInfo:
        assert self.type_info
        return self.type_info

# ------------------------------------------------------------
# DataModel
# ------------------------------------------------------------

@dataclass
class DataModel(BoundDataModelBase):

    # setup must be called first for this component, and later for others
    # bigger comes first, 0 is DotExpression default, 1 is for other copmonents default
    SETUP_PRIORITY:     ClassVar[int] = 9

    model_klass:              Union[ModelKlassType, DotExpression] = field(repr=False)

    name:               Optional[str] = field(default=None)
    contains:           Optional[List[Union[DataModelWithHandlers, Self]]] = field(repr=False, default_factory=list)

    # title:            TransMessageType

    # evaluated later
    parent:             Union[IDataModel, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    parent_name:        Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

    # Filled from from model
    type_info:          Optional[TypeInfo] = field(init=False, default=None, repr=False)
    models_with_handlers_dict: Dict[str, ModelWithHandlers] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        if isinstance(self.model_klass, DotExpression):
            if not self.name:
                self.name = "__".join([
                    camel_case_to_snake(self.__class__.__name__),
                    get_name_from_bind(self.model_klass)
                    ])
        elif is_model_klass(self.model_klass):
            if not self.name:
                self.name = "__".join([
                        camel_case_to_snake(self.__class__.__name__),
                        camel_case_to_snake(self.model_klass.__name__),
                        ])
        else:
            # Similar check is done later in container too
            raise EntitySetupValueError(owner=self,
                                        msg=f"For 'model' argument expected model class or DotExpression, got: {self.model_klass}")


    def get_type_info(self) -> TypeInfo:
        if not self.type_info:
            self._set_type_info()
        return self.type_info


    def _set_type_info(self):
        # NOTE: model_klass: DotExpression - would be hard to fill automatically
        #           when DotExpression, dexp is evaluated setup() what is a bit late in
        #           container.setup().
        assert not self.type_info
        if not (is_model_klass(self.model_klass) or isinstance(self.model_klass, DotExpression)):
            raise EntitySetupValueError(f"Model should be Model class (DC/PYD) or DotExpression, got: {self.model_klass}")

        if isinstance(self.model_klass, DotExpression):
            if not self.model_klass.IsFinished():
                raise EntityInternalError(owner=self, msg=f"model not setup: {self.model_klass}")
            self.type_info = self.model_klass._evaluator.last_node().type_info
        else:
            self.type_info = TypeInfo.get_or_create_by_type(
                                    py_type_hint=self.model_klass,
                                    caller=self,
                                    )

    def setup(self, setup_session:ISetupSession):
        super().setup(setup_session=setup_session)
        if not self.type_info:
            self._set_type_info()

        # self._register_nested_models(setup_session)

        self._finished = True




# ------------------------------------------------------------
# DataModelHandler
# ------------------------------------------------------------

# @dataclass
# class DataModelHandler(EntityHandlerFunction):
#     pass

# def save(self, *args, **kwargs):
#     return self.fn_save(*args, **kwargs)

# check save handler params (args)
# save_type_info_dict = TypeInfo.extract_function_arguments_type_info_dict(function=self.save_handler.function)
# save_params_expected = sorted(list(self.save_handler.inject_params.keys()) + [self.name])
# save_params_found = sorted(save_type_info_dict.keys())
# if save_params_found != save_params_expected:
#     raise EntitySetupValueError(owner=self, msg=f"save_handler={self.save_handler} has arguments '{save_params_found}' and expected '{save_params_expected}'. Check function declaration or inject_params.")

# save_type_info = save_type_info_dict[self.name]

# assert isinstance(py_type_hint, TypeInfo), py_type_hint

# if save_type_info.py_type_hint != self.type_info.py_type_hint:
#     raise EntitySetupValueError(owner=self, msg=f"save_handler={self.save_handler} argument '{self.name}' has type '{save_type_info.py_type_hint}' and expected is same as read handler return type '{self.type_info.py_type_hint}'. Check save or read function declaration.")

# TODO: read() and save() method types matches - problem is
#       read_handler M is dexpr that is not available in this
#       moment - should be checked in setup() method ...
