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
        get_available_names_example,
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
        is_model_class,
        ModelType,
        # get_model_fields,
        extract_py_type_hints,
        EmptyFunctionArguments,
        )
from .expressions import (
        DotExpression,
        ISetupSession,
        ExecResult,
        )
from .attr_nodes import (
        AttrDexpNode,
        )
from .functions import (
        CustomFunctionFactory,
        IFunction,
        )
from .base import (
    get_name_from_bind,
    BoundModelBase,
    SetupStackFrame,
    IApplyResult,
    ApplyStackFrame,
        )

# ------------------------------------------------------------

@dataclass
class ModelWithHandlers:
    name: str
    in_model: bool 
    read_handler_dexp: IFunction = field(repr=False)
    type_info: TypeInfo = field(repr=False)


class NestedBoundModelMixin:

    def _register_nested_models(self, setup_session:ISetupSession):
        # ALT: self.get_children()
        if not self.contains:
            return False

        container_parent = self.parent.get_first_parent_container(consider_self=True)

        if not container_parent or not container_parent.is_top_parent():
            raise EntitySetupValueError(owner=self, msg=f"Currently child bound models ('contains') supported only for top contaainers owners (i.e. Entity), got: {self.parent} / {container_parent}")

        if self.models_with_handlers_dict:
            raise EntityInternalError(owner=self, msg=f"models_with_handlers_dict should be empty, got: {to_repr(self.models_with_handlers_dict)}. Have you called setup for 2nd time?") 

        # model_fields = get_model_fields(self.model)
        parent_py_type_hints = extract_py_type_hints(self.model, f"{self}")

        models_registry = setup_session.get_registry(ModelsNS)

        # TODO: currently validatiojn of function argument types is done only in ApplyStackFrame() in apply(), 
        #       but should be here used for check attrs in setup() phase ... Define here: 
        #           self.local_setup_session = setup_session.create_local_setup_session(...(self.model))
        #       later reuse it and use it here to check func args types.
        for child_bound_model in self.contains:

            if not isinstance(child_bound_model, BoundModelWithHandlers):
                raise EntitySetupValueError(owner=self, msg=f"Child bound model should be BoundModelWithHandlers, got: {child_bound_model.name} : {type(child_bound_model)}")

            model_name = child_bound_model.name
            if model_name in self.models_with_handlers_dict:
                raise EntitySetupValueError(owner=self, msg=f"Child bound model should be unique, got duplicate name: {model_name}")

            # field = model_fields.get(model_name, None)
            field_py_type_hint = parent_py_type_hints.get(model_name, None)
            read_handler_type_info = child_bound_model.read_handler.get_type_info()

            if not field_py_type_hint:
                if child_bound_model.in_model:
                    raise EntitySetupValueError(owner=self, msg=f"Child bound model `{model_name}` not found in model. Choose existing model attribute name or use `in_model=False` property.")
            else:
                if not child_bound_model.in_model:
                    raise EntitySetupValueError(owner=self, msg=f"Child bound model `{model_name}` is marked with `in_model=True`, but field already exists. Unset property or use another model name.")

                field_type_info = TypeInfo.get_or_create_by_type(field_py_type_hint, caller=f"{self} ==> {model_name}")

                type_err_msg = field_type_info.check_compatible(read_handler_type_info)
                if type_err_msg:
                    raise EntitySetupValueError(owner=self, msg=f"Child bound model `{model_name}` is not compatible with underlying field: {type_err_msg}")

            # 1. if it is non-model -> Register new attribute node within M. /
            #    ModelsNS registry
            if not child_bound_model.in_model:
                model_attr_dexp_node = AttrDexpNode(
                                            name=child_bound_model.name,
                                            data=read_handler_type_info,
                                            namespace=models_registry.NAMESPACE,
                                            type_info=read_handler_type_info, 
                                            th_field=None,
                                            )

                models_registry.register_attr_node(attr_node=model_attr_dexp_node)

            # 2. Create function object and register that read handlers (with
            #    type_info) in local registry - will be called in apply phase
            assert read_handler_type_info

            with setup_session.use_stack_frame(
                    SetupStackFrame(
                        container = setup_session.current_frame.container,
                        component = self, 
                        # bound_model_type_info=read_handler_type_info,
                    )):
                read_handler_dexp = child_bound_model.read_handler.create_function(
                                        func_args  = EmptyFunctionArguments,
                                        setup_session = setup_session,
                                        name       = f"{child_bound_model.name}__{child_bound_model.read_handler.name}")

            read_handler_dexp.finish()

            model_with_handlers = ModelWithHandlers(
                        name=model_name,
                        in_model=child_bound_model.in_model,
                        read_handler_dexp=read_handler_dexp,
                        type_info=read_handler_type_info,
                        )

            self.models_with_handlers_dict[model_name] = model_with_handlers


    # ------------------------------------------------------------


    def _apply_nested_models(self, apply_session: IApplyResult, instance: ModelType):
        children_bound_models = self.get_children()
        if not children_bound_models:
            return 

        if not isinstance(apply_session, IApplyResult):
            raise EntityInternalError(owner=self, msg=f"apply_session is not IApplySession, got: {apply_session}")

        if not isinstance(instance, self.model):
            raise EntityInternalError(owner=self, msg=f"Type of instance is not '{self.model}', got: {to_repr(instance)}")

        children_bound_models_dict = {
                child_bound_model.name: child_bound_model 
                for child_bound_model in children_bound_models
                }

        local_setup_session = apply_session.setup_session \
                                .create_local_setup_session_for_this_instance(
                                        model_class=self.model,
                                        )

        container = self.get_first_parent_container(consider_self=False)

        with apply_session.use_stack_frame(
                ApplyStackFrame(
                    container = container, 
                    component = self, 
                    instance = instance,
                    instance_new = None,
                    local_setup_session=local_setup_session,
                    )):

            for model_name, model_with_handler in self.models_with_handlers_dict.items():
                model_with_handler: ModelWithHandlers = model_with_handler

                # TODO: warn: 
                #   current_value = getattr(instance, model_with_handler.name, UNDEFINED)
                #   if current_value is UNDEFINED and model_with_handler.in_model:
                #   elif current_value is not UNDEFINED and not model_with_handler.in_model:

                # NOTE: can check 'model_with_handler.type_info'

                rh_dexp_result = model_with_handler.read_handler_dexp.execute_node(
                                    apply_session=apply_session, 
                                    dexp_result=ExecResult(),
                                    prev_node_type_info=None,
                                    is_last=True)

                child_instances = rh_dexp_result.value

                # apply_session.config.logger.warn(f"set bound model read_handler to instance: {to_repr(instance)}.{model_with_handler.name} = {to_repr(rh_dexp_result.value)}")
                setattr(instance, model_with_handler.name, child_instances)

                if child_instances:
                    # TODO: if expected list and result not list then convert to list and vice versa
                    child_bound_model = children_bound_models_dict.get(model_name, None)

                    if not child_bound_model:
                        names_avail = get_available_names_example(model_name, children_bound_models_dict.keys())
                        raise EntityInternalError(owner=self, msg=f"Child bound model '{model_name}' not found, available: {names_avail}")

                    # ------------------------------------------------------------
                    # RECURSION
                    # ------------------------------------------------------------
                    if not isinstance(child_instances, (list, tuple)):
                        child_instances = [child_instances]

                    for child_instance in child_instances:
                        child_bound_model._apply_nested_models(
                                                apply_session=apply_session, 
                                                instance=child_instance)


# ------------------------------------------------------------
# BoundModelWithHandlers
# ------------------------------------------------------------

@dataclass
class BoundModelWithHandlers(NestedBoundModelMixin, BoundModelBase):
    # return type of this function is used as model
    read_handler : CustomFunctionFactory

    name         : Optional[str] = field(default=None)
    title        : Optional[str] = field(default=None, repr=False)
    in_model     : bool = field(default=True)
    contains     : Optional[List[Self]] = field(repr=False, default_factory=list)

    # --- evaluated later
    # Filled from from .read_hanlder -> (.type_info: TypeInfo).type_
    model         : ModelType = field(init=False, metadata={"skip_traverse": True})
    parent        : Union[BoundModelBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    parent_name   : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

    type_info     : Union[TypeInfo, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    models_with_handlers_dict : Dict[str, ModelWithHandlers] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):

        if not isinstance(self.read_handler, CustomFunctionFactory):
            raise EntitySetupValueError(owner=self, msg=f"read_handler={self.read_handler} should be instance of CustomFunctionFactory. Maybe wrap plain python function with Function(? ")
        # if not isinstance(self.save_handler, CustomFunctionFactory):
        #     raise EntitySetupValueError(owner=self, msg=f"save_handler={self.save_handler} should be instance of CustomFunctionFactory")

        # ------------------------------------------------------------
        # TODO: do it better 
        # ------------------------------------------------------------
        self.type_info = self.read_handler.get_type_info() # factory
        self.model = self.type_info.type_

        if not is_model_class(self.model):
            raise EntitySetupValueError(f"Model got from read_handler output type - should not be Model class (DC/PYD), got: {self.model}")

        if not self.name:
            self.name = "__".join([
                camel_case_to_snake(self.__class__.__name__),
                get_name_from_bind(self.model)
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
                raise EntitySetupValueError(owner=self, msg=f"BoundModel* nesting (attribute 'contains') is not supported for '{type(container)}'")

        self._register_nested_models(setup_session)

        self._finished = True


    def get_type_info(self):
        assert self.type_info
        return self.type_info

# ------------------------------------------------------------
# BoundModel
# ------------------------------------------------------------

@dataclass
class BoundModel(NestedBoundModelMixin, BoundModelBase):

    # setup must be called first for this component, and later for others
    # bigger comes first, 0 is DotExpression default, 1 is for other copmonents default
    SETUP_PRIORITY  : ClassVar[int] = 9

    model           : Union[ModelType, DotExpression] = field(repr=False)

    name            : Optional[str] = field(default=None)
    contains        : Optional[List[BoundModelWithHandlers]] = field(repr=False, default_factory=list)

    # title           : TransMessageType

    # evaluated later
    parent          : Union[BoundModelBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    parent_name     : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

    # Filled from from model
    type_info : Optional[TypeInfo] = field(init=False, default=None, repr=False)
    models_with_handlers_dict : Dict[str, ModelWithHandlers] = field(init=False, repr=False, default_factory=dict)


    def __post_init__(self):
        if isinstance(self.model, DotExpression):
            if not self.name:
                self.name = "__".join([
                    camel_case_to_snake(self.__class__.__name__),
                    get_name_from_bind(self.model)
                    ])
        elif is_model_class(self.model):
            if not self.name:
                self.name = "__".join([
                        camel_case_to_snake(self.__class__.__name__),
                        camel_case_to_snake(self.model.__name__),
                        ])
        else:
            # Similar check is done later in container too
            raise EntitySetupValueError(owner=self, 
                    msg=f"For 'model' argument expected model class or DotExpression, got: {self.model}")


    def get_type_info(self):
        if not self.type_info:
            self._set_type_info()
        return self.type_info


    def _set_type_info(self):
        # NOTE: model: DotExpression - would be hard to fill automatically
        #           when DotExpression, dexp is evaluated setup() what is a bit late in
        #           container.setup().
        assert not self.type_info
        if not (is_model_class(self.model) or isinstance(self.model, DotExpression)):
            raise EntitySetupValueError(f"Model should be Model class (DC/PYD) or DotExpression, got: {self.model}")

        if isinstance(self.model, DotExpression):
            self.type_info = self.model._evaluator.last_node().type_info
        else:
            self.type_info = TypeInfo.get_or_create_by_type(
                                    py_type_hint=self.model,
                                    caller=self,
                                    )

    def setup(self, setup_session:ISetupSession):
        super().setup(setup_session=setup_session)
        if not self.type_info:
            self._set_type_info()

        self._register_nested_models(setup_session)

        self._finished = True




# ------------------------------------------------------------
# BoundModelHandler
# ------------------------------------------------------------

# @dataclass
# class BoundModelHandler(EntityHandlerFunction):
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
