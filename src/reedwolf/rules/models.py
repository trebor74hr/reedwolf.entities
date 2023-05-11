from typing       import (
        Optional, 
        List, 
        Union,
        Dict,
        )
from dataclasses  import (
        dataclass, 
        field,
        )
from .utils import (
        UNDEFINED,
        UndefinedType,
        )
from .exceptions import (
        RuleSetupValueError,
        )
from .meta import (
        TypeInfo,
        is_model_class,
        ModelType,
        get_model_fields,
        )
from .base        import (
        BoundModelBase,
        )
from .expressions import (
        ValueExpression,
        IRegistries,
        )
from .functions import (
        CustomFunctionFactory,
        )


# ------------------------------------------------------------

@dataclass
class ModelWithHandlers:
    name: str
    in_model: bool 
    read_handler: CustomFunctionFactory = field(repr=False)
    type_info: TypeInfo = field(repr=False)


# ------------------------------------------------------------
# BoundModel
# ------------------------------------------------------------

@dataclass
class BoundModel(BoundModelBase):
    name            : str
    # label           : TransMessageType

    model           : Union[ModelType, ValueExpression] = field(repr=False)
    contains        : Optional[List[BoundModelBase]] = field(repr=False, default_factory=list)

    # evaluated later
    owner           : Union[BoundModelBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    owner_name      : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)
    # Filled from from model
    type_info : Optional[TypeInfo] = field(init=False, default=None, repr=False)

    models_with_handlers_dict : Dict[str, ModelWithHandlers] = field(init=False, default_factory=dict)
    

    def get_type_info(self):
        if not self.type_info:
            self._set_type_info()
        return self.type_info

    def _set_type_info(self):
        # NOTE: model: ValueExpression - would be hard to fill automatically
        #           when ValueExpression, vexp is evaluated setup() what is a bit late in
        #           container.setup().
        assert not self.type_info
        if not (is_model_class(self.model) or isinstance(self.model, ValueExpression)):
            raise RuleSetupValueError(f"Model should be Model class (DC/PYD) or ValueExpression, got: {self.model}")

        if isinstance(self.model, ValueExpression):
            self.type_info = self.model._evaluator.last_node().type_info
        else:
            self.type_info = TypeInfo.get_or_create_by_type(
                                    py_type_hint=self.model,
                                    )

    def setup(self, registries:IRegistries):
        super().setup(registries=registries)
        if not self.type_info:
            self._set_type_info()

        # ALT: self.get_children()
        if self.contains:
            if not self.owner.is_top_owner:
                raise RuleSetupValueError(owner=self, msg=f"Child bound models supported only for top contaainers, got: {self.contains}")

            assert not self.models_with_handlers_dict

            # TODO: cache this, it is used multiple times ... 
            model_fields = get_model_fields(self.model)

            for child_bound_model in self.contains:
                if not isinstance(child_bound_model, BoundModelWithHandlers):
                    raise RuleSetupValueError(owner=self, msg=f"Child bound model should be BoundModelWithHandlers, got: {BoundModelWithHandlers}")

                model_name = child_bound_model.name
                if model_name in self.models_with_handlers_dict:
                    raise RuleSetupValueError(owner=self, msg=f"Child bound model should be unique, got duplicate name: {model_name}")

                field = model_fields.get(model_name, None)
                read_handler_type_info = child_bound_model.read_handler.get_type_info()

                if not field:
                    if child_bound_model.in_model:
                        raise RuleSetupValueError(owner=self, msg=f"Child bound model `{model_name}` not found in model. Choose existing model attribute name or use `in_model=False` property.")
                else:
                    if not child_bound_model.in_model:
                        raise RuleSetupValueError(owner=self, msg=f"Child bound model `{model_name}` is marked with `in_model=True`, but field already exists. Unset property or use another model name.")
                    field_type_info = TypeInfo.get_or_create_by_type(field.type)

                    type_err_msg = field_type_info.check_compatible(read_handler_type_info)
                    if type_err_msg:
                        raise RuleSetupValueError(owner=self, msg=f"Child bound model `{model_name}` is not compatible with underlying field: {type_err_msg}")

                # NOTE: currently copies from BoundModelWithHandlers, convert to BoundModelWithHandlers reference
                self.models_with_handlers_dict[model_name] = \
                        ModelWithHandlers(
                            name=model_name,
                            in_model=child_bound_model.in_model,
                            read_handler=child_bound_model.read_handler,
                            type_info=read_handler_type_info,
                            )


        self._finished = True


# ------------------------------------------------------------
# BoundModelWithHandlers
# ------------------------------------------------------------

@dataclass
class BoundModelWithHandlers(BoundModelBase):
    # TODO: razdvoji save/read/.../unique check
    name         : str
    label        : str # TransMsg
    # return type is used as model
    read_handler : CustomFunctionFactory
    in_model     : bool = field(default=True)

    # --- evaluated later
    # Filled from from .read_hanlder -> (.type_info: TypeInfo).type_
    model        : ModelType = field(init=False, metadata={"skip_traverse": True})
    owner        : Union[BoundModelBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    owner_name   : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)
    type_info    : Union[TypeInfo, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)


    def __post_init__(self):

        if not isinstance(self.read_handler, CustomFunctionFactory):
            raise RuleSetupValueError(owner=self, msg=f"read_handler={self.read_handler} should be instance of CustomFunctionFactory. Maybe wrap plain python function with Function(? ")
        # if not isinstance(self.save_handler, CustomFunctionFactory):
        #     raise RuleSetupValueError(owner=self, msg=f"save_handler={self.save_handler} should be instance of CustomFunctionFactory")

        # ------------------------------------------------------------
        # TODO: do it better 
        # ------------------------------------------------------------
        self.type_info = self.read_handler.get_type_info() # factory
        self.model = self.type_info.type_

        if not is_model_class(self.model):
            raise RuleSetupValueError(f"Model got from read_handler output type - should not be Model class (DC/PYD), got: {self.model}")

        super().__post_init__()

        # self.read_handler

        # TODO: check read handler params (args)

        #       read_type_info_dict = TypeInfo.extract_function_arguments_type_info_dict(function=self.read_handler.function)
        #       read_params_expected = sorted(self.read_handler.inject_params.keys())
        #       read_params_found = sorted(read_type_info_dict.keys())
        #       if read_params_found != read_params_expected:
        #           raise RuleSetupValueError(owner=self, msg=f"read_handler={self.read_handler} has arguments '{read_params_found}' and expected '{read_params_expected}'. Check function declaration or inject_params.")

    def get_type_info(self):
        assert self.type_info
        return self.type_info

    # def read(self, *args, **kwargs):
    #     return self.fn_read(*args, **kwargs)

# ------------------------------------------------------------
# BoundModelHandler
# ------------------------------------------------------------

# @dataclass
# class BoundModelHandler(RulesHandlerFunction):
#     pass

# def save(self, *args, **kwargs):
#     return self.fn_save(*args, **kwargs)

# check save handler params (args)
# save_type_info_dict = TypeInfo.extract_function_arguments_type_info_dict(function=self.save_handler.function)
# save_params_expected = sorted(list(self.save_handler.inject_params.keys()) + [self.name])
# save_params_found = sorted(save_type_info_dict.keys())
# if save_params_found != save_params_expected:
#     raise RuleSetupValueError(owner=self, msg=f"save_handler={self.save_handler} has arguments '{save_params_found}' and expected '{save_params_expected}'. Check function declaration or inject_params.")

# save_type_info = save_type_info_dict[self.name]

# assert isinstance(py_type_hint, TypeInfo), py_type_hint

# if save_type_info.py_type_hint != self.type_info.py_type_hint:
#     raise RuleSetupValueError(owner=self, msg=f"save_handler={self.save_handler} argument '{self.name}' has type '{save_type_info.py_type_hint}' and expected is same as read handler return type '{self.type_info.py_type_hint}'. Check save or read function declaration.")

# TODO: read() and save() method types matches - problem is
#       read_handler M is vexpr that is not available in this
#       moment - should be checked in setup() method ...
