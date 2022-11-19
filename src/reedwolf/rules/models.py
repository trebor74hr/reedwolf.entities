from typing       import Optional, List, Union
from dataclasses  import dataclass, field

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
        )
from .base        import (
        BoundModelBase,
        IRegistries,
        )
from .expressions import (
        ValueExpression,
        )
from .functions import (
        CustomFunctionFactory,
        )

class BoundModelMixin:

    pass

# ------------------------------------------------------------
# BoundModel
# ------------------------------------------------------------

@dataclass
class BoundModel(BoundModelMixin, BoundModelBase):
    name            : str
    # label           : TransMessageType

    model           : Union[ModelType, ValueExpression] = field(repr=False)
    contains        : Optional[List['BoundModel']] = field(repr=False, default_factory=list)

    # evaluated later
    owner           : Union[BoundModelBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    owner_name      : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)
    # Filled from from model
    type_info : Optional[TypeInfo] = field(init=False, default=None, repr=False)


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
        self._finished = True

    # def __post_init__(self):
    #     super().__post_init__()

# ------------------------------------------------------------
# BoundModelWithHandlers
# ------------------------------------------------------------

@dataclass
class BoundModelWithHandlers(BoundModelMixin, BoundModelBase):
    # TODO: razdvoji save/read/.../unique check
    name         : str
    label        : str # TransMsg
    # return type is used as model
    read_handler : CustomFunctionFactory
    # save_handler : Optinoal[CustomFunctionFactory]

    # --- evaluated later
    # Filled from from .read_hanlder -> (.type_info: TypeInfo).type_
    model        : ModelType = field(init=False, metadata={"skip_traverse": True})
    owner        : Union[BoundModelBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    owner_name   : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)
    type_info : Union[TypeInfo, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)


    def __post_init__(self):

        if not isinstance(self.read_handler, CustomFunctionFactory):
            raise RuleSetupValueError(owner=self, msg=f"read_handler={self.read_handler} should be instance of CustomFunctionFactory. Maybe wrap plain python function with Function(? ")
        # if not isinstance(self.save_handler, CustomFunctionFactory):
        #     raise RuleSetupValueError(owner=self, msg=f"save_handler={self.save_handler} should be instance of CustomFunctionFactory")

        # ------------------------------------------------------------
        # TODO: do it better 
        # ------------------------------------------------------------
        self.type_info = self.read_handler.get_output_type_info() # factory
        self.model = self.type_info.type_

        if not is_model_class(self.model):
            raise RuleSetupValueError(f"Model got from read_handler output type - should not be Model class (DC/PYD), got: {self.model}")

        super().__post_init__()

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
#       read_handler M.company is vexpr that is not available in this
#       moment - should be checked in setup() method ...
