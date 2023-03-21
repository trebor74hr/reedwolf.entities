from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import (
        Callable, 
        Any,
        Dict,
        )
from dataclasses import dataclass

from .exceptions import (
        RuleApplyError,
        RuleValidationError,
        RuleInternalError,
        )
from .meta import (
        UNDEFINED,
        NoneType,
        TypeInfo,
        )
from .base import (
        ComponentBase,
        IRegistries,
        IApplySession,
        StackFrame,
        ValidationFailure,
        )
from .expressions import (
        VexpResult,
        ValueExpression,
        )
from .fields import (
        FieldBase,
        )
from .components import (
        EvaluationBase,
        ValidationBase,
        )
from .containers import (
        ContainerBase,
        Rules,
        )


MAX_RECURSIONS = 30

@dataclass
class ApplyResult(IApplySession):

    def __post_init__(self):
        if not isinstance(self.rules, Rules):
            raise RuleApplyError(owner=self, msg=f"Component object '{self.rules}' is not top container - Rules.")

        if self.component_name_only:
            # Will raise if component is not found
            self.component_only = self.rules.get_component(self.component_name_only)
            if not self.component_only.can_apply_partial():
                raise RuleApplyError(owner=self, msg=f"Component '{self.component_only}' does not support partial apply. Use Extension, FieldGroup or similar.")

        self.bound_model = getattr(self.rules, "bound_model")
        if not self.bound_model:
            raise RuleApplyError(owner=self, msg=f"Component object '{self.rules}' has no bound model")

        self.model = self.bound_model.model
        if not self.model:
            raise RuleInternalError(owner=self, item=component, msg=f"Bound model '{self.bound_model}' has empty model.")

        if not isinstance(self.instance, self.model):
            raise RuleApplyError(owner=self, msg=f"Object '{self.instance}' is not instance of bound model '{self.model}'.")

        if self.rules.context_class:
            if not self.context:
                raise RuleApplyError(owner=self, msg=f"Pass context object, instance of context class '{component.context_class}'.")
            if not isinstance(self.context, component.context_class):
                raise RuleApplyError(owner=self, msg=f"Context object '{self.context}' is not instance of context class '{component.context_class}'.")
        else:
            if self.context:
                raise RuleApplyError(owner=self, msg=f"Given context object '{self.context}', but context class in component is not setup. Provide 'context_class' to Rules object and try again.")

        # ----------------------------------------
        # see IApplySession for description 
        self.binary_operations_type_adapters[(str, int)] = str


    def is_ok(self) -> bool:
        return bool(self.finished) and not bool(self.errors)

    # ------------------------------------------------------------

    def register_initial_from_bind(self, component: ComponentBase) -> VexpResult:
        init_bind_vexp_result = component.get_vexp_result_from_instance(apply_session=self)
        init_value = init_bind_vexp_result.value

        self.register_instance_attr_change(
                component = component, 
                vexp_result = init_bind_vexp_result, 
                new_value = init_value,
                is_from_init_bind = True)

        if isinstance(component, FieldBase):
            new_value = component.try_adapt_value(init_value)
            if (init_value != new_value):
                self.register_instance_attr_change(
                        component=component, 
                        # how to mark init -> adaptaion change
                        vexp_result=None,
                        new_value=new_value
                        )

        return init_bind_vexp_result

    # ------------------------------------------------------------

    def raise_if_failed(self):
        if not self.finished:
            raise RuleApplyError(owner=self, msg=f"Apply process is not finished")

        if self.errors:
            raise RuleValidationError(owner=self.rules, errors=self.errors)

    # ------------------------------------------------------------

    def execute_evaluation(self, component: ComponentBase, evaluation:EvaluationBase) -> VexpResult:
        """ Execute evaluation and if new value is different from existing
            value, update current instance """
        # TODO: this is funny how much "evaluate.." names are referenced/used
        #       problem is that two terms have similar naming: 
        #           1. class Evaluation (as cleaner) 
        #           2. .evaluate() method as verb which calculates something
        #              (and _evaluator which executes this)

        assert component == self.current_frame.component

        # evaluation_vexp = evaluation.value
        # assert isinstance(evaluation_vexp, ValueExpression)
        # eval_vexp_result  = evaluation_vexp._evaluator.evaluate(apply_session=self)
        eval_vexp_result  = evaluation.evaluate(apply_session=self)

        eval_value = eval_vexp_result.value

        if isinstance(component, FieldBase):
            eval_value = component.try_adapt_value(eval_value)

        # ALT: bind_vexp_result = component.get_vexp_result_from_instance(apply_session)
        orig_value = component.get_current_value_from_history(apply_session=self)

        if (orig_value != eval_value):
            self.register_instance_attr_change(
                    component=component, 
                    vexp_result=eval_vexp_result,
                    new_value=eval_value
                    )

        return eval_vexp_result

    # ------------------------------------------------------------

    # value: Any, 
    def execute_validation(self, component: ComponentBase, validation:ValidationBase) -> Optional[ValidationFailure]:
        """ Execute validaion - if returns False value then register error and
            mark component and children invalid to prevent further rules execution
        """
        assert component == self.current_frame.component

        # value=value, 
        validation_failure = validation.validate(apply_session=self)
        if validation_failure:
            self.register_instance_validation_failed(component, validation_failure)

        return validation_failure

    # ------------------------------------------------------------

    def _apply(self, 
               parent: Optional[ComponentBase], 
               component: ComponentBase, 
               in_component_only_tree: bool = False,
               depth:int=0, 
               extension_list_mode:bool = False, 
               ):
        assert not self.finished

        if self.component_only:
            if component == self.component_only:
                if not extension_list_mode and in_component_only_tree:
                    raise RuleInternalError(owner=self, msg=f"{parent} -> {component}: in_component_only_tree should be False, got {in_component_only_tree}")
                in_component_only_tree = True

        # print ("here", in_component_only_tree, component.name, component == self.component_only)

        if depth > MAX_RECURSIONS:
            raise RecursionError("Maximum recursion depth exceeded ({depth})")

        new_frame = None
        if depth==0:
            assert parent is None
            assert component == self.rules
            instance = self.instance
            new_frame = StackFrame(
                            container=component, 
                            component=component, 
                            instance=instance)
            self.push_frame_to_stack(new_frame)

        elif not extension_list_mode and component.is_extension():
            assert isinstance(component.bound_model.model, ValueExpression), \
                    component.bound_model.model

            vexp_result: VexpResult = \
                                component \
                                .bound_model \
                                .model \
                                ._evaluator.evaluate(apply_session=self)
            instance = vexp_result.value

            if isinstance(instance, (list, tuple)):
                instance_list = instance
                # TODO: validate cardinality before or after changes

                # Apply for all items
                for index0, instance in enumerate(instance_list, 0):
                    item_frame = StackFrame(
                                        container=component,
                                        component=component, 
                                        instance=instance, 
                                        index0=index0)
                    self.push_frame_to_stack(item_frame)

                    # Recursion with prevention to hit this code again
                    self._apply(parent=parent, 
                                component=component, 
                                in_component_only_tree=in_component_only_tree,
                                depth=depth+1,
                                # prevent is_extension_logic again -> infinitive recursion
                                extension_list_mode=True)

                    frame_popped = self.pop_frame_from_stack()
                    assert frame_popped == item_frame

                return
                # --------------------

            # single item extension
            new_frame = StackFrame(
                            container=component, 
                            component=component, 
                            instance=instance)
            self.push_frame_to_stack(new_frame)

        # ------------------------------------------------------------

        if not new_frame:
            # register non-container frame - only component is new. take instance from previous frame
            new_frame = StackFrame(
                            container = self.current_frame.container, 
                            component = component, 
                            instance = self.current_frame.instance,
                            index0 = self.current_frame.index0)
            self.push_frame_to_stack(new_frame)



        if not (self.component_only and not in_component_only_tree):
            self.call_cleaners(component=component)

        # ------------------------------------------------------------
        # REMOVE_THIS: used only for test if all Vexp values could evaluate ...
        if True:
            for attr_name, attr_value in vars(component).items():
                if isinstance(attr_value, ValueExpression):
                    vexp_result: VexpResult = \
                            attr_value._evaluator.evaluate(
                                    apply_session=self)
                    # print(f"{parent.name if parent else ''}.{component.name}.{attr_name} = VExp[{attr_value}] -> {vexp_result}")

        # ------------------------------------------------------------
        # --- Recursive walk down - for each child call _apply
        # ------------------------------------------------------------
        for child in component.get_children():
            self._apply(parent=component, 
                        in_component_only_tree=in_component_only_tree,
                        component=child, 
                        depth=depth+1)

        # -- remove frame if new one is set
        assert new_frame
        frame_popped = self.pop_frame_from_stack()
        assert frame_popped == new_frame

        if depth==0:
            assert len(self.frames_stack)==0

        # print(f"depth={depth}, comp={component.name}, bind={bind} => {vexp_result}")

        return

    # ------------------------------------------------------------

    def apply(self) -> ApplyResult:
        """
        Main function for parsing, validating and evaluating input instance.
        Returns ApplyResult object.

        if all ok - Result.instance contains a new instance (clone + update) of the bound model type 
        if not ok - errors contain all details.

        """
        self._apply(parent=None, component=self.rules)
        self.set_finished()

        return self

    # ------------------------------------------------------------

    def call_cleaners(self, component: ComponentBase):
        # ------------------------------------------------------------
        # --- call cleaners - validations and evaluations
        # ------------------------------------------------------------
        # Fill initial value from instance

        # TODO: if isinstance(component, FieldBase):
        init_bind_vexp_result = self.register_initial_from_bind(component) if getattr(component, "bind", None) else None

        # TODO: provide last value to all evaluations and validations 
        #       but be careful with vexp_result.value - it coluld be unadapted
        #       see: field.try_adapt_value(eval_value)

        if component.cleaners:
            for cleaner in component.cleaners:
                if isinstance(cleaner, ValidationBase):
                    # value=value, 
                    self.execute_validation(component=component, validation=cleaner)
                elif isinstance(cleaner, EvaluationBase):
                    if not init_bind_vexp_result:
                        # TODO: this belongs to Setup phase
                        raise RuleApplyError(owner=self, msg=f"Evaluator can be defined only for components with 'bind' defined. Remove 'Evaluation' or define 'bind'.")
                    self.execute_evaluation(component=component, evaluation=cleaner)
                else:
                    raise RuleApplyError(owner=self, msg=f"Unknown cleaner type {type(cleeaner)}. Expected Evaluation or Validation.")

        # NOTE: initial value from instance is not checked - only
        #       intermediate and the last value
        self.validate_type(component)

