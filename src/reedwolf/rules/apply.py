from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import (
        Callable, 
        Any,
        Dict,
        )
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum

from .exceptions import (
        RuleApplyError,
        RuleValidationError,
        RuleInternalError,
        RuleApplyValueError,
        )
from .meta import (
        UNDEFINED,
        NoneType,
        TypeInfo,
        ModelType,
        is_model_class,
        )
from .base import (
        ComponentBase,
        IRegistries,
        IApplySession,
        StackFrame,
        ValidationFailure,
        StructEnum,
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


class UseStackFrame(AbstractContextManager):
    " with() ... custom context manager "
    # ALT: from contextlib import contextmanager
    def __init__(self, apply_session: IApplySession, frame: StackFrame):
        self.apply_session = apply_session
        self.frame = frame

    def __enter__(self):
        self.apply_session.push_frame_to_stack(self.frame)
        return self.frame

    def __exit__(self, exc_type, exc_value, exc_tb):
        frame_popped = self.apply_session.pop_frame_from_stack()
        if not exc_type and frame_popped != self.frame:
            raise RuleInternalError(f"Something wrong with frame stack, got {frame_popped}, expected {self.frame}")



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

        # self.model = self.bound_model.model
        # if not self.model:
        #     raise RuleInternalError(owner=self, item=component, msg=f"Bound model '{self.bound_model}' has empty model.")

        if not isinstance(self.instance, self.bound_model.model):
            raise RuleApplyError(owner=self, msg=f"Object '{self.instance}' is not instance of bound model '{self.bound_model.model}'.")

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

        if self.instance_new is not None and not self.component_name_only:
            self._detect_instance_new_struct_type(self.rules)


    def is_ok(self) -> bool:
        return bool(self.finished) and not bool(self.errors)

    # ------------------------------------------------------------

    def use_stack_frame(self, frame: StackFrame) -> UseStackFrame:
        return UseStackFrame(apply_session=self, frame=frame)

    # ------------------------------------------------------------

    def _apply_bind_vexp(self, component: ComponentBase) -> VexpResult:
        " get initial vexp value, if instance_new try to updated/overwrite with it"

        if not isinstance(component, FieldBase):
            raise RuleInternalError(owner=self, msg=f"Expected FieldBase field, got: {component}")

        bind_vexp_result = component.get_vexp_result_from_instance(apply_session=self)
        init_value = bind_vexp_result.value

        self.register_instance_attr_change(
                component = component, 
                vexp_result = bind_vexp_result, 
                new_value = init_value,
                is_from_init_bind = True)


        # try adapt of initial value changed value?
        last_value = component.try_adapt_value(init_value)

        updated = False
        if self.current_frame.instance_new is not None:
            if self.instance_new_struct_type == StructEnum.MODELS_LIKE:

                with self.use_stack_frame(
                        StackFrame(container=self.current_frame.container, 
                                   component=self.current_frame.component, 
                                   # only this is changed
                                   instance=self.current_frame.instance_new,
                                   instance_new=UNDEFINED, 
                        )):
                    instance_new_bind_vexp_result = \
                            component.get_vexp_result_from_instance(apply_session=self)
                    new_value = instance_new_bind_vexp_result.value

            elif self.instance_new_struct_type == StructEnum.RULES_LIKE:
                raise NotImplementedError()
            else: 
                raise RuleInternalError(owner=self, msg=f"Invalid instance_new_struct_type = {self.instance_new_struct_type}")

            new_value = component.try_adapt_value(new_value)

            # adapted new instance value diff from adapted initial value
            if new_value != last_value:
                self.register_instance_attr_change(
                        component = component, 
                        vexp_result = instance_new_bind_vexp_result, 
                        new_value = new_value,
                        )
                last_value = new_value 
                bind_vexp_result = instance_new_bind_vexp_result
                updated = True

        elif init_value != last_value:
            # diff initial value from adapted
            self.register_instance_attr_change(
                    component=component, 
                    # TODO: how to mark init -> adaptaion change?
                    vexp_result=None,
                    new_value=last_value
                    )

        return bind_vexp_result

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
               # -- recursion - internal props
               # partial apply
               in_component_only_tree: bool = False,
               depth:int=0, 
               extension_list_mode:bool = False, 
               ):
        assert not self.finished

        if self.component_only and component == self.component_only:
            # partial apply - detected component
            if not extension_list_mode and in_component_only_tree:
                raise RuleInternalError(owner=self, 
                        msg=f"{parent} -> {component}: in_component_only_tree should be False, got {in_component_only_tree}")

            if self.instance_new is not None and not self.instance_new_struct_type:
                self._detect_instance_new_struct_type(component)

            # NOTE: self.component_only should be found 
            in_component_only_tree = True


        if depth > MAX_RECURSIONS:
            raise RecursionError("Maximum recursion depth exceeded ({depth})")

        new_frame = None


        if depth==0:
            assert parent is None
            new_frame = StackFrame(
                            container = component, 
                            component = component, 
                            instance = self.instance,
                            instance_new = self.instance_new,
                            )

        elif not extension_list_mode and component.is_extension():
            assert isinstance(component.bound_model.model, ValueExpression), \
                    component.bound_model.model

            # original instance
            vexp_result: VexpResult = component.bound_model.model \
                                        ._evaluator.evaluate(apply_session=self)
            instance = vexp_result.value

            # new instance if any
            current_instance_new = self._get_current_instance_new(component=component,
                                            in_component_only_tree=in_component_only_tree)


            if isinstance(instance, (list, tuple)):
                # == Extension with item List ==

                instance_list = instance

                # TODO: validate cardinality before or after changes

                current_instances_by_key = None
                if current_instance_new is not None:
                    if not isinstance(current_instance_new, (list, tuple)):
                        raise RuleApplyValueError(owner=self, msg=f"{component}: Expected list/tuple in the new instance, got: {current_instance_new}")

                    current_instances_by_key = {}
                    for index0, item_instance_new in enumerate(current_instance_new, 0):
                        key = component.get_key_pairs(item_instance_new) if component.keys else index0
                        if key in current_instances_by_key:
                            raise RuleApplyValueError(owner=self, msg=f"{component}: Duplicate key {key}, first item is: {current_instances_by_key[key]}")
                        current_instances_by_key[key] = item_instance_new

                # Apply for all items
                for index0, instance in enumerate(instance_list, 0):
                    if current_instance_new is not None:
                        key = component.get_key_pairs(instance) if component.keys else index0
                        item_instance_new = current_instances_by_key.get(key, UNDEFINED)
                        if item_instance_new is UNDEFINED:
                            # TODO: deleted -> currently ignore
                            item_instance_new = None
                    else:
                        item_instance_new = None

                    with self.use_stack_frame(
                            StackFrame(
                                container = component,
                                component = component, 
                                instance = instance, 
                                index0 = index0,
                                instance_new = item_instance_new,
                        )):

                        # Recursion with prevention to hit this code again
                        self._apply(parent=parent, 
                                    component=component, 
                                    in_component_only_tree=in_component_only_tree,
                                    depth=depth+1,
                                    # prevent is_extension_logic again -> infinitive recursion
                                    extension_list_mode=True)

                # finished, processed all children
                return
                # --------------------



            # == Extension with single item ==

            if not is_model_class(instance):
                raise RuleApplyValueError(owner=self, msg=f"Expected list/tuple or model type, got: {instance}")


            new_frame = StackFrame(
                            container = component, 
                            component = component, 
                            instance = instance,
                            instance_new = current_instance_new,
                            )

        # ------------------------------------------------------------

        if not new_frame:
            # register non-container frame - only component is new. take instance from previous frame
            new_frame = StackFrame(
                            container = self.current_frame.container, 
                            component = component, 
                            # copy
                            instance = self.current_frame.instance,
                            index0 = self.current_frame.index0,
                            instance_new = self.current_frame.instance_new,
                            )

        with self.use_stack_frame(new_frame):
                # self.push_frame_to_stack(new_frame)

            # obnly when full apply or partial apply
            if not (self.component_only and not in_component_only_tree):
                self._update_and_clean(component=component)

            # ------------------------------------------------------------
            # REMOVE_THIS: used only for test if all Vexp values could evaluate ...
            # if True:
            #     for attr_name, attr_value in vars(component).items():
            #         if isinstance(attr_value, ValueExpression):
            #             vexp_result: VexpResult = \
            #                     attr_value._evaluator.evaluate(
            #                             apply_session=self)
            #             # print(f"{parent.name if parent else ''}.{component.name}.{attr_name} = VExp[{attr_value}] -> {vexp_result}")

            # ------------------------------------------------------------
            # --- Recursive walk down - for each child call _apply
            # ------------------------------------------------------------
            for child in component.get_children():
                self._apply(parent=component, 
                            in_component_only_tree=in_component_only_tree,
                            component=child, 
                            depth=depth+1)


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

    def _detect_instance_new_struct_type(self, component: ComponentBase) -> StructEnum:

        assert self.instance_new is not None

        if isinstance(component, ContainerBase):
            # full or partial on container
            model = component.bound_model.type_info.type_
        else:
            # FieldGroup supported only - partial with matched compoonent
            # raise NotImplementedError(f"TODO: currently not supported: {component}")
            container = component.get_container_owner()
            model = container.bound_model.type_info.type_

        if isinstance(self.instance_new, (list, tuple)):
            if not self.instance_new:
                raise RuleInternalError(owner=self, msg=f"Instance is an empty list, can not detect type of base  structure")
            # test only first
            instance_to_test = self.instance_new[0]
        else:
            instance_to_test = self.instance_new

        if isinstance(instance_to_test, model):
            instance_new_struct_type = StructEnum.MODELS_LIKE
        elif is_model_class(instance_to_test.__class__):
            # TODO: it could be StructEnum.MODELS_LIKE too, but how to detect this? input param or?
            instance_new_struct_type = StructEnum.RULES_LIKE
        else:
            raise RuleApplyError(owner=self, 
                    msg=f"Object '{instance_to_test}' is not instance of bound model '{model}' and not model class: {type(instance_to_test)}.")

        self.instance_new_struct_type = instance_new_struct_type
        # return instance_new_struct_type


    # ------------------------------------------------------------


    def _get_current_instance_new(self, component: ComponentBase, in_component_only_tree:bool):
        if self.instance_new_struct_type is None:
            current_instance_new = None
        elif self.instance_new_struct_type == StructEnum.MODELS_LIKE:
            assert isinstance(component.bound_model.model, ValueExpression), component.bound_model.model

            if self.current_frame.instance_new is not None:

                # if partial - then vexp must know - this value is set only in this case
                if in_component_only_tree and component == self.component_only:
                    # Extension or FieldGroup is root
                    on_component_only = component
                else:
                    on_component_only = None

                # container = component.get_container_owner() if not component.is_extension() else component
                with self.use_stack_frame(
                        StackFrame(
                            container = self.current_frame.container, 
                            component = self.current_frame.component, 
                            # only this is changed
                            instance = self.current_frame.instance_new,
                            # should not be used
                            instance_new = UNDEFINED, 
                            on_component_only=on_component_only,
                        )) as frame:
                    vexp_result: VexpResult = \
                                        component \
                                        .bound_model \
                                        .model \
                                        ._evaluator.evaluate(
                                                apply_session=self, 
                                                )
                # set new value
                current_instance_new = vexp_result.value
                if frame.bound_model_root.type_info.is_list and not isinstance(current_instance_new, (list, tuple)):
                    current_instance_new = [current_instance_new]
            else:
                current_instance_new = None

        elif self.instance_new_struct_type == StructEnum.RULES_LIKE:
            raise NotImplementedError()
        else: 
            raise RuleInternalError(owner=self, msg=f"Invalid instance_new_struct_type = {self.instance_new_struct_type}")

        return current_instance_new

    # ------------------------------------------------------------

    def _update_and_clean(self, component: ComponentBase):
        # ----------------------------------------------------------------
        # 1. init change history by bind of self.instance
        # 2. update value by bind of self.instance_new
        # 3. call cleaners - validations and evaluations in given order
        # ----------------------------------------------------------------

        # Fill initial value from instance and try to update if instance_new is
        # provided and is different
        init_bind_vexp_result = self._apply_bind_vexp(component) \
                                    if getattr(component, "bind", None) else None

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


