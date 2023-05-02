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
from collections import OrderedDict, defaultdict

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
        is_model_instance,
        )
from .base import (
        KeyString,
        ComponentBase,
        IRegistries,
        IApplySession,
        StackFrame,
        ValidationFailure,
        StructEnum,
        ChangeOpEnum,
        InstanceChange,
        get_instance_key_string_attrname_pair,
        )
from .expressions import (
        ExecResult,
        NotAvailableExecResult,
        ValueExpression,
        execute_available_vexp,
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
        MissingKey,
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
                raise RuleApplyError(owner=self.rules, msg=f"Pass context object to .apply*(). Context should be instance of '{self.rules.context_class}'.")
            if not isinstance(self.context, self.rules.context_class):
                raise RuleApplyError(owner=self, msg=f"Context object '{self.context}' is not instance of context class '{self.rules.context_class}'.")
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

    def raise_if_failed(self):
        if not self.finished:
            raise RuleApplyError(owner=self, msg=f"Apply process is not finished")

        if self.errors:
            raise RuleValidationError(owner=self.rules, errors=self.errors)

    # ------------------------------------------------------------

    def execute_evaluation(self, component: ComponentBase, evaluation:EvaluationBase) -> ExecResult:
        """ Execute evaluation and if new value is different from existing
            value, update current instance """
        assert component == self.current_frame.component

        # evaluation_vexp = evaluation.value
        # assert isinstance(evaluation_vexp, ValueExpression)
        # eval_vexp_result  = evaluation_vexp._evaluator.execute(apply_session=self)
        eval_vexp_result  = evaluation.execute(apply_session=self)

        if eval_vexp_result.is_not_available():
            return eval_vexp_result

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

            if self.instance_new not in (None, UNDEFINED) and not self.instance_new_struct_type:
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

            # NOTE: this one has no 'available' attribute

            # original instance
            vexp_result: ExecResult = component.bound_model.model \
                                        ._evaluator.execute_vexp(apply_session=self)
            instance = vexp_result.value

            # new instance if any
            current_instance_new = self._get_current_instance_new(component=component,
                                            in_component_only_tree=in_component_only_tree)


            if isinstance(instance, (list, tuple)):
                # == Extension with item List ==

                instance_list = instance

                # TODO: validate cardinality before or after changes

                new_instances_by_key = None
                if current_instance_new not in (None, UNDEFINED):
                    if not isinstance(current_instance_new, (list, tuple)):
                        raise RuleApplyValueError(owner=self, msg=f"{component}: Expected list/tuple in the new instance, got: {current_instance_new}")

                    new_instances_by_key = {}
                    for index0, item_instance_new in enumerate(current_instance_new, 0):
                        key = component.get_key_pairs_or_index0(item_instance_new, index0)
                        if key in new_instances_by_key:
                            raise RuleApplyValueError(owner=self, msg=f"{component}: Duplicate key {key}, first item is: {new_instances_by_key[key]}")
                        new_instances_by_key[key] = item_instance_new


                # NOTE: considered to use dict() since dictionaries are ordered in Python 3.6+ 
                #       ordering-perserving As of Python 3.7, this is a guaranteed, i.e.  Dict keeps insertion order
                #       https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6
                instances_by_key = OrderedDict()

                for index0, instance in enumerate(instance_list, 0):
                    key = component.get_key_pairs_or_index0(instance, index0)
                    if component.keys:
                        missing_keys = [kn for kn, kv in key if isinstance(kv, MissingKey)]
                        if missing_keys:
                            raise RuleApplyValueError(owner=self, msg=f"Instance {instance} has key(s) with value None, got: {', '.join(missing_keys)}")

                    if current_instance_new not in (None, UNDEFINED):
                        item_instance_new = new_instances_by_key.get(key, UNDEFINED)
                        if item_instance_new is UNDEFINED:
                            key_string = component.get_key_string_by_instance(
                                    apply_session = self, 
                                    instance = instance, 
                                    index0 = index0)

                            self.changes.append(
                                    InstanceChange(
                                        key_string = key_string,
                                        key_pairs = key,
                                        operation = ChangeOpEnum.DELETE,
                                        instance = instance,
                                    ))
                            item_instance_new = None
                    else:
                        item_instance_new = None

                    if key in instances_by_key:
                        raise RuleApplyValueError(owenr=self, msg=f"Found duplicate key {key}:\n  == {instance}\n  == {instances_by_key[key]}")

                    instances_by_key[key] = (instance, index0, item_instance_new)

                
                if new_instances_by_key:
                    index0_new = len(instances_by_key)
                    new_keys = [key for key in new_instances_by_key.keys() if key not in instances_by_key]
                    # register new items and add to processing
                    for key in new_keys:
                        item_instance_new = new_instances_by_key[key]

                        key_string = component.get_key_string_by_instance(
                                apply_session = self, 
                                instance = item_instance_new, 
                                index0 = index0_new)

                        self.changes.append(
                                InstanceChange(
                                    key_string = key_string,
                                    key_pairs = key,
                                    operation = ChangeOpEnum.CREATE,
                                    instance = item_instance_new,
                                ))
                        instances_by_key[key] = (item_instance_new, index0_new, None)
                        index0_new += 1

                for key, (instance, index0, item_instance_new) in instances_by_key.items():
                    # Apply for all items
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

            if not is_model_instance(instance):
                # import pdb;pdb.set_trace() 
                raise RuleApplyValueError(owner=self, msg=f"Expected list/tuple or model instance, got: {instance} : {type(instance)}")


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

        process_further = True

        with self.use_stack_frame(new_frame):
                # self.push_frame_to_stack(new_frame)

            # obnly when full apply or partial apply
            if not (self.component_only and not in_component_only_tree):
                # ============================================================
                # Update, validate, evaluate
                # ============================================================
                process_further = self._update_and_clean(component=component)
                # also if validation fails ...

                if process_further and getattr(component, "enables", None):
                    assert not getattr(component, "contains", None), component
                    bind_vexp_result = component.get_vexp_result_from_instance(apply_session=self)
                    value = bind_vexp_result.value
                    if not isinstance(value, (bool, NoneType)):
                        raise RuleApplyValueError(owner=component, 
                                msg=f"Component.enables can be applied only for Boolean or None values, got: {value} : {type(value)}")
                    if not value: 
                        process_further = False


            # ------------------------------------------------------------
            # NOTE: used only for test if all Vexp values could evaluate ...
            #       self.__check_component_all_vexps(component)

            # ------------------------------------------------------------
            # --- Recursive walk down - for each child call _apply
            # ------------------------------------------------------------
            if process_further:
                for child in component.get_children():
                    self._apply(parent=component, 
                                in_component_only_tree=in_component_only_tree,
                                component=child, 
                                depth=depth+1)


        if depth==0:
            assert len(self.frames_stack)==0

            # ---------------------------------------------------------------------
            # Check which attributes are changed and collect original + new value
            # ---------------------------------------------------------------------
            updated_values_dict: Dict[KeyString, Dict[AttrName, Tuple[AttrValue, AttrValue]] ] = defaultdict(dict)
            for key_string, instance_attr_values in self.update_history.items():
                if len(instance_attr_values)>1:
                    first_val = instance_attr_values[0]
                    last_val = instance_attr_values[-1]
                    if last_val.value != first_val.value:
                        instance_key_string, attrname = \
                                get_instance_key_string_attrname_pair(key_string)
                        updated_values_dict[instance_key_string][attrname] = (first_val.value, last_val.value)

            for instance_key_string, updated_values in updated_values_dict.items():
                instance_updated = self.get_instance_by_key_string(instance_key_string)
                # TODO: key_pairs are currently not set - should be cached, not so easy to get in this moment
                #   key = component.get_key_pairs_or_index0(instance_updated, index0)
                self.changes.append(
                        InstanceChange(
                            key_string = instance_key_string,
                            # see above
                            key_pairs = None,
                            operation = ChangeOpEnum.UPDATE,
                            instance = instance_updated,
                            updated_values = updated_values,
                        ))


        # print(f"depth={depth}, comp={component.name}, bind={bind} => {vexp_result}")

        return

    # ------------------------------------------------------------

    def __check_component_all_vexps(self, component: ComponentBase):
        # used only for testing
        for attr_name, attr_value in vars(component).items():
            if isinstance(attr_value, ValueExpression):
                vexp_result: ExecResult = \
                        attr_value._evaluator.execute_vexp(
                                apply_session=self)
                # print(f"{parent.name if parent else ''}.{component.name}.{attr_name} = VExp[{attr_value}] -> {vexp_result}")


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

    def get_instance_by_key_string(self, key_string: str) -> ModelType:
        " must exist in cache - see previous method which sets Container.get_key_string_by_instance "
        return self.instance_by_key_string_cache[key_string]

    # ------------------------------------------------------------

    def _detect_instance_new_struct_type(self, component: ComponentBase) -> StructEnum:

        assert self.instance_new not in (None, UNDEFINED)

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

            if self.current_frame.instance_new not in (None, UNDEFINED):

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
                    vexp_result: ExecResult = \
                                        component \
                                        .bound_model \
                                        .model \
                                        ._evaluator.execute_vexp(
                                                apply_session=self, 
                                                )
                # set new value
                current_instance_new = vexp_result.value
                if frame.bound_model_root.type_info.is_list and not isinstance(current_instance_new, (list, tuple)):
                    # TODO: 
                    current_instance_new = [current_instance_new]
            else:
                current_instance_new = None

        elif self.instance_new_struct_type == StructEnum.RULES_LIKE:
            exec_result = self.get_attr_value_by_comp_name(
                                component=component, 
                                instance=self.current_frame.instance_new)
            current_instance_new = exec_result.value
        else: 
            raise RuleInternalError(owner=self, msg=f"Invalid instance_new_struct_type = {self.instance_new_struct_type}")

        return current_instance_new


    # ============================================================
    # _update_and_clean - Update, validate, evaluate
    # ============================================================

    def _update_and_clean(self, component: ComponentBase) -> bool:
        """ returns if children should be processed 
                False - when available yields False
                False - when validation fails
                True - everything is clean
        """
        # ----------------------------------------------------------------
        # 1. init change history by bind of self.instance
        # 2. update value by bind of self.instance_new
        # 3. call cleaners - validations and evaluations in given order
        # ----------------------------------------------------------------

        if getattr(component, "available", None):
            not_available_vexp_result = execute_available_vexp(
                                                component.available, 
                                                apply_session=self)
            if not_available_vexp_result: 
                return False

        if getattr(component, "bind", None):
            # Fill initial value from instance 
            init_bind_vexp_result = self._init_by_bind_vexp(component)

            # try to update if instance_new is provided and yields different value
            bind_vexp_result, _ = self._try_update_by_instance(
                    component=component, 
                    init_bind_vexp_result=init_bind_vexp_result)
        else:
            bind_vexp_result = None


        # TODO: provide last value to all evaluations and validations 
        #       but be careful with vexp_result.value - it coluld be unadapted
        #       see: field.try_adapt_value(eval_value)
        all_ok = True
        if component.cleaners:
            for cleaner in component.cleaners:
                if isinstance(cleaner, ValidationBase):
                    # returns validation_failure
                    if self.execute_validation(component=component, validation=cleaner):
                        all_ok = False
                elif isinstance(cleaner, EvaluationBase):
                    if not bind_vexp_result:
                        # TODO: this belongs to Setup phase
                        raise RuleApplyError(owner=self, msg=f"Evaluator can be defined only for components with 'bind' defined. Remove 'Evaluation' or define 'bind'.")
                    self.execute_evaluation(component=component, evaluation=cleaner)
                else:
                    raise RuleApplyError(owner=self, msg=f"Unknown cleaner type {type(cleeaner)}. Expected Evaluation or Validation.")

        # NOTE: initial value from instance is not checked - only
        #       intermediate and the last value
        # returns validation_failure
        if self.validate_type(component):
            all_ok = False

        return all_ok

    # ------------------------------------------------------------

    def _init_by_bind_vexp(self, component: ComponentBase) -> ExecResult:
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

        return bind_vexp_result


    # ------------------------------------------------------------

    def _try_update_by_instance(self, component: ComponentBase, init_bind_vexp_result: ExecResult) \
            -> Tuple[ExecResult, bool]:
        """
        try to update if instance_new is provided and yields different value
        bool -> returns if updated or not, but not if value is adapted
        """
        init_value = init_bind_vexp_result.value
        bind_vexp_result = init_bind_vexp_result

        # try adapt of initial value changed value?
        last_value = component.try_adapt_value(init_value)

        updated = False
        if self.current_frame.instance_new not in (None, UNDEFINED):
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
                instance_new_bind_vexp_result = \
                        self.get_attr_value_by_comp_name(
                                component=component, 
                                instance=self.current_frame.instance_new)
                new_value = instance_new_bind_vexp_result.value
            else: 
                raise RuleInternalError(owner=self, msg=f"Invalid instance_new_struct_type = {self.instance_new_struct_type}")

            if new_value is not UNDEFINED:
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

        if not updated and init_value != last_value:
            # adapted value => updated = False
            # diff initial value from adapted
            self.register_instance_attr_change(
                    component=component, 
                    # TODO: how to mark init -> adaptation change?
                    vexp_result=None,
                    new_value=last_value
                    )

        return bind_vexp_result, updated

    # ------------------------------------------------------------

    def get_attr_value_by_comp_name(self, component:ComponentBase, instance: ModelType) -> ExecResult:
        attr_name = component.name
        if not hasattr(instance, attr_name):
            # TODO: depending of self.rules strategy or apply(strategy) 
            #   - raise error
            #   - return NotAvailableExecResult() / UNDEFINED (default)
            #   - return None (default)
            return NotAvailableExecResult.create(reason="Missing instance attribute")

        value =  getattr(instance, attr_name)

        exec_result = ExecResult()
        exec_result.set_value(value, attr_name, changer_name=f"{component.name}.ATTR")
        return exec_result

