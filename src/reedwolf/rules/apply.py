from __future__ import annotations

from dataclasses import dataclass
from typing import (
        Dict,
        Optional,
        Tuple,
        )
from contextlib import AbstractContextManager
from collections import OrderedDict, defaultdict

from .exceptions import (
        RuleApplyError,
        RuleValidationError,
        RuleInternalError,
        RuleApplyValueError,
        )
from .expressions import (
        ExecResult,
        NotAvailableExecResult,
        DotExpression,
        execute_available_dexp,
        )
from .meta import (
        UNDEFINED,
        NoneType,
        ModelType,
        is_model_class,
        is_model_instance,
        )
from .base import (
        AttrValue,
        AttrName,
        GlobalConfig,
        KeyString,
        ComponentBase,
        IApplySession,
        ApplyStackFrame,
        ValidationFailure,
        StructEnum,
        ChangeOpEnum,
        InstanceAttrValue,
        InstanceChange,
        InstanceAttrCurrentValue,
        get_instance_key_string_attrname_pair,
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
        Extension,
        )


MAX_RECURSIONS = 30


class UseApplyStackFrame(AbstractContextManager):
    " with() ... custom context manager. Very similar to UseSetupStackFrame "

    # ALT: from contextlib import contextmanager
    def __init__(self, apply_session: IApplySession, frame: ApplyStackFrame):
        self.apply_session = apply_session
        self.frame = frame

        self.copy_from_previous_frame()


    def copy_from_previous_frame(self):
        """
        if the instance is the same - consider from last frame 
        container (copy/check), index0 (copy/check), component ...
        """
        if not self.apply_session.frames_stack:
            return

        previous_frame = self.apply_session.frames_stack[0]

        # do not use ==, compare by instance (ALT: use id(instance) ) 
        if self.frame.instance is previous_frame.instance:
            self._copy_attr_from_previous_frame(previous_frame, "container", may_be_copied=False)
            self._copy_attr_from_previous_frame(previous_frame, "bound_model_root", may_be_copied=False)
            self._copy_attr_from_previous_frame(previous_frame, "instance_new")
            self._copy_attr_from_previous_frame(previous_frame, "index0")
            # only these can be copied
            self._copy_attr_from_previous_frame(previous_frame, "parent_instance")
            self._copy_attr_from_previous_frame(previous_frame, "parent_instance_new")

            # do not use ==, compare by instance (ALT: use id(instance) ) 
            if self.frame.component is previous_frame.component:
                self._copy_attr_from_previous_frame(previous_frame, "on_component_only", may_be_copied=False)
                self._copy_attr_from_previous_frame(previous_frame, "key_string", may_be_copied=False)

                # NOTE: not this for now:
                #   self._copy_attr_from_previous_frame(previous_frame, "local_setup_session")

            # check / init again 
            self.frame.clean()


    def _copy_attr_from_previous_frame(self, previous_frame: ApplyStackFrame, attr_name: str, may_be_copied: bool = True):

        if not hasattr(self.frame, attr_name):
            raise RuleInternalError(owner=self, msg=f"This frame {self.frame}.{attr_name} not found") 
        if not hasattr(previous_frame, attr_name):
            raise RuleInternalError(owner=self, msg=f"Previous frame {previous_frame}.{attr_name} not found") 

        this_frame_attr_value = getattr(self.frame, attr_name)
        prev_frame_attr_value = getattr(previous_frame, attr_name)

        if this_frame_attr_value in (None, UNDEFINED):
            if prev_frame_attr_value not in (None, UNDEFINED):
                if not may_be_copied:
                    raise RuleInternalError(owner=self, 
                        msg=f"Attribute '{attr_name}' value in previous frame is non-empty and current frame has empty value:\n  {previous_frame}\n    = {prev_frame_attr_value}\n<>\n  {self.frame}\n    = {this_frame_attr_value} ") 
                # Copy from previous frame
                # apply_session.config.loggeer.debugf"setattr '{attr_name}' current_frame <= previous_frame := {prev_frame_attr_value} (frame={self.frame})")
                setattr(self.frame, attr_name, prev_frame_attr_value)
        else:
            # in some cases id() / is should be used?
            if prev_frame_attr_value != this_frame_attr_value:
                raise RuleInternalError(owner=self, 
                    msg=f"Attribute '{attr_name}' value in previous frame is different from current:\n  {previous_frame}\n    = {prev_frame_attr_value}\n<>\n  {self.frame}\n    = {this_frame_attr_value} ") 




    def __enter__(self):
        self.apply_session.push_frame_to_stack(self.frame)
        return self.frame

    def __exit__(self, exc_type, exc_value, exc_tb):
        frame_popped = self.apply_session.pop_frame_from_stack()
        if not exc_type and frame_popped != self.frame:
            raise RuleInternalError(owner=self, msg=f"Something wrong with frame stack, got {frame_popped}, expected {self.frame}")



@dataclass
class ApplyResult(IApplySession):
    """ 
    ApplyResult is ApplySession (variable apply_session) - but is renamed to
    result since it is name exposed to external API.

    Similar is Function -> IFunctionFactory.
    """

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

    def use_stack_frame(self, frame: ApplyStackFrame) -> UseApplyStackFrame:
        if not isinstance(frame, ApplyStackFrame):
            raise RuleInternalError(owner=self, msg=f"Expected ApplyStackFrame, got frame: {frame}") 

        return UseApplyStackFrame(apply_session=self, frame=frame)

    # ------------------------------------------------------------

    def raise_if_failed(self):
        if not self.finished:
            raise RuleApplyError(owner=self, msg="Apply process is not finished")

        if self.errors:
            raise RuleValidationError(owner=self.rules, errors=self.errors)

    # ------------------------------------------------------------

    def execute_evaluation(self, component: ComponentBase, evaluation:EvaluationBase) -> ExecResult:
        """ Execute evaluation and if new value is different from existing
            value, update current instance """
        assert isinstance(evaluation, EvaluationBase)
        assert component == evaluation.owner

        if not component == self.current_frame.component:
            raise RuleInternalError(owner=self.current_frame.component, 
                    msg=f"Component in frame {self.current_frame.component} must match component: {component}") 

        # evaluation_dexp = evaluation.value
        # assert isinstance(evaluation_dexp, DotExpression)
        # eval_dexp_result  = evaluation_dexp._evaluator.execute(apply_session=self)
        eval_dexp_result  = evaluation.execute(apply_session=self)

        if eval_dexp_result.is_not_available():
            return eval_dexp_result

        eval_value = eval_dexp_result.value

        if isinstance(component, FieldBase):
            eval_value = component.try_adapt_value(eval_value)

        # ALT: bind_dexp_result = component.get_dexp_result_from_instance(apply_session)
        orig_value = self.get_current_value(component)

        if (orig_value != eval_value):
            self.register_instance_attr_change(
                    component=component, 
                    dexp_result=eval_dexp_result,
                    new_value=eval_value
                    )

        return eval_dexp_result

    # ------------------------------------------------------------

    # value: Any, 
    def execute_validation(self, component: ComponentBase, validation:ValidationBase) -> Optional[ValidationFailure]:
        """ Execute validaion - if returns False value then register error and
            mark component and children invalid to prevent further rules execution
        """
        assert isinstance(validation, ValidationBase)
        assert component == validation.owner

        if not component == self.current_frame.component:
            raise RuleInternalError(owner=self.current_frame.component, 
                    msg=f"Component in frame {self.current_frame.component} must match component: {component}") 

        # value=value, 
        validation_failure = validation.validate(apply_session=self)
        if validation_failure:
            self.register_instance_validation_failed(component, validation_failure)

        return validation_failure

    # ------------------------------------------------------------

    def register_instance_attr_change(self, 
            component: ComponentBase, 
            dexp_result: ExecResult,
            new_value: Any,
            is_from_init_bind:bool=False) -> InstanceAttrValue:

        # NOTE: new_value is required - since dexp_result.value
        #       could be unadapted (see field.try_adapt_value()

        assert component == self.current_frame.component

        if new_value is UNDEFINED:
            raise RuleInternalError(owner=component, msg="New value should not be UNDEFINED, fix the caller")

        # key_str = component.get_key_string(apply_session=self)
        key_str = self.get_key_string(component)

        if key_str not in self.update_history:
            if not is_from_init_bind:
                raise RuleInternalError(owner=component, msg=f"key_str '{key_str}' not found in update_history and this is not initialization")

            self.update_history[key_str] = []

            assert key_str not in self.current_values
            self.current_values[key_str] = InstanceAttrCurrentValue(
                                                key_string=key_str, 
                                                component=component)

            # NOTE: initial value from instance is not checked - only
            #       intermediate and the last value
            #   self.validate_type(component, new_value)
        else:
            assert key_str in self.current_values

            if is_from_init_bind:
                raise RuleInternalError(owner=component, msg=f"key_str '{key_str}' found in update_history and this is initialization")

            if not self.update_history[key_str]:
                raise RuleInternalError(owner=component, msg=f"change history for key_str='{key_str}' is empty")

            # -- check if current value is different from new one
            value_current = self.update_history[key_str][-1].value
            if value_current == new_value:
                raise RuleApplyError(owner=component, msg=f"register change failed, the value is the same: {value_current}")

            # is this really necessary
            self.validate_type(component, new_value)

            # -- parent instance
            # parent_raw_attr_value = dexp_result.value_history[-2]
            # parent_instance = parent_raw_attr_value.value

            parent_instance = self.current_frame.instance
            # TODO: not sure if this validation is ok
            if not isinstance(parent_instance, 
                    self.current_frame.container.bound_model.get_type_info().type_):
                raise RuleInternalError(owner=self, msg=f"Parent instance {parent_instance} has wrong type")


            # -- attr_name - fetch from initial bind dexp (very first)
            init_instance_attr_value = self.update_history[key_str][0]
            if not init_instance_attr_value.is_from_bind:
                raise RuleInternalError(owner=self, msg=f"{init_instance_attr_value} is not from bind")
            init_bind_dexp_result = init_instance_attr_value.dexp_result
            # attribute name is in the last item
            init_raw_attr_value = init_bind_dexp_result.value_history[-1]
            attr_name = init_raw_attr_value.attr_name

            if not hasattr(parent_instance, attr_name):
                raise RuleInternalError(owner=self, msg=f"Missing {parent_instance}.{attr_name}")

            # ----------------------------------------
            # Finally change instance value
            # ----------------------------------------
            setattr(parent_instance, attr_name, new_value)

            # NOTE: bind_dexp_result last value is not changed
            #       maybe should be changed but ... 

        # TODO: pass input arg value_owner_name - component.name does not have
        #       any purpose
        instance_attr_value = InstanceAttrValue(
                                value_owner_name=component.name, 
                                value=new_value,
                                dexp_result=dexp_result,
                                is_from_bind = is_from_init_bind,
                                # TODO: source of change ...
                                )

        self.update_history[key_str].append(instance_attr_value)
        self.current_values[key_str].set_value(new_value)

        return instance_attr_value


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
            # ---- Rules case -----

            assert parent is None

            # NOTE: frame not set so 'self.current_frame.instance' is not available
            #       thus sending 'instance' param
            component.bound_model._apply_nested_models(
                                        apply_session=self, 
                                        instance=self.instance
                                        )

            new_frame = ApplyStackFrame(
                            container = component, 
                            component = component, 
                            instance = self.instance,
                            instance_new = self.instance_new,
                            )


        elif not extension_list_mode and component.is_extension():
            # ---- Extension case -----

            component : Extension = component
            if not isinstance(component.bound_model.model, DotExpression):
                raise RuleInternalError(owner=self, msg=f"For Extension `bound_model` needs to be DotExpression, got: {component.bound_model.model}") 

            if getattr(component.bound_model, "contains", None):
                raise RuleInternalError(owner=self, msg=f"For Extension complex `bound_model` is currently not supported (e.g. `contains`), use simple BoundModel, got: {component.bound_model}") 

            # original instance
            dexp_result: ExecResult = component.bound_model.model \
                                        ._evaluator.execute_dexp(apply_session=self)
            instance = dexp_result.value

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
                        key = component.get_key_pairs_or_index0(instance=item_instance_new, index0=index0)
                        if key in new_instances_by_key:
                            raise RuleApplyValueError(owner=self, msg=f"{component}: Duplicate key {key}, first item is: {new_instances_by_key[key]}")
                        new_instances_by_key[key] = item_instance_new


                # NOTE: considered to use dict() since dictionaries are ordered in Python 3.6+ 
                #       ordering-perserving As of Python 3.7, this is a guaranteed, i.e.  Dict keeps insertion order
                #       https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6
                instances_by_key = OrderedDict()

                for index0, instance in enumerate(instance_list, 0):
                    key = component.get_key_pairs_or_index0(instance=instance, index0=index0)
                    if component.keys:
                        missing_keys = [kn for kn, kv in key if isinstance(kv, MissingKey)]
                        if missing_keys:
                            raise RuleApplyValueError(owner=self, msg=f"Instance {instance} has key(s) with value None, got: {', '.join(missing_keys)}")

                    if current_instance_new not in (None, UNDEFINED):
                        item_instance_new = new_instances_by_key.get(key, UNDEFINED)
                        if item_instance_new is UNDEFINED:
                            key_string = self.get_key_string_by_instance(
                                    component = component, 
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

                        key_string = self.get_key_string_by_instance(
                                        component = component,
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

                # Apply for all items
                for key, (instance, index0, item_instance_new) in instances_by_key.items():
                    # Go one level deeper 
                    with self.use_stack_frame(
                            ApplyStackFrame(
                                container = component,
                                component = component, 
                                index0 = index0,
                                # main instance - original values
                                instance = instance, 
                                parent_instance=self.current_frame.instance,
                                # new instance - new values (when update mode)
                                instance_new = item_instance_new, 
                                parent_instance_new=self.current_frame.instance_new,
                                )):
                        # Recursion with prevention to hit this code again
                        self._apply(parent=parent, 
                                    component=component, 
                                    in_component_only_tree=in_component_only_tree,
                                    depth=depth+1,
                                    # prevent is_extension_logic again -> infinitive recursion
                                    extension_list_mode=True)

                # ========================================
                # Finished, processed all children
                # ========================================
                return
                # ========================================



            # == Extension with single item ==

            if instance is None:
                # must be type_info.is_optional ...
                pass
            elif not is_model_instance(instance):
                raise RuleApplyValueError(owner=self, msg=f"Expected list/tuple or model instance, got: {instance} : {type(instance)}")


            new_frame = ApplyStackFrame(
                            container = component, 
                            component = component, 
                            instance = instance,
                            instance_new = current_instance_new,
                            )

        # ------------------------------------------------------------

        if not new_frame:
            # register non-container frame - only component is new. take instance from previous frame
            new_frame = ApplyStackFrame(
                            component = component, 
                            # copy
                            instance = self.current_frame.instance,
                            container = self.current_frame.container, 
                            # automatically copied
                            #   instance_new = self.current_frame.instance_new,
                            #   index0 = self.current_frame.index0,
                            )

        process_further = True

        with self.use_stack_frame(new_frame):
            # obnly when full apply or partial apply
            if not (self.component_only and not in_component_only_tree):
                # ============================================================
                # Update, validate, evaluate
                # ============================================================
                process_further = self._update_and_clean(component=component)
                # also if validation fails ...

                if process_further and getattr(component, "enables", None):
                    assert not getattr(component, "contains", None), component
                    bind_dexp_result = component.get_dexp_result_from_instance(apply_session=self)
                    value = bind_dexp_result.value
                    if not isinstance(value, (bool, NoneType)):
                        raise RuleApplyValueError(owner=component, 
                                msg=f"Component.enables can be applied only for Boolean or None values, got: {value} : {type(value)}")
                    if not value: 
                        process_further = False

            # if parent yields None - do not process on children
            if self.current_frame.instance in (None, UNDEFINED):
                process_further = False

            # ------------------------------------------------------------
            # NOTE: used only for test if all Dexp values could evaluate ...
            #       self.__check_component_all_dexps(component)

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
            updated_values_dict: Dict[KeyString, Dict[AttrName, Tuple[AttrValue, AttrValue]]] = defaultdict(dict)
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

        # TODO: logger: apply_session.config.logger.debug(f"depth={depth}, comp={component.name}, bind={bind} => {dexp_result}")

        return

    # ------------------------------------------------------------

    def __check_component_all_dexps(self, component: ComponentBase):
        # used only for testing
        for attr_name, attr_value in vars(component).items():
            if isinstance(attr_value, DotExpression):
                # dexp_result: ExecResult = 
                attr_value._evaluator.execute_dexp(apply_session=self)
                # TODO: apply_session.config.logger.debug(f"{parent.name if parent else ''}.{component.name}.{attr_name} = DExp[{attr_value}] -> {dexp_result}")

    # ------------------------------------------------------------

    def apply(self) -> ApplyResult:
        """
        Main function for parsing, validating and evaluating input instance.
        Returns ApplyResult object.

        if all ok - Result.instance contains a new instance (clone + update) of the bound model type 
        if not ok - errors contain all details.
        """
        self._apply(parent=None, component=self.rules)
        self.finish()

        return self

    # ------------------------------------------------------------

    def get_instance_by_key_string(self, key_string: str) -> ModelType:
        " must exist in cache - see previous method which sets self.get_key_string_by_instance(container) "
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
            container = component.get_container_owner(consider_self=True)
            model = container.bound_model.type_info.type_

        if isinstance(self.instance_new, (list, tuple)):
            if not self.instance_new:
                raise RuleInternalError(owner=self, msg="Instance is an empty list, can not detect type of base  structure")
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
            assert isinstance(component.bound_model.model, DotExpression), component.bound_model.model

            if self.current_frame.instance_new not in (None, UNDEFINED):

                # if partial - then dexp must know - this value is set only in this case
                if in_component_only_tree and component == self.component_only:
                    # Extension or FieldGroup is root
                    on_component_only = component
                else:
                    on_component_only = None

                # container = component.get_container_owner(consider_self=True) if not component.is_extension() else component
                with self.use_stack_frame(
                        ApplyStackFrame(
                            container = self.current_frame.container, 
                            component = self.current_frame.component, 
                            # only this is changed
                            instance = self.current_frame.instance_new,
                            # should not be used
                            instance_new = UNDEFINED, 
                            on_component_only=on_component_only,
                        )) as frame:
                    dexp_result: ExecResult = \
                                        component \
                                        .bound_model \
                                        .model \
                                        ._evaluator.execute_dexp(
                                                apply_session=self, 
                                                )
                # set new value
                current_instance_new = dexp_result.value
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
            not_available_dexp_result = execute_available_dexp(
                                                component.available, 
                                                apply_session=self)
            if not_available_dexp_result: 
                return False

        if getattr(component, "bind", None):
            # Fill initial value from instance 
            init_bind_dexp_result = self._init_by_bind_dexp(component)

            # try to update if instance_new is provided and yields different value
            bind_dexp_result, _ = self._try_update_by_instance(
                                        component=component, 
                                        init_bind_dexp_result=init_bind_dexp_result)
        else:
            bind_dexp_result = None


        # TODO: provide last value to all evaluations and validations 
        #       but be careful with dexp_result.value - it coluld be unadapted
        #       see: field.try_adapt_value(eval_value)
        all_ok = True
        if component.cleaners:
            for cleaner in component.cleaners:
                # TODO: if something is wrong - cleaner is not known. maybe
                #       something like this:
                #   with self.use_stack_frame(
                #           ApplyStackFrame(
                #               container = self.current_frame.container,
                #               instance = self.current_frame.instance,
                #               component = self.current_frame.component,
                #               cleaner = cleaner, 
                #               )):
                if isinstance(cleaner, ValidationBase):
                    # returns validation_failure
                    if self.execute_validation(component=component, validation=cleaner):
                        all_ok = False
                elif isinstance(cleaner, EvaluationBase):
                    if not bind_dexp_result:
                        # TODO: this belongs to Setup phase
                        raise RuleApplyError(owner=self, msg="Evaluator can be defined only for components with 'bind' defined. Remove 'Evaluation' or define 'bind'.")
                    self.execute_evaluation(component=component, evaluation=cleaner)
                else:
                    raise RuleApplyError(owner=self, msg=f"Unknown cleaner type {type(cleaner)}. Expected Evaluation or Validation.")

        # NOTE: initial value from instance is not checked - only
        #       intermediate and the last value
        # returns validation_failure
        if self.validate_type(component):
            all_ok = False

        return all_ok

    # ------------------------------------------------------------

    def _init_by_bind_dexp(self, component: ComponentBase) -> ExecResult:
        " get initial dexp value, if instance_new try to updated/overwrite with it"

        if not isinstance(component, FieldBase):
            raise RuleInternalError(owner=self, msg=f"Expected FieldBase field, got: {component}")

        bind_dexp_result = component.get_dexp_result_from_instance(apply_session=self)
        init_value = bind_dexp_result.value

        self.register_instance_attr_change(
                component = component, 
                dexp_result = bind_dexp_result, 
                new_value = init_value,
                is_from_init_bind = True)

        return bind_dexp_result


    # ------------------------------------------------------------

    def _try_update_by_instance(self, component: ComponentBase, init_bind_dexp_result: ExecResult) \
            -> Tuple[ExecResult, bool]:
        """
        try to update if instance_new is provided and yields different value
        bool -> returns if updated or not, but not if value is adapted
        """
        init_value = init_bind_dexp_result.value
        bind_dexp_result = init_bind_dexp_result

        # try adapt of initial value changed value?
        last_value = component.try_adapt_value(init_value)

        updated = False
        if self.current_frame.instance_new not in (None, UNDEFINED):
            if self.instance_new_struct_type == StructEnum.MODELS_LIKE:

                with self.use_stack_frame(
                        ApplyStackFrame(container=self.current_frame.container, 
                                   component=self.current_frame.component, 
                                   # only this is changed
                                   instance=self.current_frame.instance_new,
                                   instance_new=UNDEFINED, 
                        )):
                    instance_new_bind_dexp_result = \
                            component.get_dexp_result_from_instance(apply_session=self)
                    new_value = instance_new_bind_dexp_result.value

            elif self.instance_new_struct_type == StructEnum.RULES_LIKE:
                instance_new_bind_dexp_result = \
                        self.get_attr_value_by_comp_name(
                                component=component, 
                                instance=self.current_frame.instance_new)
                new_value = instance_new_bind_dexp_result.value
            else: 
                raise RuleInternalError(owner=self, msg=f"Invalid instance_new_struct_type = {self.instance_new_struct_type}")

            if new_value is not UNDEFINED:
                new_value = component.try_adapt_value(new_value)

                # adapted new instance value diff from adapted initial value
                if new_value != last_value:
                    self.register_instance_attr_change(
                            component = component, 
                            dexp_result = instance_new_bind_dexp_result, 
                            new_value = new_value,
                            )
                    last_value = new_value 
                    bind_dexp_result = instance_new_bind_dexp_result
                    updated = True

        if not updated and init_value != last_value:
            # adapted value => updated = False
            # diff initial value from adapted
            self.register_instance_attr_change(
                    component=component, 
                    # TODO: how to mark init -> adaptation change?
                    dexp_result=None,
                    new_value=last_value
                    )

        return bind_dexp_result, updated

    # ------------------------------------------------------------

    def get_key_string(self, component: ComponentBase):
        # apply_session: IApplySession
        # TODO: is caching possible? 
        # TODO: consider moving to ApplySession/ApplyResult?
        if component.is_container():
            # raise RuleInternalError(owner=component, msg=f"Expecting non-container, got: {component}") 
            " uses cache, when not found then gets intances and index0 from current frame "
            key_string = self.get_key_string_by_instance(
                            component=component,
                            instance = self.current_frame.instance, 
                            index0 = self.current_frame.index0)
        else:
            container = component.get_container_owner(consider_self=True)
            # container_key_string = container.get_key_string(apply_session)
            # recursion
            container_key_string = self.get_key_string(container)

            key_string = GlobalConfig.ID_NAME_SEPARATOR.join(
                    [container_key_string, component.name] 
                    )
        return key_string

    # ------------------------------------------------------------

    def get_key_string_by_instance(self, component: ComponentBase, instance: ModelType, index0: Optional[int]) -> str:
        # apply_session:IApplySession,  -> self
        """
        Two cases - component has .keys or not:

        a) with keys:
            For containers which have keys defined, it is assumed that one key is
            globally unique within Extension components, so no need to prefix key
            with parent/owner key_string. Example:

                 address_set_ext[id2=1]

        b) without keys - index0 based:
            In other cases item index in list is used (index0), and then this is
            only locally within one instance of parent/owner, therefore parent
            key_string is required. Example:

                 company_rules::address_set_ext[0]

        """
        if not component.is_container():
            raise RuleInternalError(owner=component, msg=f"Expecting container, got: {component}") 

        instance_id = id(instance)
        key_string = self.key_string_container_cache.get(instance_id, None)
        if key_string is None:

            if component.keys:
                key_pairs = component.get_key_pairs(instance)
                assert key_pairs
                key_string = "{}[{}]".format(
                                component.name, 
                                GlobalConfig.ID_NAME_SEPARATOR.join(
                                    [f"{name}={value}" for name, value in key_pairs]
                                ))
            elif index0 is not None:
                key_string = f"{component.name}[{index0}]"

                if self.current_frame.parent_instance:
                    parent_instance = self.current_frame.parent_instance
                    parent_id = id(parent_instance)
                    if parent_id not in self.key_string_container_cache:
                        # must_be_in_cache
                        if not self.component_name_only:
                            raise RuleInternalError(owner=component, msg=f"Parent instance's key not found in cache, got: {parent_instance}") 
                        parent_key_string = f"__PARTIAL__{self.component_name_only}"
                    else:
                        parent_key_string = self.key_string_container_cache[parent_id]
                    key_string = GlobalConfig.ID_NAME_SEPARATOR.join([parent_key_string, key_string])

            else:
                key_string = component.name

            self.key_string_container_cache[instance_id] = key_string
            self.instance_by_key_string_cache[key_string] = instance
        #     from_cache = "new"
        # else:
        #     from_cache = "cache"

        # TODO: self.config.logger.debug("cont:", component.name, key_string, f"[{from_cache}]")

        return key_string


    # ------------------------------------------------------------

    def get_current_value_instance(self, component: ComponentBase) -> InstanceAttrCurrentValue:
        """ if not found will return UNDEFINED
            Probaly a bit faster, only dict queries.
        """
        key_str = self.get_key_string(component)
        if not key_str in self.current_values:
            return UNDEFINED
        instance_attr_current_value = self.current_values[key_str]
        return instance_attr_current_value

    # ------------------------------------------------------------

    def get_current_value(self, component: ComponentBase) -> LiteralType:
        # apply_session:IApplySession
        """ Could work on non-stored fields.
            Probaly a bit faster, only dict queries.
        """
        # ALT: from update_history: 
        #       instance_attr_value = apply_session.update_history[key_str][-1]
        #       return instance_attr_value.value
        # ALT: fetch from:
        #       bind_dexp: DotExpression = getattr(component, "bind", None)
        #       bind_dexp._evaluator.execute()
        # key_str = component.get_key_string(apply_session=self)
        key_str = self.get_key_string(component)
        if not key_str in self.current_values:
            raise RuleInternalError(owner=component, msg=f"{key_str} not found in current values") 
        instance_attr_current_value = self.current_values[key_str]
        return instance_attr_current_value.value

    # ------------------------------------------------------------

    def get_attr_value_by_comp_name(self, component:ComponentBase, instance: ModelType) -> ExecResult:
        attr_name = component.name
        if not hasattr(instance, attr_name):
            # TODO: depending of self.rules strategy or apply(strategy) 
            #   - raise error
            #   - return NotAvailableExecResult() / UNDEFINED (default)
            #   - return None (default)
            return NotAvailableExecResult.create(reason="Missing instance attribute")

        value = getattr(instance, attr_name)

        exec_result = ExecResult()
        exec_result.set_value(value, attr_name, changer_name=f"{component.name}.ATTR")
        return exec_result


# ------------------------------------------------------------
# OBSOLETE
# ------------------------------------------------------------

    # def get_current_value(self, apply_session:iapplysession) -> any:
    #     """ Fetch ExecResult from component.bind from APPLY_SESSION.UPDATE_HISTORY
    #         last record.
    #         !!! ExecResult.value could be unadapted :( !!!
    #         Could work on non-stored fields.
    #         Probaly a bit faster, only dict queries.
    #     """
    #     key_str = self.get_key_string(apply_session=apply_session)
    #     if not key_str in apply_session.update_history:
    #         raise RuleInternalError(owner=self, msg=f"{key_str} not found in current values") 
    #     instance_attr_value = apply_session.update_history[key_str][-1]
    #     return instance_attr_value.value
