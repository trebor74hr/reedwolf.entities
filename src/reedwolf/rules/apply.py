from __future__ import annotations

from enum import Enum
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
from .utils import (
        get_available_names_example,
        UNDEFINED,
        NA_DEFAULTS_MODE,
        NA_IN_PROGRESS,
        NOT_APPLIABLE,
        )
from .meta import (
        NoneType,
        ModelType,
        is_model_class,
        is_model_instance,
        ValuesTree,
        ComponentTreeWValuesType,
        )
from .base import (
        MAX_RECURSIONS,
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
        if not self.apply_session.stack_frames:
            return

        previous_frame = self.apply_session.stack_frames[0]

        self._copy_attr_from_previous_frame(previous_frame, "in_component_only_tree", 
                                            if_set_must_be_same=False)
        self._copy_attr_from_previous_frame(previous_frame, "depth", 
                                            if_set_must_be_same=False)
        self._copy_attr_from_previous_frame(previous_frame, "parent_values_subtree", 
                                            if_set_must_be_same=False)

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


    def _copy_attr_from_previous_frame(self, 
            previous_frame: ApplyStackFrame, 
            attr_name: str, 
            may_be_copied: bool = True,
            if_set_must_be_same: bool = True):

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
            if if_set_must_be_same and prev_frame_attr_value != this_frame_attr_value:
                raise RuleInternalError(owner=self, 
                    msg=f"Attribute '{attr_name}' value in previous frame is different from current:\n  {previous_frame}\n    = {prev_frame_attr_value}\n<>\n  {self.frame}\n    = {this_frame_attr_value} ") 




    def __enter__(self):
        self.apply_session.push_frame_to_stack(self.frame)
        return self.frame

    def __exit__(self, exc_type, exc_value, exc_tb):
        frame_popped = self.apply_session.pop_frame_from_stack()
        if not exc_type and frame_popped != self.frame:
            raise RuleInternalError(owner=self, msg=f"Something wrong with frame stack, got {frame_popped}, expected {self.frame}")


# ============================================================


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

        if self.rules.context_class:
            if not self.context:
                raise RuleApplyError(owner=self.rules, msg=f"Pass context object to .apply*(). Context should be instance of '{self.rules.context_class}'.")
            if not isinstance(self.context, self.rules.context_class):
                raise RuleApplyError(owner=self, msg=f"Context object '{self.context}' is not instance of context class '{self.rules.context_class}'.")
        else:
            if self.context:
                raise RuleApplyError(owner=self, msg=f"Given context object '{self.context}', but context class in component is not setup. Provide 'context_class' to Rules object and try again.")

        # self.model = self.bound_model.model
        # if not self.model:
        #     raise RuleInternalError(owner=self, item=component, msg=f"Bound model '{self.bound_model}' has empty model.")

        if self.defaults_mode:
            if not (self.instance is NA_DEFAULTS_MODE and self.instance_new is None and not self.component_name_only):
                raise RuleInternalError(owner=self, msg=f"Defaults mode does not expect instance or instance_new or component_name_only, got: {self.instance} / {self.instance_new} / {self.component_name_only}") 
        else:
            if not isinstance(self.instance, self.bound_model.model):
                raise RuleApplyError(owner=self, msg=f"Object '{self.instance}' is not instance of bound model '{self.bound_model.model}'.")


            if self.instance_new is not None and not self.component_name_only:
                self._detect_instance_new_struct_type(self.rules)

        # ----------------------------------------
        # see IApplySession for description 
        self.binary_operations_type_adapters[(str, int)] = str


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
        orig_value = self.get_current_value(component, strict=False)

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
        assert not self.defaults_mode

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

        if new_value is UNDEFINED:
            raise RuleInternalError(owner=component, msg="New value should not be UNDEFINED, fix the caller")

        # key_str = component.get_key_string(apply_session=self)
        key_str = self.get_key_string(component)

        if self.update_history.get(key_str, UNDEFINED) == UNDEFINED:
            if not is_from_init_bind:
                raise RuleInternalError(owner=component, msg=f"key_str '{key_str}' not found in update_history and this is not initialization")

            self.update_history[key_str] = []

            # Can be various UndefinedType: NA_IN_PROGRESS, NOT_APPLIABLE
            if self.current_values.get(key_str):
                raise RuleInternalError(owner=self, msg=f"current_values[{key_str}] ==  {self.current_values[key_str]}") 

            self.current_values[key_str] = InstanceAttrCurrentValue(
                                                key_string=key_str, 
                                                component=component)

            # NOTE: initial value from instance is not checked - only
            #       intermediate and the last value
            #   self.validate_type(component, new_value)
        else:
            assert key_str in self.current_values
            assert component == self.current_frame.component

            if is_from_init_bind:
                raise RuleInternalError(owner=component, msg=f"key_str '{key_str}' found in update_history and this is initialization")

            if not self.update_history[key_str]:
                raise RuleInternalError(owner=component, msg=f"change history for key_str='{key_str}' is empty")

            # -- check if current value is different from new one
            value_current = self.update_history[key_str][-1].value
            if value_current == new_value:
                raise RuleApplyError(owner=component, msg=f"register change failed, the value is the same: {value_current}")

            # TODO: is this really necessary - will be done in apply() later
            self.validate_type(component, strict=False, value=new_value)

            # -- parent instance
            # parent_raw_attr_value = dexp_result.value_history[-2]
            # parent_instance = parent_raw_attr_value.value

            parent_instance = self.current_frame.instance

            if parent_instance is not NA_DEFAULTS_MODE:
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
        # set current value 
        self.current_values[key_str].set_value(new_value)

        return instance_attr_value


    # ------------------------------------------------------------
    # _apply() -> main apply function
    # ------------------------------------------------------------

    def _apply(self, 
               # parent: Optional[ComponentBase], 
               component: ComponentBase, 

               # -- RECURSION -- internal props

               # see dox below
               mode_extension_list:bool = False, 

               # see dox below
               mode_dexp_dependency: bool = False,

               entry_call: bool = False,

               # Moved to stack
               #    # partial apply - passed through recursion further
               #    in_component_only_tree: bool = False,
               #    # recursion depth
               #    depth:int=0, 
               ):
        """
        Main entry function 'apply' logic.
        Recursion in 3 main modes:

            normal_mode 
               for each child in children -> ._apply()

            mode_extension_list
               if extension and list - for each item in items -> ._apply()
               caller is extension_list processing - NOT passed through recursion further

            mode_dexp_dependency 
               if cleaner.DotExpression references component which is not yet
               processed - it skip normal order. 

               Example:
                ._apply() 
                -> DotExpression 
                -> get_current_value_instance() 
                -> ._apply()

              this component and its subtreee will be skipped later when it
              comes again in normal order (cases A / B).
              In this case frame . depth does not correspond with component's
              tree depth.

        """
        # TODO: self.config.loger.debug(...)
        assert not (mode_extension_list and mode_dexp_dependency)

        comp_container = component.get_container_owner(consider_self=True)
        comp_container_model = comp_container.bound_model.get_type_info().type_

        if mode_dexp_dependency:
            # TODO: self.config.logger.info(f"apply - mode_dexp_dependency - {self.current_frame.component.name} depends on {component.name} - calling apply() ...")

            # instance i.e. containers must match
            assert self.stack_frames
            caller_container = self.current_frame.component.get_container_owner(consider_self=True)
            if comp_container is not caller_container:
                raise RuleInternalError(owner=component, msg=f"Componenent's container '{comp_container}' must match caller's '{self.current_frame.component.name}' container: {caller_container.name}") 

        if self.finished:
            raise RuleInternalError(owner=self, msg=f"Already finished") 

        if self.stack_frames:
            assert not entry_call
            depth = self.current_frame.depth
            in_component_only_tree = self.current_frame.in_component_only_tree 

            # check if instance model is ok
            if not component.is_extension():
                if self.defaults_mode:
                    if self.current_frame.instance is not NA_DEFAULTS_MODE:
                        raise RuleInternalError(owner=self, msg=f"Defaults mode - current frame's instance's model must be NA_DEFAULTS_MODE, got: {type(self.instance)}") 
                else:
                    if not isinstance(self.current_frame.instance, comp_container_model):
                        raise RuleInternalError(owner=self, msg=f"Current frame's instance's model does not corresponds to component's container's model. Expected: {comp_container_model}, got: {type(self.instance)}") 

            # TODO: for extension() any need to check this at all in this phase?
        else:
            # no stack around -> initial call -> depth=0
            assert entry_call
            depth = 0
            in_component_only_tree = False

            if self.defaults_mode:
                if not self.instance is NA_DEFAULTS_MODE:
                    raise RuleInternalError(owner=self, msg=f"Defaults mode - instance must be NA_DEFAULTS_MODE, got: {type(self.instance)}") 
            else:
                if not isinstance(self.instance, comp_container_model):
                    raise RuleInternalError(owner=self, msg=f"Rules instance's model does not corresponds to component's container's model. Expected: {comp_container_model}, got: {type(self.instance)}") 


        if self.component_only and component == self.component_only:
            # partial apply - detected component
            if not mode_extension_list and in_component_only_tree:
                raise RuleInternalError(owner=self, 
                        # {parent} -> 
                        msg=f"{component}: in_component_only_tree should be False, got {in_component_only_tree}")

            if self.instance_new not in (None, UNDEFINED) and not self.instance_new_struct_type:
                self._detect_instance_new_struct_type(component)

            # NOTE: self.component_only should be found 
            in_component_only_tree = True


        if depth > MAX_RECURSIONS:
            raise RuleInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

        new_frame = None

        parent_values_subtree = None
        if depth==0:
            # ---- Rules case -----

            # assert parent is None

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
                            in_component_only_tree=in_component_only_tree,
                            )

        elif not mode_extension_list and component.is_extension():
            # ---- Extension case -> process single or iterate all items -----


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
            current_instance_new = self._get_current_instance_new(
                                            component=component,
                                            in_component_only_tree=in_component_only_tree)


            if isinstance(instance, (list, tuple)):
                # enters recursion -> _apply() -> ...
                # parent_values_subtree = self.current_frame.parent_values_subtree

                self._apply_extension_list(
                        component=component,
                        # parent=parent,
                        in_component_only_tree=in_component_only_tree,
                        instance_list=instance,
                        current_instance_list_new=current_instance_new,
                        depth=depth,
                        )
                # ========================================
                # Finished, processed all children items
                # ========================================
                return
                # ========================================


            # ========================================
            # == Extension with single item ==
            #    will be processed as any other fields
            # ========================================
            if instance is None:
                # TODO: check that type_info.is_optional ...
                ...
            elif not is_model_instance(instance):
                raise RuleApplyValueError(owner=self, msg=f"Expected list/tuple or model instance, got: {instance} : {type(instance)}")

            new_frame = ApplyStackFrame(
                            container = component, 
                            component = component, 
                            # must inherit previous instance
                            parent_instance=self.current_frame.instance,
                            instance = instance,
                            instance_new = current_instance_new,
                            in_component_only_tree=in_component_only_tree,
                            )

        # ------------------------------------------------------------

        if not new_frame:
            # -- Fallback case --
            # register non-container frame - only component is new. take instance from previous frame
            new_frame = ApplyStackFrame(
                            component = component, 
                            # copy
                            instance = self.current_frame.instance,
                            container = self.current_frame.container, 
                            in_component_only_tree=in_component_only_tree,
                            # automatically copied
                            #   instance_new = self.current_frame.instance_new,
                            #   index0 = self.current_frame.index0,
                            )


        process_further = True

        # one level deeper
        new_frame.depth = depth + 1

        # ------------------------------------------------------------
        # ----- Main processing - must use new stack frame
        # ------------------------------------------------------------
        with self.use_stack_frame(new_frame):

            comp_key_str = self.get_key_string(component)

            # try to initialize (create value instance), but if exists, check for circular dependency loop
            current_value_instance = self.current_values.get(comp_key_str, UNDEFINED)

            if current_value_instance is NA_IN_PROGRESS:
                 raise RuleApplyError(owner=component, 
                            msg=f"The component is already in progress state. Probably circular dependency issue. Fix problematic references to this component and try again.") 
            elif current_value_instance is UNDEFINED:
                self.current_values[comp_key_str] = NA_IN_PROGRESS

            # only when full apply or partial apply
            if not (self.component_only and not in_component_only_tree):

                # TODO: maybe a bit too late to check this but this works
                if current_value_instance:
                    process_further = False
                else:
                    # ============================================================
                    # Update, validate, evaluate
                    # ============================================================
                    process_further, current_value_instance_new = self._update_and_clean(component=component, key_string=comp_key_str)
                    # ============================================================
                    if current_value_instance_new is not None:
                        current_value_instance = current_value_instance_new

                if process_further and getattr(component, "enables", None):
                    assert not getattr(component, "contains", None), component

                    value = self.get_current_value(component, strict=False)

                    # OLD: delete this - not good in mode_dexp_dependency mode
                    #       bind_dexp_result = component.get_dexp_result_from_instance(apply_session=self)
                    #       value = bind_dexp_result.value
                    if not self.defaults_mode:
                        if not isinstance(value, (bool, NoneType)):
                            raise RuleApplyValueError(owner=component, 
                                    msg=f"Component.enables can be applied only for Boolean or None values, got: {value} : {type(value)}")
                        if not value: 
                            process_further = False

            # if parent yields None - do not process on children
            if self.current_frame.instance is None \
              or self.current_frame.instance is UNDEFINED:
                process_further = False

            # ------------------------------------------------------------
            # NOTE: used only for test if all Dexp values could evaluate ...
            #       self.__check_component_all_dexps(component)

            # -- fill values dict
            # TODO: consider: recursive=True - currently causes lot of issues
            self._fill_values_dict(filler="_apply", is_init=(depth==0), process_further=process_further) 

            if process_further:
                # ------------------------------------------------------------
                # --- Recursive walk down - for each child call _apply
                # ------------------------------------------------------------
                for child in component.get_children():
                    self._apply(
                                # parent=component, 
                                # in_component_only_tree=in_component_only_tree,
                                component=child, 
                                # depth=depth+1,
                                )
                # TODO: consider to reset - although should not influence since stack_frame will be disposed
                # self.current_frame.set_parent_values_subtree(parent_values_subtree)

            # if self.current_frame.component.is_container():
            # Fill internal cache for possible later use by some `dump_` functions



        if depth==0:
            assert len(self.stack_frames)==0

            # ---------------------------------------------------------------------
            # For changed attributes -> collect original + new value
            # ---------------------------------------------------------------------
            updated_values_dict: Dict[KeyString, Dict[AttrName, Tuple[AttrValue, AttrValue]]] = defaultdict(dict)
            for key_string, instance_attr_values in self.update_history.items():
                if len(instance_attr_values)>1:
                    # any change?
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

    def _apply_extension_list(self, 
            component: ComponentBase,
            # parent: Optional[ComponentBase], 
            in_component_only_tree:bool,
            instance_list: List[ModelType],
            current_instance_list_new: Union[NoneType, UNDEFINED, ModelType],
            depth: int,
            ):
        """
        Extension with item List
        Recursion -> _apply(mode_extension_list=True) -> ...
        """

        if not isinstance(instance_list, (list, tuple)):
            raise RuleApplyValueError(owner=self, msg=f"{component}: Expected list/tuple in the new instance, got: {current_instance_list_new}")

        # instance_list = instance

        # TODO: validate cardinality before or after changes

        new_instances_by_key = None
        if current_instance_list_new not in (None, UNDEFINED):
            if not isinstance(current_instance_list_new, (list, tuple)):
                raise RuleApplyValueError(owner=self, msg=f"{component}: Expected list/tuple in the new instance, got: {current_instance_list_new}")

            new_instances_by_key = {}
            for index0, item_instance_new in enumerate(current_instance_list_new, 0):
                key = component.get_key_pairs_or_index0(instance=item_instance_new, index0=index0)
                if key in new_instances_by_key:
                    raise RuleApplyValueError(owner=self, msg=f"{component}: Duplicate key {key}, first item is: {new_instances_by_key[key]}")
                new_instances_by_key[key] = item_instance_new


        parent_instance = self.current_frame.instance

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

            if current_instance_list_new not in (None, UNDEFINED):
                item_instance_new = new_instances_by_key.get(key, UNDEFINED)
                if item_instance_new is UNDEFINED:
                    key_string = self.get_key_string_by_instance(
                            component = component, 
                            instance = instance, 
                            parent_instance=parent_instance,
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
                                parent_instance=parent_instance,
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

        if instances_by_key:

            # -- fill values dict
            self._fill_values_dict(filler="ext_list", component=component, is_init=False, process_further=True, extension_items_mode=True)

            for key, (instance, index0, item_instance_new) in instances_by_key.items():
                # Go one level deeper 
                with self.use_stack_frame(
                        ApplyStackFrame(
                            container = component,
                            component = component, 
                            index0 = index0,
                            # main instance - original values
                            instance = instance, 
                            parent_instance=parent_instance,
                            # new instance - new values (when update mode)
                            instance_new = item_instance_new, 
                            parent_instance_new=self.current_frame.instance_new,
                            in_component_only_tree=in_component_only_tree,
                            depth=depth+1,
                            )):
                    # ------------------------------------------------
                    # Recursion with prevention to hit this code again
                    # ------------------------------------------------
                    self._apply(
                                # parent=parent, 
                                component=component, 
                                mode_extension_list=True,
                                # in_component_only_tree=in_component_only_tree,
                                # depth=depth+1,
                                # prevent is_extension_logic again -> infinitive recursion
                                )

            # TODO: consider to reset - although should not influence since stack_frame will be disposed
            # self.current_frame.set_parent_values_subtree(parent_values_subtree)

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
        self._apply(
                # parent=None, 
                component=self.rules,
                entry_call=True,
                )
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

    def _update_and_clean(self, component: ComponentBase, key_string=KeyString) -> (bool, InstanceAttrCurrentValue):
        """ returns if children should be processed 
                False - when available yields False
                False - when validation fails
                True - everything is clean, update went well

            2nd return value:
                current_value_instance - object which holds evaluated value

        it does following:
            1. init change history by bind of self.instance
            2. update value by bind of self.instance_new
            3. call cleaners - validations and evaluations in given order
            4. validate type is ok?
        """
         
        if getattr(component, "available", None):
            not_available_dexp_result = execute_available_dexp(
                                                component.available, 
                                                apply_session=self)
            if not_available_dexp_result: 
                return False, None

        if getattr(component, "bind", None):
            # --- 1. Fill initial value from instance 
            key_str = self.get_key_string(component)

            # TODO: NA_DEFAULTS_MODE
            if self.current_values.get(key_str, None) not in (
                    None, NA_IN_PROGRESS, NOT_APPLIABLE):
                # Call to "self._init_by_bind_dexp()" is already done 
                # in building tree values in some validations. 
                # In this case fetch that existing value.
                assert len(self.update_history[key_str]) == 1, "expected only initial value, got updated value too"
                instance_attr_value = self.update_history[key_str][-1]
                init_bind_dexp_result = instance_attr_value.dexp_result
            else:
                init_bind_dexp_result = self._init_by_bind_dexp(component)

            # --- 2. try to update if instance_new is provided and yields different value
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
                    # --- 3.a. run validations
                    # returns validation_failure
                    if not self.defaults_mode:
                        if self.execute_validation(component=component, validation=cleaner):
                            all_ok = False
                elif isinstance(cleaner, EvaluationBase):
                    # --- 3.b. run evaluation
                    if not bind_dexp_result:
                        # TODO: this belongs to Setup phase
                        raise RuleApplyError(owner=self, msg="Evaluator can be defined only for components with 'bind' defined. Remove 'Evaluation' or define 'bind'.")
                    self.execute_evaluation(component=component, evaluation=cleaner)
                else:
                    raise RuleApplyError(owner=self, msg=f"Unknown cleaner type {type(cleaner)}. Expected Evaluation or Validation.")

        if self.defaults_mode and isinstance(component, FieldBase):
            # NOTE: change NA_DEFAULTS_MODE to most reasonable and
            #       common empty value -> None
            value = self.get_current_value(component, strict=True)
            if value is NA_DEFAULTS_MODE:
                self.register_instance_attr_change(
                        component=component, 
                        dexp_result=None,
                        new_value=None
                        )


        # --- 4.1 finalize last value and mark as finished
        current_value_instance = self.current_values[key_string]
        if current_value_instance is NA_IN_PROGRESS:
            current_value_instance = NOT_APPLIABLE
            self.current_values[key_string] = current_value_instance

        elif current_value_instance is not UNDEFINED \
          and current_value_instance is not NOT_APPLIABLE:

            if not isinstance(current_value_instance, InstanceAttrCurrentValue):
                raise RuleInternalError(owner=self, msg=f"Unexpected current value instance found for {key_string}, got: {current_value_instance}")  
            current_value_instance.mark_finished()


        # --- 4.2 validate type is ok?
        # NOTE: initial value from instance is not checked - only
        #       intermediate and the last value (returns validation_failure)
        if self.validate_type(component, strict=True):
            all_ok = False

        return all_ok, current_value_instance

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

    def get_key_string(self, component: ComponentBase, depth:int=0, force:bool=False) -> KeyString:
        """
        Recursion
        Caching is on containers only - by id(indstance)
        for other components only attach name to it.
        Container - when not found then gets intances and index0 
        from current frame
        """
        if depth > MAX_RECURSIONS:
            raise RuleInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

        # Started to process extension, but not yet positioned on any extension instance item 
        # the key will have no extension::<instance_id>, just owner_key_string::owner_key_string::extension_name
        extension_no_instance_case = (component.get_container_owner(consider_self=True) 
                                      != 
                                      self.current_frame.component.get_container_owner(consider_self=True))

        # NOTE: this could be different 
        #       component == self.current_frame.component

        if component.is_container() and not extension_no_instance_case:
            instance = self.current_frame.instance
            parent_instance = self.current_frame.parent_instance
            index0 = self.current_frame.index0

            key_string = self.get_key_string_by_instance(
                            component = component,
                            instance = instance, 
                            parent_instance=parent_instance,
                            index0 = index0,
                            force=force,
                            )
        else:
            consider_self = False if extension_no_instance_case else True
            container = component.get_container_owner(consider_self=consider_self)

            # Recursion
            container_key_string = self.get_key_string(container, depth=depth+1)

            # construct
            key_string = GlobalConfig.ID_NAME_SEPARATOR.join(
                    [container_key_string, component.name] 
                    )

        return key_string

    # ------------------------------------------------------------

    def get_key_string_by_instance(self, component: ComponentBase, instance: ModelType, parent_instance: ModelType, index0: Optional[int], force:bool=False) -> str:
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
        if key_string is None or force:

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
            else:
                key_string = component.name

            if parent_instance:
                # prepend parent key_string(s)

                # parent_instance = self.current_frame.parent_instance
                parent_id = id(parent_instance)
                # if parent_id not in self.key_string_container_cache:
                #     # must_be_in_cache
                #     if not self.component_name_only:
                #         raise RuleInternalError(owner=component, msg=f"Parent instance's key not found in cache, got: {parent_instance}") 
                #     parent_key_string = f"__PARTIAL__{self.component_name_only}"
                # else:
                parent_key_string = self.key_string_container_cache[parent_id]
                key_string = GlobalConfig.ID_NAME_SEPARATOR.join([parent_key_string, key_string])
            else:
                container_owner = component.get_container_owner(consider_self=True)
                if container_owner.is_extension():
                    raise RuleInternalError(owner=component, msg=f"Owner container {container_owner.name} is an extension and parent_instance is empty") 


            self.key_string_container_cache[instance_id] = key_string
            self.instance_by_key_string_cache[key_string] = instance
        #     from_cache = "new"
        # else:
        #     from_cache = "cache"

        # TODO: self.config.logger.debug("cont:", component.name, key_string, f"[{from_cache}]")

        return key_string

    # ------------------------------------------------------------

    # def is_component_instance_processed(self, component: ComponentBase) -> Optional[KeyString]:
    #     # instance is grabbed from current_frame
    #     assert self.stack_frames
    #     key_str = self.get_key_string(component)
    #     return key_str if self.current_values.get(key_str, None) else None

    # def is_apply_component_instance_in_progress(self, component: ComponentBase) -> Optional[KeyString]:
    #     if self.stack_frames:
    #         key_str = self.get_key_string(component)
    #         if self.current_values.get(key_str, None) is NA_IN_PROGRESS:
    #             return key_str
    #     return None

    # ------------------------------------------------------------

    def get_current_value_instance(self, 
                                   component: ComponentBase, 
                                   init_when_missing:bool=False
                                   ) -> InstanceAttrCurrentValue:
        """ if not found will return UNDEFINED
            Probaly a bit faster, only dict queries.
        """
        key_str = self.get_key_string(component)
        if not key_str in self.current_values:
            if not init_when_missing:
                raise RuleInternalError(owner=component, msg=f"Value fetch too early") 

            # ------------------------------------------------------------
            # NOTE: apply() / mode_dexp_dependency 
            #       complex case - value is required from node that was not yet
            #       processed. So process it now and later will be skipped when
            #       it comes in normal order.
            #       Example:
            #       apply() -> ._apply() -> DotExpression 
            #               -> get_current_value_instance() 
            #               -> ._apply()
            # 
            # This yields RECURSION! See doc for _apply() - mode_dexp_dependency
            # ------------------------------------------------------------
            self._apply(component=component, mode_dexp_dependency=True)
            # self._init_by_bind_dexp(component)
            
        attr_current_value_instance = self.current_values[key_str]
        return attr_current_value_instance

    # ------------------------------------------------------------

    def get_current_value(self, component: ComponentBase, strict:bool) -> LiteralType:
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
        attr_current_value_instance = self.current_values[key_str]
        return attr_current_value_instance.get_value(strict=strict)

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

    def get_values_tree(self, key_string: Optional[KeyString] = None) -> ComponentTreeWValuesType:
        # component: ComponentBase
        """
        will go recursively through every children and
        fetch their "children" and collect to output structure.
        selects all nodes, put in tree, includes self
        for every node bind (M.<field>) is evaluated
        """
        if self.values_tree is None:
            raise RuleInternalError(owner=self, msg=f"_get_values_tree() did not filled _values_tree cache") 
        if key_string is None:
            component = self.rules
            tree = self.values_tree
        else:
            if key_string not in self.values_tree_by_key_string.keys():
                if key_string not in self.current_values:
                    names_avail = get_available_names_example(key_string, self.values_tree_by_key_string.keys())
                    raise RuleInternalError(owner=self, msg=f"Key string not found in values tree, got: {key_string}. Available: {names_avail}") 

                # -- fill values dict
                assert not (self.current_frame.component==self.rules)
                # NOTE: can trigger recursion 
                # extension_items_mode = False, component = component, 
                self._fill_values_dict(filler="get_values_tree", is_init=False, recursive=True)

            tree = self.values_tree_by_key_string[key_string]

        return tree

    # ------------------------------------------------------------

    def _fill_values_dict(self, 
                          filler:str,
                          is_init:bool, 
                          process_further:bool=True,
                          extension_items_mode: bool = False,
                          component: Optional[ComponentBase] = None, 
                          recursive: bool = False,
                          depth: int=0,
                          ) -> ComponentTreeWValuesType:
        " see unit test for example - test_dump.py "
        if depth > MAX_RECURSIONS:
            raise RuleInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

        if not component:
            component = self.current_frame.component
        # else:
        #     # fetching values not allowed since stack is maybe not set up correctly
        #     assert not getattr(component, "bind", None), component

        # process_further currently not used
        if is_init:
            assert not extension_items_mode
            assert self.values_tree_by_key_string is None
            self.values_tree_by_key_string = {}
        else:
            assert self.values_tree_by_key_string is not None

        # -- fill cache by key_string 
        key_string = self.get_key_string(component)

        if key_string in self.values_tree_by_key_string:
            # already in cache
            values_dict = self.values_tree_by_key_string[key_string] 
            return values_dict

        # -- create a values_dict for this component
        filler = filler + ("+RECURSIVE" if recursive else "") + (f"+{depth}" if depth or recursive else "") 

        values_dict = {}

        # set immediatelly - to avoid duplicate calls in 
        self.values_tree_by_key_string[key_string] = values_dict

        values_dict["name"] = component.name
        # DEBUG: values_dict["filler"] = filler
        attr_current_value_instance = None
        if getattr(component, "bind", None):
            # can trigger recursion - filling tree 
            attr_current_value_instance = \
                    self.get_current_value_instance(
                        component=component, init_when_missing=True)

            if attr_current_value_instance is not NOT_APPLIABLE:
                if attr_current_value_instance is UNDEFINED:
                    raise RuleInternalError(owner=component, msg="Not expected to have undefined current instance value") 
            else:
                attr_current_value_instance = None

        if attr_current_value_instance is not None:
            values_dict["attr_current_value_instance"] = attr_current_value_instance
            # values_dict["key_string"] = attr_current_value_instance.key_string

        # -- add to parent object or call recursion and fill the tree completely

        if extension_items_mode:
            self.current_frame.parent_values_subtree.append(values_dict)
            values_dict["extension_items"] = []
            self.current_frame.set_parent_values_subtree(values_dict["extension_items"])
            if recursive:
                raise RuleInternalError(owner=component, msg=f"Did not implement this case - extension_items + recursive") 
        else:
            if is_init:
                assert not recursive, "did not consider this case"
                # only first object is a dict
                parent_values_subtree = values_dict
                assert self.values_tree is None
                self.values_tree = parent_values_subtree
            else:
                parent_values_subtree = self.current_frame.parent_values_subtree
                # all next objects are lists
                assert isinstance(parent_values_subtree, list)

                # TODO: explain. For now: don't ask ... some special case
                #       needed to be skipped - otherwise duplicate and
                #       mislocated items
                if not (recursive and depth>0):
                    parent_values_subtree.append(values_dict)

            children = component.get_children()
            if children: 
                if recursive:
                    values_dict["contains"] = []
                    for child_component in component.get_children():
                        # recursion
                        child_values_dict = self._fill_values_dict(
                                filler=filler,
                                is_init=False,
                                component=child_component, 
                                process_further=process_further,
                                recursive=recursive,
                                depth=depth+1)
                        values_dict["contains"].append(child_values_dict)
                    # set to None to prevent further filling 
                    self.current_frame.set_parent_values_subtree(None) # parent_values_subtree)
                else:
                    # recursion on the caller, setup target, caller will fill everything
                    values_dict["contains"] = []
                    self.current_frame.set_parent_values_subtree(values_dict["contains"])


        return values_dict

    # ------------------------------------------------------------

    def dump(self) -> ValuesTree:
        """
        Recursively traverse children's tree and and collect current values to
        recursive output dict structure.
        Everything should be already cached.
        """
        tree: ComponentTreeWValuesType = self.get_values_tree()
        return self._dump_values(tree)

    # ------------------------------------------------------------

    def _dump_defaults(self) -> ValuesTree:
        """
        In defaults_mode - recursively traverse children's tree and and collect
        current values to recursive output dict structure.
        """
        assert self.defaults_mode
        tree: ComponentTreeWValuesType = self.get_values_tree()
        return self._dump_values(tree)

    # ------------------------------------------------------------


    def _dump_values(self, tree: ComponentTreeWValuesType, depth: int=0) -> ValuesTree:
        # recursion
        if depth > MAX_RECURSIONS:
            raise RuleInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

        output = {}
        # component's name
        output["name"] = tree["name"]

        if "attr_current_value_instance" in tree:
            output["value"] = tree["attr_current_value_instance"].get_value(strict=True)

        self._dump_values_children(tree=tree, output=output, key_name="contains", depth=depth)
        self._dump_values_children(tree=tree, output=output, key_name="extension_items", depth=depth)

        return output

    def _dump_values_children(self, tree: ComponentTreeWValuesType, output: ComponentTreeWValuesType, key_name: str, depth:int):
        # recursion
        children = tree.get(key_name, None)
        if children:
            # children is renamed to contains since it is meant to be publicly
            # exposed struct
            output[key_name] = []
            for child in children:
                # recursion
                child_out = self._dump_values(child, depth+1)
                output[key_name].append(child_out)

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
