from dataclasses import (
        dataclass,
        field,
        asdict,
        is_dataclass,
        )
from typing import (
        Any,
        Dict,
        Optional,
        Tuple,
        List,
        Union,
        Type,
        )
from collections import OrderedDict, defaultdict

from .exceptions import (
        EntityApplyError,
        EntityValidationError,
        EntityInternalError,
        EntityApplyValueError,
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
        to_repr,
        )
from .meta import (
    Self,
    NoneType,
    ModelType,
    is_model_class,
    is_model_instance,
    ValuesTree,
    ComponentTreeWValuesType,
    LiteralType,
    make_dataclass_with_optional_fields,
    get_dataclass_field_type_info, dataclass_from_dict,
)
from .base import (
    IField,
    UndefinedType,
    MAX_RECURSIONS,
    AttrValue,
    AttrName,
    GlobalConfig,
    KeyString,
    IComponent,
    IApplyResult,
    ApplyStackFrame,
    ValidationFailure,
    StructEnum,
    ChangeOpEnum,
    InstanceAttrValue,
    InstanceChange,
    InstanceAttrCurrentValue,
    get_instance_key_string_attrname_pair,
    UseStackFrameCtxManagerBase,
    IContainer,
)
from .fields import (
        FieldBase,
        )

from .valid_base     import ValidationBase
from .eval_base      import EvaluationBase
from .valid_field    import FieldValidationBase
from .eval_field     import FieldEvaluationBase
from .valid_children import ChildrenValidationBase
from .eval_children  import ChildrenEvaluationBase
from .valid_items    import ItemsValidationBase
from .eval_items     import ItemsEvaluationBase

from .containers import (
        ContainerBase,
        MissingKey,
        Entity,
        SubEntityItems,
        )


class UseApplyStackFrameCtxManager(UseStackFrameCtxManagerBase):
    " with() ... custom context manager. "
    owner_session: "ApplyResult"
    frame: ApplyStackFrame


# ============================================================


@dataclass
class ApplyResult(IApplyResult):
    """ 
    ApplyResult is IApplyResult (variable apply_result) - but is renamed to
    result since it is name exposed to external API.

    Similar is Function -> IFunctionFactory.
    """

    instance_none_mode:bool = field(init=False, default=False)

    STACK_FRAME_CLASS = ApplyStackFrame
    STACK_FRAME_CTX_MANAGER_CLASS = UseApplyStackFrameCtxManager

    def __post_init__(self):
        if not isinstance(self.entity, Entity):
            raise EntityApplyError(owner=self, msg=f"Component object '{self.entity}' is not top container - Entity.")

        if self.component_name_only:
            # Will raise if component is not found
            self.component_only = self.entity.get_component(self.component_name_only)
            if not self.component_only.can_apply_partial():
                raise EntityApplyError(owner=self, msg=f"Component '{self.component_only}' does not support partial apply. Use SubEntityItems, FieldGroup or similar.")

        # self.bound_model = getattr(self.entity, "bound_model")
        self.bound_model = self.entity.bound_model
        if not self.bound_model:
            raise EntityApplyError(owner=self, msg=f"Component object '{self.entity}' has no bound model")

        if self.entity.context_class:
            if not self.context:
                raise EntityApplyError(owner=self.entity, msg=f"Pass context object to .apply*(). Context should be instance of '{self.entity.context_class}'.")
            if not isinstance(self.context, self.entity.context_class):
                raise EntityApplyError(owner=self, msg=f"Context object '{self.context}' is not instance of context class '{self.entity.context_class}'.")
        else:
            if self.context:
                raise EntityApplyError(owner=self, msg=f"Given context object '{self.context}', but context class in component is not setup. Provide 'context_class' to Entity object and try again.")

        # self.model = self.bound_model.model
        # if not self.model:
        #     raise EntityInternalError(owner=self, item=component, msg=f"Bound model '{self.bound_model}' has empty model.")

        if self.defaults_mode:
            if not (self.instance is NA_DEFAULTS_MODE and self.instance_new is None and not self.component_name_only):
                raise EntityInternalError(owner=self, msg=f"Defaults mode does not expect instance or instance_new or component_name_only, got: {self.instance} / {self.instance_new} / {self.component_name_only}") 
        else:
            if self.instance is None:
                if not (self.entity.is_top_parent()):
                    raise EntityInternalError(owner=self, msg=f"Currently implemented only for  top container objects, got: {self.entity}")
                if self.instance_new is None:
                    raise EntityApplyError(owner=self, msg=f"Both 'instance' and 'instance_new' is None, at least one should be set.")

                self.instance_none_mode = True

                # Instead of making new dataclass, filling original dataclass
                # with all fields having None would be alternative, but thiw
                # way __post_init__, __init__ and similar methods that could
                # change instance state are avoided.
                # TODO: maybe this could/should be cached by dc_model?
                temp_dataclass_model = make_dataclass_with_optional_fields(self.bound_model.model)

                # self.instance_shadow_dc = temp_dataclass_model()

                # All fields are made optional and will have value == None
                # (from field's default)
                self.instance = temp_dataclass_model()

            else:
                self.entity.value_accessor_default.validate_instance_type(owner_name=self.entity.name,
                                                                          instance=self.instance,
                                                                          model_type=self.bound_model.model)

            if self.instance_new is not None and not self.component_name_only:
                self._detect_instance_new_struct_type(self.entity)

        # ----------------------------------------
        # see IApplyResult for description
        self.binary_operations_type_adapters[(str, int)] = str


    def is_ok(self) -> bool:
        return bool(self.finished) and not bool(self.errors)

    # ------------------------------------------------------------

    # def use_stack_frame(self, frame: ApplyStackFrame) -> UseApplyStackFrameCtxManager:
    #     if not isinstance(frame, ApplyStackFrame):
    #         raise EntityInternalError(owner=self, msg=f"Expected ApplyStackFrame, got frame: {frame}") 

    #     return UseApplyStackFrameCtxManager(owner_session=self, frame=frame)

    # ------------------------------------------------------------

    def raise_if_failed(self):
        if not self.finished:
            raise EntityApplyError(owner=self, msg="Apply process is not finished")

        if self.errors:
            raise EntityValidationError(owner=self.entity, errors=self.errors)

    # ------------------------------------------------------------

    def _execute_cleaners(self,
                          component: IComponent,
                          validation_class: Type[ValidationBase],
                          evaluation_class: Type[EvaluationBase],
                          ) -> ExecResult:

        assert issubclass(validation_class, ValidationBase)
        assert issubclass(evaluation_class, EvaluationBase)

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

                # TODO: self.config.logger.warning(f"{'  ' * self.current_frame.depth} clean : {component.name} -> {cleaner.name}")
                if isinstance(cleaner, validation_class):
                    # --- 3.a. run validations
                    # returns validation_failure
                    if not self.defaults_mode:
                        if self._execute_validation(component=component, validation=cleaner):
                            all_ok = False
                elif isinstance(cleaner, evaluation_class):
                    # --- 3.b. run evaluation
                    # if not bind_dexp_result:
                    #     # TODO: this belongs to Setup phase
                    #     raise EntityApplyError(owner=self, msg="Evaluation can be defined only for components with 'bind' defined. Remove 'Evaluation' or define 'bind'.")
                    self._execute_evaluation(component=component, evaluation=cleaner)
                elif not (isinstance(cleaner, ValidationBase) or  
                          isinstance(cleaner, EvaluationBase)):
                    raise EntityApplyError(owner=self, msg=f"Unknown cleaner type {type(cleaner)}. Expected *Evaluation or *Validation.")

        return all_ok

    # ------------------------------------------------------------

    def _execute_evaluation(self, component: IComponent, evaluation:EvaluationBase) -> ExecResult:
        """ Execute evaluation and if new value is different from existing
            value, update current instance """
        assert isinstance(evaluation, EvaluationBase)
        assert component == evaluation.parent

        if not component == self.current_frame.component:
            raise EntityInternalError(owner=self.current_frame.component, 
                    msg=f"Component in frame {self.current_frame.component} must match component: {component}")

        # evaluation_dexp = evaluation.value
        # assert isinstance(evaluation_dexp, DotExpression)
        # eval_dexp_result  = evaluation_dexp._evaluator.execute(apply_result=self)
        eval_dexp_result  = evaluation.execute(apply_result=self)

        if eval_dexp_result.is_not_available():
            return eval_dexp_result

        eval_value = eval_dexp_result.value

        if isinstance(component, FieldBase):
            eval_value = component.try_adapt_value(eval_value)

        # ALT: bind_dexp_result = component.get_dexp_result_from_instance(apply_result)
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
    def _execute_validation(self, component: IComponent, validation:ValidationBase) -> Optional[ValidationFailure]:
        """ Execute validaion - if returns False value then register error and
            mark component and children invalid to prevent further entity execution
        """
        assert isinstance(validation, ValidationBase)
        assert component == validation.parent
        assert not self.defaults_mode

        if not component == self.current_frame.component:
            raise EntityInternalError(owner=self.current_frame.component, 
                    msg=f"Component in frame {self.current_frame.component} must match component: {component}")

        # value=value, 
        validation_failure = validation.validate(apply_result=self)
        if validation_failure:
            self.register_instance_validation_failed(component, validation_failure)

        return validation_failure

    # ------------------------------------------------------------

    def register_instance_attr_change(self,
                                      component: IComponent,
                                      dexp_result: ExecResult,
                                      new_value: Any,
                                      is_from_init_bind:bool=False) -> InstanceAttrValue:

        # NOTE: new_value is required - since dexp_result.value
        #       could be unadapted (see field.try_adapt_value()

        if new_value is UNDEFINED:
            raise EntityInternalError(owner=component, msg="New value should not be UNDEFINED, fix the caller")

        # key_str = component.get_key_string(apply_result=self)
        key_str = self.get_key_string(component)

        if self.update_history.get(key_str, UNDEFINED) == UNDEFINED:
            if not is_from_init_bind:
                raise EntityInternalError(owner=component, msg=f"key_str '{key_str}' not found in update_history and this is not initialization")

            self.init_update_history_for_key(key_str)

            # Can be various UndefinedType: NA_IN_PROGRESS, NOT_APPLIABLE
            if self.current_values.get(key_str):
                raise EntityInternalError(owner=self, msg=f"current_values[{key_str}] ==  {self.current_values[key_str]}") 

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
                raise EntityInternalError(owner=component, msg=f"key_str '{key_str}' found in update_history and this is initialization")

            if not self.update_history[key_str]:
                raise EntityInternalError(owner=component, msg=f"change history for key_str='{key_str}' is empty")

            # -- check if current value is different from new one
            value_current = self.update_history[key_str][-1].value
            if value_current == new_value:
                raise EntityApplyError(owner=component, msg=f"register change failed, the value is the same: {value_current}")

            # TODO: is this really necessary - will be done in apply() later
            self.validate_type(component, strict=False, value=new_value)

            # -- parent instance
            # parent_raw_attr_value = dexp_result.value_history[-2]
            # parent_instance = parent_raw_attr_value.value

            parent_instance = self.current_frame.instance

            if parent_instance is not NA_DEFAULTS_MODE:
                # TODO: not sure if this validation is ok
                # NOTE: bound_model.model could be VExpr
                model = self.current_frame.container.bound_model.get_type_info().type_
                if not self.instance_none_mode \
                  and not isinstance(parent_instance, model):
                    raise EntityInternalError(owner=self, msg=f"Parent instance {parent_instance} has wrong type")

                # -- attr_name - fetch from initial bind dexp (very first)
                init_instance_attr_value = self.update_history[key_str][0]
                if not init_instance_attr_value.is_from_bind:
                    raise EntityInternalError(owner=self, msg=f"{init_instance_attr_value} is not from bind")
                init_bind_dexp_result = init_instance_attr_value.dexp_result
                # attribute name is in the last item

                # "for" loop is required for attributes from substructure that
                # is not done as SubEntity rather direct reference, like: 
                #   bind=M.access.alive
                # OLD - very simple logic: 
                #   init_raw_attr_value = init_bind_dexp_result.value_history[-1]
                #   attr_name = attr_name_path[-1]
                #   if not hasattr(parent_instance, attr_name):
                #       raise EntityInternalError(owner=self, msg=f"Missing {parent_instance}.{attr_name}")
                attr_name_path = [init_raw_attr_value.attr_name 
                        for init_raw_attr_value in init_bind_dexp_result.value_history
                        ]
                if not attr_name_path:
                    raise EntityInternalError(owner=self, msg=f"{component}: attr_name_path is empty")

                if isinstance(component, IField):
                    # TODO: what about Boolean + enables? Better to check .get_children() ?
                    # value accessor should be used from parent of the component
                    assert component.parent
                    value_accessor = component.parent.value_accessor
                else:
                    # contaainers + fieldgroup
                    value_accessor = component.value_accessor

                current_instance_parent = None
                current_instance = parent_instance
                for anr, attr_name in  enumerate(attr_name_path,0):
                    if current_instance is None:
                        if self.instance_none_mode:
                            # Create all missing intermediate empty dataclass objects 
                            assert anr > 0
                            attr_name_prev = attr_name_path[anr-1]

                            current_instance_type_info = get_dataclass_field_type_info(current_instance_parent, attr_name_prev)
                            if current_instance_type_info is None:
                                raise EntityInternalError(owner=self, msg=f"Attribute {attr_name} not found in dataclass definition of {current_instance_parent}.") 
                            if current_instance_type_info.is_list:
                                raise EntityInternalError(owner=self, msg=f"Attribute {attr_name} of {current_instance_parent} is a list: {current_instance_type_info}.") 

                            current_instance_model = current_instance_type_info.type_
                            if not is_dataclass(current_instance_model):
                                raise EntityInternalError(owner=self, msg=f"Attribute {attr_name} of {type(current_instance_parent)} is not a dataclass instance, got: {current_instance_model}") 

                            # set new value of temp instance attribute
                            # all attrs of a new instance will have None value (dc default)
                            temp_dataclass_model = make_dataclass_with_optional_fields(current_instance_model)
                            current_instance = temp_dataclass_model()
                            value_accessor.set_value(instance=current_instance_parent,
                                                     attr_name=attr_name_prev,
                                                     attr_index=None,
                                                     new_value=current_instance)
                            # ORIG: setattr(current_instance_parent, attr_name_prev, current_instance)
                        else:
                            attr_name_path_prev = ".".join(attr_name_path[:anr])
                            # TODO: fix this ugly validation message
                            raise EntityApplyValueError(owner=self, msg=f"Attribute '{attr_name}' can not be set while '{parent_instance}.{attr_name_path_prev}' is not set. Is '{attr_name_path_prev}' obligatory?")

                    current_instance_parent = current_instance
                    current_instance = value_accessor.get_value(instance=current_instance_parent,
                                                                attr_name=attr_name,
                                                                attr_index=None)
                    if current_instance is UNDEFINED:
                        raise EntityInternalError(owner=self, msg=f"Missing attribute:\n  Current: {current_instance}.{attr_name}\n Parent: {parent_instance}.{'.'.join(attr_name_path)}")
                    # ORIG:
                    # if not hasattr(current_instance, attr_name):
                    #     raise EntityInternalError(owner=self, msg=f"Missing attribute:\n  Current: {current_instance}.{attr_name}\n Parent: {parent_instance}.{'.'.join(attr_name_path)}")
                    # current_instance = getattr(current_instance_parent, attr_name)

                    # self.instance_shadow_dc = temp_dataclass_model()

                # ----------------------------------------
                # Finally change instance value
                # ----------------------------------------
                value_accessor.set_value(instance=current_instance_parent,
                                         attr_name=attr_name,
                                         attr_index=None,
                                         new_value=new_value)
                # setattr(current_instance_parent, attr_name, new_value)

            # NOTE: bind_dexp_result last value is not changed
            #       maybe should be changed but ... 

        # TODO: pass input arg value_parent_name - component.name does not have
        #       any purpose
        instance_attr_value = InstanceAttrValue(
                                value_parent_name=component.name, 
                                value=new_value,
                                dexp_result=dexp_result,
                                is_from_bind = is_from_init_bind,
                                # TODO: source of change ...
                                )

        self.register_instance_attr_value_change(key_str=key_str, instance_attr_value = instance_attr_value)

        return instance_attr_value


    # ------------------------------------------------------------
    # _apply() -> main apply function
    # ------------------------------------------------------------

    def _apply(self,
               # parent: Optional[IComponent],
               component: IComponent,
               # -- RECURSION -- internal props
               # see dox below
               mode_subentity_items:bool = False,
               # see dox below
               mode_dexp_dependency: bool = False,
               top_call: bool = False,
        ) -> bool:
        """
        Main entry function 'apply' logic.
        Return value is "process further / all ok" - not used by caller.
        Recursion in 3 main modes:

            normal_mode 
               for each child in children -> ._apply()

            mode_subentity_items
               if subentity_items and list - for each item in items -> ._apply()
               caller is subentity_items processing - NOT passed through recursion further

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
        assert not (mode_subentity_items and mode_dexp_dependency)

        comp_container = component.get_first_parent_container(consider_self=True)
        # comp_container_model = comp_container.bound_model.get_type_info().type_

        if mode_dexp_dependency:
            # TODO: self.config.logger.info(f"apply - mode_dexp_dependency - {self.current_frame.component.name} depends on {component.name} - calling apply() ...")

            # instance i.e. containers must match
            assert self.stack_frames
            caller_container = self.current_frame.component.get_first_parent_container(consider_self=True)
            if comp_container is not caller_container:
                raise EntityInternalError(owner=component, msg=f"Componenent's container '{comp_container}' must match caller's '{self.current_frame.component.name}' container: {caller_container.name}") 

        if self.finished:
            raise EntityInternalError(owner=self, msg="Already finished") 

        if self.stack_frames:
            assert not top_call
            depth = self.current_frame.depth
            in_component_only_tree = self.current_frame.in_component_only_tree 

            # check if instance model is ok
            if not component.is_subentity():
                if self.defaults_mode:
                    if self.current_frame.instance is not NA_DEFAULTS_MODE:
                        raise EntityInternalError(owner=self, msg=f"Defaults mode - current frame's instance's model must be NA_DEFAULTS_MODE, got: {type(self.instance)}") 
                else:
                    pass
                    # TODO: consider adding validation again with:
                    #   self.component.value_accessor.validate_instance_type(owner_name=self.component.name,
                    # if not self.instance_none_mode \
                    #   and not isinstance(self.current_frame.instance, comp_container_model):
                    #     raise EntityInternalError(owner=self, msg=f"Current frame's instance's model does not corresponds to component's container's model. Expected: {comp_container_model}, got: {type(self.instance)}")

            # TODO: for subentity() any need to check this at all in this phase?
        else:
            # no stack around -> initial call -> depth=0
            assert top_call
            depth = 0
            in_component_only_tree = False

            if self.defaults_mode:
                if self.instance is not NA_DEFAULTS_MODE:
                    raise EntityInternalError(owner=self, msg=f"Defaults mode - instance must be NA_DEFAULTS_MODE, got: {type(self.instance)}")
            # else:
            #     # NOTE: Dropped this check - it is done before. if it will be required:
            #     #     self.entity.value_accessor_default.validate_instance_type(instance=self.instance, bound_model=self.bound_model)
            #     if not self.instance_none_mode \
            #       and not isinstance(self.instance, comp_container_model):
            #         raise EntityInternalError(owner=self, msg=f"Entity instance's model does not corresponds to component's container's model. Expected: {comp_container_model}, got: {type(self.instance)}")

        # TODO: in partial mode this raises RecursionError:
        #       component == self.component_only
        if self.component_only and component is self.component_only:
            # partial apply - detected component
            if not mode_subentity_items and in_component_only_tree:
                raise EntityInternalError(owner=self, 
                        # {parent} -> 
                        msg=f"{component}: in_component_only_tree should be False, got {in_component_only_tree}")

            if self.instance_new not in (None, UNDEFINED) and not self.instance_new_struct_type:
                self._detect_instance_new_struct_type(component)

            # NOTE: self.component_only should be found 
            in_component_only_tree = True


        if depth > MAX_RECURSIONS:
            raise EntityInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

        if depth>0 and not mode_subentity_items and component.is_subentity_items():
            # ---- SubEntityItems - RECURSION & finish -----
            return self._apply_subentity_items(
                subentity_items=component,
                in_component_only_tree=in_component_only_tree,
                depth=depth,
            )
            # -----------------------------------------------

        new_frame = self._create_apply_stack_frame_for_component(
                                          component=component,
                                          depth=depth,
                                          in_component_only_tree=in_component_only_tree,
                                          mode_subentity_items=mode_subentity_items,
                                          )

        # ------------------------------------------------------------
        # ----- Main processing - must use new stack frame
        # ------------------------------------------------------------
        # NOTE: did not put this chunk of code in a special function
        #       in order to have simple stack trace, e.g.
        #           _apply() -> _apply() -> _apply() ....
        process_further = True

        with self.use_stack_frame(new_frame):

            comp_key_str = self.get_key_string(component)

            # try to initialize (create value instance), but if exists, check for circular dependency loop
            current_value_instance = self.current_values.get(comp_key_str, UNDEFINED)

            if current_value_instance is NA_IN_PROGRESS:
                raise EntityApplyError(owner=component, 
                            msg="The component is already in progress state. Probably circular dependency issue. Fix problematic references to this component and try again.")
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
                    all_ok, current_value_instance_new = self._update_and_clean(component=component, key_string=comp_key_str)
                    process_further = all_ok
                    # ============================================================
                    if current_value_instance_new is not None:
                        # NOTE: not used later on
                        current_value_instance = current_value_instance_new

                if process_further and getattr(component, "enables", None):
                    assert not getattr(component, "contains", None), component

                    value = self.get_current_value(component, strict=False)

                    # OLD: delete this - not good in mode_dexp_dependency mode
                    #       bind_dexp_result = component.get_dexp_result_from_instance(apply_result=self)
                    #       value = bind_dexp_result.value
                    if not self.defaults_mode:
                        if not isinstance(value, (bool, NoneType)):
                            raise EntityApplyValueError(owner=component, 
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

            # TODO: consider: recursive=True - currently causes lot of issues
            self._fill_values_dict(filler="_apply", is_init=(depth==0), process_further=process_further) 

            if process_further:
                # ------------------------------------------------------------
                # --- Recursive walk down - for each child call _apply
                # ------------------------------------------------------------
                children = component.get_children()
                for child in children:
                    # TODO: self.config.logger.warning(f"{'  ' * self.current_frame.depth} _apply: {component.name} -> {child.name}")
                    # --------------------
                    # RECURSION
                    # --------------------
                    self._apply(component=child)

                # TODO: consider to reset - although should not influence since stack_frame will be disposed
                #           self.current_frame.set_parent_values_subtree(parent_values_subtree)
                # NOTE: bind_dexp_result not used

                if children:
                    # NOTE: even in mode_subentity_items need to run ChildrenValidation
                    #       on every instance in the list - what is this moment

                    # == Create This. registry and put in local setup session
                    this_registry = component.get_this_registry()
                    with self.use_changed_current_stack_frame(
                        this_registry=this_registry
                    ):
                        all_ok = self._execute_cleaners(component,
                                validation_class=ChildrenValidationBase,
                                evaluation_class=ChildrenEvaluationBase,
                                )
                    # TODO: not used later
                    process_further = all_ok

            # if self.current_frame.component.is_container():
            # Fill internal cache for possible later use by some `dump_` functions

        if depth==0:
            self._apply_collect_changed_values()

        # TODO: logger: apply_result.config.logger.debug(f"depth={depth}, comp={component.name}, bind={bind} => {dexp_result}")

        if self.instance_none_mode and component.is_top_parent():
            # create new instance of bound_model.model class
            # based on temporary created dataclass
            kwargs = asdict(self.instance)
            self.instance = dataclass_from_dict(
                               dataclass_klass=self.bound_model.model,
                               values_dict=kwargs)

        return process_further

    # ------------------------------------------------------------

    def _get_subentity_model_instances(self,
                                       subentity: IContainer,
                                       in_component_only_tree: bool
                                       ) -> Tuple[ModelType, ModelType]:
        """
        evaluate bound_model.model DotExpression and get instance
        and instance_new. Applies to SubentityItems and SubentitySingle.
        """
        if not isinstance(subentity.bound_model.model, DotExpression):
            raise EntityInternalError(owner=self,
                                      msg=f"For SubEntityItems `bound_model` needs to be DotExpression, got: {subentity.bound_model.model}")

        if getattr(subentity.bound_model, "contains", None):
            raise EntityInternalError(owner=self,
                                      msg=f"For SubEntityItems complex `bound_model` is currently not supported (e.g. `contains`), use simple BoundModel, got: {subentity.bound_model}")

            # original instance
        dexp_result: ExecResult = subentity.bound_model.model \
            ._evaluator.execute_dexp(apply_result=self)
        instance = dexp_result.value

        # new instance if any
        current_instance_new = self._get_current_instance_new(
            component=subentity,
            in_component_only_tree=in_component_only_tree)

        return instance, current_instance_new

    # ------------------------------------------------------------

    def _create_apply_stack_frame_for_component(self,
                                                component: IComponent,
                                                depth: int,
                                                in_component_only_tree: bool,
                                                mode_subentity_items: bool,
                                                ) -> ApplyStackFrame:
        new_frame = None

        if depth==0:
            # ---- Entity case -----
            container: IContainer = component

            # NOTE: frame not yet set so 'self.current_frame.instance' is not available
            #       thus sending 'instance' param
            container.bound_model._apply_nested_models(
                apply_result=self,
                instance=self.instance
            )
            new_frame = ApplyStackFrame(
                container = container,
                component = container,
                instance = self.instance,
                instance_new = self.instance_new,
                in_component_only_tree=in_component_only_tree,
            )

        elif not mode_subentity_items and component.is_subentity():

            # ---- SubEntityItems case -> process single or iterate all items -----
            instance, current_instance_new = self._get_subentity_model_instances(
                component, in_component_only_tree)

            # ---- SubEntitySingle case
            if not component.is_subentity_single():
                raise EntityApplyValueError(owner=component, msg=f"Did not expect single instance: {to_repr(instance)}")

            # ========================================
            # == SubEntityItems with single item ==
            #    will be processed as any other fields
            # ========================================
            if instance is None:
                # TODO: check that type_info.is_optional ...
                ...
            elif isinstance(instance, (list, tuple)):
                raise EntityApplyValueError(owner=component, msg=f"Did not expected list/tuple, got: {instance} : {type(instance)}")

            elif not is_model_instance(instance):
                raise EntityApplyValueError(owner=component, msg=f"Expected single model instance, got: {instance} : {type(instance)}")

            new_frame = ApplyStackFrame(
                container = component,
                component = component,
                # must inherit previous instance
                parent_instance=self.current_frame.instance,
                instance = instance,
                instance_new = current_instance_new,
                in_component_only_tree=in_component_only_tree,
            )
        else:
            assert not new_frame
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
        assert new_frame

        # ------ common setup for the new_frame ----------
        # one level deeper
        new_frame.depth = depth + 1
        if isinstance(component, IField):
            #assert getattr(component, "bind", None)
            #assert not component.is_container()
            new_frame.set_this_registry(
                component.get_this_registry()
            )
        return new_frame

    # ------------------------------------------------------------

    def _apply_collect_changed_values(self):
        """
        For changed attributes -> collect original + new value
        """
        assert len(self.stack_frames) == 0
        updated_values_dict: Dict[KeyString, Dict[AttrName, Tuple[AttrValue, AttrValue]]] = defaultdict(dict)
        for key_string, instance_attr_values in self.update_history.items():
            # any change?
            if len(instance_attr_values) > 1:
                first_val = instance_attr_values[0]
                last_val = instance_attr_values[-1]
                if last_val.value != first_val.value:
                    instance_key_string, attrname = \
                        get_instance_key_string_attrname_pair(key_string)
                    updated_values_dict[KeyString(instance_key_string)][attrname] = \
                        (first_val.value, last_val.value)

        for instance_key_string, updated_values in updated_values_dict.items():
            instance_updated = self.get_instance_by_key_string(instance_key_string)
            # TODO: key_pairs are currently not set - should be cached, not so easy to get in this moment
            #   key = component.get_key_pairs_or_index0(instance_updated, index0)
            self.changes.append(
                InstanceChange(
                    key_string=instance_key_string,
                    # see above
                    key_pairs=None,
                    operation=ChangeOpEnum.UPDATE,
                    instance=instance_updated,
                    updated_values=updated_values,
                ))


    # ------------------------------------------------------------

    def _apply_subentity_items(self,
                               subentity_items: IContainer,
                               in_component_only_tree:bool,
                               depth: int,
                               ) -> bool:
        """
        SubEntityItems with item List
        Recursion -> _apply(mode_subentity_items=True) -> ...
        """
        instance_list, current_instance_list_new = self._get_subentity_model_instances(
                    subentity_items, in_component_only_tree)

        if instance_list is None:
            # TODO: checkk type_info is optional - similar to single case
            ...
        elif not isinstance(instance_list, (list, tuple)):
            raise EntityApplyValueError(owner=subentity_items, msg=f"Expecting list of instances, got: {to_repr(instance_list)}")

        # ---- SubEntityItems case
        if not subentity_items.is_subentity_items():
            raise EntityApplyValueError(owner=subentity_items, msg=f"Did not expect list of instances: {to_repr(instance_list)}")
            # enters recursion -> _apply() -> ...
        # parent_values_subtree = self.current_frame.parent_values_subtree

        if instance_list is None:
            # NOTE: found no better way to do it
            instance_list = []
        elif not isinstance(instance_list, (list, tuple)):
            raise EntityApplyValueError(owner=self, msg=f"{subentity_items}: Expected list/tuple in the new instance, got: {current_instance_list_new}")

        # instance_list = instance

        # TODO: validate cardinality before or after changes

        new_instances_by_key = None
        if current_instance_list_new not in (None, UNDEFINED):
            if not isinstance(current_instance_list_new, (list, tuple)):
                raise EntityApplyValueError(owner=subentity_items, msg=f"Expected list/tuple in the new instance, got: {current_instance_list_new}")

            new_instances_by_key = {}
            for index0, item_instance_new in enumerate(current_instance_list_new, 0):
                key = subentity_items.get_key_pairs_or_index0(instance=item_instance_new, index0=index0, apply_result=self)
                if key in new_instances_by_key:
                    raise EntityApplyValueError(owner=subentity_items, msg=f"Duplicate key {key}, first item is: {new_instances_by_key[key]}")
                new_instances_by_key[key] = item_instance_new


        parent_instance = self.current_frame.instance

        # NOTE: considered to use dict() since dictionaries are ordered in Python 3.6+ 
        #       ordering-perserving As of Python 3.7, this is a guaranteed, i.e.  Dict keeps insertion order
        #       https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6
        instances_by_key = OrderedDict()
        for index0, instance in enumerate(instance_list, 0):
            key = subentity_items.get_key_pairs_or_index0(instance=instance, index0=index0, apply_result=self)
            if subentity_items.keys:
                missing_keys = [kn for kn, kv in key if isinstance(kv, MissingKey)]
                if missing_keys:
                    raise EntityApplyValueError(owner=self, msg=f"Instance {instance} has key(s) with value None, got: {', '.join(missing_keys)}")

            if current_instance_list_new not in (None, UNDEFINED):
                item_instance_new = new_instances_by_key.get(key, UNDEFINED)
                if item_instance_new is UNDEFINED:
                    key_string = self.get_key_string_by_instance(
                            component = subentity_items,
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
                raise EntityApplyValueError(owenr=self, msg=f"Found duplicate key {key}:\n  == {instance}\n  == {instances_by_key[key]}")

            instances_by_key[key] = (instance, index0, item_instance_new)


        if new_instances_by_key:
            index0_new = len(instances_by_key)
            new_keys = [key for key in new_instances_by_key.keys() if key not in instances_by_key]
            # register new items and add to processing
            for key in new_keys:
                item_instance_new = new_instances_by_key[key]

                key_string = self.get_key_string_by_instance(
                                component = subentity_items,
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
            self._fill_values_dict(filler="subentity_items", component=subentity_items, is_init=False, process_further=True, subentity_items_mode=True)

            for key, (instance, index0, item_instance_new) in instances_by_key.items():
                # Go one level deeper 
                with self.use_stack_frame(
                        ApplyStackFrame(
                            container = subentity_items,
                            component = subentity_items,
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
                    # RECURSION + prevent not to hit this code again
                    # ------------------------------------------------
                    self._apply(
                                component=subentity_items,
                                mode_subentity_items=True,
                                )

            # TODO: consider to reset - although should not influence since stack_frame will be disposed
            # self.current_frame.set_parent_values_subtree(parent_values_subtree)

        with self.use_stack_frame(
                ApplyStackFrame(
                    component = subentity_items,
                    # instance is a list of items
                    instance = instance_list,
                    instance_is_list = True,
                    # ALT: self.current_frame.container
                    container = subentity_items,
                    parent_instance=self.current_frame.instance,
                    in_component_only_tree=in_component_only_tree,
                    # NOTE: instance_new skipped - (contains list of
                    #       new items) are already applied
                )) as current_frame:

            # setup this registry
            current_frame.set_this_registry(
                subentity_items.get_this_registry()
            )

            # finally apply validations on list of items
            all_ok = self._execute_cleaners(subentity_items,
                                            validation_class=ItemsValidationBase,
                                            evaluation_class=ItemsEvaluationBase,
                                            )
        return all_ok

    # ------------------------------------------------------------

    def __check_component_all_dexps(self, component: IComponent):
        # used only for testing
        for attr_name, attr_value in vars(component).items():
            if isinstance(attr_value, DotExpression):
                # dexp_result: ExecResult = 
                attr_value._evaluator.execute_dexp(apply_result=self)
                # TODO: apply_result.config.logger.debug(f"{parent.name if parent else ''}.{component.name}.{attr_name} = DExp[{attr_value}] -> {dexp_result}")

    # ------------------------------------------------------------

    def apply(self) -> Self:
        """
        Main function for parsing, validating and evaluating input instance.
        Returns ApplyResult object.

        if all ok - Result.instance contains a new instance (clone + update) of the bound model type 
        if not ok - errors contain all details.
        """
        self._apply(
                # parent=None, 
                component=self.entity,
                top_call=True,
                )
        self.finish()

        return self

    # ------------------------------------------------------------

    def get_instance_by_key_string(self, key_string: str) -> ModelType:
        " must exist in cache - see previous method which sets self.get_key_string_by_instance(container) "
        return self.instance_by_key_string_cache[key_string]

    # ------------------------------------------------------------

    def _detect_instance_new_struct_type(self, component: IComponent) -> StructEnum:

        assert self.instance_new not in (None, UNDEFINED)

        if isinstance(component, ContainerBase):
            # full or partial on container
            model = component.bound_model.type_info.type_
        else:
            # FieldGroup supported only - partial with matched compoonent
            # raise NotImplementedError(f"TODO: currently not supported: {component}")
            container = component.get_first_parent_container(consider_self=True)
            model = container.bound_model.type_info.type_

        if isinstance(self.instance_new, (list, tuple)):
            if not self.instance_new:
                raise EntityInternalError(owner=self, msg="Instance is an empty list, can not detect type of base  structure")
            # test only first
            instance_to_test = self.instance_new[0]
        else:
            instance_to_test = self.instance_new

        if isinstance(instance_to_test, model):
            instance_new_struct_type = StructEnum.MODELS_LIKE
        elif is_model_class(instance_to_test.__class__):
            # TODO: it could be StructEnum.MODELS_LIKE too, but how to detect this? input param or?
            instance_new_struct_type = StructEnum.ENTITY_LIKE
        else:
            raise EntityApplyError(owner=self, 
                    msg=f"Object '{instance_to_test}' is not instance of bound model '{model}' and not model class: {type(instance_to_test)}.")

        self.instance_new_struct_type = instance_new_struct_type
        # return instance_new_struct_type


    # ------------------------------------------------------------


    def _get_current_instance_new(self, component: IComponent, in_component_only_tree:bool):
        if self.instance_new_struct_type is None:
            current_instance_new = None
        elif self.instance_new_struct_type == StructEnum.MODELS_LIKE:
            assert isinstance(component.bound_model.model, DotExpression), component.bound_model.model

            if self.current_frame.instance_new not in (None, UNDEFINED):

                # if partial - then dexp must know - this value is set only in this case
                if in_component_only_tree and component is self.component_only:
                    # SubEntityItems or FieldGroup is root
                    on_component_only = component
                else:
                    on_component_only = None

                # container = component.get_first_parent_container(consider_self=True) if not component.is_subentity() else component
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
                                                apply_result=self, 
                                                )
                # set new value
                current_instance_new = dexp_result.value
                if frame.bound_model_root.type_info.is_list and not isinstance(current_instance_new, (list, tuple)):
                    # TODO: 
                    current_instance_new = [current_instance_new]
            else:
                current_instance_new = None

        elif self.instance_new_struct_type == StructEnum.ENTITY_LIKE:
            exec_result = self.get_attr_value_by_comp_name(
                                component=component, 
                                instance=self.current_frame.instance_new)
            current_instance_new = exec_result.value
        else: 
            raise EntityInternalError(owner=self, msg=f"Invalid instance_new_struct_type = {self.instance_new_struct_type}")

        return current_instance_new


    # ============================================================
    # _update_and_clean - Update, validate, evaluate
    # ============================================================

    def _update_and_clean(self, component: IComponent, key_string=KeyString) -> (bool, InstanceAttrCurrentValue):
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
                                                apply_result=self)
            if not_available_dexp_result: 
                return False, None

        if isinstance(component, IField):
            assert getattr(component, "bind", None)
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
            # TODO: self.config.logger.warning(f"{'  ' * self.current_frame.depth} update: {component.name}")
        else:
            bind_dexp_result = None  # noqa: F841

        # NOTE: bind_dexp_result not used
        all_ok = self._execute_cleaners(component,
                validation_class=FieldValidationBase,
                evaluation_class=FieldEvaluationBase,
                )

        if self.defaults_mode and isinstance(component, FieldBase):
            # NOTE: change NA_DEFAULTS_MODE to most reasonable and
            #       common empty value -> None
            value = self.get_current_value(component, strict=False)
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
                raise EntityInternalError(owner=self, msg=f"Unexpected current value instance found for {key_string}, got: {current_value_instance}")  
            current_value_instance.mark_finished()


        # --- 4.2 validate type is ok?
        # NOTE: initial value from instance is not checked - only
        #       intermediate and the last value (returns validation_failure)
        if self.validate_type(component, strict=True):
            all_ok = False

        return all_ok, current_value_instance

    # ------------------------------------------------------------

    def _init_by_bind_dexp(self, component: IComponent) -> ExecResult:
        " get initial dexp value, if instance_new try to updated/overwrite with it"

        if not isinstance(component, FieldBase):
            raise EntityInternalError(owner=self, msg=f"Expected FieldBase field, got: {component}")

        bind_dexp_result = component.get_dexp_result_from_instance(apply_result=self)
        init_value = bind_dexp_result.value

        self.register_instance_attr_change(
                component = component, 
                dexp_result = bind_dexp_result, 
                new_value = init_value,
                is_from_init_bind = True)

        return bind_dexp_result


    # ------------------------------------------------------------

    def _try_update_by_instance(self, component: IComponent, init_bind_dexp_result: ExecResult) \
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
                            component.get_dexp_result_from_instance(apply_result=self)
                    new_value = instance_new_bind_dexp_result.value

            elif self.instance_new_struct_type == StructEnum.ENTITY_LIKE:
                instance_new_bind_dexp_result = self.get_attr_value_by_comp_name(
                                                    component=component,
                                                    instance=self.current_frame.instance_new)
                new_value = instance_new_bind_dexp_result.value
            else: 
                raise EntityInternalError(owner=self, msg=f"Invalid instance_new_struct_type = {self.instance_new_struct_type}")

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

    def get_key_string(self, component: IComponent, depth:int=0, force:bool=False) -> KeyString:
        """
        Recursion
        Caching is on containers only - by id(indstance)
        for other components only attach name to it.
        Container - when not found then gets intances and index0 
        from current frame
        """
        if depth > MAX_RECURSIONS:
            raise EntityInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

        # Started to process subentity_items, but not yet positioned on any subentity_items instance item 
        # the key will have no subentity_items::<instance_id>, just parent_key_string::parent_key_string::subentity_items_name
        subentity_items_no_instance_case = (component.get_first_parent_container(consider_self=True) 
                                      != 
                                      self.current_frame.component.get_first_parent_container(consider_self=True))

        # NOTE: this could be different 
        #       component == self.current_frame.component

        if component.is_container() and not subentity_items_no_instance_case:
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
            consider_self = False if subentity_items_no_instance_case else True
            container = component.get_first_parent_container(consider_self=consider_self)

            # Recursion
            container_key_string = self.get_key_string(container, depth=depth+1)

            # construct
            key_string = GlobalConfig.ID_NAME_SEPARATOR.join(
                    [container_key_string, component.name] 
                    )

        return key_string

    # ------------------------------------------------------------

    def get_key_string_by_instance(self, component: IComponent, instance: ModelType, parent_instance: ModelType, index0: Optional[int], force:bool=False) -> KeyString:
        # apply_result:IApplyResult,  -> self
        """
        Two cases - component has .keys or not:

        a) with keys:
            For containers which have keys defined, it is assumed that one key is
            globally unique within SubEntityItems components, so no need to prefix key
            with parent key_string. Example:

                 address_set_ext[id2=1]

        b) without keys - index0 based:
            In other cases item index in list is used (index0), and then this is
            only locally within one instance of parent, therefore parent
            key_string is required. Example:

                 company_entity::address_set_ext[0]

        """
        if not component.is_container():
            raise EntityInternalError(owner=component, msg=f"Expecting container, got: {component}") 

        instance_id = id(instance)
        key_string = self.key_string_container_cache.get(instance_id, None)
        if key_string is None or force:

            if component.keys:
                assert component.is_container()
                key_pairs = component.get_key_pairs(instance, apply_result=self)
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
                #         raise EntityInternalError(owner=component, msg=f"Parent instance's key not found in cache, got: {parent_instance}")
                #     parent_key_string = f"__PARTIAL__{self.component_name_only}"
                # else:
                parent_key_string = self.key_string_container_cache[parent_id]
                key_string = GlobalConfig.ID_NAME_SEPARATOR.join([parent_key_string, key_string])
            else:
                container_parent = component.get_first_parent_container(consider_self=True)
                if container_parent.is_subentity():
                    raise EntityInternalError(owner=component, msg=f"Parent container {container_parent.name} is an SubEntitySingle/SubEntityItems and parent_instance is empty") 


            self.key_string_container_cache[instance_id] = key_string
            self.instance_by_key_string_cache[key_string] = instance
        #     from_cache = "new"
        # else:
        #     from_cache = "cache"

        # TODO: self.config.logger.debug("cont:", component.name, key_string, f"[{from_cache}]")

        return KeyString(key_string)

    # ------------------------------------------------------------

    # def is_component_instance_processed(self, component: IComponent) -> Optional[KeyString]:
    #     # instance is grabbed from current_frame
    #     assert self.stack_frames
    #     key_str = self.get_key_string(component)
    #     return key_str if self.current_values.get(key_str, None) else None

    # def is_apply_component_instance_in_progress(self, component: IComponent) -> Optional[KeyString]:
    #     if self.stack_frames:
    #         key_str = self.get_key_string(component)
    #         if self.current_values.get(key_str, None) is NA_IN_PROGRESS:
    #             return key_str
    #     return None

    # ------------------------------------------------------------

    def get_current_value_instance(self,
                                   component: IComponent,
                                   init_when_missing:bool=False
                                   ) -> InstanceAttrCurrentValue:
        """ if not found will return UNDEFINED
            Probaly a bit faster, only dict queries.
        """
        key_str = self.get_key_string(component)
        if key_str not in self.current_values:
            if not init_when_missing:
                raise EntityInternalError(owner=component, msg="Value fetch too early") 

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
        # TODO: self.config.logger.warning(f"{'  ' * self.current_frame.depth} value: {component.name}")
        return attr_current_value_instance

    # ------------------------------------------------------------

    def get_current_value(self, component: IComponent, strict:bool) -> LiteralType:
        # apply_result:IApplyResult
        """ Could work on non-stored fields.
            Probaly a bit faster, only dict queries.
        """
        # ALT: from update_history: 
        #       instance_attr_value = apply_result.update_history[key_str][-1]
        #       return instance_attr_value.value
        # ALT: fetch from:
        #       bind_dexp: DotExpression = getattr(component, "bind", None)
        #       bind_dexp._evaluator.execute()
        # key_str = component.get_key_string(apply_result=self)
        key_str = self.get_key_string(component)
        if key_str not in self.current_values:
            raise EntityInternalError(owner=component, msg=f"{key_str} not found in current values") 
        attr_current_value_instance = self.current_values[key_str]
        return attr_current_value_instance.get_value(strict=strict)

    # ------------------------------------------------------------

    def get_attr_value_by_comp_name(self, component:IComponent, instance: ModelType) -> ExecResult:
        attr_name = component.name
        value = component.value_accessor.get_value(instance=instance, attr_name=attr_name, attr_index=None)
        if value is UNDEFINED:
            # TODO: depending of self.entity strategy or apply(strategy)
            #   - raise error
            #   - return NotAvailableExecResult() / UNDEFINED (default)
            #   - return None (default)
            return NotAvailableExecResult.create(reason="Missing instance attribute")

        # ORIG:
        # if not hasattr(instance, attr_name):
        #     # TODO: depending of self.entity strategy or apply(strategy)
        #     #   - raise error
        #     #   - return NotAvailableExecResult() / UNDEFINED (default)
        #     #   - return None (default)
        #     return NotAvailableExecResult.create(reason="Missing instance attribute")
        # value = getattr(instance, attr_name)

        exec_result = ExecResult()
        exec_result.set_value(value, attr_name, changer_name=f"{component.name}.ATTR")
        return exec_result

    # ------------------------------------------------------------

    def get_values_tree(self, key_string: Optional[KeyString] = None) -> ComponentTreeWValuesType:
        """
        will go recursively through every children and
        fetch their "children" and collect to output structure.
        selects all nodes, put in tree, includes self
        for every node bind (M.<field>) is evaluated
        """
        if self.values_tree is None:
            raise EntityInternalError(owner=self, msg="_get_values_tree() did not filled _values_tree cache") 

        if key_string is None:
            # component = self.entity
            tree = self.values_tree
        else:
            if key_string not in self.values_tree_by_key_string.keys():
                if key_string not in self.current_values:
                    names_avail = get_available_names_example(key_string, self.values_tree_by_key_string.keys())
                    raise EntityInternalError(owner=self, msg=f"Key string not found in values tree, got: {key_string}. Available: {names_avail}") 

                # -- fill values dict
                assert not (self.current_frame.component==self.entity)
                # NOTE: can trigger recursion 
                # subentity_items_mode = False, component = component,
                self._fill_values_dict(filler="get_values_tree", is_init=False, recursive=True)

            tree = self.values_tree_by_key_string[key_string]

        return tree

    # ------------------------------------------------------------

    def _fill_values_dict(self,
                          filler:str,
                          is_init:bool,
                          process_further:bool=True,
                          subentity_items_mode: bool = False,
                          component: Optional[IComponent] = None,
                          recursive: bool = False,
                          depth: int=0,
                          ) -> ComponentTreeWValuesType:
        " recursive - see unit test for example - test_dump.py "
        if depth > MAX_RECURSIONS:
            raise EntityInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

        if not component:
            component = self.current_frame.component
        # else:
        #     # fetching values not allowed since stack is maybe not set up correctly
        #     assert not getattr(component, "bind", None), component

        # process_further currently not used
        if is_init:
            assert not subentity_items_mode
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

        if isinstance(component, IField):
            assert getattr(component, "bind", None)
            # can trigger recursion - filling tree 
            attr_current_value_instance = \
                    self.get_current_value_instance(
                        component=component, init_when_missing=True)

            if attr_current_value_instance is not NOT_APPLIABLE:
                if attr_current_value_instance is UNDEFINED:
                    raise EntityInternalError(owner=component, msg="Not expected to have undefined current instance value") 
            else:
                attr_current_value_instance = None

        if attr_current_value_instance is not None:
            values_dict["value_instance"] = attr_current_value_instance
            # values_dict["key_string"] = attr_current_value_instance.key_string

        # -- add to parent object or call recursion and fill the tree completely

        if subentity_items_mode:
            self.current_frame.parent_values_subtree.append(values_dict)
            values_dict["subentity_items"] = []
            self.current_frame.set_parent_values_subtree(values_dict["subentity_items"])
            if recursive:
                raise EntityInternalError(owner=component, msg="Did not implement this case - subentity_items + recursive") 
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
                if not isinstance(parent_values_subtree, list):
                    raise EntityInternalError(owner=self, msg=f"Expected list, got: {parent_values_subtree}") 

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
            raise EntityInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

        output = {}
        # component's name
        output["name"] = tree["name"]

        if "value_instance" in tree:
            output["value"] = tree["value_instance"].get_value(strict=True)

        self._dump_values_children(tree=tree, output=output, key_name="contains", depth=depth)
        self._dump_values_children(tree=tree, output=output, key_name="subentity_items", depth=depth)

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

