from collections import OrderedDict
from dataclasses import (
    dataclass,
    field,
    is_dataclass,
)
from typing import (
    Optional,
    Union,
    Dict,
    List,
)

from .exceptions import (
    EntityInternalError,
    EntityApplyValueError,
)
from .utils import (
    UndefinedType,
    NA_IN_PROGRESS,
    NOT_APPLIABLE,
    NA_DEFAULTS_MODE,
    UNDEFINED,
)
from .meta_dataclass import (
    MAX_RECURSIONS,
    Self,
    ComponentStatus,
)
from .meta import (
    AttrValue,
    NoneType,
    KeyString,
    AttrName,
    ModelKlassType,
    get_dataclass_field_type_info,
    make_dataclass_with_optional_fields,
    ModelInstanceType,
    ComponentTreeWValuesType,
)
from .expressions import (
    ExecResult,
)
from .base import (
    IApplyResult,
    IContainer,
    ChangeOpEnum,
    InstanceAttrValue,
    GlobalConfig,
    ValueSetPhase,
    IValueNode,
    IField,
    ApplyStackFrame,
    IDataModel, IFieldGroup, IComponent, IEntity,
)
from .value_accessors import (
    IValueAccessor,
)


@dataclass
class ValueNodeBase(IValueNode):
    """
    Don't want to store apply_result into this class,
    so "setup_key_string()" method needs to be called
    just after object instatiation, e.g.

        ValueNode(...).setup_key_string(apply_result=...)

    """
    # ----------------------------------------
    # AUTOCOMPUTED
    # ----------------------------------------
    _accessor: IValueAccessor = field(init=False, default=UNDEFINED, repr=False)

    # used only for computing .key
    _container_node: IValueNode = field(repr=False, init=False, compare=False, default=None)

    # <field>.Value
    _value: Union[AttrValue, UndefinedType] = field(repr=False, init=False, default=UNDEFINED)

    # TODO: replace .initialized and .finished with ._status
    _status: ComponentStatus = field(init=False, repr=False, default=ComponentStatus.draft)

    # initial value is filled, ready for evaluations
    initialized: bool = field(repr=False, init=False, default=False)

    # process of update/evaluate is finished - no more changes to values
    # ready for validations
    # TODO: consider compare=False ?? do not compare by this attribute (for easier unit-test checks)
    finished: bool = field(repr=False, init=False, default=False)

    # initial value - Just value - no wrapper
    init_value: Union[UndefinedType, AttrValue] = \
        field(repr=False, init=False, default=UNDEFINED)

    # NOTE: used only in .set_instance_attr_to_value()
    #       to set new value on model instance.
    #       Result of Field.bind_to=M.<field> DotExpression execution.
    # TODO: consider ALT: Could be done with bind_to.Path() too.
    #       attr_name_path: Optional[List[AttrName]] = field(init=False, repr=False)
    init_dexp_result: Union[UndefinedType, NoneType, ExecResult] = \
        field(repr=False, init=False, default=UNDEFINED)

    # Should be used only for better introspection / debug / trace / log
    # set later by calling .setup(apply_result)
    key_string: KeyString = field(init=False, repr=True, default=UNDEFINED)
    # TODO: consider to have key_pairs too?

    # Setup later when ApplyStackFrame() is created, set with .set_apply_stack_frame()
    # this is cross reference, since stack_frame has ref to ValueNode too
    apply_stack_frame: Union[ApplyStackFrame, UndefinedType] = field(repr=False, default=UNDEFINED)

    # interrnal structures
    value_history: Union[UndefinedType, List[InstanceAttrValue]] = \
        field(repr=False, init=False, default=UNDEFINED)

    # cached values used in set_value() logic
    # see . _init_instance_attr_access_objects()
    _instance_parent: Union[ModelInstanceType, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)
    _attr_name_last: Union[AttrName, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)

    def __post_init__(self):
        self.name = self.component.name

        if self.component.is_entity() or self.component.is_data_model():
            assert self.parent_node is None
            self.top_node = self
            self.parent_container_node = None
            self._container_node = self
        else:
            if not isinstance(self.component, IComponent):
                raise EntityInternalError(owner=self, msg=f"Expected IComponent, got: {self.component}")

            # if GlobalSettings.is_development:
            assert self.parent_node is not None
            assert self.parent_node.top_node.parent_node is None
            self.top_node = self.parent_node.top_node

            self.parent_container_node = self.parent_node._container_node
            assert self.parent_container_node.component.is_container(), self.parent_container_node
            assert self.parent_container_node and self.parent_container_node != self

            if not self.copy_mode and isinstance(self.parent_container_node, ValueNode):
                # only for single containers are unique and can be fetched
                self.parent_container_node.add_to_container_children(self)

            if self.component.is_container():
                self._container_node = self
            else:
                assert self.parent_node and self.parent_node._container_node
                self._container_node = self.parent_container_node

        if self.index0 is not None and self.index0 < 0:
            raise EntityInternalError(owner=self, msg=f"index0 must be integer >= 1, got: {self.index0}")

    def setup(self, apply_result: "IApplyResult") -> Self:
        """
        will:
            1) calculate and set key_string
            2) register within apply_result

        Calculate and set key_string
        ----------------------------
        Two cases - component has .keys or not:

        a) with keys:
            For containers which have keys defined, it is assumed that one key is
            globally unique within SubEntityItems components, so no need to prefix key
            with parent key_string. Example:

                 address_set_ext[id2=1]

            if for each key that is not not set in instance, a special
            key is generated using:

                apply_result._get_new_id()

        b) without keys - index0 based:
            In other cases item index in list is used (index0), and then this is
            only locally within one instance of parent, therefore parent
            key_string is required. Example:

                 company_entity::address_set_ext[0]

        """
        if self.key_string:
            raise EntityInternalError(owner=self, msg="value node already setup")

        component = self.component

        # ---- 1) calculate and set key_string

        # default case
        key_string = component.name
        if not self.is_list() and isinstance(component, IContainer):
            # Part of key_string that includes instance identificators
            # i.e. key_pairs / index in a list
            if component.keys:
                key_pairs = component.get_key_pairs(self.instance, apply_result=apply_result)
                assert key_pairs
                key_string = "{}[{}]".format(
                    component.name,
                    GlobalConfig.ID_NAME_SEPARATOR.join(
                        [f"{name}={value}" for name, value in key_pairs]
                    ))
            elif self.index0 is not None:
                key_string = f"{component.name}[{self.index0}]"

        # ALT: not (self._container_node and self._container_node is not self)
        # ALT: not (self.parent_node is None)
        if self is not self.top_node:
            if self._container_node is self:
                container_node = self._container_node.parent_node._container_node
            else:
                container_node = self._container_node
            # TODO: consider to remove_this, just to have the same logic as before
            # ignore self.parent_node.is_list() parent value_node - will introduce redunant markup
            if container_node.is_list():
                container_node = container_node.parent_node._container_node

            key_string = f"{container_node.key_string}{GlobalConfig.ID_NAME_SEPARATOR}{key_string}"

        self.key_string = KeyString(key_string)

        # ---- 2) register within apply_result - regsiter node to value_node_list
        apply_result.register_value_node(value_node=self)

        self._accessor = apply_result._apply_accessor if apply_result._apply_accessor else self.component._accessor

        return self

    def set_apply_stack_frame(self, apply_stack_frame: ApplyStackFrame):
        """
        Called when creating ApplyStackFrame - but only in 2 places (what is enough).
        Only used in _finish_component() and is necessary.
        TODO: check if this is really necessary - circular dependency Stack -> ValueNode -> Stack
        """
        assert apply_stack_frame
        if not apply_stack_frame.value_node == self:
            raise EntityInternalError(owner=self, msg=f"apply_stack_frame has diff value_node: {self.apply_stack_frame.value_node}")
        if self.apply_stack_frame:
            raise EntityInternalError(owner=self, msg=f"apply_stack_frame is already set: {self.apply_stack_frame}")
        # TODO: make a copy to preserve original values. Maybe is not necessary.
        apply_stack_frame_copy = apply_stack_frame.clone()
        self.apply_stack_frame = apply_stack_frame_copy

    # def is_top_node(self):
    #     return self is self.top_node

    def is_changed(self) -> bool:
        """
        redundant code in mark_finished()
        """
        if self.change_op is not None:
            return True
        return self._value!=self.init_value


    # ------------------------------------------------------------

    def set_instance_attr_to_value(self):
        """
        will set instance.<attr-name> = self._value
        attr-name could be nested (e.g. M.access.can_delete)
        Functino name is odd.
        """
        # TODO: not sure if this validation is ok
        type_info = self.container.data_model.get_type_info()
        if self.container.entity.is_unbound():
            # TODO: check if it has attribute and it has corresponding type
            #       check with _accessor
            # attr_name = self._attr_name_last, model=
            pass
        else:
            model_klass = type_info.type_
            # TODO: check with _accessor?
            if not self.instance_none_mode \
                    and not isinstance(self.instance, model_klass):
                raise EntityInternalError(owner=self, msg=f"Parent instance {self.instance} has wrong type")

        # -- attr_name - fetch from initial bind_to dexp (very first)
        # TODO: save dexp_result in ValueNode or get from component.bind_to ?
        init_bind_dexp_result = self.init_dexp_result
        if not init_bind_dexp_result:
            raise EntityInternalError(owner=self, msg=f"init_bind_dexp_result is not set, got: {self} . {init_bind_dexp_result}")

        if self._instance_parent is UNDEFINED:
            self._init_instance_attr_access_objects()

        assert not isinstance(self._value, UndefinedType)

        # Finally change instance value by last attribute name
        assert self._attr_name_last
        self._accessor.set_value(instance=self._instance_parent,
                                 attr_name=self._attr_name_last,
                                 attr_index=None,
                                 new_value=self._value)

    # ------------------------------------------------------------

    def _init_instance_attr_access_objects(self):
        """
        initialize:
            self._instance_parent
            self._attr_name_last

        used in call to:
           self._accessor.set_value()
        """
        assert self._instance_parent is UNDEFINED
        if not self._accessor:
            raise EntityInternalError(owner=self, msg=f"_accessor not set")

        # "for" loop is required for attributes from substructure that
        # is not done as SubEntity rather direct reference, like:
        #   bind_to=M.access.alive
        attr_name_path = [init_raw_attr_value.attr_name
                          for init_raw_attr_value in self.init_dexp_result.dexp_value_list
                          ]
        if not attr_name_path:
            raise EntityInternalError(owner=self, msg=f"{self.component}: attr_name_path is empty")

        current_instance_parent = None
        parent_instance = self.instance
        current_instance = parent_instance
        attr_name_last = UNDEFINED

        # Attribute path example: "M.access.alive".
        # Only last, i.e. "alive" in this example, will be updated,
        # and this for loop reaches instance and current value in this case.
        for anr, attr_name in enumerate(attr_name_path, 0):
            attr_name_last = attr_name
            if current_instance is None:
                if self.instance_none_mode:
                    # -------------------------------------------------------
                    # Create all missing intermediate empty dataclass objects
                    # -------------------------------------------------------
                    assert anr > 0
                    attr_name_prev = attr_name_path[anr - 1]
                    temp_dataclass_model= self._make_dataclass_with_opt_fields(
                        current_instance_parent=current_instance_parent,
                        attr_name_prev=attr_name_prev,
                        attr_name=attr_name,
                    )
                    current_instance = temp_dataclass_model()
                    self._accessor.set_value(instance=current_instance_parent,
                                             attr_name=attr_name_prev,
                                             attr_index=None,
                                             new_value=current_instance)
                else:
                    attr_name_path_prev = ".".join(attr_name_path[:anr])
                    # TODO: fix this ugly validation message
                    raise EntityApplyValueError(owner=self,
                                                msg=f"Attribute '{attr_name}' can not be set while '{parent_instance}.{attr_name_path_prev}' is not set. Is '{attr_name_path_prev}' obligatory?")

            current_instance_parent = current_instance
            current_instance = self._accessor.get_value(instance=current_instance_parent,
                                                        attr_name=attr_name,
                                                        attr_index=None)
            if current_instance is UNDEFINED:
                raise EntityInternalError(owner=self,
                                          msg=f"Missing attribute:\n  Current: {current_instance}.{attr_name}\n Parent: {parent_instance}.{'.'.join(attr_name_path)}")

        self._instance_parent = current_instance_parent
        # attribute name is in the last item
        self._attr_name_last = attr_name_last


    # ------------------------------------------------------------
    def _make_dataclass_with_opt_fields(self,
                                        current_instance_parent: ModelInstanceType,
                                        attr_name_prev: AttrName,
                                        attr_name: AttrName,
                                        ) -> ModelKlassType:
        current_instance_type_info = get_dataclass_field_type_info(current_instance_parent, attr_name_prev)
        if current_instance_type_info is None:
            raise EntityInternalError(owner=self,
                                      msg=f"Attribute {attr_name} not found in dataclass definition of {current_instance_parent}.")
        if current_instance_type_info.is_list:
            raise EntityInternalError(owner=self,
                                      msg=f"Attribute {attr_name} of {current_instance_parent} is a list: {current_instance_type_info}.")

        current_instance_model = current_instance_type_info.type_
        if not is_dataclass(current_instance_model):
            raise EntityInternalError(owner=self,
                                      msg=f"Attribute {attr_name} of {type(current_instance_parent)} is not a dataclass instance, got: {current_instance_model}")

            # set new value of temp instance attribute
        # all attrs of a new instance will have None value (dc default)
        temp_dataclass_model = make_dataclass_with_optional_fields(current_instance_model)
        return temp_dataclass_model

    def mark_initialized(self):
        if self.initialized:
            raise EntityInternalError(owner=self, msg=f"Init phase already marked, last value: {self._value}")
        if self.finished: # and self._value is not NA_DEFAULTS_MODE:
            raise EntityInternalError(owner=self, msg=f"Invalid state, already marked as finished, last value: {self._value}")
        self.initialized = True
        self._status = ComponentStatus.did_init

    def mark_finished(self):
        """
        redundant code in is_changed()
        """
        if not self.initialized:
            raise EntityInternalError(owner=self, msg=f"Init phase should have been finished first, last value: {self._value}")
        if self.finished: # and self._value is not NA_DEFAULTS_MODE:
            raise EntityInternalError(owner=self, msg=f"Current value already finished, last value: {self._value}")

        if self.change_op is None and self._value != self.init_value:
            self.change_op = ChangeOpEnum.UPDATE
        self.finished = True
        self._status = ComponentStatus.finished

    def clean(self):
        if self.component.is_entity() or self.component.is_data_model():
            assert self.parent_node is None
        else:
            if self.parent_node is None:
                raise EntityInternalError(owner=self, msg=f"parent_node not set")

            self.parent_node._clean(child_node=self)

            # # subentity item instance - with children
            # if self.parent_node.is_list() and not self.parent_node.component == self.component:
            #     # print(f"Dump:\n{self.top_node.dump_to_str()}")
            #     raise EntityInternalError(owner=self, msg=f"value:node.is_list(), parent_node - component does not match:"
            #                                               f"\n {self.component.parent}"
            #                                               f"\n !=\n {self.parent_node.component}")

            # elif not self.parent_node.is_list() and not self.parent_node.component == self.component.parent:
            #     # print(f"Dump:\n{self.top_node.dump_to_str()}")
            #     raise EntityInternalError(owner=self, msg=f"value:node parent_node component does not match:"
            #                                               f"\n {self.component.parent}"
            #                                               f"\n !=\n {self.parent_node.component}")

    def dump_to_str(self) -> str:
        return "\n".join(self.dump_to_strlist())

    def dump_to_strlist(self, depth=0) -> List[str]:
        lines = []
        indent = "  " * depth
        line = f"{indent}{self.name}{'(is_list)' if self.is_list() else ''}"
        value = self.get_value(strict=False)
        if isinstance(value, UndefinedType):
            line += f" = {value}"
            nr_updates = len(self.value_history) - 1
            if nr_updates > 0:
                # orig_value = self.value_history[0].value
                init_value = self.init_value
                # assert init_value!=value
                value_type = value.__class__.__name__
                init_type = init_value.__class__.__name__
                if value_type!=init_type and not isinstance(value, (NoneType, UndefinedType)):
                    line += f":{value_type}"
                line += f"  (orig: {init_value}"
                if value_type!=init_type and not isinstance(init_value, (NoneType, UndefinedType)):
                    line += f":{init_type}"
                line += f", ch: {nr_updates})"
        lines.append(line)
        lines_ex = self._dump_to_strlist(lines=lines, indent=indent, depth=depth)
        lines.extend(lines_ex)
        return lines

    def dump_to_dict(self, depth: int=0) -> ComponentTreeWValuesType:
        """
        RECURSIVE - see unit test for example - test_dump.py
        """
        if depth > MAX_RECURSIONS:
            raise EntityInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({depth})")

        # is_init = (depth == 0)
        component = self.component

        # -- create a values_dict for this component
        values_dict = {}
        values_dict["name"] = component.name
        current_value = UNDEFINED
        if isinstance(component, IField):
            current_value = self.get_value(strict=True)
            if current_value not in (NOT_APPLIABLE, NA_DEFAULTS_MODE):
                if current_value is UNDEFINED:
                    raise EntityInternalError(owner=component, msg="Not expected to have undefined current instance value")
            else:
                current_value = None

        if current_value is not UNDEFINED:
            values_dict["value_instance"] = current_value

        if isinstance(self, SubentityItemsValueNode):
            # TODO: antipattern -> put in SubentityItemsValueNode
            values_dict["subentity_items"] = []
            for item_value_node in self.items:
                # RECURSION
                child_values_dict = item_value_node.dump_to_dict(depth=depth + 1)
                values_dict["subentity_items"].append(child_values_dict)
        elif self.children is not UNDEFINED:
            # TODO: antipattern -> put in ValueNode
            children = self.children.values()
            if children:
                values_dict["contains"] = []
                for child_value_node in children:
                    # RECURSION
                    child_values_dict = child_value_node.dump_to_dict(depth=depth+1)
                    values_dict["contains"].append(child_values_dict)

        return values_dict


@dataclass
class ValueNode(ValueNodeBase):

    # only these types, other component types have specialized ValueNode
    component:  Union[IField, IFieldGroup] = field(repr=False, default=None)

    # has_items: bool = field(init=False, repr=True, default=False)

    # <field>.Children - when component has children - initialized with empty {}, later filled
    children:   Union[UndefinedType, Dict[AttrName, IValueNode]] = field(repr=False, init=False, default=UNDEFINED)

    # F.<field-name> filled only if container - contains all fields (flattens FieldGroup and BooleanFeild children field tree)
    container_children: Dict[AttrName, IValueNode] = field(repr=False, init=False, default_factory=dict)


    # TODO: # F.<field-name>
    # TODO: # originally initialized in conteiner_node and all children (except SubEntityItems),
    # TODO: # all children use the same dictionary.
    # TODO: # TODO: When name clashes (e.g. SubentitySingle), then ...
    # TODO: fields_dict:   Union[UndefinedType, Dict[AttrName, Self]] = field(repr=False, init=False, default=UNDEFINED)

    def __post_init__(self):
        # self.component can be SubEntityItems - the node with Children
        super().__post_init__()

        if self.component.get_children():
            # TODO: with py 3.7 {} has ordered items as default behaviour,
            #       so std {} could be used instead
            self.children = OrderedDict()

    @staticmethod
    def is_list() -> bool:
        return False

    # def get_items(self) -> List[Self]:
    #     raise EntityInternalError(owner=self, msg=f"get_items() available only on SubentityItemsValueNode() instances")

    def get_self_or_items(self) -> Union[Self, List[IValueNode]]:
        return self

    def set_value(self, value: AttrValue, dexp_result: Optional[ExecResult], value_set_phase: ValueSetPhase) -> NoneType:
        """
        - when setting initial value dexp_result is stored into init_dexp_result
        - dexp_result can be None too - in NA_* some undefined values
        - fills value_history objects only in "trace" mode - only in the case when value is not of type UndefinedType
        """
        if self.finished:
            raise EntityInternalError(owner=self, msg=f"Current value already finished, last value: {self._value}")

        if dexp_result is not None and self.init_dexp_result is UNDEFINED:
            assert value_set_phase == ValueSetPhase.INIT_BY_BIND, value_set_phase
            if not self._value in (NA_IN_PROGRESS, UNDEFINED):
                raise EntityInternalError(owner=self, msg=f"self._value already set to: {self._value}")
            self.init_dexp_result = dexp_result
        else:
            assert value_set_phase != ValueSetPhase.INIT_BY_BIND, value_set_phase

        if not self.initialized and not value_set_phase.startswith("INIT_"):
            raise EntityInternalError(owner=self, msg=f"Setting value in init phase '{value_set_phase}' can be done before marking 'initialized'. Last_value: {self._value}")
        elif self.initialized and not value_set_phase.startswith("EVAL_"):
            raise EntityInternalError(owner=self, msg=f"Setting value in eval phase '{value_set_phase}' can be done after marking 'initialized'. Last_value: {self._value}")

        # TODO: if settings.trace - then update values_history

        self._value = value
        # if self._value is NA_DEFAULTS_MODE:
        #     # TODO: check if finish immediatelly
        #     self.mark_finished()

        # --- value_history - add new value
        # TODO: pass input arg value_parent_name - component.name does not have any purpose

        if self.trace_value_history and value is not NA_IN_PROGRESS \
                and value is not NOT_APPLIABLE and value is not NA_DEFAULTS_MODE:
            instance_attr_value = InstanceAttrValue(
                value_parent_name=self.component.name,
                value=value,
                dexp_result=dexp_result,
                value_set_phase=value_set_phase
                # TODO: source of change ...
            )
            self.value_history.append(instance_attr_value)
        # else: instance_attr_value = None
        return  #  instance_attr_value: Optional[InstanceAttrValue]

    def get_value(self, strict:bool) -> AttrValue:
        # Items has its own
        if strict and not self.finished:
            raise EntityInternalError(owner=self, msg=f"Current value is not finished, last value: {self._value}")
        return self._value


    def add_to_container_children(self, child: IValueNode):
        if child.name is None:
            raise EntityInternalError(owner=self, msg=f"Child {child} name not set")
        if child.name in self.container_children:
            raise EntityInternalError(owner=self, msg=f"Child {child} already in {self.container_children}: {self.container_children[child.name]}")
        self.container_children[child.name] = child

    def add_child(self, value_node: IValueNode):
        if value_node.index0 is not None:
            raise EntityInternalError(owner=self, msg=f"value_node shouldn't have index0 set, got: {value_node}")

        if self.children is UNDEFINED:
            raise EntityInternalError(owner=self, msg=f"failed to add {value_node}, already children dict not initialized")
        # TODO: assert value_node is not self
        # TODO: assert value_node.parent_node is not self
        if value_node.name in self.children:
            raise EntityInternalError(owner=self, msg=f"failed to add {value_node}, already child added under same name, got: {self.children[value_node.name]}")
        self.children[value_node.name] = value_node

    def _clean(self, child_node: "ValueNodeBase"):
        if not self.component == child_node.component.parent:
            # print(f"Dump:\n{self.top_node.dump_to_str()}")
            raise EntityInternalError(owner=self, msg=f"value:node parent_node component does not match:"
                                                      f"\n {child_node.component.parent}"
                                                      f"\n !=\n {self.component}")

    def _dump_to_strlist(self, depth, indent: str, lines: List[str]):
        if self.children != UNDEFINED:
            lines.append(f"{indent}- Children[{len(self.children)}]:")
            for child_node in self.children.values():
                add_lines = child_node.dump_to_strlist(depth+1)
                lines.extend(add_lines)


@dataclass
class EntityValueNode(ValueNode):
    """
    Just to have different type
    """
    component: IEntity = field(repr=False, default=None)

    # <field>.Parent
    # - empty only on top tree node (Entity component)
    parent_node:  Optional[Self] = field(init=False, repr=False, default=None)

    # autocomputed
    container: IContainer = field(init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        assert self.component.is_entity()
        self.container = self.component


@dataclass
class SubentityItemsValueNode(ValueNodeBase):

    # TODO: ISubentityItems
    component: IContainer = field(repr=False, default=None)

    # has_items: bool = field(init=False, repr=True, default=True)

    # <field>.Items
    # - when component has items - initialized with empty {}, later filled
    items:      Union[UndefinedType, List[IValueNode]] = field(repr=False, init=False, default=UNDEFINED)

    def __post_init__(self):
        super().__post_init__()
        assert self.component.is_subentity_items()
        # TODO: items collide with dict.items(). maybe a rename to _items + get_items() would be better?
        self.items = []

    @staticmethod
    def is_list() -> bool:
        return True

    # def get_items(self) -> List[Self]:
    #     return self.items
    def add_child(self, value_node: IValueNode):
        raise EntityInternalError(owner=self, msg=f"Should not be called")

    def set_value(self, value: AttrValue, dexp_result: Optional[ExecResult], value_set_phase: ValueSetPhase) -> NoneType:
        raise EntityInternalError(owner=self, msg=f"Should not be called")

    def get_self_or_items(self) -> Union[Self, List[IValueNode]]:
        return self.items

    def get_value(self, strict: bool) -> List[IValueNode]:
        # raise EntityInternalError(owner=self, msg=f"get_value() not available on SubentityItemsValueNode() instances")
        if strict and not self.finished:
            raise EntityInternalError(owner=self, msg=f"Current value is not finished, last value: {self._value}")
        if self.finished:
            # contains internal status ... UNDEFINED, NA_... and similar, real value is in "items"
            # i.e. it is not possible to convert/extract simply.
            raise EntityInternalError(owner=self, msg=f"Current value is not possible in ItemsValueNode after finished, last value: {self._value}")
        return self._value

    def add_item(self, value_node: IValueNode):
        # TODO: assert value_node is not self
        # TODO: assert value_node.parent_node is not self
        if value_node.index0 is None:
            raise EntityInternalError(owner=self, msg=f"value_node should have index0 set, got: {value_node}")
        self.items.append(value_node)

    def _clean(self, child_node: "ValueNodeBase"):
        # subentity item instance - with children
        if not self.component == child_node.component:
            # print(f"Dump:\n{self.top_node.dump_to_str()}")
            raise EntityInternalError(owner=child_node, msg=f"value:node.is_list(), parent_node - component does not match:"
                                                      f"\n {child_node.component.parent}"
                                                      f"\n !=\n {self.component}")

    def _dump_to_strlist(self, depth, indent: str, lines: List[str]):
        lines.append(f"{indent}- Items[{len(self.items)}]:")
        for item_node in self.items:
            add_lines = item_node.dump_to_strlist(depth+1)
            lines.extend(add_lines)

# ------------------------------------------------------------


class SubentityValueNode(ValueNode):
    # TODO: ISubentity
    component: IContainer = field(repr=False, default=None)


class DataModelValueNode(ValueNode):
    component:  IDataModel = field(repr=False)
