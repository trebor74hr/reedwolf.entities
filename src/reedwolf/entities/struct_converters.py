# --------------------------------------------------------------------------------------
# TOOD: consider to rewrite this to use ._apply() and apply_result to get all values ...
#       similar to logic NA_DEFAuLTS ... get_defaults() ..., no cleaners call,
#       just parse and fill values, then dump to wanted struct.
# --------------------------------------------------------------------------------------

# dump_defaults
# instance = NA_DEFAULTS_MODE,
from dataclasses import (
    field,
    dataclass,
)
from typing import (
    Type,
    Dict,
    Optional,
    ClassVar,
    List,
    Union,
)

from .exceptions import (
    EntityInternalError,
    EntityValueError,
    EntityTypeError,
)
from .expressions import DotExpression
from .meta import (
    MAX_RECURSIONS,
    ModelKlassType,
    AttrValue,
    UNDEFINED,
    is_model_klass,
    get_model_fields,
    ModelField,
    is_pydantic,
    NoneType,
    AttrName, ModelInstanceType,
)
from .base import (
    DTO_STRUCT_CHILDREN_SUFFIX,
    UseStackFrameCtxManagerBase,
    IStackFrame,
    IComponent,
    IStackOwnerSession,
    IContainer,
)
from .fields import (
    FieldBase,
)
from .utils import UndefinedType


# ------------------------------------------------------------
# Structure converter
# TODO: put this in a new module
# ------------------------------------------------------------

class UseStructConverternStackFrameCtxManager(UseStackFrameCtxManagerBase):
    ...


@dataclass
class StructConverterStackFrame(IStackFrame):

    component: IComponent = field(repr=False)
    path_names: List[str] = field()
    instance: ModelInstanceType = field(repr=False)
    dto_class: Type[ModelKlassType] = field(repr=False)
    # None will not trigger creation
    dto_kwargs: Union[NoneType, Dict[str, AttrValue]] = field(repr=False)
    depth: Optional[int] = field(repr=True)  # 0 based

    # autocomputed
    component_name: str = field(init=False, repr=True)
    # set manually
    dto_instance: Union[ModelInstanceType, UndefinedType] = field(repr=False, init=False, default=UNDEFINED)

    def __post_init__(self):
        if not isinstance(self.component, IComponent):
            raise EntityInternalError(owner=self, msg=f"Expected Component, got: {self.component}")
        self.component_name = self.component.name
        if self.depth > MAX_RECURSIONS:
            raise EntityInternalError(owner=self, msg=f"Maximum recursion depth exceeded ({self.depth})")

        self.path_names = self.path_names[:]
        self.path_names.append(self.component.name)
        if not len(self.path_names) == self.depth + 1:
            raise EntityInternalError(owner=self, msg=f"Depth is {self.depth} and length of path-names not matched: {self.path_names}")


    def clean(self):
        pass

    def post_clean(self):
        pass

    def set_dto_instance(self, dto_instance: ModelInstanceType):
        if self.dto_instance != UNDEFINED:
            raise EntityInternalError(owner=self, msg=f"Instance already set {self.dto_instance}")
        self.dto_instance = dto_instance

    def ensure_dto_kwargs(self):
        if self.dto_kwargs is None:
            self.dto_kwargs = {}

    def set_dto_kwarg(self, name: AttrName, value: AttrValue):
        if self.dto_kwargs is None:
            self.ensure_init_dto_kwargs()
        elif name in self.dto_kwargs:
            raise EntityInternalError(owner=self, msg=f"Attribute {name} already set {self.dto_kwargs[name]}")

        self.dto_kwargs[name] = value


@dataclass
class StructConverterRunner(IStackOwnerSession):
    """
    Will convert instance of self.data_model.model
    to DTO structures
        StructEnum.ENTITY_LIKE

    Although ApplyResult.get_values_tree()) returns all values, it is designed for apply process.
    Potentially it could be used, it holds component information, but will require special apply()
    process without calling cleaners (just like .

    So this logic is a bit redundant, but is more general since it does not depend on apply process at all.
    """

    STACK_FRAME_CLASS: ClassVar[type] = StructConverterStackFrame
    STACK_FRAME_CTX_MANAGER_CLASS: ClassVar[type] = UseStructConverternStackFrameCtxManager

    # autocomputed
    stack_frames: List[StructConverterStackFrame] = field(repr=False, init=False, default_factory=list)
    current_frame: Optional[StructConverterStackFrame] = field(repr=False, init=False, default=None)
    finished: bool = field(repr=False, init=False, default = False)


    def create_dto_instance_from_model_instance(self,
                                                component: IComponent,
                                                instance: ModelInstanceType,
                                                dto_class: Type[ModelKlassType],
                                                ) -> ModelKlassType:
        with self.use_stack_frame(
                StructConverterStackFrame(
                    component=component,
                    instance=instance,
                    dto_class=dto_class,
                    dto_kwargs=None,
                    path_names=[],
                    depth=0,
                )) as frame:
            self._create_dto_instance_from_model_instance()
            dto_instance = self.current_frame.dto_instance

        return dto_instance


    def _create_dto_instance_from_model_instance(self) -> ModelInstanceType:
        """
        pokriti:
            instance change - container - tj. is_container()
        """
        assert self.current_frame is not None
        component = self.current_frame.component
        instance = self.current_frame.instance
        dto_class = self.current_frame.dto_class

        if is_pydantic(dto_class):
            # to resolve ForwardRef()
            # https://stackoverflow.com/questions/76589764/pydantic-forwardref-in-fieldinfo-not-updated-by-update-forward-refs
            dto_class.update_forward_refs()
            dto_class.construct()

        # TODO: logger ("frame.path_names:", self.current_frame.path_names, self.current_frame.dto_class, self.current_frame.dto_kwargs)

        is_field = isinstance(component, FieldBase)
        if is_field:
            field: FieldBase = component
            # TODO: can be nested, e.g. M.access.can_access, pass all
            if len(field.bind_to.Path)!=1:
                raise NotImplementedError(f"Currently nestted binds not supported: {field.bind_to}")
            model_attr_name = field.bind_to._name
            instance_value = getattr(instance, model_attr_name, UNDEFINED)
            if instance_value != UNDEFINED:
                dto_attr_name = component.name
                self.current_frame.set_dto_kwarg(dto_attr_name, instance_value)

        is_field_with_children = False
        children = component.get_children()

        if children: #  and component.can_have_children():
            # for boolean without children - dto_class will be this will be {}
            if not is_model_klass(dto_class):
                raise EntityInternalError(owner=self, msg=f"dto_class not a model, got: {dto_class}")
            dto_class_field_dict = get_model_fields(dto_class)

            is_field_with_children = isinstance(component, FieldBase)

            for child in children:
                if isinstance(child, FieldBase) and child.can_have_children():
                    # BooleanField + .enables
                    child_dto_attr_name = f"{child.name}{DTO_STRUCT_CHILDREN_SUFFIX}"
                else:
                    child_dto_attr_name = child.name
                # will create {} on first instance set
                # TODO: this is pydantic specific
                child_dto_model_field: ModelField = dto_class_field_dict.get(child_dto_attr_name, UNDEFINED)
                # child_dto_model_field.type_._evaluate(globals(), locals())
                if child_dto_model_field == UNDEFINED:
                    child_dto_class = UNDEFINED
                    # raise EntityTypeError(owner=self, msg=f"Field {child_dto_attr_name} not available {dto_class}")
                else:
                    child_dto_class = child_dto_model_field.type_

                if child.is_container():
                    # if not is_model_klass(child_dto_class):
                    #     raise EntityTypeError(owner=self, msg=f"Field {child_dto_attr_name} not a class in {dto_class}, got: {child_dto_class}")
                    container: IContainer = child
                    # TODO: container.get_data_model_attr_node()
                    assert isinstance(container.data_model.model_klass, DotExpression), container.data_model.model_klass
                    if len(container.data_model.model_klass.Path) != 1:
                        raise NotImplementedError(f"Currently nestted binds not supported: {container.data_model.model_klass}")

                    model_attr_name = container.data_model.model_klass._name

                    child_instance = getattr(instance, model_attr_name, UNDEFINED)
                    if child_instance == UNDEFINED:
                        raise EntityTypeError(owner=self, msg=f"Instance {model_attr_name} not available in {instance}")
                else:
                    # if is_model_klass(child_dto_class):
                    #     raise EntityTypeError(owner=self, msg=f"Field {child_dto_attr_name} should not be a class in {dto_class}, got: {child_dto_class}")
                    # may check a type child_dto_class and type(instance)
                    child_instance = instance

                if child.is_subentity_items() and child_instance is not None:
                    assert not is_field
                    if not isinstance(child_instance, (list, tuple)):
                        raise EntityInternalError(owner=self, msg=f"Expectred list, got: {child_instance}")
                    child_instance_list = []
                    for child_item_instance in child_instance:
                        child_dto_kwargs = None
                        item_stack_frame = StructConverterStackFrame(
                            component=child,
                            instance=child_item_instance,
                            dto_class=child_dto_class,
                            dto_kwargs=child_dto_kwargs,
                            path_names=self.current_frame.path_names,
                            depth=self.current_frame.depth + 1,
                        )
                        with self.use_stack_frame(item_stack_frame):
                            # recursion
                            self._create_dto_instance_from_model_instance()
                            if self.current_frame.dto_instance != UNDEFINED:
                                child_instance_list.append(self.current_frame.dto_instance)
                    self.current_frame.set_dto_kwarg(child_dto_attr_name, child_instance_list)
                else:
                    if isinstance(child, FieldBase):
                        self.current_frame.ensure_dto_kwargs()
                        child_dto_kwargs = self.current_frame.dto_kwargs
                    else:
                        child_dto_kwargs = None

                    stack_frame = StructConverterStackFrame(
                        component=child,
                        instance=child_instance,
                        dto_class=child_dto_class,
                        dto_kwargs=child_dto_kwargs,
                        path_names=self.current_frame.path_names,
                        depth=self.current_frame.depth + 1,
                    )
                    with self.use_stack_frame(stack_frame):
                        # recursion
                        self._create_dto_instance_from_model_instance()
                        child_instance = self.current_frame.dto_instance

                    if child_instance != UNDEFINED:
                        if child_dto_class == UNDEFINED:
                            raise EntityTypeError(owner=self, msg=f"Field {child_dto_attr_name} not available {dto_class}")
                        self.current_frame.set_dto_kwarg(child_dto_attr_name, child_instance)

        # TODO: uff, not nice term:
        if (not is_field or is_field_with_children) and bool(self.current_frame.dto_kwargs):
            # TODO: change term to self.current_frame.dto_kwargs is not None:
            try:
                dto_instance = dto_class(**self.current_frame.dto_kwargs)
            except Exception as ex:
                raise EntityValueError(owner=self, msg=f"Failed to create: {self.current_frame.dto_class}({self.current_frame.dto_kwargs}): {ex}")
            self.current_frame.set_dto_instance(dto_instance)


