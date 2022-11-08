from typing import (
        Any, 
        Dict, 
        List, 
        Optional,
        Union,
        ClassVar,
        )
from dataclasses import dataclass, field, is_dataclass, fields as dc_fields

from .types import (
        TransMessageType, 
        STANDARD_TYPE_LIST,
        )
from .utils import (
        is_pydantic, 
        get_available_vars_sample,
        UNDEFINED,
        UndefinedType,
        )
from .base import (
        ComponentBase,
        BoundModelBase,
        extract_field_meta,
        BoundVar,
        TypeHintField,
        )
from .exceptions import (
        RuleSetupNameError,
        RuleSetupError,
        RuleInternalError,
        RuleNameNotFoundError,
        )
from .namespaces import (
        Namespace,
        FieldsNS, 
        DataProvidersNS,
        ContextNS,
        GlobalNS,
        ModelsNS,
        ThisNS,
        )
from .expressions import(
        ValueExpression,
        Operation,
        )
from .models import (
        BoundModel,
        BoundModelWithHandlers,
        )
from .validations import (
        CardinalityValidation,
        ChildrenValidation,
        )
from .variables import (
        Variable,
        VariablesHeap, 
        )
from .components import (
        BooleanField,
        ChoiceField,
        Component,
        DataVar,
        EnumField,
        Field,
        Section, 
        Validation, 
        )

# ------------------------------------------------------------
# Rules 
# ------------------------------------------------------------

@dataclass
class ContainerBase(ComponentBase):

    # def is_finished(self):
    #     return hasattr(self, "_finished")

    # def finish(self):
    #     print(self.name)
    #     import pdb;pdb.set_trace() 
    #     if self.is_finished():
    #         raise RuleSetupError(owner=self, msg="finish() should be called only once.")
    #     self._finished = True

    def add_section(self, section:Section):
        if self.is_finished():
            raise RuleSetupError(owner=self, msg="Section can not be added after setup() is called.")
        found = [sec for sec in self.contains if sec.name==section.name]
        if found:
            raise RuleSetupError(owner=self, msg=f"Section {section.name} is already added.")
        self.contains.append(section)

    def is_extension(self):
        # TODO: if self.owner is not None could be used as the term, put validation somewhere
        " if start model is value expression - that mean that the the Rules is Extension "
        return isinstance(self.bound_model.model, ValueExpression)

    def setup(self):
        # components are flat list, no recursion/hierarchy browsing needed
        if self.bound_model is None:
            raise RuleSetupError(owner=self, msg="bound_model not set. Initialize in constructor or call bind_to() first.")

        if self.is_finished():
            raise RuleSetupError(owner=self, msg="setup() should be called only once")

        if self.heap is not None:
            raise RuleSetupError(owner=self, msg="Heap.setup() should be called only once")

        self.heap = VariablesHeap(owner=self)

        # ----------------------------------------
        # A. 1st level variables
        # ----------------------------------------

        # ------------------------------------------------------------
        # A.1. MODELS - collect variables from managed models
        # ------------------------------------------------------------
        self.models = self.bound_model.fill_models()

        if not self.models:
            raise RuleSetupError(owner=self, msg="Rules(models=List[models]) is required.")

        # model_fields_variables = {}
        # th_field = extract_field_meta(self, "bound_model"):
        is_extension = self.is_extension()

        for bound_model_name, bound_model in self.models.items():
            assert bound_model_name == bound_model.name
            # ex. th_field.metadata.get("bind_to_owner_heap")
            is_main_model = (bound_model==self.bound_model)
            is_extension_main_model = (is_extension and is_main_model)

            # is_list = False
            if not isinstance(bound_model, BoundModelBase):
                raise RuleSetupError(owner=self, msg=f"{bound_bound_model_name}: Needs to be Boundbound_model* instance, got: {bound_model}")

            model = bound_model.model

            variable = None
            if is_extension_main_model:
                if not isinstance(model, ValueExpression):
                    raise RuleSetupError(owner=self, msg=f"{bound_model_name}: For Extension main bound_model needs to be ValueExpression: {bound_model.model}")

            alias_saved = False
            is_list = False

            # if bound_model_name=="device_types": import pdb;pdb.set_trace() 
            if isinstance(model, ValueExpression):
                if model.GetNamespace()!=ModelsNS:
                    raise RuleSetupError(owner=self, msg=f"{bound_model_name}: ValueExpression should be in ModelsNS namespace, got: {model.GetNamespace()}")

                if is_extension_main_model:
                    # bound variable
                    assert hasattr(self, "owner_heap")
                    # TODO: GetVariable + Setup - 3 time repetead (1x in components) - DRY it
                    # variable = model.GetVariable(heap=self.owner_heap, strict=False)
                    variable = self.owner_heap.get_var_by_vexp(vexp=model)
                    if not variable:
                        variable = model.Setup(heap=self.owner_heap, owner=bound_model, parent=None)
                        # add bounded variable - from owner heap to this heap
                        self.heap.add(variable, alt_var_name=bound_model_name)

                    variable.add_bound_var(BoundVar(self.heap.name, variable.namespace, bound_model_name))
                    alias_saved = True
                else:
                    # variable = self.heap.get_var(namespace=ModelsNS, var_name=model)
                    # variable = model.GetVariable(heap=self.heap, strict=False)
                    variable = self.heap.get_var_by_vexp(vexp=model)
                    if not variable:
                        variable = model.Setup(heap=self.heap, owner=bound_model, parent=None)

                if not variable:
                    raise RuleInternalError(owner=self, msg=f"Variable not recognized: {model}")

                if not isinstance(variable.data, TypeHintField):
                    raise RuleInternalError(owner=self, msg=f"Variable data is not TypeHintField, got: {type(variable.data)} / {variable.data}")

                model   = variable.data.klass 
                is_list = variable.data.is_list
                # OLD: model, is_list = self.heap.get_vexp_type(vexp=model)

            if not is_dataclass(model) and not is_pydantic(model) and not (is_list and model in STANDARD_TYPE_LIST):
                raise RuleSetupError(owner=self, msg=f"Managed model {bound_model_name} needs to be a @dataclass, pydantic.BaseModel or List[{STANDARD_TYPE_LIST}], got: {type(model)}")

            if not variable:
                # standard variable
                # when available Instead model, pass type_hint_field - it holds more information
                if not bound_model.type_hint_field:
                    bound_model.set_type_hint_field()
                assert bound_model.type_hint_field.klass==model
                variable = Variable(bound_model_name, bound_model.type_hint_field, namespace=ModelsNS) # , is_list=is_list)
                self.heap.add(variable)

            if not alias_saved and variable.name!=bound_model_name:
                # save variable with alias 
                self.heap.add(variable, alt_var_name=bound_model_name)
                variable.add_bound_var(BoundVar(self.heap.name, variable.namespace, bound_model_name))
                alias_saved = True

            if is_extension_main_model:
                self.bound_variable = variable

            # NOTE: bound_model not stored in heap, just bound_model.model
            # heap.add(current_variable, alt_var_name=copy_to_heap.var_name)


        # ------------------------------------------------------------
        # A.2. DATAPROVIDERS - Collect all variables from dataproviders section
        # ------------------------------------------------------------
        for data_var in self.dataproviders:
            assert isinstance(data_var, DataVar)
            self.heap.add(Variable(data_var.name, data_var, namespace=DataProvidersNS)) 

        # ------------------------------------------------------------
        # Traverse the whole tree (recursion) and collect all components into
        # simple flat list. It will set owner for each child component.
        # ------------------------------------------------------------
        self.components = self.fill_components()

        # A.3. COMPONENTS - collect variables - previously flattened (recursive function fill_components)
        for component_name, component in self.components.items():
            if isinstance(component, (Field, DataVar)):
                denied = False
                deny_reason = ""
            # containers and validations
            elif isinstance(component, (BoundModel, BoundModelWithHandlers, Validation, Section, Extension, Rules, ChildrenValidation)):
                # stored - but should not be used
                denied = True
                deny_reason = "Component of type {component.__class__.__name__} can not be referenced in ValueExpressions"
            else: 
                valid_types = ', '.join([t.__class__.__name__ for t in (Field, ChoiceField, BooleanField, EnumField, Validation, Section, Extension, Rules, ChildrenValidation)])
                raise RuleSetupError(owner=self, msg=f"RuleSetup does not support type {type(component)}: {repr(component)[:100]}. Valid type of objects are: {valid_types}")

            variable = Variable(component_name, component, namespace=FieldsNS, 
                                denied=denied, deny_reason=deny_reason)
            self.heap.add(variable) # , is_list=False))

        # NOTE: ContextNS, UtilsNS, ThisNS and GlobalNS - are used on-the-fly later 
        # TODO: do boot/load validations of ValueExpressions on these too (ThisNS, UtilsNS especially)

        # ----------------------------------------
        # B. other level variables - recursive
        # ----------------------------------------
        # now when all variables are set, now setup() can be called for all
        # components recursively:
        #   it will validate every component attribute
        #   if attribute is another component - it will call recursively component.setup()
        #   if component attribute is ValueExpression -> will call vexp.Setup()
        #   if component is another container i.e. is_extension() - it will
        #       process only that component and will not go deeper. later
        #       extension.setup() will do this within own tree dep (own .components / .heap)

        if not self.contains:
            raise RuleSetupError(owner=self, msg=f"{self}: needs 'contains' attribute with list of components")

        self._setup(heap=self.heap)

        for component_name, component in self.components.items():
            # TODO: maybe bound Field.bind -> Model variable?
            if not component.is_finished():
                raise RuleInternalError(owner=self, msg=f"{component} not finished")

        self.heap.finish() 

    def get_bound_model_var(self) -> Variable:
        # TODO: rename this method to _get_bound_model_var
        return self.heap.get_var_by_bound_model(bound_model=self.bound_model)


    def get_component(self, name:str) -> ComponentBase:
        # TODO: currently components are retrieved only from contains - but should include validations + cardinality 
        if name not in self.components:
            vars_avail = get_available_vars_sample(name, self.components.keys())
            raise RuleNameNotFoundError(owner=self, msg=f"{self}: component '{name}' not found, some valid_are: {vars_avail}")
        return self.components[name]


    def print_components(self):
        if not hasattr(self, "components"): raise RuleError(owner=self, msg="Call .setup() first")
        for k,v in self.components.items():
            print(k, repr(v)[:100]) # noqa: T001


# ------------------------------------------------------------


@dataclass
class Rules(ContainerBase):
    name            : str
    label           : TransMessageType
    contains        : List[Component]            = field(repr=False)
    # --- can be bound later with bind_to
    bound_model     : Optional[BoundModel]       = field(repr=False, default=None)
    # --- can be bound later with bind_to
    dataproviders   : Optional[List[DataVar]]    = field(repr=False, default_factory=list)

    validations     : Optional[List[Validation]] = field(repr=False, default_factory=list)

    # --- Evaluated later
    heap            : Optional[VariablesHeap]    = field(init=False, repr=False, default=None)
    components      : Optional[List[Component]]  = field(repr=False, default=None)
    models          : Dict[str, Union[type, ValueExpression]] = field(repr=False, init=False, default_factory=dict)
    # in Rules (top object) this case allway None - since it is top object
    owner           : Union[None, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    owner_name      : Union[str, UndefinedType]  = field(init=False, default=UNDEFINED)

    def bind_to(self, bound_model:BoundModel, dataproviders:Optional[List[DataVar]]=None):
        """
        late binding, will do setup()
        """
        if self.bound_model is not None:
            raise RuleSetupError(owner=self, msg="bound_model already set.")
        if self.dataproviders:
            raise RuleSetupError(owner=self, msg="dataproviders set, late binding not allowed.")
        self.bound_model = bound_model
        if dataproviders:
            self.dataproviders = dataproviders
        self.setup()

# ------------------------------------------------------------

@dataclass
class Extension(ContainerBase):
    """ can not be used individually - must be directly embedded into Other
        Extension or top Rules """

    # required since if it inherit name from BoundModel then the name will not
    # be unique in self.components (Extension and BoundModel will share the same name)
    name            : str 
    bound_model     : Union[BoundModel, BoundModelWithHandlers] = field(repr=False) 
    # metadata={"bind_to_owner_heap" : True})
    label           : TransMessageType

    cardinality     : CardinalityValidation
    contains        : List[Component]            = field(repr=False)
    dataproviders   : Optional[List[DataVar]]    = field(repr=False, default_factory=list)
    validations     : Optional[List[Validation]] = field(repr=False, default_factory=list)

    # --- Evaluated later
    heap            : Optional[VariablesHeap]    = field(init=False, repr=False, default=None)
    components      : Optional[List[Component]]  = field(repr=False, default=None)
    models          : Dict[str, Union[type, ValueExpression]] = field(repr=False, init=False, default_factory=dict)
    owner           : Union[ComponentBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    owner_name      : Union[str, UndefinedType]  = field(init=False, default=UNDEFINED)

    # extension specific
    owner_container : Union[ContainerBase, UndefinedType] = field(init=False, default=UNDEFINED, repr=False)
    owner_heap      : Optional[VariablesHeap]   = field(init=False, repr=False, default=None)
    bound_variable: Union[Variable, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)

    # Class attributes
    # namespace_only  : ClassVar[Namespace] = ThisNS

    def set_owner(self, owner:ContainerBase):
        super().set_owner(owner=owner)
        self.owner_container = self.get_owner_container()

    def setup(self, heap:VariablesHeap):
        # NOTE: heap is not used, can be reached with owner.heap(). left param
        #       for same function signature as for components.
        self.owner_heap = heap
        super().setup()
        self.cardinality.validate_setup()


