# Copied and adapted from Reedwolf project (project by robert.lujo@gmail.com - git@bitbucket.org:trebor74hr/reedwolf.git)
from __future__ import annotations

from enum import Enum
from collections import namedtuple
from typing import List, Any, Dict, get_type_hints, Union, Set
from dataclasses import dataclass, is_dataclass, Field as DcField, field
from functools import partial

from .namespaces import RubberObjectBase, ModelsNS
from .types import STANDARD_TYPE_LIST
from .utils import (
        is_pydantic, 
        is_enum,
        PydModelField, 
        is_function, 
        get_available_vars_sample,
        snake_case_to_camel,
        UNDEFINED, # TODO: change all imports to use .utils
        )
from .exceptions import (
        RuleError, 
        RuleSetupError, 
        RuleSetupValueError, 
        RuleInternalError, 
        RuleSetupNameError,
        RuleSetupNameNotFoundError,
        )
from .expressions import (
        ValueExpression,
        Operation,
        VExpStatusEnum,
        )

# ------------------------------------------------------------

YAML_INDENT = "  "
PY_INDENT = "    "

def warn(msg):
    print(f"WARNING: {msg}") # noqa: T001

def repr_obj(obj, limit=100):
    out = str(obj)
    if len(out)>limit-3:
        out=out[:limit-3] + "..."
    return out

def add_indent_to_strlist(out):
    return (f"\n{YAML_INDENT}".join(out)).splitlines()

def obj_to_strlist(obj, path=[]):
    return obj.to_strlist(path) if not isinstance(obj, RubberObjectBase) and getattr(obj, "to_strlist", None) else [str(obj)]

def list_to_strlist(args, before, after):
    out = []
    if len(args) == 0:
        out.append(f"{before}{after}")
    elif len(args) == 1:
        vstr = obj_to_strlist(args[0])
        out.append(f"{before}{vstr[0]}{after}")
    else:
        out.append(before)
        for v in args:
            out.extend(add_indent_to_strlist(obj_to_strlist(v)))
        out.append(after)
    return out

# ------------------------------------------------------------

def extract_field_meta(inspect_object: Any, var_name: Optional[str]):
    """ returns th_field, fields - if var_name is None then th_field is None
    """
    # TODO: Any -> pydatntic / dataclass base model, return -> Union[DcField|pydfield]
    th_field = None
    if is_dataclass(inspect_object):
        fields = inspect_object.__dataclass_fields__
        if var_name is not None:
            th_field = fields.get(var_name, None)
            if th_field:
                assert type(th_field)==DcField
    elif is_pydantic(inspect_object):
        fields = inspect_object.__fields__
        if var_name is not None:
            th_field = fields.get(var_name, None)
            if th_field:
                assert PydModelField
                assert type(th_field)==PydModelField
    else:
        raise RuleSetupError(item=inspect_object, msg=f"Class should be Dataclass or Pydantic ({var_name})")
    return th_field, fields

#------------------------------------------------------------

# TODO: add proper typing
# TODO: return proper object not tuple 
def extract_is_list_is_optional_and_type(py_type_hint: Any) -> Tuple(bool, bool, Any): 
    origin_type = getattr(py_type_hint, "__origin__", None)

    if origin_type==Union and len(getattr(py_type_hint, "__args__", []))==2 and py_type_hint.__args__[1]==None.__class__:
        # e.g. Optional[SomeClass] === Union[SomeClass, NoneType]
        is_optional = True
        origin_type = py_type_hint.__args__[0]
        origin_type_new = getattr(origin_type, "__origin__", None)
        if origin_type_new: 
            py_type_hint = origin_type
            origin_type = origin_type_new
    else: 
        is_optional = False

    if origin_type==list:
        # List[some_type]
        is_list = True
        if len(py_type_hint.__args__)!=1:
            raise RuleSetupNameError(item=py_type_hint, msg=f"Variable's annotation should have single List argument, got: {py_type_hint.__args__}.")
        inner_type = py_type_hint.__args__[0]
    elif origin_type is not None: 
        is_list=False
        inner_type = origin_type
    else:
        is_list=False
        inner_type = py_type_hint

    return is_list, is_optional, inner_type


# TODO: add proper typing
def extract_py_type_hints(inspect_object:Any, caller_name:str, strict=True) -> Any:
    # todo: check if module variable, class, class attribute, function or method.
    # NOTE: type_hint = function.__annotations__ - will not evalate types
    # NOTE: .get("return", None) - will get return function type hint
    try:
        return get_type_hints(inspect_object)
    except Exception as ex:
        if strict:
            # NOTE: sometimes there is NameError because some referenced types are not available in this place???
            #       when custom type with ValueExpression alias are defined, then
            #       get_type_hints falls into problems producing NameError-s
            #         Object <class 'type'> / '<class 'reedwolf.rules.components.BooleanField'>' 
            #         type hint is not possible/available: name 'OptionalBoolOrVExp' is not defined.
            raise RuleSetupValueError(item=inspect_object, msg=f"{caller_name}: Object type hint is not possible/available: {ex}."
                                 + " Please verify that object is properly type hinted class attribute, function or method,"
                                 + " and type hint should not include not standard type Type alias (see rules/types.py comment).")
        return {"__exception__" : ex}

# ------------------------------------------------------------
# BASE CLASSES/ENGINE OF BUILDING BLOCKS OF RULES (later)
# ------------------------------------------------------------

# @dataclass
# class CopyToHeap:
#     heap_bind_from: VariableHeap
#     var_name: str

BoundVar = namedtuple("BoundVar", ["heap_name", "namespace", "var_name"])

# @dataclass
# class ComponentAttribute:
#     var_name : str
#     var_path: str
#     value : Any
#     th_field: Any
ComponentAttribute = namedtuple('ComponentAttribute', [
    'var_name',
    'var_path',
    'value',
    'th_field',
    ])

# ------------------------------------------------------------


class BaseOnlyArgs:  # noqa: SIM119

    def __init__(self, *args):
        self.args = args
        self.name = self.__class__.__name__

    def __str__(self):
        return "\n".join(self.to_strlist())

    def __repr__(self):
        return f"{self.__class__.__name__}({repr_obj(self.args)})"

    def to_strlist(self):
        return list_to_strlist(self.args, before=f"{self.__class__.__name__}(", after=")")


# ------------------------------------------------------------
# RulesHandlerFunction
# ------------------------------------------------------------

@dataclass
class RulesHandlerFunction:
    function : Callable[..., Any]
    inject_params: Optional[Dict[str, ValueExpression]] = None
    model_param_name: Optional[str] = None



# ------------------------------------------------------------
# TYPE HINT CLASSES
# ------------------------------------------------------------

class AttributeTypeEnum(Enum):
    DC_FIELD  = 61
    PYD_FIELD = 71
    FUNCTION  = 81
    DIRECT_ARG= 91

# ------------------------------------------------------------
# TypeHintField
# ------------------------------------------------------------

@dataclass
class TypeHintField:
    # python type hint -> klass
    py_type_hint: Any # TODO: proper type pydantic or dataclass
    parent_object: type
    # None when function
    th_field: Union[DcField, PydModelFieldType, None] 

    # evaluated later
    klass: Union[type, UndefinedType] = field(init=False, default=UNDEFINED)
    is_list: Union[bool, UndefinedType] = field(init=False, repr=False, default=UNDEFINED) 
    is_optional: Union[bool, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)
    var_type: Union[AttributeTypeEnum, UndefinedType] = field(init=False, repr=False, default=UNDEFINED)

    def __post_init__(self):
        is_list, is_optional, py_type_hint = extract_is_list_is_optional_and_type(self.py_type_hint)
        if not is_list and not is_optional:
            if self.py_type_hint!=py_type_hint:
                raise RuleInternalError(item=self, msg=f"Extract_is_list_is_optional_and_type issue: {self.py_type_hint}!={py_type_hint}")
        # normalize 
        self.klass = py_type_hint
        self.is_list = is_list
        self.is_optional = is_optional

        if self.parent_object is None:
            assert self.th_field is None
            self.var_type = AttributeTypeEnum.DIRECT_ARG
        elif is_dataclass(self.parent_object):
            self.var_type = AttributeTypeEnum.DC_FIELD
        elif is_pydantic(self.parent_object):
            self.var_type = AttributeTypeEnum.PYD_FIELD
        elif is_function(self.parent_object):
            self.var_type = AttributeTypeEnum.FUNCTION
        else:
            raise RuleSetupError(owner=self, msg=f"Currently only pydantic/dataclass parent classes are supported, got: {self.parent_object} / {type(self.parent_object)}")

    def is_pydantic(self):
        return self.var_type==AttributeTypeEnum.PYD_FIELD

    def is_dataclass(self):
        return self.var_type==AttributeTypeEnum.DC_FIELD

    # ------------------------------------------------------------

    @staticmethod
    def extract_function_return_type_hint_field(function:Callable[..., Any]) -> TypeHintField:
        """ returns: function return type + if it returns list or not """
        assert is_function(function), function

        if isinstance(function, partial):
            function = function.func

        name = getattr(function, "__name__", None)
        if not name:
            name = function.__class__.__name__

        if not hasattr(function, "__annotations__"):
            raise RuleSetupNameError(item=function, msg=f"Heap: Variable FUNCTION '{name}' is not valid, it has no __annotations__ / type hints metainfo.")

        # e.g. Optional[List[SomeCustomClass]] or SomeCustomClass or ...
        py_type_hint = extract_py_type_hints(function, f"Function {name}").get("return", None)
        if not py_type_hint:
            raise RuleSetupNameError(item=function, msg=f"Variable FUNCTION '{name}' is not valid, it has no return type hint (annotations).")
        if py_type_hint in (None.__class__,):
            raise RuleSetupNameError(item=function, msg=f"Heap: Variable FUNCTION '{name}' is not valid, returns None (from annotation).")

        type_hint_field = TypeHintField(py_type_hint=py_type_hint, parent_object=function, th_field=None)
        return type_hint_field

        # is_list, is_optional, py_type_hint = extract_is_list_is_optional_and_type(py_type_hint)
        # return SimpleTypeHint(klass=py_type_hint, is_list=is_list, is_optional=is_optional)

    # ------------------------------------------------------------

    @staticmethod
    def extract_type_hint_field(var_name:str, inspect_object: Any) -> TypeHintField:
        is_list = False
        if isinstance(inspect_object, BoundModelBase):
            inspect_object = inspect_object.model

        if isinstance(inspect_object, RulesHandlerFunction):
            inspect_object = inspect_object.function

        # function - callable and not class and not pydantic?
        if is_function(inspect_object):
            fun_return_py_type_hint_field = TypeHintField.extract_function_return_type_hint_field(inspect_object)
            # ignore fun_return_py_type_hint.is_list, fun_return_py_type_hint.is_optional
            parent_object = fun_return_py_type_hint_field.klass
        else:
            # normal object - hopefully with with type hinting 
            parent_object = inspect_object 

        if isinstance(parent_object, TypeHintField):
            # go one level deeper
            parent_object_klass = parent_object.klass
            if not isinstance(parent_object_klass, type): 
                raise RuleSetupValueError(item=inspect_object, msg=f"Inspected object's type hint is not a class object/type: {parent_object.klass}.{parent_object.th_field.name} : {parent_object_klass}, got: {type(parent_object_klass)} ('.{var_name}' process)")
            if not is_dataclass(parent_object_klass) and not is_pydantic(parent_object_klass):
                # TODO: this probably should not be restriction, maybe only suggestion?
                raise RuleSetupValueError(item=inspect_object, msg=f"Inspected object's type hint type is not 'dataclass'/'Pydantic.BaseModel' type: {parent_object.klass}.{parent_object.th_field.name}, got: {parent_object_klass} ('.{var_name}' process)")
            parent_object = parent_object_klass

        if not hasattr(parent_object, "__annotations__"):
            raise RuleSetupValueError(item=inspect_object, msg=f"Object '{parent_object}' is not valid, it has no __annotations__ / type hints metainfo.")

        if is_dataclass(parent_object) or is_pydantic(parent_object):
            # === Dataclass / pytdantic field metadata
            # TODO: can it be a List[]?
            th_field, fields = extract_field_meta(inspect_object=parent_object, var_name=var_name)
            if not th_field:
                vars_avail = get_available_vars_sample(var_name, fields.keys())
                raise RuleSetupNameNotFoundError(item=inspect_object, msg=f"Variable name '{var_name}' not found in fields of {parent_object}. Valid are: {vars_avail}")
            
            # === Type hint
            parent_py_type_hints = extract_py_type_hints(parent_object, f"heap->{var_name}:DC/PYD")
            py_type_hint = parent_py_type_hints.get(var_name, None)
            if not py_type_hint:
                raise RuleSetupNameNotFoundError(item=inspect_object, msg=f"Object {type(parent_object)} / '{parent_object}' has no '{var_name}' (internal issue?).")
        else:
            raise NotImplementedError(f"Heap: Variable name '{var_name}' is not valid. Type currently not supported: {type(parent_object)} -> {parent_object}")

        return TypeHintField(py_type_hint=py_type_hint, parent_object=parent_object, th_field=th_field)


# ------------------------------------------------------------
# SetOwnerMixin
# ------------------------------------------------------------

class SetOwnerMixin:
    """ requires:
        name
        owner_name
        owner
    """

    # ------------------------------------------------------------

    def set_owner(self, owner:ComponentBase):
        if not self.owner==UNDEFINED:
            raise RuleInternalError(owner=self, msg=f"Owner already defined, got: {owner}")

        assert owner is None or isinstance(owner, ComponentBase), owner
        self.owner = owner

        if not self.owner_name==UNDEFINED:
            raise RuleInternalError(owner=self, msg=f"Owner name already defined, got: {owner}")

        self.owner_name=owner.name if owner else ""
        if self.name==UNDEFINED:
            suffix = self.__class__.__name__.lower()
            self.name = f"{self.owner_name}__{suffix}"


# ------------------------------------------------------------
# ComponentBase
# ------------------------------------------------------------

class ComponentBase(SetOwnerMixin):

    def as_str(self):
        return "\n".join(self.to_strlist())

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self):
        return str(self)

    # ------------------------------------------------------------

    # TODO: def clone(self):
    #       quite complex to do right, due cross-references (parent->child and opposite, siblings etc.)

    def to_strlist(self, path=None):
        if path is None:
            path = []
        out = []
        out.append(f"{self.__class__.__name__}(")
        # vars(self.kwargs).items():
        if len(path)>15:
            raise RuleSetupError("Maximum object tree depth reached, not allowed depth more than 15.")
        for name, field in self.__dataclass_fields__.items():
            # if name.startswith("_") or callable(k):
            #     continue
            value = getattr(self, name)
            if type(field) in (list, tuple):
                out.extend(
                    add_indent_to_strlist(
                        list_to_strlist(
                            value,
                            before=f"{name}=[,",
                            after="],",
                        )
                    )
                )
            elif value==self:
                out.append(f"{name}=[Self...],")
            elif name in path:
                out.append(f"{name}=[...],")
            else:
                vstr = obj_to_strlist(value, path=path+[name])
                if len(vstr) <= 1:
                    out.append(f"{name}={vstr[0]},")
                else:
                    # vstr = add_indent_to_strlist(vstr)
                    out.append(f"{name}=")
                    for v2 in vstr:
                        out.append(f"{YAML_INDENT}{v2}")
        out.append(")")
        return add_indent_to_strlist(out)

    # ------------------------------------------------------------

    def get_children(self):
        children = getattr(self, "contains", None)
        if not children:
            children = getattr(self, "enables", None)
        else:
            assert not hasattr(self, "enables"), self
        return children if children else []

    # ------------------------------------------------------------

    def _add_component(self, component:ComponentBase, components:List[ComponentBase]):
        if not (component.name and isinstance(component.name, str)):
            raise RuleSetupValueError(owner=self, item=component, msg=f"Component's name needs to be some string value, got: {component.name}': ")
        if component.name in components:
            raise RuleSetupNameError(owner=self, item=component, msg=f"Duplicate name '{component.name}': " 
                        + repr(components[component.name])[:100] 
                        + " --------- AND --------- " 
                        + repr(component)[:100])
        # Save top container too - to preserve name and for completness (if is_top)
        components[component.name] = component 


    def fill_components(self, components:Optional[List[ComponentBase]]=None, owner:Optional[ComponentBase]=None) -> Dict[str, Component]:
        """ recursive -> flat dict 
        component can be Component, Dataprovider, ...
        """

        is_top = bool(components is None)
        if is_top:
            components = {}
            # assert not owner

        # for children/contains attributes - owner is set here
        if not hasattr(self, "name"):
            raise RuleSetupError(owner=self, msg=f"Component should have 'name' attribute, got class: {self.__class__.__name__}")

        if not is_top: 
            # Component
            assert owner
            assert owner!=self
            self.set_owner(owner)
        else:
            if self.owner not in (None, UNDEFINED):
                # Extension()
                assert not owner
            else:
                # Rules()
                assert not owner
                self.set_owner(None)

        self._add_component(component=self, components=components)

        # if self.name=="eva_dts_readout_time_value": import pdb;pdb.set_trace() 

        for component_name, component_path, component, th_field in self._get_subcomponents_list():
            if isinstance(component, ValueExpression):
                pass
            elif hasattr(component, "fill_components"):
                if hasattr(component, "is_extension") and component.is_extension():
                    # for extension container don't go deeper into tree (call fill_components)
                    # it will be called later in container.setup() method
                    component.set_owner(owner=self)
                    # save only container (top) object
                    self._add_component(component=component, components=components)
                else:
                    component.fill_components(components=components, owner=self)
            elif hasattr(component, "set_owner"):
                component.set_owner(owner=self)
                self._add_component(component=component, components=components)
                # e.g. BoundModel.model - can be any custom Class(dataclass/pydantic)


        # print(f"RT: {owner} -> {self}")
        # if is_top:
        #     print("component:")
        #     for n,v in components.items():
        #         print(f"  {n} [{type(v)}] = {repr(v)[:100]}")

        return components

    # ------------------------------------------------------------

    def _invoke_component_setup(self, subcomponent_name:str, subcomponent:ComponentBase, heap:'VariableHeap'): 
        # copy_to_heap:Optional[CopyToHeap]=None):
        # TODO: remove dep somehow - interface?
        from .components import Component

        called = False
        if isinstance(subcomponent, (ValueExpression, Operation)):
            # copy_to_heap=copy_to_heap, 
            if subcomponent.GetNamespace()==ModelsNS \
                and subcomponent._status!=VExpStatusEnum.INITIALIZED:
                # Setup() was called in container.setup() before
                called = False
            else:
                subcomponent.Setup(heap=heap, owner=self, parent=None)
                called = True
        elif isinstance(subcomponent, ComponentBase):
            assert not "Rules(" in repr(subcomponent)
            # assert not isinstance(subcomponent, Rules), subcomponent
            subcomponent.setup(heap=heap) # , owner=self)
            called = True
        elif isinstance(subcomponent, (dict, list, tuple)):
            assert False, f"{self}: dicts/lists/tuples not supported {subcomponent}"
        else:
            assert not hasattr(subcomponent, "Setup"), f"{self.name}.{subcomponent_name} has attribute that is not Component/ValueExpression/Operation: {type(subcomponent)}"
        return called

    # ------------------------------------------------------------

    def setup(self, heap:'VariableHeap'):
        return self._setup(heap=heap)

    # ------------------------------------------------------------
    def _get_subcomponents_list(self) -> List[ComponentAttribute]:
        # returns name, subcomponent
        _, fields = extract_field_meta(self.__class__, var_name=None)
        output = []

        # NOTE: with vars() not the best way, other is to put metadata in field() 
        for subcomponent_name, subcomponent in vars(self).items():
            # Skip procesing only following
            # TODO: do this better - check type hint (init=False) and then decide 
            # TODO: collect names once and store it internally on class instance level? 
            # type_hint = type_hints.get(subcomponent_name)
            th_field = fields.get(subcomponent_name)

            if th_field and th_field.metadata.get("skip_traverse", False):
                continue

            if is_function(subcomponent):
                continue
            if (subcomponent_name in ("owner", "owner_name", "owner_container", "owner_heap",
                                      "name", "label", "datatype", "components", "type", "autocomplete", 
                                      "heap", 
                                      # NOTE: maybe in the future will have value expressions too
                                      "evaluate", "error", "description", "hint", "enum",
                                      # now is evaluated from bound_model, bound_model is processed
                                      "models", "py_type_hint", "type_hint_field",
                                      "bound_variable",
                                      ) 
                or subcomponent_name[0]=="_"):
                continue

            if not th_field or getattr(th_field, "init", True)==False:
                # warn(f"TODO: _get_subcomponents_list: {self} -> {subcomponent_name} -> {th_field}")
                raise RuleInternalError(owner=subcomponent, msg=f"Should '{subcomponent_name}' field be excluded from processing: {th_field}")

            # TODO: models should not be dict()
            if subcomponent_name not in ("models", "dataproviders") and th_field \
                    and "Component" not in str(th_field.type) \
                    and "Container" not in str(th_field.type) \
                    and "ValueExpression" not in str(th_field.type) \
                    and "Validation" not in str(th_field.type) \
                    and "BoundModel" not in str(th_field.type) \
                    and "Operation" not in str(th_field.type):
                raise RuleInternalError(owner=subcomponent, msg=f"Should this field be excluded from processing: {subcomponent_name}: {th_field}")
                

            # if subcomponent_name=="bind" and self.__class__.__name__=="EnumField":
            #     import pdb;pdb.set_trace() 
            if isinstance(subcomponent, (list, tuple)):
                for nr, sub_subcomponent in enumerate(subcomponent):
                    output.append(ComponentAttribute(f"{subcomponent_name}__{nr}", f"{subcomponent_name}[{nr}]", sub_subcomponent, th_field))
            elif isinstance(subcomponent, (dict,)):
                for ss_name, sub_subcomponent in subcomponent.items():
                    # NOTE: bind_to_models case - key value will be used as
                    #       variable name - should be heap unique
                    output.append(ComponentAttribute(ss_name, f"{subcomponent_name}.{ss_name}", sub_subcomponent, th_field))
            else:
                output.append(ComponentAttribute(subcomponent_name, subcomponent_name, subcomponent, th_field))
        return output

    # ------------------------------------------------------------

    def _setup(self, heap:'VariableHeap'):
        if self.owner==UNDEFINED:
            raise RuleInternalError(owner=self, msg=f"Owner not set")

        if self.is_finished():
            raise RuleInternalError(owner=self, msg=f"Setup already called")

        for subcomponent_name, subcomponent_path, subcomponent, th_field in self._get_subcomponents_list():
            self._invoke_component_setup(subcomponent_name, subcomponent=subcomponent, heap=heap)

        self.finish()

    # ------------------------------------------------------------

    def finish(self):
        if self.is_finished():
            raise RuleSetupError(owner=self, msg="finish() should be called only once.")
        self._finished = True

    # ------------------------------------------------------------

    def is_finished(self):
        return hasattr(self, "_finished")


    # ------------------------------------------------------------

    def get_name_from_bind(cls, bind:ValueExpression):
        # rename function to _get_name_from_bind
        if len(bind.Path)<=2:
            # Vexpr(Person.name) -> name
            name = bind._name
        else:
            # Vexpr(Person.address.street) -> address.street
            # TODO: this is messy :( - should be one simple logic ...
            name = "__".join([bit._name for bit in bind.Path][1:])
        assert name
        return name


    def get_owner_container(self) -> 'ContainerBase':
        """ traverses up the component tree and find first container 
        """
        # TODO: remove dep somehow - interface?
        from .containers import ContainerBase

        if self.owner==UNDEFINED:
            raise RuleSetupError(owner=self, msg=f"Owner is not set. Have setup() been called?")

        if isinstance(self, ContainerBase):
            return self

        owner_container = self.owner
        while owner_container!=None:
            if isinstance(owner_container, ContainerBase):
                break
            owner_container = owner_container.owner
        if owner_container in (None, UNDEFINED):
            raise RuleSetupError(owner=self, msg=f"Did not found container in parents. Every component needs to be in some container object tree (Rules/Extension).")

        return owner_container

# ------------------------------------------------------------
# BoundModelBase
# ------------------------------------------------------------

class BoundModelBase(ComponentBase):

    def fill_models(self, models=None):
        if models is None:
            models={}
        if self.name in models:
            raise RuleSetupNameError(owner=self, msg=f"Currently model names in tree dependency should be unique. Model name {self.name} is not, found: {models[self.name]}")
        models[self.name] = self
        if hasattr(self, "contains"):
            for dep_bound_model in self.contains:
                # recursion
                dep_bound_model.fill_models(models=models)
        return models
                    
    def get_var(self, heap: 'VariableHeap') -> Union[Variable, UndefinedType]:
        return heap.get_var_by_bound_model(bound_model=self)



# @dataclass
# class SimpleTypeHint:
#     klass: Any # maybe type?
#     is_list: bool 
#     is_optional: bool

