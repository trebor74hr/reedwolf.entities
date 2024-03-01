import inspect
from abc import ABCMeta
from dataclasses import (
    is_dataclass,
    fields as dc_fields,
    MISSING as DC_MISSING,
)
from enum import IntEnum
from typing import (
    List,
    Optional,
    Dict,
    Type,
    Union,
    Any,
    Callable, NewType, Iterable,
)
# immutable dict - python 3.3+
# https://stackoverflow.com/questions/11014262/how-to-create-an-immutable-dictionary-in-python
from types import MappingProxyType

from .exceptions import (
    EntityInitError,
    EntityImmutableError,
    EntityCopyError,
    EntityInternalError,
    EntityNameNotFoundError,
)
from .utils import (
    UNDEFINED,
)
from .global_settings import (
    GlobalSettings,
)

# ------------------------------------------------------------

try:
    from typing import Self
except ImportError:
    # TODO: consider using typing_extensions for python version < 3.11
    Self = NewType("Self", Any)

MAX_RECURSIONS: int = 30

SELF_ARG_NAME = "self"

ERR_MSG_IMMUTABLE = ("Change is not allowed. Instance is setup and is in immutable state. "
                     "You can .copy() and change the new instance before .setup() is called.")

class ComponentStatus(IntEnum):
    draft = 0
    did_init = 1
    did_phase_one = 2
    finished = 3



# ------------------------------------------------------------
# ReedwolfMetaclass
# ------------------------------------------------------------

class ReedwolfMetaclass(ABCMeta):
    """
    This metaclass enables:
        - storing initial args and kwargs to enable copy() later (see ReedwolfDataclassBase
        - TODO: storing extra arguments - to enable custom attributes features
    References:
        - https://realpython.com/python-metaclasses/
    """
    def __new__(cls, name, bases, dct):
        # kwargs: name, bases, dct
        new_class = super().__new__(cls, name, bases, dct)

        if getattr(new_class, "DENY_POST_INIT", None) and hasattr(new_class, "__post_init__"):
            raise EntityInitError(owner=new_class,
                                  msg=f"Class '{new_class.__name__}' should not have '__post_init__' method." 
                                      "Initialization checks and instance configuration is be done in setup phase. "
                                      "Move logic to init() method (don't forget to call super().init() method).")
        return new_class

    # def __call__(cls, *args, **kwargs):
    #     """
    #     NOTE: this method is called in each instance creation, so it is crucial
    #           not to add logic that will slow down the whole system.
    #           Therefore a several cache vars are added to klass.

    #     Several uses:
    #        1) create instance and save initial arguments to instance._rwf_kwargs.
    #           args are matched to function arguments and converted to kwargs.
    #           not passed init arguments with default_factory are also added to kwargs
    #           to cover some cases.
    #           ._rwf_kwargs are used later in copy().
    #        2) Fill klass._RWF_ARG_NAMES with __init__() argument names.
    #           Must do it in lazy-init fashion - on first instance creation.
    #           Class argument will be used in copy().
    #           Can not do it in __new__, it is too early, I need result of decorator,
    #           i.e. dataclass(new_class), and in that moment fields/__init__ are not created.
    #        3) fill klass._RWF_INIT_FUNC_ARGS - list of __init__ function arguments
    #           used internally for first 2 purposes.
    #        3) fill klass._RWF_DC_FACTORY_ARGS - list of dataclass init fields
    #           with default_factory
    #     """
    #     # extra_kwargs = {name: kwargs.pop(name)
    #     #                 for name in list(kwargs.keys())
    #     #                 if name not in cls.__annotations__}
    #     try:
    #         instance = super().__call__(*args, **kwargs)
    #     except TypeError as ex:
    #         if "__init__()" in str(ex):
    #             all_args: List[str] = []
    #             if args:
    #                 all_args.append(", ".join(map(repr, args)))
    #             if kwargs:
    #                 all_args.append(", ".join([f"{k}={v!r}" for k,v in kwargs.items()]))
    #             raise EntityError(msg=f"Failed to construct:\n    {cls.__name__}({', '.join(all_args)})\n  with error:\n    {ex}") from ex
    #         raise

    #     instance._immutable = False

    #     # ------------------------------------------------------------
    #     # instance._rwf_kwargs = kwargs

    #     # klass = instance.__class__
    #     # if args:
    #     #     # NOTE: presuming order is preserved
    #     #     if not hasattr(klass, "_RWF_INIT_FUNC_ARGS"):
    #     #         klass._RWF_INIT_FUNC_ARGS = get_func_arguments(instance.__init__)

    #     #     # ------------------------------------------------------------
    #     #     # merge: args + kwargs => self._rwf_kwargs
    #     #     kwargs_from_args = {param_name: arg_value  for param_name, arg_value in zip(klass._RWF_INIT_FUNC_ARGS, args)}
    #     #     same_params = set(kwargs_from_args.keys()).intersection(set(kwargs))
    #     #     if same_params:
    #     #         raise EntityInternalError(owner=instance, msg=f"Params overlap: {same_params}")
    #     #     instance._rwf_kwargs.update(kwargs_from_args)

    #     # if is_dataclass(instance):
    #     #     if not hasattr(klass, "_RWF_DC_FACTORY_ARGS"):
    #     #         klass._RWF_DC_FACTORY_ARGS = tuple(fld for fld in dc_fields(instance)
    #     #                                            if fld.init and fld.default_factory != DC_MISSING)

    #     #     # lists/dicts - if not passed take their instances to make from them later
    #     #     #   cases to cover: entity.Entity(); entity.contains.append(StringField())
    #     #     args_with_default_factory = {fld.name: getattr(instance, fld.name)
    #     #                                  for fld in klass._RWF_DC_FACTORY_ARGS
    #     #                                  if fld.name not in instance._rwf_kwargs}
    #     #     instance._rwf_kwargs.update(args_with_default_factory)

    #     # # ------------------------------------------------------------
    #     # # fill klass._RWF_ARG_NAMES
    #     # if not hasattr(instance.__class__, "_RWF_ARG_NAMES"):
    #     #     # code duplication for speed optimization
    #     #     if not hasattr(klass, "_RWF_INIT_FUNC_ARGS"):
    #     #         klass._RWF_INIT_FUNC_ARGS = get_func_arguments(instance.__init__)

    #     #     # Should produce the same result as __init__ params parsing
    #     #     # if is_dataclass(instance):
    #     #     #     arg_names = [fld.name for fld in dc_fields(instance) if fld.init]
    #     #     klass._RWF_ARG_NAMES = tuple(klass._RWF_INIT_FUNC_ARGS.keys()) \
    #     #             if hasattr(klass, "__init__") else ()

    #     return instance


# ------------------------------------------------------------
# ReedwolfDataclassBase
# ------------------------------------------------------------

class ReedwolfDataclassBase(metaclass=ReedwolfMetaclass):

    # def __raise_immutable_error(self, key=None, value=None):
    #     raise EntityImmutableError(owner=self, msg=ERR_MSG_IMMUTABLE)

    def _make_immutable_and_finish(self):
        """
        Deny change of object attributes.
        This is limited implementation, covers:
            - .change() method can not be called
            - all list attributes are converted to tuples (no change allowed)
            - all dict attributes are converted to immutable dict - MappingProxyType (no change allowed)
            - deny change of all instance attributes (but python's internals)
              but ONLY IN UNIT-TEST CASE: method __setattr__ which prevents attribute change
              is set with ._setup_setattr_immutable_guard() when RWF_IS_UNIT_TEST env var is set.
        """
        # assert not getattr(self, "_immutable", None), self
        # assert not self._immutable, self
        assert self._status != ComponentStatus.finished, self

        klass = self.__class__

        rwf_dc_cache_field_names_varname = klass._get_cache_rwf_varname("_RWF_DC_CACHE_FIELD_NAMES")
        if not hasattr(klass, rwf_dc_cache_field_names_varname):
            raise EntityInternalError(owner=self, msg=f"{rwf_dc_cache_field_names_varname} is not yet set.")

        # ALT: only arguments from __init__
        # for arg_name in self.__get_rwf_arg_names():
        #     arg_val = getattr(self, arg_name, UNDEFINED)
        for arg_name, arg_val in vars(self).items():
            if arg_name.endswith("__"):
                # don't touch python internals
                continue
            if isinstance(arg_val, list):
                # NOTE: x:list != tuple(x) - see IComponent._dump_meta
                setattr(self, arg_name, tuple(arg_val))
            elif isinstance(arg_val, dict):
                # NOTE: x:dict == MappingProxyType(x)
                setattr(self, arg_name, MappingProxyType(arg_val))

        # self._immutable = True
        self._status = ComponentStatus.finished


    def __setattr_implementation(self, key, value):
        """
        Ensure no attribute can be changed after ._immutable attribute is set.

        Having __setattr__ function defined for all attr access adds 5-8% performance penalty.
        So this function is included only in unit test mode - special env var RWF_IS_UNIT_TEST.
        """
        # if getattr(self, "_immutable", None):
        # if self._immutable:
        if self._status == ComponentStatus.finished \
          and not (getattr(self, key, UNDEFINED) == UNDEFINED
                   and key in getattr(self.__class__, self._get_cache_rwf_varname("_RWF_DC_CACHE_FIELD_NAMES"))):
            # cache values are allowed to be set only once (not existing or initialized to UNDEFINED)
            raise EntityImmutableError(owner=self, msg=ERR_MSG_IMMUTABLE)
        super().__setattr__(key, value)

    @classmethod
    def _setup_setattr_immutable_guard(cls):
        """
        Done like this to have better control when to set up this.
        ALT:
            if GlobalSettings.is_development:
                def __setattr__(self, ...):
        """
        cls.__setattr__ = cls.__setattr_implementation


    @classmethod
    def __get_rwf_arg_names(cls) -> Iterable[str]:
        """
        Lazy init klass._RWF_ARG_NAMES
        """
        rwf_arg_names_varname = cls._get_cache_rwf_varname("_RWF_ARG_NAMES")
        if not hasattr(cls, rwf_arg_names_varname):
            rwf_init_func_args = get_func_arguments(cls.__init__)
            # Should be the same as klass._RWF_DC_FIELD_NAMES
            arg_names = tuple(rwf_init_func_args.keys()) if hasattr(cls, "__init__") else ()
            setattr(cls, rwf_arg_names_varname, arg_names)
            return arg_names

        return getattr(cls, rwf_arg_names_varname)


    def change(self, **kwargs) -> Self:
        # if getattr(self, "_immutable", False):
        # if self._immutable:
        if self._status == ComponentStatus.finished:
            raise EntityImmutableError(owner=self, msg=ERR_MSG_IMMUTABLE)

        rwf_arg_names = self.__get_rwf_arg_names()
        unknown = set(kwargs.keys()) - set(rwf_arg_names)
        if unknown:
            raise EntityNameNotFoundError(owner=self,
                                          msg=f"Unknown arguments: {', '.join(sorted(unknown))}. "
                                              f"Supported arguments: {', '.join(rwf_arg_names)}")

        # old: self._rwf_kwargs.update(kwargs)
        for k,v in kwargs.items():
            setattr(self, k, v)

        return self

    def copy(self, change: Optional[Dict]=None, traverse: bool = True, as_class: Optional[Type]=None) -> Union[Self, Any]:
        """
        change - dictionary with list of attributes which need to be overridden/passed to constructor for new instance
        traverse - to copy deep every ReedwolfDataclassBase. TODO: not nice and clean solution :(
        as_class - when constructor is not self.__class__ but some other class. Class must support all provided fields

        TODO: pass additional arrguments to modify new instance - e.g. EntityChange(...)
        """

        return self._copy(traverse=traverse, as_class=as_class, change=change)

    @classmethod
    def _get_cache_rwf_varname(cls, name_prefix: str) -> str:
        """
        When class A inherits other class B, on first A() creation the process could use B klass variable
        since it is not yet created for A klass. Therefore I need to put klass name into klass variable, to be sure to
        get/set i.e. use proper variable.
        Example:
            EntityValueNode -> ValueNode
        """
        return f"{name_prefix}__{cls.__module__}__{cls.__name__}"


    def _getset_rwf_kwargs(self):
        """
        collects original values for this object and all dependent objects
        recursive!
        """
        # return self._getset_rwf_kwargs_impl()
        # def _getset_rwf_kwargs_impl(self, depth=0, instances_copied=None):
        # if depth==0:
        #     instances_copied = {}
        # elif depth>MAX_RECURSIONS:
        #     raise EntityInternalError(owner=self, msg=f"Too deep recursion: {depth}")

        if not is_dataclass(self):
            raise EntityCopyError(owner=self, msg=f"Only dataclass instances can be copied.")

        if not hasattr(self, "_status"):
            # currently only IComponent
            raise EntityInternalError(owner=self, msg="Instance should have _status:ComponentStatus dataclass field set in init() method.")

        # self_id = id(self)
        # if self_id not in instances_copied and \
        # if not self._did_init:
        if self._status == ComponentStatus.draft:
            # TODO: resolve this properly
            # from .expressions import DotExpression

            klass = self.__class__
            rwf_dc_fields_varname = klass._get_cache_rwf_varname("_RWF_DC_FIELDS")
            if not hasattr(klass, rwf_dc_fields_varname):
                dc_field_set = dc_fields(self)
                rwf_dc_fields = [fld for fld in dc_field_set if fld.init]
                setattr(klass, rwf_dc_fields_varname, rwf_dc_fields)
                setattr(klass, klass._get_cache_rwf_varname("_RWF_DC_CACHE_FIELD_NAMES"),
                        [fld.name for fld in dc_field_set if fld.metadata.get("cache", False)])
            else:
                rwf_dc_fields = getattr(klass, rwf_dc_fields_varname)

            # Tried to optimize - no diff (a bit slower, though)
            # rwf_kwargs = [(fld, getattr(self, fld.name, UNDEFINED)) for fld in klass._RWF_DC_FIELDS]
            # rwf_kwargs = {fld.name: attr_val for fld, attr_val in rwf_kwargs
            #               if hasattr(attr_val, "_is_dexp_or_ns") or (
            #                 attr_val not in (UNDEFINED, DC_MISSING) and attr_val is not fld.default)}
            rwf_kwargs = {}
            for fld in rwf_dc_fields:
                attr_name = fld.name
                attr_val = getattr(self, attr_name, UNDEFINED)
                # if isinstance(attr_val, DotExpression) or Namespace
                if hasattr(attr_val, "_is_dexp_or_ns") or (attr_val not in (UNDEFINED, DC_MISSING) and attr_val is not fld.default):
                    # if hasattr(attr_val, "_getset_rwf_kwargs"):
                    #     # recursion: coollect for dependent objects too
                    #     attr_val._getset_rwf_kwargs()
                    # collects fld.default_factory != DC_MISSING
                    rwf_kwargs[fld.name] = attr_val
            self._rwf_kwargs = rwf_kwargs
            # instances_copied[self_id] = True
        else:
            if not hasattr(self, "_rwf_kwargs"):
                raise EntityInternalError(owner=self, msg="Internal attribute '_rwf_kwargs' not set. Did IComponent.init() has been called (super().init())?")
        return self._rwf_kwargs

    def _copy(self,
              traverse: bool,
              depth=0,
              instances_copied=None,
              change: Optional[Dict] = None,
              as_class: Optional[Type]=None,
              ):

        rwf_kwargs = self._getset_rwf_kwargs()

        if change:
            rwf_kwargs = rwf_kwargs.copy()
            rwf_kwargs.update(change)
        else:
            rwf_kwargs = rwf_kwargs

        if not as_class:
            as_class = self.__class__
        # else:
        # # NOTE: this is not the best way how to handle case when constructor has not the same signature
        #     if issubclass(as_class, self.__class__):
        #         raise EntityInternalError(owner=self, msg=f"Class provided in as_class={as_class} is not subclass of: {self.__class__}")

        if depth==0:
            instances_copied = {}
        elif depth>MAX_RECURSIONS:
            raise EntityInternalError(owner=self, msg=f"Too deep recursion: {depth}")

        self_id = id(self)
        instance_copy = instances_copied.get(self_id, UNDEFINED)
        if instance_copy is UNDEFINED:
            # Recursion x 2
            # not yet copied within this session
            # OLD: args = [self._try_call_copy(aval=aval, depth=depth, instances_copied=instances_copied, traverse=traverse)
            #         for aval in self._rwf_args]
            kwargs = {aname: self._try_call_copy(aval=aval, depth=depth, instances_copied=instances_copied, traverse=traverse)
                      for aname, aval in rwf_kwargs.items()}
            # create instance from the same class with *identical* arguments
            # OLD: instance_copy = as_class(*args, **kwargs)
            # TODO: consider wrapping it in try: except: and enrich it
            instance_copy = as_class(**kwargs)
            instances_copied[self_id] = instance_copy

        return instance_copy

    @classmethod
    def _try_call_copy(cls, traverse:bool, aval: Any, depth: int, instances_copied: Dict) -> Any:
        # from .expressions import DotExpression

        # if isinstance(aval, DotExpression): # must be first
        # if hasattr(aval, "Clone"): # DotExpression - must be first
        if getattr(aval, "_is_dexp", None):
            aval_new = aval.Clone()
        elif isinstance(aval, (tuple, list)):
            aval_new = [cls._try_call_copy(traverse=traverse, aval=aval_item, depth=depth+1, instances_copied=instances_copied)
                        for aval_item in aval]
        elif isinstance(aval, (dict,)):
            aval_new = {aval_name: cls._try_call_copy(traverse=traverse, aval=aval_item, depth=depth+1, instances_copied=instances_copied)
                        for aval_name, aval_item in aval.items()}
        # ALT: elif hasattr(aval, "_copy") and callable(aval._copy):
        elif traverse and isinstance(aval, ReedwolfDataclassBase):
            aval_new = aval._copy(traverse=traverse, depth=depth + 1, instances_copied=instances_copied)
        else:
            # TODO: if not primitive type - use copy.deepcopy() instead?
            # Namespace case
            aval_new = aval
        return aval_new


if GlobalSettings.is_development:
    ReedwolfDataclassBase._setup_setattr_immutable_guard()

# ------------------------------------------------------------

# TODO: Any -> Parameter
def get_func_arguments(func: Callable) -> Dict[str, Any]:
    # presuming order is preserved
    return {k: v for k, v in inspect.signature(func).parameters.items() if k != SELF_ARG_NAME}

# ------------------------------------------------------------
# OBSOLETE
# ------------------------------------------------------------
# this does not work - method is not called:
#   self.__setattr__ = self.__raise_immutable_error
# make all attributes attributes with getter with @property
#   does not work correctly
# for fld in dc_fields(self):
#     # only for dataclass / list / dicts - for others... hm...
#     attr_name, attr_val = fld.name, getattr(self, fld.name, UNDEFINED)
#     if attr_val is UNDEFINED:
#         continue
#     setattr(self, attr_name, property(lambda self: attr_val))
# so the only solution is to have standard __setattr__
# which denies change when ._immutable

