# NOTE: naming: storage vs accessor - what to do? I chose accessor - feels more natural

# TODO: Single child instance accessor - Items(contains=[IntField(M.day_in_week])) -> parse: [1,2,3])
# TODO: autodetect

from abc import abstractmethod
from inspect import isclass
from typing import (
    Dict,
    List,
    Any,
)

from .exceptions import (
    EntityTypeError,
)
from .meta import (
    ModelType,
    AttrName,
    AttrIndex,
    AttrValue,
    UNDEFINED,
)
from .utils import to_repr

# ATTR_GETTER_CALLABLE_TYPE = Callable[[ModelType, AttrName, AttrIndex], AttrValue]
# ATTR_SETTER_CALLABLE_TYPE = Callable[[ModelType, AttrName, AttrIndex, AttrValue], None]


class IValueAccessor:
    """
    Valuue accessor has role of reading and setting value of the data instance.
    Currently only 2 functions are used, one for
        get_getter - getting the value from the instance based on the "attribute" name or attribute index
        set_setter - setting the value to the instance based on the "attribute" name or attribute index
    """
    @staticmethod
    @abstractmethod
    def get_code() -> str:
        """
        code of the accessor - must be unique, can be used in Component.value_accessor
        """
        ...

    @staticmethod
    @abstractmethod
    def validate_instance_type(owner_name: str, instance: Any, model_type: ModelType) -> None:
        """
        will raise EntityTypeError if type of instance is not adequate
        """
        ...

    @staticmethod
    @abstractmethod
    def get_value(instance: ModelType, attr_name: AttrName, attr_index: AttrIndex) -> AttrValue:
        ...

    @staticmethod
    @abstractmethod
    def set_value(instance: ModelType, attr_name: AttrName, attr_index: AttrIndex, new_value: AttrValue) -> None:
        ...


# ------------------------------------------------------------
# Attribute - instance getter / setter
# ------------------------------------------------------------

class AttributeValueAccessor(IValueAccessor):

    @staticmethod
    def get_code() -> str:
        return "attribute"

    def validate_instance_type(self, owner_name: str, instance: Any, model_type: ModelType) -> None:
        if not isinstance(instance, model_type):
            raise EntityTypeError(owner=self, msg=f"Expecting instance of bound model '{model_type}', got: {type(instance)} / {to_repr(instance)}.")

    @staticmethod
    def get_value(instance: ModelType, attr_name: AttrName, attr_index: AttrIndex) -> AttrValue:
        return getattr(instance, attr_name, UNDEFINED)

    @staticmethod
    def set_value(instance: ModelType, attr_name: AttrName, attr_index: AttrIndex, new_value: AttrValue) -> None:
        setattr(instance, attr_name, new_value)


# ------------------------------------------------------------
# Dictionary - instance getter / setter
# ------------------------------------------------------------

class DictValueAccessor(IValueAccessor):

    @staticmethod
    def get_code() -> str:
        return "dict"

    @staticmethod
    def validate_instance_type(owner_name: str, instance: Any, model_type: ModelType) -> None:
        if not isinstance(instance, dict):
            raise EntityTypeError(owner=owner_name, msg=f"Expecting a dict instance, got: {type(instance)} / {to_repr(instance)}.")

    @staticmethod
    def get_value(instance: Dict, attr_name: AttrName, attr_index: AttrIndex) -> AttrValue:
        if not isinstance(instance, dict):
            raise EntityTypeError(msg=f"Expecting a dict, got: {attr_name} : {type(instance)} == {to_repr(instance)}")
        return instance.get(attr_name, UNDEFINED)

    @staticmethod
    def set_value(instance: Dict, attr_name: AttrName, attr_index: AttrIndex, new_value: AttrValue) -> None:
        if not isinstance(instance, dict):
            raise EntityTypeError(msg=f"Expecting a dict, got: {attr_name} : {type(instance)} == {to_repr(instance)}")
        instance[attr_name] = new_value


# ------------------------------------------------------------
# Tuple by index getter / setter
# ------------------------------------------------------------

# TODO:
# class ListByIndexValueAccessor(IValueAccessor):
#
#     @staticmethod
#     def get_code() -> str:
#         return "list_by_index"
#
#     @staticmethod
#     def validate_instance_type(owner_name: str, instance: Any, model_type: ModelType) -> None:
#         # Tuples not supported - can not be extended in-place, set is not possible when it misses
#         if not isinstance(instance, list):
#             raise EntityTypeError(owner=owner_name, msg=f"Expecting list instance, got: {type(instance)} / {to_repr(instance)}.")
#
#     @staticmethod
#     def get_value(instance: List, attr_name: AttrName, attr_index: AttrIndex) -> AttrValue:
#         if not isinstance(instance, list) or attr_index is None:
#             raise EntityTypeError(msg=f"Expecting a list and attr_index is not None, got: "
#                       f"{attr_name} : {type(instance)} == {to_repr(instance)} / attr_index == {attr_index}")
#         return instance[attr_index] if attr_index > len(instance) else UNDEFINED
#
#     @staticmethod
#     def set_value(instance: List, attr_name: AttrName, attr_index: AttrIndex, new_value: AttrValue) -> None:
#         if not isinstance(instance, list) or attr_index is None:
#             raise EntityTypeError(msg=f"Expecting a list and attr_index is not None, got: "
#                                       f"{attr_name} : {type(instance)} == {to_repr(instance)} / attr_index == {attr_index}")
#         diff = len(instance) - (attr_index + 1)
#         if diff >= 1:
#             # extern
#             instance[attr_index] += [UNDEFINED for _ in range(diff)]
#         instance[attr_index] = new_value

# ------------------------------------------------------------
# Autodetect by index getter / setter
# ------------------------------------------------------------

class AutodetectValueAccessor(IValueAccessor):

    @staticmethod
    def get_code() -> str:
        return "autodetect"

    def validate_instance_type(self, owner_name: str, instance: Any, model_type: ModelType) -> None:
        if not (isinstance(instance, model_type) or isinstance(instance, dict)): #  or isinstance(instance, list)):
            raise EntityTypeError(owner=self, msg=f"Expecting instance of bound model '{model_type}' or dict , got: {type(instance)} / {to_repr(instance)}.")

    @staticmethod
    def get_value(instance: Any, attr_name: AttrName, attr_index: AttrIndex) -> AttrValue:
        """  not DRY-ed to make it more performant """
        if isinstance(instance, dict):
            return instance.get(attr_name, UNDEFINED)
        # elif isinstance(instance, list):
        #     if attr_index is None:
        #         raise NotImplementedError(f"Not implemented case - attr_index is None: {attr_name} : {type(instance)} == {to_repr(instance)}")
        #     return instance[attr_index] if attr_index > len(instance) else UNDEFINED
        else:
            # not dry to make it more performant
            return getattr(instance, attr_name, UNDEFINED)

    @staticmethod
    def set_value(instance: Any, attr_name: AttrName, attr_index: AttrIndex, new_value: AttrValue) -> None:
        """  not DRY-ed to make it more performant (except for list - a bit complex logic) """
        if isinstance(instance, dict):
            instance[attr_name] = new_value
        # elif isinstance(instance, list):
        #     ListByIndexValueAccessor.get_value(instance, attr_name, attr_index)
        else:
            setattr(instance, attr_name, new_value)

# ------------------------------------------------------------
# REGISTRY
# ------------------------------------------------------------
STANDARD_VALUE_ACCESSOR_CLASS_REGISTRY : Dict[str, IValueAccessor] = {
    obj.get_code(): obj
    for _, obj in globals().items()
    if isclass(obj) and issubclass(obj, IValueAccessor) and obj!=IValueAccessor
}
DEFAULT_VALUE_ACCESSOR_CLASS = AttributeValueAccessor

