"""
validations used for [subentity_]items records
e.g. in container cardinality.

TODO: Check if they could be implemented in cleaners instead of cardinality ...
        SubEntityItems(
            ...
            cleeaners = [
                Cardinality.Single()
            ]

TODO: implement as normal validations, now there are leftovers ...

"""
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from dataclasses import dataclass, field

from .exceptions import (
        EntitySetupError,
        EntityValidationCardinalityError,
        EntitySetupTypeError,
        )
from .utils import (
        to_int,
        UNDEFINED,
        UndefinedType,
        )
from .meta import (
        TransMessageType,
        NoneType,
        )
from .expressions   import (
        DotExpression,
        )
from .valid_base import (
        ValidationBase,
        )
from .base import (
    ValidationFailure,
    IApplyResult,
        )


def _validate_setup_common(validation, allow_none:Optional[bool]=None) -> 'AttrDexpNode':  # noqa: F821
    model_attr_node = validation.parent.get_bound_model_attr_node()
    if allow_none is not None:
        if allow_none and not model_attr_node.isoptional():
            raise EntitySetupTypeError(owner=validation, msg="Type hint is not Optional and cardinality allows None. Add Optional or set .allow_none=False/min=1+")
        if not allow_none and model_attr_node.isoptional():
            raise EntitySetupTypeError(owner=validation, msg="Type hint is Optional and cardinality does not allow None. Remove Optional or set .allow_none=True/min=0")
    return model_attr_node


class ItemsValidationBase(ValidationBase, ABC):
    ...

@dataclass
class ItemsValidation(ItemsValidationBase):
    ensure          : DotExpression
    available       : Optional[Union[bool, DotExpression]] = field(repr=False, default=True)

    name            : Optional[str] = field(default=None)
    error           : Optional[TransMessageType] = field(repr=False, default=None)
    title           : Optional[TransMessageType] = field(repr=False, default=None)

    def validate(self, apply_session: IApplyResult) -> Union[NoneType, ValidationFailure]:
        raise NotImplementedError()
        # TODO: check which namespaces are used, ...
        # TODO: items case - run on items ... 
        # assert apply_session.current_frame.component.is_subentity_items():
        #   output = []
        #   for item in apply_session.current_frame.get_subentity_items():
        #       out = self._validate_common_impl(apply_session=apply_session)
        #       if out:
        #           output.append(out)
        #       return output



class ICardinalityValidation(ItemsValidationBase, ABC): # count

    @abstractmethod
    def validate_setup(self):
        """
        if not ok,
            raises EntitySetupTypeError
        """

    # TODO: @abstractmethod
    def validate(self, items_count:bool, raise_err:bool=True) -> bool:
        """
        takes nr. of items and validates
        if ok, returns True
        if not ok,
            if raise_err -> raises EntityValidationCardinalityError
            else -> return false
        """
        raise NotImplementedError("abstract method")


# ------------------------------------------------------------

class Cardinality: # namespace holder


    @dataclass
    class Range(ICardinalityValidation):
        """
            at least one (min or max) arg is required
            min=None -> any number (<= max)
            max=None -> any number (>= min)
            min=0    -> same as allow_none in other validations
        """
        min             : Optional[Union[int, DotExpression]] = None
        max             : Optional[Union[int, DotExpression]] = None
        available       : Optional[Union[bool, DotExpression]] = field(repr=False, default=True)

        name            : Optional[str] = field(default=None)
        error           : Optional[TransMessageType] = field(repr=False, default=None)
        title           : Optional[TransMessageType] = field(repr=False, default=None)

        # autocomputed
        parent           : Union['ContainerBase', UndefinedType] = field(init=False, default=UNDEFINED, repr=False)  # noqa: F821
        parent_name      : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

        def __post_init__(self):
            if self.min is None and self.max is None:
                raise EntitySetupError(owner=self, msg="Please provide min and/or max")
            if self.min is not None and (to_int(self.min) is None or to_int(self.min)<0):
                raise EntitySetupError(owner=self, msg="Please provide integer min >=0 ")
            if self.max is not None and (to_int(self.max) is None or to_int(self.max)<0):
                raise EntitySetupError(owner=self, msg="Please provide integer max >=0 ")
            if self.min is not None and self.max is not None and self.max<self.min:
                raise EntitySetupError(owner=self, msg="Please provide min <= max")
            if self.max is not None and self.max==1:
                raise EntitySetupError(owner=self, msg="Please provide max>1 or use Single.")

        def validate_setup(self):
            # [] is allowed, and that is not same as None ( allow_none=(self.min==0)) )
            model_attr_node = _validate_setup_common(validation=self, allow_none=False)
            if not model_attr_node.islist():
                raise EntitySetupTypeError(owner=self, msg="Type hint is not List and should be. Change to Single or add List[] type hint ")

        def validate(self, items_count:int, raise_err:bool=True):
            if self.min and items_count < self.min:
                if raise_err:
                    raise EntityValidationCardinalityError(owner=self, msg=f"Expected at least {self.min} item(s), got {items_count}.")
                return False
            if self.max and items_count < self.max:
                if raise_err:
                    raise EntityValidationCardinalityError(owner=self, msg=f"Expected at most {self.max} items, got {items_count}.")
                return False
            return True

    @dataclass
    class Multi(ICardinalityValidation):
        " [0,1]:N "
        allow_none      : Optional[Union[bool, DotExpression]] = True
        available       : Optional[Union[bool, DotExpression]] = field(repr=False, default=True)

        name            : Optional[str] = field(default=None)
        error           : Optional[TransMessageType] = field(repr=False, default=None)
        title           : Optional[TransMessageType] = field(repr=False, default=None)

        # autocomputed
        parent          : Union['ContainerBase', UndefinedType] = field(init=False, default=UNDEFINED, repr=False)  # noqa: F821
        parent_name     : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

        def validate_setup(self):
            model_attr_node = _validate_setup_common(validation=self, allow_none=self.allow_none)
            if not model_attr_node.islist():
                raise EntitySetupTypeError(owner=self, msg="Type hint is not a List and should be. Change to Single or add List[] type hint")

        def validate(self, items_count:int, raise_err:bool=True):
            if items_count==0 and not self.allow_none:
                if raise_err:
                    raise EntityValidationCardinalityError(owner=self, msg="Expected at least one item, got none.")
                return False
            return True


# ------------------------------------------------------------
# other validations
# ------------------------------------------------------------
class IUniqueValidation(ItemsValidationBase, ABC):
    ...


class Unique: # namespace holder

    @dataclass
    class Global(IUniqueValidation):
        " globally - e.g. within table "
        # TODO: do it with M. DotExpression, e.g. fields=[M.name, M.surname]
        fields          : List[str]
        ignore_none     : Optional[Union[bool, DotExpression]] = field(default=True)
        available       : Optional[Union[bool, DotExpression]] = field(repr=False, default=True)

        name            : Optional[str] = field(default=None)
        error           : Optional[TransMessageType] = field(repr=False, default=None)
        title           : Optional[TransMessageType] = field(repr=False, default=None)

        # autocomputed
        parent          : Union['ContainerBase', UndefinedType] = field(init=False, default=UNDEFINED, repr=False)  # noqa: F821
        parent_name     : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

        def validate(self, apply_session: IApplyResult) -> Optional[ValidationFailure]:
            raise NotImplementedError()

    @dataclass
    class Items(IUniqueValidation):
        " within subentity_items records "
        # TODO: do it with M. DotExpression, e.g. fields=[M.name, M.surname]
        fields          : List[str]
        ignore_none     : Optional[Union[bool, DotExpression]] = field(default=True)
        available       : Optional[Union[bool, DotExpression]] = field(repr=False, default=True)

        name            : Optional[str] = field(default=None)
        error           : Optional[TransMessageType] = field(repr=False, default=None)
        title           : Optional[TransMessageType] = field(repr=False, default=None)

        # autocomputed
        parent          : Union['ContainerBase', UndefinedType] = field(init=False, default=UNDEFINED, repr=False)  # noqa: F821
        parent_name     : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

        def validate(self, apply_session: IApplyResult) -> Optional[ValidationFailure]:
            raise NotImplementedError()

# ------------------------------------------------------------

@dataclass
class SingleValidation(ValidationBase):
    " Cardinality validation for for SubEntitySingle case, does not really belong to this module "
    allow_none      : Union[bool, DotExpression] = True
    available       : Optional[Union[bool, DotExpression]] = field(repr=False, default=True)

    name            : Optional[str] = field(default=None)
    error           : Optional[TransMessageType] = field(repr=False, default=None)
    title           : Optional[TransMessageType] = field(repr=False, default=None)

    # autocomputed
    parent          : Union['ContainerBase', UndefinedType] = field(init=False, default=UNDEFINED, repr=False)  # noqa: F821
    parent_name     : Union[str, UndefinedType] = field(init=False, default=UNDEFINED)

    def validate_setup(self):
        model_attr_node = _validate_setup_common(validation=self, allow_none=self.allow_none)
        if model_attr_node.islist():
            raise EntitySetupTypeError(owner=self, msg="Type hint is List and should be single instance. Change to Range/Multi or remove type hint List[]")

    def validate(self, apply_session: IApplyResult) -> Optional[ValidationFailure]:
        return None
        # TODO:implement this 
        #   raise NotImplementedError()
        #   if items_count==0 and not self.allow_none:
        #       if raise_err:
        #           raise EntityValidationCardinalityError(owner=self, msg="Expected exactly one item, got none.")
        #       return False
        #   if items_count!=1:
        #       if raise_err:
        #           raise EntityValidationCardinalityError(owner=self, msg="Expected exactly one item, got {items_count}.")
        #       return False
        #   return True
