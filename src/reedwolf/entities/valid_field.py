# ------------------------------------------------------------
# Validations (components)
# ------------------------------------------------------------
"""
Validation-s are used in cleaners.
Base is value:DotExpression which shuuld return bool value 
if False - validation.error should be registered/raised


TODO: novi Validation - je li se promijenio u bazi u međuvremenu - provjeri
      timestamp polje koje se update-a (vidi evaluations.py na vrhu)
TODO: novi Validation - je li isti korisnik mijenja podatak
TODO: novi Validation - koja grupa korisnika smije mijenjati zapis?
TODO: novi Validation - je li podatak lock-an do nekog datuma 

new custom validations could be done like this::

    @dataclass
    class ValidationHourValue(FieldValidation):
        def __init__(self, name:str, title:TransMessageType):
            super().__init__(
                    name=name, title=title,
                    ensure=((This.value>=0) & (This.value<=23)),
                    error=_("Need valid hour value (0-23)"),
                    )
"""
from abc import ABC
from typing import (
        Union, 
        Optional,
        )
from dataclasses import (
        dataclass, 
        field,
        )

from .namespaces import (
        Fn, 
        This,
        )
from .utils import (
        to_int,
        message_truncate,
        is_none_value,
        )
from .meta import (
        StandardType,
        STANDARD_TYPE_LIST,
        TransMessageType,
        NoneType,
        )
from .exceptions import (
        EntitySetupError,
        )
from .expressions import (
        DotExpression,
        )
from .base import (
    IApplyResult,
    ValidationFailure,
    _,
        )
from .valid_base import (
        ValidationBase,
        )


class FieldValidationBase(ValidationBase, ABC):
    ...


@dataclass
class FieldValidation(FieldValidationBase):
    """ generic validation runned on field """
    ensure:         DotExpression
    name:           Optional[str] = field(default=None)
    error:          Optional[TransMessageType] = field(repr=False, default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    title:          Optional[TransMessageType] = field(repr=False, default=None)

    def validate(self, apply_result: IApplyResult) -> Union[NoneType, ValidationFailure]:
        return self._validate_common_impl(apply_result=apply_result)

    # def __repr__(self):
    #     # due multiple inheritance - to force calling ComponentBase implementation
    #     return super(ValidationBase).__repr__()


@dataclass
class Required(FieldValidationBase):
    name:           Optional[str] = None
    error:          Optional[TransMessageType] = field(repr=False, default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    title:          Optional[TransMessageType] = field(repr=False, default=None)

    def __post_init__(self):
        if not self.error:
            self.error = "The value is required"
        super().__post_init__()

    def validate(self, apply_result: IApplyResult) -> Optional[ValidationFailure]:
        component = apply_result.current_frame.component
        value = apply_result.get_current_value(component, strict=False)
        if value is None:
            return ValidationFailure(
                            component_key_string = apply_result.get_key_string(component),
                            error=self.error, 
                            validation_name=self.name,
                            validation_title=self.title,
                            details="The value is required."
                            )
        return None


@dataclass
class Readonly(FieldValidationBase):
    """ 
    After filled with initial value can the value be cheanged.
    Synonym for: Editable(False) / Frozen.
    """
    # TODO: check if autocomputed = ALLWAYS should not be enabled
    value:          Optional[Union[bool, DotExpression]] = True
    name:           Optional[str] = None
    error:          Optional[TransMessageType] = field(repr=False, default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    title:          Optional[TransMessageType] = field(repr=False, default=None)

    def __post_init__(self):
        if not isinstance(self.value, (bool, DotExpression)):
            raise EntitySetupError(owner=self, msg=f"ensure must be DotExpression or bool, got: {type(self.value)} / {self.value}")

        if not self.error:
            self.error = "The value is readonly"
        super().__post_init__()


    def validate(self, apply_result: IApplyResult) -> Optional[ValidationFailure]:
        component = apply_result.current_frame.component

        if isinstance(self.value, DotExpression):
            dexp_result = self.value._evaluator.execute_dexp(apply_result)
            is_readonly = dexp_result.value
        else:
            is_readonly = self.value

        if not is_readonly:
            return None

        key_string = apply_result.get_key_string(component)
        update_history = apply_result.update_history.get(key_string)
        if update_history and len(update_history) > 1:
            initial_value = update_history[0].value
            last_value = update_history[-1].value
            if initial_value != last_value:
                return ValidationFailure(
                                component_key_string = key_string,
                                error=self.error, 
                                validation_name=self.name,
                                validation_title=self.title,
                                details="Readonly value is changed" 
                                " from '{message_truncate(initial_value, 15)}'"
                                " to '{message_truncate(last_value, 15)}'"
                                )
        return None


@dataclass
class MinValue(FieldValidationBase):
    """ could be implemented with std. FieldValidator This.Value > 0 - but then
        right side could not be DotExpression 
    """
    # TODO: unit test needed
    # TODO: MaxValue
    # TODO: DRY - lot of similar code in all these classes
    value:          Union[StandardType, DotExpression]
    name:           Optional[str] = None
    error:          Optional[TransMessageType] = field(repr=False, default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    title:          Optional[TransMessageType] = field(repr=False, default=None)

    def __post_init__(self):
        # TODO: allow DotExpression 
        if type(self.value) not in STANDARD_TYPE_LIST:
            raise EntitySetupError(owner=self, msg="Argument 'value' must be some standard type (TODO: DotExpression), got: {type(self.value)} : {self.value}")
        if not self.error:
            self.error = "The value it too big"
        super().__post_init__()

    def validate(self, apply_result: IApplyResult) -> Optional[ValidationFailure]:
        # TODO: evaluate self.value when DotExpression
        component = apply_result.current_frame.component
        value = apply_result.get_current_value(component, strict=False)
        if not is_none_value(value) and value < self.value:
            return ValidationFailure(
                            component_key_string = apply_result.get_key_string(component),
                            error=self.error, 
                            validation_name=self.name,
                            validation_title=self.title,
                            details=_("The value should be less than '{self.valueis}', got: {value}"),
                            )
        return None

@dataclass
class ExactLength(FieldValidationBase):
    value:          Union[int, DotExpression]
    name:           Optional[str] = None
    error:          Optional[TransMessageType] = field(repr=False, default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    title:          Optional[TransMessageType] = field(repr=False, default=None)

    def __post_init__(self):
        self._check_dot_expression_or_positive_int("value", self.value)
        if not self.error:
            self.error = f"Provide value with length at most {self.value}"
        super().__post_init__()

    def validate(self, apply_result: IApplyResult) -> Optional[ValidationFailure]:
        # TODO: evaluate self.value when DotExpression
        component = apply_result.current_frame.component
        value = apply_result.get_current_value(component, strict=False)
        if value and hasattr(value, "__len__") and len(value) != self.value:
            return ValidationFailure(
                            component_key_string = apply_result.get_key_string(component),
                            error=self.error, 
                            validation_name=self.name,
                            validation_title=self.title,
                            details=f"Value's length of {len(value)} must be exactly {self.value}" 
                                    f" (value is '{message_truncate(value)}')"
                            )
        return None

@dataclass
class MaxLength(FieldValidationBase):
    value:          Union[int, DotExpression]
    name:           Optional[str] = None
    error:          Optional[TransMessageType] = field(repr=False, default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    title:          Optional[TransMessageType] = field(repr=False, default=None)

    def __post_init__(self):
        self._check_dot_expression_or_positive_int("value", self.value)
        if not self.error:
            self.error = f"Provide value with length at most {self.value}"
        super().__post_init__()

    def validate(self, apply_result: IApplyResult) -> Optional[ValidationFailure]:
        # TODO: evaluate self.value when DotExpression
        component = apply_result.current_frame.component
        value = apply_result.get_current_value(component, strict=False)
        if value and hasattr(value, "__len__") and len(value) > self.value:
            return ValidationFailure(
                            component_key_string = apply_result.get_key_string(component),
                            error=self.error, 
                            validation_name=self.name,
                            validation_title=self.title,
                            details=f"Value's length of {len(value)} is greater of maximum allowed {self.value}" 
                                    f" (value is '{message_truncate(value)}')"
                            )
        return None

@dataclass
class MinLength(FieldValidationBase):
    value:          Union[int, DotExpression]
    name:           Optional[str] = None
    error:          Optional[TransMessageType] = field(repr=False, default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    title:          Optional[TransMessageType] = field(repr=False, default=None)

    def __post_init__(self):
        self._check_dot_expression_or_positive_int("value", self.value)
        if not self.error:
            self.error = f"Provide value with length at least {self.value}"

        super().__post_init__()

    def validate(self, apply_result: IApplyResult) -> Optional[ValidationFailure]:
        # TODO: evaluate self.value when DotExpression
        component = apply_result.current_frame.component
        value = apply_result.get_current_value(component, strict=False)
        if value and hasattr(value, "__len__") and len(value) < self.value:
            return ValidationFailure(
                            component_key_string = apply_result.get_key_string(component),
                            error=self.error, 
                            validation_name=self.name,
                            validation_title=self.title,
                            details=f"Value's length of {len(value)} is smaller of minimum allowed {self.value}" 
                                    f"({message_truncate(value)})")
        return None

@dataclass
class RangeLength(FieldValidationBase):
    # see MaxLength NOTE:
    min:            Union[int, DotExpression]
    max:            Union[int, DotExpression]
    name:           Optional[str] = None
    error:          Optional[TransMessageType] = field(repr=False, default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    title:          Optional[TransMessageType] = field(repr=False, default=None)

    def __post_init__(self):
        if self.min is None and self.max is None:
            raise EntitySetupError(owner=self, msg="Please provide min and/or max")
        if self.min is not None:
            self._check_dot_expression_or_positive_int("min", self.min)
        if self.max is not None:
            self._check_dot_expression_or_positive_int("max", self.max)
            if to_int(self.min) and self.max<self.min:
                raise EntitySetupError(owner=self, msg="Please provide min <= max")

        if not self.error:
            if self.min and self.max:
                self.error = f"Provide value with length between {self.min} and {self.max}"
            elif self.min:
                self.error = f"Provide value with length at least {self.min}"
            elif self.max:
                self.error = f"Provide value with length at most {self.max}"
            else:
                assert False

        super().__post_init__()

    # value: Any, component: "ComponentBase", 
    def validate(self, apply_result: IApplyResult) -> Optional[ValidationFailure]:
        # TODO: evaluate self.value when DotExpression
        component = apply_result.current_frame.component
        value = apply_result.get_current_value(component, strict=False)
        if hasattr(value, "__len__"):
            if self.min and value and len(value) < self.min:
                return ValidationFailure(
                                component_key_string = apply_result.get_key_string(component),
                                error=self.error, 
                                validation_name=self.name,
                                validation_title=self.title,
                                details=f"Value's length of {len(value)} is smaller of minimum allowed {self.min}" 
                                        f"({message_truncate(value)})")
            elif self.max and value and len(value) > self.max:
                return ValidationFailure(
                                component_key_string = apply_result.get_key_string(component),
                                error=self.error, 
                                validation_name=self.name,
                                validation_title=self.title,
                                details=f"Value's length of {len(value)} is greater of maximum allowed {self.max}" 
                                        f"({message_truncate(value)})")
        return None

# ------------------------------------------------------------
# Some simple / common validations
# ------------------------------------------------------------

@dataclass
class HourFieldValidation(FieldValidation):
    def __init__(self, **kwargs):
        super().__init__(
            ensure=(Fn.is_int(This.Value) & (This.Value >= 0) & (This.Value <= 23)),
            error=_("Hour value needs to be a number between 0 and 23."),
            **kwargs,
        )
