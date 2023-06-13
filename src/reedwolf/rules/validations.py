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

"""
from __future__ import annotations

from abc import ABC
from typing import (
        Union, 
        Optional,
        )
from dataclasses import (
        dataclass, 
        field,
        )
from .meta import (
        TransMessageType,
        NoneType,
        )
from .base import (
        IApplySession,
        ValidationFailure,
        ExecResult,
        )
from .components    import (
        ValidationBase, 
        )
from .expressions   import (
        DotExpression,
        NotAvailableExecResult,
        execute_available_dexp,
        )
from .exceptions    import RuleSetupError
from .utils         import (
        to_int,
        message_truncate,
        )

@dataclass
class Validation(ValidationBase):
    """
    new custom validations could be done like this:

        @dataclass
        class ValidationHourValue(Validation):
            def __init__(self, name:str, label:TransMessageType):
                super().__init__(
                        name=name, label=label,
                        ensure=((This.value>=0) & (This.value<=23)),
                        error=_("Need valid hour value (0-23)"),
                        )

    """
    ensure:         DotExpression
    name:           Optional[str] = field(default=None)
    error:          Optional[TransMessageType] = field(repr=False, default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    label:          Optional[TransMessageType] = field(repr=False, default=None)

    def __post_init__(self):
        if not isinstance(self.ensure, DotExpression):
            raise RuleSetupError(owner=self, msg=f"ensure must be DotExpression, got: {type(self.ensure)} / {self.ensure}")
        self._fill_name_when_missing()
        if self.error is None:
            self.error = f"Validation failed: {self.ensure}"

        super().__post_init__()

    def validate(self, apply_session: IApplySession) -> Union[NoneType, ValidationFailure]:
        not_available_dexp_result: NotAvailableExecResult  = execute_available_dexp(self.available, apply_session=apply_session)
        if not_available_dexp_result: 
            # TODO: log ...
            return None

        component = apply_session.current_frame.component
        dexp_result: ExecResult = self.ensure._evaluator.execute_dexp(apply_session)
        if not bool(dexp_result.value):
            error = self.error if self.error else "Validation failed"
            return ValidationFailure(
                            component_key_string = apply_session.get_key_string(component),
                            error=error, 
                            validation_name=self.name,
                            validation_label=self.label,
                            details=f"The validation returned '{dexp_result.value}'"
                            )
        return None



@dataclass
class SimpleValidationBase(ValidationBase, ABC):
    name:           Optional[str] = None
    error:          Optional[TransMessageType] = field(repr=False, default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    label:          Optional[TransMessageType] = field(repr=False, default=None)

@dataclass
class Required(SimpleValidationBase):

    def __post_init__(self):
        if not self.error:
            self.error = "The value is required"
        super().__post_init__()

    def validate(self, apply_session: IApplySession) -> Optional[ValidationFailure]:
        component = apply_session.current_frame.component
        value = apply_session.get_current_value(component, strict=False)
        if value is None:
            return ValidationFailure(
                            component_key_string = apply_session.get_key_string(component),
                            error=self.error, 
                            validation_name=self.name,
                            validation_label=self.label,
                            details="The value is required."
                            )
        return None

@dataclass
class Readonly(SimpleValidationBase):
    # synonym for: Editable(False)
    # TODO: define standard label and error message
    # returns True, False, only-when-not-empty
    # NOTE: Editable / Frozen
    #   after filled with initial value can the value be cheanged
    # TODO: if autocomputed = ALLWAYS should not be ...
    value: Optional[Union[bool, DotExpression]] = True

    def __post_init__(self):
        if not isinstance(self.value, (bool, DotExpression)):
            raise RuleSetupError(owner=self, msg=f"ensure must be DotExpression or bool, got: {type(self.value)} / {self.value}")

        if not self.error:
            self.error = "The value is readonly"
        super().__post_init__()


    def validate(self, apply_session: IApplySession) -> Optional[ValidationFailure]:
        component = apply_session.current_frame.component

        if isinstance(self.value, DotExpression):
            dexp_result = self.value._evaluator.execute_dexp(apply_session)
            is_readonly = dexp_result.value
        else:
            is_readonly = self.value

        if not is_readonly:
            return None

        key_string = apply_session.get_key_string(component)
        update_history = apply_session.update_history.get(key_string)
        if update_history and len(update_history) > 1:
            initial_value = update_history[0].value
            last_value = update_history[-1].value
            if initial_value != last_value:
                return ValidationFailure(
                                component_key_string = key_string,
                                error=self.error, 
                                validation_name=self.name,
                                validation_label=self.label,
                                details="Readonly value is changed" 
                                " from '{message_truncate(initial_value, 15)}'"
                                " to '{message_truncate(last_value, 15)}'"
                                )
        return None


@dataclass
class ExactLength(ValidationBase):
    value:          int = None
    name:           Optional[str] = None
    error:          Optional[TransMessageType] = field(repr=False, default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    label:          Optional[TransMessageType] = field(repr=False, default=None)

    def __post_init__(self):
        if not to_int(self.value, 0) > 0:
            raise RuleSetupError(owner=self, msg="'value' must be integer > 0")
        if not self.error:
            self.error = f"Provide value with length at most {self.value}"
        super().__post_init__()

    def validate(self, apply_session: IApplySession) -> Optional[ValidationFailure]:
        component = apply_session.current_frame.component
        value = apply_session.get_current_value(component, strict=False)
        if value and hasattr(value, "__len__") and len(value) != self.value:
            return ValidationFailure(
                            component_key_string = apply_session.get_key_string(component),
                            error=self.error, 
                            validation_name=self.name,
                            validation_label=self.label,
                            details=f"Value's length of {len(value)} must be exactly {self.value}" 
                                    f" (value is '{message_truncate(value)}')"
                            )
        return None

@dataclass
class MaxLength(ValidationBase):
    # NOTE: list all attributes, do not reuse SimpleValidationBase, due 
    #       goal of having proper types and order: single required positional
    #       argument 'value'
    value:          int = None
    name:           Optional[str] = None
    error:          Optional[TransMessageType] = field(repr=False, default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    label:          Optional[TransMessageType] = field(repr=False, default=None)

    def __post_init__(self):
        # TODO: define standard label and error message
        if not to_int(self.value, 0) > 0:
            raise RuleSetupError(owner=self, msg="'value' must be integer > 0")
        if not self.error:
            self.error = f"Provide value with length at most {self.value}"
        super().__post_init__()

    def validate(self, apply_session: IApplySession) -> Optional[ValidationFailure]:
        component = apply_session.current_frame.component
        value = apply_session.get_current_value(component, strict=False)
        if value and hasattr(value, "__len__") and len(value) > self.value:
            return ValidationFailure(
                            component_key_string = apply_session.get_key_string(component),
                            error=self.error, 
                            validation_name=self.name,
                            validation_label=self.label,
                            details=f"Value's length of {len(value)} is greater of maximum allowed {self.value}" 
                                    f" (value is '{message_truncate(value)}')"
                            )
        return None

@dataclass
class MinLength(ValidationBase):
    # see MaxLength NOTE:
    value:          int
    name:           Optional[str] = None
    error:          Optional[TransMessageType] = field(repr=False, default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    label:          Optional[TransMessageType] = field(repr=False, default=None)

    def __post_init__(self):
        # TODO: define standard label and error message
        if not to_int(self.value, 0) > 0:
            raise RuleSetupError(owner=self, msg="'value' must be integer > 0")
        if not self.error:
            self.error = f"Provide value with length at least {self.value}"

        super().__post_init__()

    def validate(self, apply_session: IApplySession) -> Optional[ValidationFailure]:
        component = apply_session.current_frame.component
        value = apply_session.get_current_value(component, strict=False)
        if value and hasattr(value, "__len__") and len(value) < self.value:
            return ValidationFailure(
                            component_key_string = apply_session.get_key_string(component),
                            error=self.error, 
                            validation_name=self.name,
                            validation_label=self.label,
                            details=f"Value's length of {len(value)} is smaller of minimum allowed {self.value}" 
                                    f"({message_truncate(value)})")
        return None

@dataclass
class RangeLength(ValidationBase):
    # see MaxLength NOTE:
    min:            Optional[int] = None
    max:            Optional[int] = None
    name:           Optional[str] = None
    error:          Optional[TransMessageType] = field(repr=False, default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    label:          Optional[TransMessageType] = field(repr=False, default=None)

    def __post_init__(self):
        if self.min is None and self.max is None:
            raise RuleSetupError(owner=self, msg="Please provide min and/or max")
        if self.min is not None and (to_int(self.min) is None or to_int(self.min)<0):
            raise RuleSetupError(owner=self, msg="Please provide integer min >=0 ")
        if self.max is not None and (to_int(self.max) is None or to_int(self.max)<0):
            raise RuleSetupError(owner=self, msg="Please provide integer max >=0 ")
        if self.min is not None and self.max is not None and self.max<self.min:
            raise RuleSetupError(owner=self, msg="Please provide min <= max")
        # TODO: define standard label and error message
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
    def validate(self, apply_session: IApplySession) -> Optional[ValidationFailure]:
        component = apply_session.current_frame.component
        value = apply_session.get_current_value(component, strict=False)
        if hasattr(value, "__len__"):
            if self.min and value and len(value) < self.min:
                return ValidationFailure(
                                component_key_string = apply_session.get_key_string(component),
                                error=self.error, 
                                validation_name=self.name,
                                validation_label=self.label,
                                details=f"Value's length of {len(value)} is smaller of minimum allowed {self.min}" 
                                        f"({message_truncate(value)})")
            elif self.max and value and len(value) > self.max:
                return ValidationFailure(
                                component_key_string = apply_session.get_key_string(component),
                                error=self.error, 
                                validation_name=self.name,
                                validation_label=self.label,
                                details=f"Value's length of {len(value)} is greater of maximum allowed {self.max}" 
                                        f"({message_truncate(value)})")
        return None

