"""
Validations for Children fields.
In SubEntityItems case, this will be running against every SubEntity in list.
"""
from abc import ABC
from typing import (
        Optional, 
        Union,
        )
from dataclasses import dataclass, field

from .meta import (
        TransMessageType,
        NoneType,
        )
from .expressions import (
        DotExpression,
        )
from .valid_base import (
        ValidationBase,
        )
from .base import (
    ValidationFailure,
    IApplyResult,
        )


class ChildrenValidationBase(ValidationBase, ABC):
    ...


@dataclass
class ChildrenValidation(ChildrenValidationBase):
    ensure:         DotExpression
    name:           Optional[str] = field(default=None)
    error:          Optional[TransMessageType] = field(repr=False, default=None)
    available:      Optional[Union[bool, DotExpression]] = field(repr=False, default=True)
    title:          Optional[TransMessageType] = field(repr=False, default=None)

    def validate(self, apply_result: IApplyResult) -> Union[NoneType, ValidationFailure]:
        # TODO: check which namespaces are used, ...
        # if apply_result.current_frame.component.is_subentity_items():
        #     #   output = []
        #     #   for item in apply_result.current_frame.get_subentity_items():
        #     #       out = self._validate_common_impl(apply_result=apply_result)
        #     #       if out:
        #     #           output.append(out)
        #     #       return output
        #     # RT: raise NotImplementedError()
        #     return False
        return self._validate_common_impl(apply_result=apply_result)
