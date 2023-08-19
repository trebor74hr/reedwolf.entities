from abc import ABC
from typing import (
    Optional,
    Dict,
    )
from .utils import (
    message_truncate,
    )
from .namespaces import (
    DynamicAttrsBase,
    )


# TODO: check https://snarky.ca/unravelling-from/ - how to convert/transform exception

# Base errors

class EntityError(Exception, ABC):
    # TODO: validate that every call is marked for translations, check in constructor or using mypy
    def __init__(self, msg: str, owner: Optional['ComponentBase'] = None, item: Optional['Item'] = None):  # noqa: F821
        self.parent, self.item = owner, item
        self.set_msg(msg)

    def set_msg(self, msg: str):
        self.msg = msg
        self.full_msg = self._get_full_msg() + (f" (item={repr(self.item)[:50]})" if self.item else "")

    def _get_full_msg(self) -> str:
        # TODO: fix: ApplyResult('ApplyResult()')
        out = []

        if self.parent and isinstance(self.parent, str):
            out.append(f"{self.parent} -> ")
        elif self.parent is not None:
            out.append(f"{self.parent.__class__.__name__}")
            if not isinstance(self.parent, DynamicAttrsBase) and getattr(self.parent, "name", None):
                out.append(f"('{self.parent.name}') -> ")
            else:
                out.append(f"('{str(self.parent)}') -> ")
        out.append(f"{self.msg}")
        return "".join(out)

    def __str__(self):
        return f"{self.full_msg}"

    def __repr__(self):
        return f"{self.full_msg}"


# ------------------------------------------------------------
# General and internal errors
# ------------------------------------------------------------
class EntityInternalError(EntityError):
    pass


class EntityNameNotFoundError(EntityError):
    pass


# ------------------------------------------------------------
# Entity Setup phase (boot time) validation errors
# ------------------------------------------------------------
class EntitySetupError(EntityError):
    pass


class EntitySetupValueError(EntitySetupError):
    pass


class EntitySetupNameError(EntitySetupError):
    pass


class EntitySetupNameNotFoundError(EntitySetupNameError):
    pass


class EntitySetupTypeError(EntitySetupError):
    pass


# ------------------------------------------------------------
# Entity Apply phase validation errors
# ------------------------------------------------------------
class EntityApplyError(EntityError):
    pass


class EntityApplyValueError(EntityApplyError):
    pass


class EntityApplyNameError(EntityApplyError):
    pass


class EntityApplyNameNotFoundError(EntityApplyNameError):
    pass


class EntityApplyTypeError(EntityApplyError):
    pass


# ------------------------------------------------------------
# Entity Load phase validation errors
# ------------------------------------------------------------
class EntityLoadError(EntityError):
    pass


class EntityLoadValueError(EntityLoadError):
    pass


class EntityLoadNameError(EntityLoadError):
    pass


class EntityLoadNameNotFoundError(EntityLoadNameError):
    pass


class EntityLoadTypeError(EntityLoadError):
    pass


# ------------------------------------------------------------
# Codegen (code generator) errors
# ------------------------------------------------------------
class EntityCodegenError(EntityError):
    pass

# class EntityCodegenValueError(EntityCodegenError):
#     pass
# 
# class EntityCodegenNameError(EntityCodegenError):
#     pass
# 
# class EntityCodegenNameNotFoundError(EntityCodegenNameError):
#     pass
# 
# class EntityCodegenTypeError(EntityCodegenError):
#     pass


# ------------------------------------------------------------
# Validations
# ------------------------------------------------------------

class EntityValidationError(EntityError):
    MAX_ERR_DESC_ONE = 200
    MAX_ERR_DESC_ALL = 1000

    def __init__(self, errors: Dict[str, 'ValidationFailure'], owner: 'ComponentBase'):  # noqa: F821
        """owner is required """
        self.errors = errors
        msg = []
        for component_key, validation_failure_list in self.errors.items():
            err_msg = [vf.error for vf in validation_failure_list]
            msg.append("{} => {}".format(
                    component_key, 
                    message_truncate("; ".join(err_msg), self.MAX_ERR_DESC_ONE)))
        msg = f"Validation failed ({len(msg)}): \n  - " + message_truncate("\n  - ".join(msg), self.MAX_ERR_DESC_ALL)

        super().__init__(msg=msg, owner=owner)

# class EntityValidationFieldError(EntityError):
#     def __init__(self, msg:str, owner:'Field', item : Optional['Item'] = None):  # noqa: F821
#         " owner must be field and is required "
#         super().__init__(msg=msg, owner=owner, item=item)
# 
# class EntityValidationValueError(EntityValidationError):
#     pass


class EntityValidationCardinalityError(EntityValidationError):
    pass
