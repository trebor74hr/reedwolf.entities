import re
from typing import (
        Callable, 
        Any, 
        List,
        Optional,
        )
from functools import reduce
from threading import Lock

# ------------------------------------------------------------
# UNDEFINED
# ------------------------------------------------------------

class UndefinedType:

    instance:'UndefinedType'= None

    def __init__(self):
        if self.__class__.instance:
            raise ValueError("Undefined already instantiated, this is a singleton class")
        self.__class__.instance = self

    def __str__(self):
        return "UNDEFINED"

    def __bool__(self):
        return False

    def __eq__(self, other):
        # if other is None: return False
        # same only to same or other UNDEFINED
        if isinstance(other, self.__class__):
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    __repr__ = __str__


UNDEFINED = UndefinedType()

# ------------------------------------------------------------
# Utility functions ...
# ------------------------------------------------------------

# nije ok ...
def composite_functions(*func:Callable[..., Any]) -> Callable[..., Any]:
    # inspired https://www.geeksforgeeks.org/function-composition-in-python/
    # TODO: see kwargs exmple at: https://mathieularose.com/function-composition-in-python
    """ accepts N number of function as an
        argument and then compose them
        returning single function that can be applied with """
    def compose(f, g):
        return lambda x : f(g(x))

    return reduce(compose, func, lambda x : x)



def get_available_names_example(name:str, name_list:List[str], max_display:int = 5) -> str:
    # assert isinstance(name_list, list), name_list
    if not name_list:
        return "no available names"

    len_names_all = len(name_list)
    names_ellipsis = "..." if len_names_all > max_display else ""
    # names_all      = ', '.join([p for p in name_list][:max_display])

    # filter out private names
    name_list_all = [p for p in name_list if not p.startswith("_")]
    name_list = name_list_all

    if name:
        new = []
        name_bits = [bit.strip("_") for bit in name.split("_")] if "_" in name else None
        for name_cand in name_list:
            if name_cand.startswith(name[:3]) or name[:4] in name_cand:
                new.append(name_cand)
            elif name_bits:
                if bool([nb for nb in name_bits if len(nb)>=3 and nb[:3] in name_cand]):
                    new.append(name_cand)
        name_list = new

    diff = max_display - len(name_list)
    if diff>0:
        name_list += [name for name in name_list_all if name not in name_list][:diff]

    len_name_list = len(name_list)
    if len_name_list >= max_display:
        names_avail = ', '.join(name_list[:max_display])
        # avail_ellipsis = "..."
    else:
        names_avail = ', '.join(name_list)
        # avail_ellipsis = ""

    names_avail = f"{names_avail} {names_ellipsis}".strip()

    # if not name_list:
    #     names_avail = f"{names_all} {names_ellipsis}".strip()
    # elif len_name_list == len_names_all:
    #     names_avail = f"{names_all}"
    # elif len_name_list<=3:
    #     names_avail = f"{names_all} {names_ellipsis} (check: {name_list} {avail_ellipsis})".strip()
    # else:
    #     names_avail = f"... {name_list} {avail_ellipsis}".strip()
    return names_avail


def to_int(value:Any, default=None) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return default

def snake_case_to_camel(name: str) -> str:
    out = []
    for name_bit in name.split("."):
        class_name = []
        for bit in name_bit.split("_"):
            bit = bit[0].upper()+bit[1:]
            class_name.append(bit)
        out.append("".join(class_name))
    return ".".join(out)


def varname_to_title(varname:str) -> str:
    out = varname.replace("_", " ")
    out = re.sub(" +", " ", out).strip()
    return out.capitalize()


def format_arg_name_list(arg_name_list: List[str]):
    if not arg_name_list:
        return "<no arguments>"
    arg_name_list = [f"'{arg_name}'" for arg_name in arg_name_list]
    if len(arg_name_list)==1:
        return arg_name_list[0]
    return ", ".join([arg_name for arg_name in arg_name_list[:-1]]) + f" and {arg_name_list[-1]}"


def be_conjugate(count:int) -> str:
    return "is" if count<=1 else "are"

def plural_suffix(count:int) -> str:
    return "" if count<=1 else "s"

def message_truncate(message: Optional[str], max_length:int = 50) -> str:
    if message in (None,):
        return ""
    dots = ".."
    message = str(message)
    if len(message) > max_length:
        message = message[:(max_length - len(dots))] + dots
    return message


class ThreadSafeCounter:
    "  thread safe counter class based on https://superfastpython.com/thread-safe-counter-in-python/ "
    def __init__(self):
        self._counter = 0
        self._lock = Lock()
 
    def get_new(self):
        with self._lock:
            self._counter += 1
            return self._counter
 
    # def value(self):
    #     with self._lock:
    #         return self._counter

REPR_MAX_LEN = 100

def to_repr(instance: Any, max_len: int = REPR_MAX_LEN):
    out = repr(instance)
    if len(out) > max_len:
        out = out[:max_len-5] + ".." + out[-3:]
    return out



