import os
import re
import sys
from typing import Any

DataclassModel = Any


def report_and_exit(msg: str) -> None:
    error(msg, err_code=0)


CONSOLE_MODE: bool = False


def set_console_mode(value: bool = True) -> None:
    global CONSOLE_MODE
    CONSOLE_MODE = value


def indent_str_lines(prefix: str, msg: str) -> str:
    indent: str = " " * len(prefix)
    msg = f"{prefix}{msg}"
    return "\n".join(
        [(f"{indent}{line}" if nr > 1 else line) for nr, line in enumerate(msg.splitlines(), 1)]
    )


def error(msg: str, err_code: int = 1) -> None:
    if err_code:
        msg = indent_str_lines("ERROR: ", msg)
    else:
        msg = f"{msg}"

    if CONSOLE_MODE:
        sys.stderr.write(f"{msg}\n")
        sys.exit(err_code)
    raise Exception(msg)


def info(msg: str) -> None:
    print(indent_str_lines("INFO: ", msg))  # noqa: T201


def info_debug(msg: str) -> None:
    print(indent_str_lines("DBG : ", msg))  # noqa: T201


def warn(msg: str) -> None:
    print(indent_str_lines("WARN: ", msg))  # noqa: T201


def ensure_python_version(version_gte: tuple[int, int]) -> None:
    # TODO: make decorator of this
    if sys.version_info[:2] < version_gte:
        error(
            f"Please run script on Python {version_gte}+, current is {'.'.join(map(str, sys.version_info[:3]))}"
        )



def is_unit_test_session() -> bool:
    # https://stackoverflow.com/questions/25188119/test-if-code-is-executed-from-within-a-py-test-session
    return (
        "PYTEST_CURRENT_TEST"
        in os.environ
        # These ones could be tricky
        #   or "pytest" in sys.modules
        #   or "unittest" in sys.modules
    )


def right_strip(value: str, extension: str, strict: bool = False) -> str:
    if value.endswith(extension):
        value = value[: -len(extension)]
    elif strict:
        raise ValueError(f"Value '{value}' does not endswwith '{extension}'")
    return value


def extract_lines_that_match(
    lines: list[str], regex_expr: str, all_when_not_matched: bool = False
) -> list[str]:
    matched_iidx_list = [nr for nr, line in enumerate(lines) if re.search(regex_expr, line)]
    if matched_iidx_list:
        lines = lines[matched_iidx_list[0] :]
    elif not all_when_not_matched:
        lines = []
    return lines
