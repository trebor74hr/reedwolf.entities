# Adaptation to reuse Reedwolf project (project by robert.lujo@gmail.com - git@bitbucket.org:trebor74hr/reedwolf.git)
"""
Yaml entities to python source code
TODO: generated code is obsolete
TODO: missing yaml example(s) and unit-tests
"""
# NOTE: This is obsolete - could be adapted and reused
import os
import sys

import yaml

INDENT = "  "

raise NotImplementedError("The module from_yaml is not up-to-date with current Entity object system")

class ParseException(Exception):
    pass


def warn(msg):
    print(f"WARNING: {msg}")  # noqa: T001


def get_indent(length):
    return INDENT * length


def escape_string(val):
    val = val.replace("'", r"\'")
    return val


def escape_term(val):
    if not isinstance(val, str):
        val = str(val)
    else:
        val = val.replace("'", '"')
    return val


def name_to_label(name):
    label = name.replace("_", " ")
    label = label.capitalize()
    return label


def is_value_expression(aval):
    if not isinstance(aval, str):
        return False
    return (  # noqa: SIM106
        "F." in aval
        or "D." in aval
        or "M." in aval
        or "MySettings." in aval
        or "Functions." in aval
        or "This." in aval
    )


UNDEFINED = object()


def dump_rule_attr(  # noqa: C901
    item,
    output,
    indent_depth,
    attr_name,
    value_type,
    available=None,
    value_if_missing=UNDEFINED,
    # value=UNDEFINED
    return_value=False,
):
    """
    available:
        None - can be present or not
        True - must be present
        False - must not be present
    """
    # if value != UNDEFINED:
    #     aval = value
    # else:
    aval = item.pop(attr_name, UNDEFINED)

    if aval == UNDEFINED and value_if_missing != UNDEFINED:
        # assert not aval, aval
        aval = value_if_missing

    if aval in (UNDEFINED, None):
        if available:
            raise ValueError(f"{attr_name}::{value_type} value '{attr_name}' is not present")
        aval = None
    else:
        if available == False:  # noqa: E712
            raise ValueError(
                f"{attr_name}::{value_type} value '{attr_name}' should not be present"
            )

        if value_type in (
            "detect_or_interpret",
            "int_or_interpret",
            "list_or_interpret",
            "bool_or_interpret",
        ):
            # Python bug or wanted feature: 0 in (True, False) -> True
            if type(aval) == bool:
                vt = "bool"
            elif type(aval) in (tuple, list):
                vt = "list"
            elif type(aval) in (int, float):
                vt = "int"
            else:
                vt = "interpret"

            if vt != "interpret" and value_type != "detect_or_interpret":
                if value_type == "list_or_interpret":
                    assert vt in ("list",), f"{attr_name}::{value_type} -> {vt} not list: {item}"
                elif value_type == "bool_or_interpret":
                    assert vt in ("bool",), f"{attr_name}::{value_type} -> {vt} not bool: {item}"
                elif value_type == "int_or_interpret":
                    assert vt in ("int",), f"{attr_name}::{value_type} -> {vt} not int: {item}"
                else:
                    assert False, f"{attr_name}::{value_type} -> {vt} unknown type : {item}"
            value_type = vt

        elif value_type == "message_or_interpret":
            if type(aval) == dict:
                val_type = aval.pop("type", None)
                assert val_type in ("interpret",), f"val_type err: {val_type}"
                val_val = aval.pop("value", None)
                assert val_val and type(val_val) == str, f"val_type err: {val_val}"

                value_type = val_type
                aval = val_val
            else:
                assert type(aval) == str, f"val_type err: {aval}"
                value_type = "message"

        if value_type == "message":
            if "{" in aval:
                aval = f"msg(_('{escape_string(aval)}'))"  # check vars in params and prepare
            else:
                aval = f"_('{escape_string(aval)}')"
        elif value_type == "interpret":
            if type(aval) in (int, float, bool):
                aval = aval  # literal
            # TODO: ovo je primitivno
            else:
                assert type(aval) == str
                if is_value_expression(aval):  # noqa: SIM106
                    aval = f"( {aval} )"  # literal - expression
                else:
                    # if not aval[0].isupper():
                    raise ValueError(
                        f"{attr_name}::{value_type} value '{attr_name}' is interpret - solve it ..."
                    )
                    # old: aval = f"interpret('{escape_term(aval)}')"

        elif value_type == "int":
            if type(aval) != int:
                raise ValueError(f"{attr_name}::{value_type} value {aval} is not int")
            aval = f"{aval}"
        elif value_type == "bool":
            if aval not in (True, False):
                raise ValueError(f"{attr_name}::{value_type} value {aval} is not bool")
            aval = f"{aval}"
        elif value_type == "list":
            if type(aval) not in (tuple, list):
                raise ValueError(f"{attr_name}::{value_type} value {aval} is not list/tuple")
            # list of dicts?
            if aval and isinstance(aval[0], dict):
                out2 = []
                for a2 in aval:
                    out = []
                    for k2, v2 in a2.items():
                        if is_value_expression(v2):
                            v2 = f"( {v2} )"  # literal - expression
                        else:
                            v2 = f"'{v2}'"
                        out.append(f"'{k2}': {v2}")
                    out = "{ %s }" % (", ".join(out),)
                    out2.append(out)
                aval = "[ " + (f"\n{get_indent(indent_depth+4)}, ".join(out2)) + " ]"
            else:
                aval = "[" + f"\n{get_indent(indent_depth+3)}".join([f"{a}," for a in aval]) + "]"
        elif value_type == "literal":
            aval = f"{aval}"
        elif value_type == "quote":  # noqa: SIM106
            aval = aval.replace("'", r"\'")
            aval = f"'{aval}'"
        else:
            raise ValueError(f"{attr_name}::{value_type} invalid")
        aname = attr_name.replace("-", "_")
        output.append(f"{get_indent(indent_depth+2)}{aname}={aval},")
    if return_value:
        return (value_type, aval) if aval else (None, None)
    return value_type if aval else None


# ------------------------------------------------------------


def dump_cleaners(parents_str, name, type, item, indent_depth, output):
    cleaners = item.pop("cleaners", None)
    if cleaners:
        if not isinstance(cleaners, list):
            warn(f"{parents_str}::{name} ({type}) CLEANERS is not a list: {cleaners}")
        else:
            output.append(f"{get_indent(indent_depth+2)}cleaners=[")
            for vnr, cleaner in enumerate(cleaners, 1):
                raise NotImplementedError("TODO: needs to implement Validation vs Evaluation")

                output.append(f"{get_indent(indent_depth+3)}Validation(")
                validation = cleaner
                if not (
                    dump_rule_attr(
                        validation, output, indent_depth + 4, "ensure", "list_or_interpret"
                    )
                    or dump_rule_attr(
                        validation, output, indent_depth + 4, "min", "int_or_interpret"
                    )
                    or dump_rule_attr(
                        validation, output, indent_depth + 4, "max", "int_or_interpret"
                    )
                ):
                    warn(
                        f"{parents_str}::{name} ({type}) CLEANERS - not found ensure/min/max: {vnr} -> {validation}"
                    )
                dump_rule_attr(validation, output, indent_depth + 4, "error", "message")
                output.append(f"{get_indent(indent_depth+3)}),")
            output.append(f"{get_indent(indent_depth+2)}],")
    return bool(cleaners)


# ------------------------------------------------------------


def parse_item(item, outputs, nrs, indent_depth, objects, parents=None):  # noqa: C901

    output = outputs["py"]
    output_txt = outputs["txt"]

    if parents is None:
        parents = ()
    assert nrs is not None
    if not parents:
        parents_str = "ROOT"
    else:
        parents_str = "/".join([p["name"] for p in parents])

    item_repr = repr(item)[:80] + "..."
    if not isinstance(item, dict):
        warn(f"{parents_str} item not a dict: {item_repr}")  # noqa: T001
        return

    name = item.get("name", "")  # attr will be deleted from item dict later
    if not name:
        warn(f"{parents_str} item has no name: {item_repr}")  # noqa: T001
        return

    type = item.pop("type", "")
    if not type:
        warn(f"{parents_str}::{name} item has no type: {item_repr}")  # noqa: T001
        return

    ALL_TYPES = (
        "checkbox",
        "input",
        "email",
        "file",
        "password",
        "date",
        "datetime",
        "dynamic",
        "button",
        "number",
        "label",
        "form",
        "fieldgroup",
        "select",
        "data",
        "validation",
    )

    try:
        if type not in ALL_TYPES:
            raise ValueError(f"{name} '{type}' is not valid, not one of: {ALL_TYPES}")

        if len(nrs) > 5 and (
            "some-exception-name" not in parents_str
        ):  # iznimka
            warn(f"Too deep, remove fieldgroup? {parents_str} / {name} / / {item_repr}")  # noqa: T001

        children = item.pop("contains", None)
        enables = item.pop("enables", None)
        assert not (enables and children)

        # label = item.pop("label", None)
        output_label = []
        label, label_type = dump_rule_attr(
            item=item,
            output=output_label,
            indent_depth=indent_depth,
            attr_name="label",
            value_type="message_or_interpret",
            return_value=True,
            value_if_missing=name_to_label(name),  # f"_('{escape_string(label)}')",
        )

        if not parents or children or enables:
            msg = f"{parents_str} -> {name} : {type} : {label}"
            if children or enables:
                space = " " * (indent_depth - 2)
                output_txt.append("")  # noqa: T001
                output_txt.append(f"{space} {msg}")  # noqa: T001
                output_txt.append("{} {}".format(space, "-" * len(msg)))  # noqa: T001
                if enables:
                    output_txt.append(f"{space} enables:")  # noqa: T001
                if children:
                    output_txt.append(f"{space} contains:")  # noqa: T001

            else:
                output_txt.append(msg)  # noqa: T001
        else:
            # space = " " * len(parents_str)
            space = " " * (indent_depth - 2)
            output_txt.append(f"{space} - {name} : {type} : {label}")  # noqa: T001

        if name in objects:
            warn(f"{parents_str} item name {name} is duplicate: {item_repr}")  # noqa: T001
            return

        objects[name] = item

        close_element = True
        if type in ("label", "button"):
            # ignored for now
            return False

        elif type not in ("fieldgroup", "form"):
            subtype = ""

            # class name
            if type == "data":
                class_name = "StaticData"
            elif type == "evaluation":
                class_name = "Evaluation"
            elif type == "validation":
                class_name = "Validation"
            elif type == "dynamic":
                class_name = "FieldsGenerator"
            # elif type == "data":
            #     class_name = "DataSource"
            else:
                class_name = "Field"
                subtype = f" type='{type}',"

            output.append(f"{get_indent(indent_depth+1)}{class_name}(name='{name}',{subtype}")
            # exception
            output.extend(output_label)
            # dump_rule_attr(
            #     item=item,
            #     output=output,
            #     indent_depth=indent_depth,
            #     attr_name="label",
            #     value_type="literal",
            #     value_if_missing=f"_('{escape_string(label)}')",
            # )

            # attributes
            if type in ("data",):
                dump_rule_attr(
                    item=item,
                    output=output,
                    indent_depth=indent_depth,
                    attr_name="datatype",
                    value_type="literal",
                )
                dump_rule_attr(
                    item=item,
                    output=output,
                    indent_depth=indent_depth,
                    attr_name="value",
                    value_type="interpret",
                )
                dump_rule_attr(
                    item=item,
                    output=output,
                    indent_depth=indent_depth,
                    attr_name="transform",
                    value_type="literal",
                )
                # dump_rule_attr(item=item, output=output, indent_depth=indent_depth, attr_name="source"   , value_type="list_or_interpret")
                # dump_rule_attr(
                #     item=item,
                #     output=output,
                #     indent_depth=indent_depth,
                #     attr_name="evaluate",
                #     value_type="bool",
                # )

            elif type in ("evaluation",):
                class_name = "Evaluation"
                raise NotImplementedError("TODO: evaluation")

            elif type in ("validation",):
                class_name = "Validation"
                dump_rule_attr(
                    item=item,
                    output=output,
                    indent_depth=indent_depth,
                    attr_name="ensure",
                    value_type="list_or_interpret",
                )
                dump_rule_attr(
                    item=item,
                    output=output,
                    indent_depth=indent_depth,
                    attr_name="available",
                    value_type="interpret",
                )
                dump_rule_attr(
                    item=item,
                    output=output,
                    indent_depth=indent_depth,
                    attr_name="error",
                    value_type="message",
                )
                # dump_rule_attr(item, output, indent_depth, "data", "list_or_interpret")
            else:
                dump_rule_attr(
                    item=item,
                    output=output,
                    indent_depth=indent_depth,
                    attr_name="label",
                    value_type="message",
                )
                dump_rule_attr(
                    item=item,
                    output=output,
                    indent_depth=indent_depth,
                    attr_name="available",
                    value_type="interpret",
                )
                dump_rule_attr(
                    item=item,
                    output=output,
                    indent_depth=indent_depth,
                    attr_name="disabled",
                    value_type="interpret",
                )
                dump_rule_attr(
                    item=item,
                    output=output,
                    indent_depth=indent_depth,
                    attr_name="default",
                    value_type="literal",
                )
                dump_rule_attr(
                    item=item,
                    output=output,
                    indent_depth=indent_depth,
                    attr_name="value",
                    value_type="interpret",
                )
                dump_rule_attr(
                    item=item,
                    output=output,
                    indent_depth=indent_depth,
                    attr_name="required",
                    value_type="bool_or_interpret",
                )
                dump_rule_attr(
                    item=item,
                    output=output,
                    indent_depth=indent_depth,
                    attr_name="description",
                    value_type="message",
                )
                # dump_rule_attr(
                #     item=item,
                #     output=output,
                #     indent_depth=indent_depth,
                #     attr_name="autocomplete",
                #     value_type="bool",
                # )
                dump_rule_attr(
                    item=item,
                    output=output,
                    indent_depth=indent_depth,
                    attr_name="hidden",
                    value_type="bool_or_interpret",
                )
                dump_rule_attr(
                    item=item,
                    output=output,
                    indent_depth=indent_depth,
                    attr_name="hint",
                    value_type="message_or_interpret",
                )
                # dump_rule_attr(item, output, indent_depth, "data-case", "quote")

                dump_cleaners(
                    parents_str=parents_str,
                    name=name,
                    type=type,
                    item=item,
                    indent_depth=indent_depth,
                    output=output,
                )

            if type == "select":
                options = item.pop("options", None)
                if not options:
                    raise ValueError(f"{name} {type} is select, options not found: {item_repr}")
                output.append(f"{get_indent(indent_depth+2)}options=SelectOptions(")
                iterate_iter_type = dump_rule_attr(
                    options,
                    output,
                    indent_depth + 3,
                    "iterate",
                    "list_or_interpret",
                    available=True,
                )
                dump_rule_attr(options, output, indent_depth + 3, "disabled", "interpret")
                dump_rule_attr(
                    options, output, indent_depth + 3, "value", "interpret", available=True
                )
                dump_rule_attr(
                    options,
                    output,
                    indent_depth + 3,
                    "option",
                    "interpret",
                    available=(iterate_iter_type == "interpret"),
                )
                dump_rule_attr(
                    options,
                    output,
                    indent_depth + 3,
                    "label",
                    "interpret",
                    available=(iterate_iter_type == "interpret"),
                )
                dump_rule_attr(options, output, indent_depth, "label_empty", "message")
                dump_rule_attr(options, output, indent_depth, "value_empty", "interpret")

                if options:
                    warn(
                        f"{parents_str}::{name} ({type}) OPTIONS has unused attributes: {options}"
                    )

                output.append(f"{get_indent(indent_depth+2)}),")

            if type == "dynamic":
                # iter_type =
                dump_rule_attr(
                    item, output, indent_depth + 0, "iterate", "interpret", available=True
                )
                options = item.pop("output", None)
                if not options:
                    raise ValueError(f"{name} {type} is dynamic, options not found: {item_repr}")
                output.append(f"{get_indent(indent_depth+2)}options=DynamicOutput(")
                dump_rule_attr(options, output, indent_depth, "type", "quote")
                dump_rule_attr(options, output, indent_depth, "name", "quote")
                dump_rule_attr(options, output, indent_depth, "value", "interpret")
                dump_rule_attr(options, output, indent_depth, "label", "interpret")
                dump_rule_attr(options, output, indent_depth, "disabled", "bool_or_interpret")
                if options:
                    warn(f"{parents_str}::{name} ({type}) OUTPUT has unused attributes: {options}")
                output.append(f"{get_indent(indent_depth+2)}),")

            if children:
                warn(  # noqa: T001
                    f"{parents_str} item {name} not fieldgroup and has children: {item_repr}"
                )
                return

        if type in ("fieldgroup", "form") or enables:
            # output.append(f"{get_indent(indent_depth+1)}")
            if enables:
                children = enables
                class_name = None
                output.append(f"{get_indent(indent_depth+2)}enables=[")
            else:
                if type == "fieldgroup":
                    class_name = "FieldGroup"
                elif type == "form":
                    class_name = "FieldGroupForm"
                output.append("")
                output.append(f"{get_indent(indent_depth+1)}{class_name}(name='{name}',")
                dump_rule_attr(item, output, indent_depth, "available", "interpret")
                if type == "form":
                    dump_rule_attr(item, output, indent_depth, "method", "quote")
                    dump_rule_attr(item, output, indent_depth, "url", "quote")
                dump_cleaners(
                    parents_str=parents_str,
                    name=name,
                    type=type,
                    item=item,
                    indent_depth=indent_depth,
                    output=output,
                )
                output.append(f"{get_indent(indent_depth+2)}contains=[")

            if not children:
                warn(f"{parents_str} item {name} has no children: {item_repr}")  # noqa: T001
                return

            for nr2, child in enumerate(children, 1):
                parse_item(
                    child,
                    nrs=nrs + [nr2],
                    indent_depth=indent_depth + 3,
                    outputs=outputs,
                    objects=objects,
                    parents=parents + (item,),
                )
            if enables:
                output.append(f"{get_indent(indent_depth+1)}]),")
                close_element = False
            else:
                output.append(f"{get_indent(indent_depth+1)}]),")
                close_element = False

        if close_element:
            output.append(f"{get_indent(indent_depth+1)}),")

        item.pop("name", "")
        if item:
            warn(f"{parents_str}::{name} ({type}) - has unused attributes: {item}")
    except ParseException:
        raise
    except Exception as ex:
        raise ParseException(f"ERROR: {parents_str}::{name} ({type}) - raised exception: {ex}")

    return True


# ------------------------------------------------------------


def main(filename):
    # example in c-s.yaml
    print("=" * 80)  # noqa: T001
    filename = os.path.absdir(filename)
    path = os.path.dirname(filename)
    basename = os.path.basename(filename)
    with open(filename) as fin:
        spec = yaml.safe_load(fin)
        objects = {}
        outputs = {"py": [], "txt": []}

        # DataSource,
        outputs["py"].append(
            "from reedwolf.entities.base import _, Entity, FieldGroup, Field, SelectOptions, DynamicOutput, "
            + "Validation, Evaluation, FieldsGenerator, FieldGroupForm, msg, "
            + "StaticData, option, This, F, D, M, MySettings, Utils"
        )
        outputs["py"].append("")
        name = "TODO:"  # noqa: F841
        outputs["py"].append("entity = Entity(name='{name}',")
        outputs["py"].append(f"{INDENT*2}models = ['TODO'],")
        outputs["py"].append(f"{INDENT*2}contains = [")
        for nr, item in enumerate(spec, 1):
            parse_item(item, indent_depth=2, outputs=outputs, nrs=[nr], objects=objects)
        outputs["py"].append(f"{INDENT}])")

    fname_out = os.path.join(path, f"{basename}.py")
    with open(fname_out, "w") as fout:
        fout.write("\n".join(outputs["py"]))
        print(f"Output in {fname_out}")  # noqa: T001

    fname_txt = os.path.join(path, f"{basename}.txt")
    with open(fname_txt, "w") as fout:
        fout.write("\n".join(outputs["txt"]))
        print(f"Output in {fname_txt}")  # noqa: T001

    # try:
    #     from . import {basename}  # noqa: F401

    #     print("TEST: import is OK")  # noqa: T001
    # except Exception as ex:
    #     print(f"ERROR: Failed to import {basename}: {ex}")  # noqa: T001


if __name__ == "__main__":
    if len(sys.argv)!=1:
        print("Usage: ... yaml-filename")
    else:
        main(sys.argv[0])

