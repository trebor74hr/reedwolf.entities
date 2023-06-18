# Copied and adapted from Reedwolf project (project by robert.lujo@gmail.com - git@bitbucket.org:trebor74hr/reedwolf.git)
# Uses https://getbootstrap.com/ - version 5.1
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

from ..base import (
        ComponentBase,
        )
from ..components import Field, FieldGroup, FieldTypeEnum, ValidationBase, EvaluationBase
from ..containers import SubEntityItems, Entitiy


HTML_INDENT = "  "

# ------------------------------------------------------------

@dataclass
class DumpHtmlLines:

    lines        : List[str] = field(init=False, default_factory=list)

    def add_lines(self, lines: List[str]):
        self.lines.extend(lines)

# ------------------------------------------------------------

def dump_to_html(component:ComponentBase, fname:str):
    lines = dump_html_models_to_str(component=component)
    len_lines = len(lines)
    code = "\n".join(lines)
    with open(fname, "w") as fout:
        fout.write(code)

    print(f"Output in {fname}, {len_lines} lines.")
    return

def prefix_indent(indent, lines):
    return [f"{indent}{line}" for line in lines]

def get_html_attrs(common_attrs):
    common_attrs_str = []
    for k,v in common_attrs.items():
        if v is not None:
            common_attrs_str.append(f'{k}="{v}"')
        else:
            common_attrs_str.append(f'{k}')
    common_attrs_str = " ".join(common_attrs_str)
    return common_attrs_str

# ------------------------------------------------------------

def dump_html_models_to_str(
                         component:ComponentBase,
                         path:List[str]=None,
                         depth:int=0,
                         component_dict: Dict[str, ComponentBase]=None,
                         inline:bool=False,
                         ) -> List[str]:

    if depth==0:
        assert path is None
        path = []
        component_dict = {}

    assert component.name not in component_dict
    component_dict[component.name] = component

    indent = HTML_INDENT * depth
    # indent_next = HTML_INDENT * (depth+1)

    lines = []

    children = component.get_children()

    lines_children = {} # ordered
    for child_comp in children:
        assert child_comp.name not in lines_children
        # recursion
        lines_children[child_comp.name] = dump_html_models_to_str(child_comp, path=path[:],
                                                depth=depth+1,
                                                component_dict=component_dict,
                                                inline=isinstance(component, SubEntityItems),
                                                )

    # guard
    if depth==1 and not isinstance(component, (FieldGroup,)):
        raise Exception(f"TODO: no supported, currenly only structure Entitiy+FieldGroups supported, got {component}")

    # py_name = None
    if depth==0:
        assert isinstance(component, (Entitiy,)), component
        lines.append(f"<h1 id='{component.name}'>{component.label}</h1>")
        lines.append('<nav>')
        lines.append('  <div class="nav nav-tabs" id="nav-tab" role="tablist">')

        area_selected = 'aria-selected="true"'
        for nr, (child_name, child_lines) in enumerate(lines_children.items(),1):
            child_comp = component_dict[child_name]
            line = ('    <button class="nav-link {"active" if nr==1 else ""}" '
                    f' id="nav-{child_name}-tab" data-bs-toggle="tab" '
                    f' data-bs-target="#nav-{child_name}" type="button" role="tab" '
                    f' aria-controls="nav-home" {area_selected if nr==1 else ""}>'
                    f'{child_comp.label}</button>')
            lines.append(line)
        lines.append('  </div>')
        lines.append('</nav>')


        lines.append('<div class="tab-content" id="nav-tabContent">')
        for nr, (child_name, child_lines) in enumerate(lines_children.items(),1):
            lines.append(f'  <div class="container tab-pane fade {"show active" if nr==1 else ""}"'
                         f' id="nav-{child_name}" role="tabpanel" aria-labelledby="nav-{child_name}-tab">')
            lines.append('    <form class="row g-3">')
            lines.extend(prefix_indent("    ", child_lines))
            lines.append('      <div class="col-12">')
            lines.append('        <button type="submit" class="btn btn-primary">Save</button>')
            lines.append('      </div>')
            lines.append('    </form>')
            lines.append('  </div>')
        lines.append('</div>')

    elif isinstance(component, (SubEntityItems,)):
        nr_examples = 3

        common_attrs = {}
        common_attrs["data-bs-toggle"]="tooltip"
        common_attrs["data-bs-placement"]="right"
        common_attrs["title"]= f'Cardinality: {component.cardinality}'
        # TODO: this too?:
        #   Is list: {component.bound_attr_node.data.is_list}'
        #   Is optional: {component.bound_attr_node.data.is_list}'
        #                component.cardinality.allow_none

        common_attrs_str = get_html_attrs(common_attrs)

        lines.append(f'<ul class="my-list-unstyled" {common_attrs_str} >')

        for row_nr in range(1, nr_examples+1):
            is_add = (row_nr==nr_examples)
            lines.append('    <ul class="list-inline" >')
            lines.append(f'      <li class="list-inline-item" id="name-{component.name}-{row_nr}">{row_nr}. </li>')
            for nr, (child_name, child_lines) in enumerate(lines_children.items(),1):
                lines.append(f'      <li class="list-inline-item" id="container-{child_name}-{row_nr}">')
                lines.extend(prefix_indent("      ", child_lines))
                lines.append('      </li>')

            lines.append(f'      <li class="list-inline-item" id="container-{child_name}-{row_nr}">')
            lines.append(f'        <button type="button" class="btn {"btn-success" if is_add else "btn-secondary"} " id="add-{component.name}">{"Add" if is_add else "Edit"}</button>')
            lines.append('      </li>')

            lines.append('    </ul>')
            lines.append('  </li>')
        lines.append('</ul>')


    elif isinstance(component, (FieldGroup, Entitiy)):
        assert lines_children
        lines.append('<ul class="my-list-unstyled">')
        title = f"<h3>{component.label}</h3>" if depth<=1 else f"<strong>{component.label}</strong>"
        lines.append(f'  <li id="title-{component.name}">{title}</li>')
        for nr, (child_name, child_lines) in enumerate(lines_children.items(),1):
            lines.append(f'  <li id="container-{child_name}">')
            lines.extend(prefix_indent("  ", child_lines))
            lines.append('  </li>')
        lines.append('</ul>')


    elif isinstance(component, (Field,)):
        html = []

        tooltip = []
        common_attrs = {}

        tooltip.append(f"Type: {component.type.name}")


        if component.required:
            common_attrs["required"] = None
            if component.required in (True,):
                tooltip.append("Required")
            else:
                tooltip.append(f"Required when: {component.required}")
        else:
            tooltip.append("Optional")


        if component.default is not None:
            tooltip.append(f"Default: {component.default}")

        # TODO: check for cleaners.Readonly(...)
        # if component.editable not in (True, None):
        #     tooltip.append(f"Editable when: {component.editable}")

        # if component.autocomplete not in (True, None):
        #     tooltip.append(f"Autocomplete: {component.autocomplete}")

        if component.available not in (True, None):
            tooltip.append(f"Available term(s): {component.available}")

        if component.cleaners:
            out = []
            for cleaner in component.cleaners:
                if isinstance(cleaner, ValidationBase):
                    out.append(f"Validate {cleaner.label} => {repr(cleaner.ensure)}")
                elif isinstance(cleaner, EvaluationBase):
                    out.append(f"Evaluate {cleaner.label} => {repr(cleaner.ensure)}")
                else:
                    raise NotImplementedError(f"{cleaner} not implemented")
            tooltip.append(f"Cleaners: {' AND '.join(out)}")

        if component.enables:
            out = []
            for child_comp in component.enables:
                out.append(f"{child_comp.label}") # ({child_comp.name})")
            tooltip.append(f"Enables components: {', '.join(out)}")

        if component.description:
            tooltip.append(f"Description: {component.description}")


        if tooltip:
            common_attrs["data-bs-toggle"]="tooltip"
            common_attrs["data-bs-html"]="true"
            common_attrs["data-bs-placement"]="left"
            common_attrs["title"]='<ul>{}</ul>'.format("\n".join([f"<li>{tt}</li>" for tt in tooltip]))

        common_attrs_str = get_html_attrs(common_attrs)

        common_class_str = f'mb-3 {"col-md-12" if inline else "col-md-6"}'

        if component.type in (FieldTypeEnum.CHOICE, FieldTypeEnum.ENUM):
            html.append(f'<div class="form-floating {common_class_str} ">')
            html.append(f'  <select class="form-select" id="{component.name}" aria-label="{component.name}" {common_attrs_str} >')
            if component.type==FieldTypeEnum.CHOICE:
                choices_name = getattr(component.choices, "__qualname__",
                                 getattr(component.choices, "__name__",
                                   getattr(component.choices, "_name",
                                     component.choices.__class__.__name__)))
                choices = [("", f"Choices from {choices_name}")]+[(f"choice-{x}", f"Some choice {x}") for x in range(1,4)]
            else:
                # use get_display_values?
                choices = [("", f"Choices from {component.enum.__name__}")] + [(ev.value, ev.name) for ev in component.enum]
            for nr, (k,v) in enumerate(choices,1):
                html.append(f'    <option {"selected" if nr==1 else ""} value="{k}">{v}</option>')
            html.append('  </select>')
            html.append(f'  <label for="{component.name}">{component.label}</label>')
            html.append('</div>')

        elif component.type==FieldTypeEnum.BOOL  :
            # NOTE: not using: {common_class_str}
            html.append(f'<div class="form-check mb-1 col-md-6 " {common_attrs_str} >')
            html.append(f'  <input class="form-check-input" type="checkbox" value="" id="{component.name}" checked >')
            html.append(f'  <label class="form-check-label" for="{component.name}">')
            html.append(f'    {component.label}')
            html.append('  </label>')
            html.append('</div>')
        elif component.type==FieldTypeEnum.FILE:
            html.append(f'<div class="{common_class_str}">')
            html.append(f'  <label for="{component.name}" class="form-label">{component.label}</label>')
            html.append(f'  <input class="form-control" type="file" id="{component.name}" {common_attrs_str} >')
            html.append('</div>')
        else:
            value = ""
            if component.type==FieldTypeEnum.EMAIL:
                input_type = "email"
            elif component.type==FieldTypeEnum.NUMBER:
                input_type = "number"
            elif component.type==FieldTypeEnum.INPUT:
                input_type = "text"
            elif component.type==FieldTypeEnum.DATE:
                input_type = "date"
            elif component.type==FieldTypeEnum.PASSWORD:
                input_type = "password"
            else:
                # fallback - unhandled type
                input_type = "text"
                value = f"Field type = {component.type}"

            html.append(f'<div class="form-floating {common_class_str}">')
            html.append(f'  <input type="{input_type}" class="form-control" id="{component.name}" value="{value}"'
                            f' placeholder="{component.label}" {common_attrs_str} >')
            html.append(f'  <label for="floatingInput">{component.label}</label>')
            html.append('</div>')

        # TODO: forgot f"" or some other issue?
        #       assert "{" not in "\n".join(html), html

        if lines_children:
            lines.append(f"<div id='wrapper-{component.name}'>")
            lines.extend(prefix_indent("  ", html))
            # lines.append(f'  <h{depth}>{component.label}</h{depth}>')
            lines.append(f'  <ul class="my-list-unstyled" id="children-{component.name}">')
            for nr, (child_name, child_lines) in enumerate(lines_children.items(),1):
                lines.append('    <li>')
                lines.extend(prefix_indent("    ", child_lines))
                lines.append('    </li>')
            lines.append('  </ul>')
            lines.append('</div>')
        else:
            lines.extend(html)

    else:
        assert False, component


    if depth==0:
        all_lines = []
        all_lines.extend([
            '<!DOCTYPE html>',
            '<html lang="en">',
            '  <head>',
            '    <meta charset=UTF-8>',
            f'    <title>{component.label}</title>',
            '    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">',
            '    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p" crossorigin="anonymous"></script>'
            '    <style media="all">',
            '        ul.my-list-unstyled { list-style: none; }',
            '        input:optional {',
            '          border-color: gray;',
            '        }',
            '        select:required, input:required {',
            '          border-color: black;',
            '          border-width: 1px;',
            '        }',
            '    </style>'
            "  </head>",
            "<body>",
            ])

        all_lines.extend(lines)

        all_lines.append("</body>")
        all_lines.extend([
            '<script>',
            """
            var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
            var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
              return new bootstrap.Tooltip(tooltipTriggerEl)
            })
            """,
            '</script>',
            ])
        all_lines.append("</html>")

        out = all_lines
    else:
        lines = prefix_indent(indent, lines)
        out = lines

    return out


