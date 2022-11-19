# Reedwolf.rules project

Reedwolf.rules is a [Python](https://www.python.org/) library that extends
and enrich existing structured and unstructured data model with declarative
constraints and automatic evaluation logic.

Some examples can be found in
[tests directory](https://github.com/trebor74hr/reedwolf.rules/blob/main/tests).

**This is currently unfinished project.** First feature complete and stable
release will include verbose explanation and documentation with examples.

Reedwolf.rules is first published part of larger **Reedwolf** project which
aims solving common software design and implementation tasks in more-or-less
unique way.


# Rules Components

## Building blocks of rules

Building blocks of rules are:

 * CONTAINERS  - are top-level objecs, can contain components or same/lower
               level containers. Rules is top object.
 * EXTENSIONS  - specialized components used for one/many to one/many inner
               data structures.
 * COMPONENTS  - can be embedded into Container (contains)
 * FIELDS      - can read/store data from/to storage. Check fields.py.
 * SECTIONS    - logical groupihg and common functionality/dependency of other
               components
 * VALIDATIONS - data constraints based on field/validator expressions.
               There are some predefined validators.
 * EVALUATIONS - automatic data evaluations (computation) based on
               field/evaluator expressions. There are some predefined
               evaluations.

**NOTES**: 

 * VALIDATIONS and EVALUATIONS are together defined as data **CLEANERS**
 * CardinalityValidators are special Validations category used in Extensions

## Bindng

Binding to existing data structures and functions is done on containers only with:

 * MODELS      - are bound to containers and their fields to components'
               fields. 
 * DATA PROVIDERS - external functions that return data lists that couuld be
                  used in ChoiceField, EnumFields and similar fields


## Internal objects

Internal objects are:

 * HEAP - holds all variables and bounded (used) variables
 * VARIABLES - are used to proccess fields/component values
 * EXPRESSIONS - expressions ... **TODO:**
 * FUNCTIONS - functions that could be used in expressions
 * TYPEINFO - object that wraps around data tyus (python type hinting)
 
 * VALIDATORS - predefined validators that can be used in validatons
 * EVALUATORS - predefined evaluators that can be used in evaluations

## Customization

User can add and use custom:
 * validations
 * evaluations
 * validators
 * evaluators
 * functions


