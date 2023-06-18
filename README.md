# Reedwolf.Entities project

Reedwolf.Entities is a [Python](https://www.python.org/) library that extends
and enrich existing structured and unstructured data model with declarative
constraints and automatic evaluation logic.

**This is currently unfinished project.** First feature complete and stable
release will include verbose explanation and documentation with examples.

Reedwolf.entities is first published part of larger **Reedwolf** project which
aims solving common software design and implementation tasks in more-or-less
unique way.


# Components

## Building blocks

Building blocks are:

 * CONTAINERS  - are top-level objecs, can contain components or same/lower
               level containers. **Entity** is top object.
 * EXTENSIONS  - specialized components used for one/many to one/many inner
               data structures.
 * COMPONENTS  - can be embedded into Container (contains)
 * FIELDS      - can read/store data from/to storage. Check fields.py.
 * FIELDGROUP  - logical groupihg and common functionality/dependency of other
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

## Internal objects

Internal objects are:

 * VALUEEXPRESSIONS - expressions ... **TODO:**
 * FUNCTIONS - functions that could be used in expressions
 * TYPEINFO - object that wraps around data tyus (python type hinting)
 
## Customization

User can add and use custom:
 * validations
 * evaluations
 * functions


