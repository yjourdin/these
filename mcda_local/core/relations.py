from abc import ABC, abstractmethod
from inspect import isclass
from typing import Any, Iterator, cast

from graphviz import Digraph
from numpy import identity, triu_indices
from pandas import DataFrame

from .matrices import BinaryAdjacencyMatrix


class Relation(ABC):
    """This class represents a pairwise relation between two elements.

    :param a: first element
    :param b: second element

    :attribute a:
    :attribute b:
    :attribute DRAW_STYLE: (class) key args for plotting all instances
    """

    _RELATION_TYPE = ""
    DRAW_STYLE: dict[str, Any] = {"style": "invis"}

    def __init__(self, a: Any, b: Any):
        self.a = a
        self.b = b
        self.validate()

    def __str__(self) -> str:
        """Return string representation of object.

        :return:
        """
        return f"{self.a} {self._RELATION_TYPE} {self.b}"

    def __repr__(self) -> str:  # pragma: nocover
        """Return representation of object.

        :return:
        """
        return f"{self.__class__.__name__}({self.a}, {self.b})"

    @property
    def elements(self) -> tuple[Any, Any]:
        """Return elements of the relation"""
        return self.a, self.b

    def validate(self):
        """Check whether a relation is valid or not."""
        pass

    def same_elements(self, relation: "Relation") -> bool:
        """Check whether the relations are about the same pair of alternatives.

        :param relation: second relation
        :return:
            ``True`` if both relations share the same elements pair, ``False``
            otherwise

        .. warning:: Does not check for relations' validity!
        """
        return set(self.elements) == set(relation.elements)

    def __eq__(self, other: Any) -> bool:
        """Check whether relations are equal.

        :param other:
        :return: check result

        .. warning:: Does not check for relations' validity!
        """
        if type(other) is type(self):
            return self.elements == other.elements
        return False

    def __add__(self, other: "Relation") -> "PreferenceStructure":
        """Build new preference structure as addition of both relations.

        :return: relations added to new preference structure
        """
        if not isinstance(other, Relation):
            raise TypeError("can only add one other Relation object")
        return PreferenceStructure([self, other])

    def __hash__(self) -> int:
        """Hash object based on its unordered list of elements"""
        return hash(self.a) + hash(self.b)

    def compatible(self, other: "Relation") -> bool:
        """Check whether both relations can coexist in the same preference
        structure.

        Relations are compatible if equal or having different elements pair.

        :param other:
        :return: check result

        .. warning:: Does not check for relations' validity!
        """
        return self == other or not self.same_elements(other)

    @classmethod
    def types(cls) -> list:
        """Return list of relation types.

        :return:
        """
        return cls.__subclasses__()

    def _draw(self, graph: Digraph):
        """Draw relation on provided graph"""
        graph.edge(str(self.a), str(self.b), **self.DRAW_STYLE)


class PreferenceRelation(Relation):
    """This class represents a preference relation between two elements.

    A relation is read `aPb`.

    :param a: first element
    :param b: second element

    :attribute a:
    :attribute b:
    :attribute DRAW_STYLE: (class) key args for plotting all instances

    .. note:: this relation is antisymmetric and irreflexive
    """

    _RELATION_TYPE = "P"
    DRAW_STYLE: dict[str, Any] = {}

    def validate(self):
        """Check whether a relation is valid or not.

        :raise ValueError: if relation is reflexive
        """
        if self.a == self.b:
            raise ValueError(
                f"Preference relations are irreflexive: {self.a} == {self.b}"
            )


class IndifferenceRelation(Relation):
    """This class represents an indifference relation between two elements.

    A relation is read `aIb`.

    :param a: first element
    :param b: second element

    :attribute a:
    :attribute b:
    :attribute DRAW_STYLE: (class) key args for plotting all instances

    .. note:: this relation is symmetric and reflexive
    """

    _RELATION_TYPE = "I"
    DRAW_STYLE = {"arrowhead": "none"}

    __hash__ = Relation.__hash__

    def __eq__(self, other):
        """Check whether relations are equal.

        :param other:
        :return: check result

        .. warning:: Does not check for relations' validity!
        """
        if type(other) is type(self):
            return self.same_elements(other)
        return False


class IncomparableRelation(Relation):
    """This class represents an incomparable relation between two elements.

    A relation is read `aRb`.

    :param a: first element
    :param b: second element

    :attribute a:
    :attribute b:
    :attribute DRAW_STYLE: (class) key args for plotting all instances

    .. note:: this relation is symmetric and irreflexive
    """

    _RELATION_TYPE = "R"
    DRAW_STYLE = {"arrowhead": "none", "style": "dotted"}

    __hash__ = Relation.__hash__

    def __eq__(self, other):
        """Check whether relations are equal.

        :param other:
        :return: check result

        .. warning:: Does not check for relations' validity!
        """
        if type(other) is type(self):
            return self.same_elements(other)
        return False

    def validate(self):
        """Check whether a relation is valid or not.

        :raise ValueError: if relation is reflexive
        """
        if self.a == self.b:
            raise ValueError(
                f"Incomparable relations are irreflexive: {self.a} == {self.b}"
            )


class IPreferenceStructure(ABC):
    """This interface describes preference structures.

    It enables conversion between any type of preference structure through
    defined conversions from and to a :class:`PreferenceStructure`.
    """

    @property
    @abstractmethod
    def preference_structure(self) -> "PreferenceStructure":  # pragma: nocover
        """Convert object to :class:`PreferenceStructure`"""
        pass

    @classmethod
    @abstractmethod
    def _from_preference_structure(
        cls, preference_structure: "PreferenceStructure"
    ) -> "IPreferenceStructure":  # pragma: nocover
        """Convert a :class:`PreferenceStructure` to current class.

        :param preference_structure:
        :return:
        """
        pass

    @classmethod
    def cast_from(
        cls, data: "IPreferenceStructure"
    ) -> "IPreferenceStructure":  # pragma: nocover
        """Convert any preference structure subclass instance to current type.

        :param data: instance of a preference structure subclass
        :return:
        """
        return cls._from_preference_structure(data.preference_structure)


class PreferenceStructure(IPreferenceStructure):
    """This class represents a list of relations.

    Any type of relations is accepted, so this represents the union of P, I and
    R.

    :param data:
    """

    def __init__(
        self,
        data: "list[Relation] | Relation | PreferenceStructure | None" = None,
    ):
        data = data or []
        match data:
            case Relation():
                relations = [data]
            case PreferenceStructure():
                relations = data.relations
            case _:
                relations = data
        self._relations = relations
        # self._relations = list(set(relations))
        # self.validate()

    @property
    def preference_structure(self) -> "PreferenceStructure":
        """Return current preference structure.

        .. note::
            necessary for implementing interface :class:`IPreferenceStructure`
        """
        return self

    @classmethod
    def _from_preference_structure(
        cls, preference_structure: "PreferenceStructure"
    ) -> "PreferenceStructure":
        """Return copy of current preference structure.

        .. note::
            necessary for implementing interface :class:`IPreferenceStructure`
        """
        return cls(preference_structure)

    @classmethod
    def cast_from(cls, data: "IPreferenceStructure") -> "PreferenceStructure":
        """Convert any preference structure subclass instance to current type.

        :param data: instance of a preference structure subclass
        :return:
        """
        return cls._from_preference_structure(data.preference_structure)

    @property
    def elements(self) -> list[Any]:
        """Return elements present in relations list."""
        return list(set(e for r in self._relations for e in r.elements))

    @property
    def relations(self) -> list[Relation]:
        """Return copy of relations list."""
        return self._relations.copy()

    def validate(self):
        """Check whether the relations are all valid.

        :raise ValueError: if at least two relations are incompatible
        """
        for i, r1 in enumerate(self._relations):
            for r2 in self._relations[(i + 1) :]:
                if not r1.compatible(r2):
                    raise ValueError(f"incompatible relations: {r1}, {r2}")

    @property
    def is_total_preorder(self) -> bool:
        """Check whether relations list is a total preorder or not"""
        return (
            len(PreferenceStructure(self.transitive_closure[IncomparableRelation])) == 0
        )

    @property
    def is_total_order(self) -> bool:
        """Check whether relations list is a total order or not"""
        res = self.transitive_closure
        return (
            len(PreferenceStructure(res[IncomparableRelation]))
            + len(PreferenceStructure(res[IndifferenceRelation]))
            == 0
        )

    def __eq__(self, other: Any):
        """Check if preference structure is equal to another.

        Equality is defined as having the same set of relations.

        :return:

        .. note:: `other` type is not coerced
        """
        if isinstance(other, PreferenceStructure):
            return set(other.relations) == set(self._relations)
        return False

    def __len__(self) -> int:
        """Return number of relations in the preference structure.

        :return:
        """
        return len(self._relations)

    def __str__(self) -> str:
        """Return string representation of relations.

        :return:
        """
        return "[" + ", ".join([str(r) for r in self._relations]) + "]"

    def __repr__(self) -> str:  # pragma: nocover
        """Return representation of relations contained in structure

        :return:
        """
        return f"{self.__class__.__name__}({repr(self._relations)})"

    def _relation(
        self,
        *args: Any,
    ) -> "Relation | PreferenceStructure | None":
        """Return all relations between given elements of given types.

        If no relation type is supplied, all are considered.
        If no element is supplied, all are considered.

        :param *args:
        :return:

        .. warning:: Does not check for a relation's validity or redundancy!
        """
        elements = []
        types = []
        for arg in args:
            if isclass(arg) and issubclass(arg, Relation):
                types.append(arg)
            else:
                elements.append(arg)
        elements = elements or self.elements
        types = types or Relation.types()
        res = None
        for r in self._relations:
            if r.a in elements and r.b in elements and r.__class__ in types:
                res = cast(Relation, res) + r if res else r
        return res

    def _element_relations(self, a: Any) -> "Relation | PreferenceStructure | None":
        """Return all relations involving given element.

        :param a: element
        :return:

        .. warning:: Does not check for a relation's validity or redundancy!
        """
        res = None
        for r in self._relations:
            if a in r.elements:
                res = cast(Relation, res) + r if res else r
        return res

    def __getitem__(self, item: Any) -> "Relation | PreferenceStructure | None":
        """Return all relations matching the request

        :param item:
        :return:
            Depending on `item` type:
                * pair of elements: search first relation with this elements
                pair
                * element: all relations involving element
                * relation class: all relations of this class
        """
        if isinstance(item, tuple):
            return self._relation(*item)
        if isclass(item):
            return self._relation(item)
        return self._element_relations(item)

    def __delitem__(self, item: Any):
        """Remove all relations matching the request

        :param item:
        :return:
            Depending on `item` type:
                * pair of elements: search first relation with this elements
                pair
                * element: all relations involving element
                * relation class: all relations of this class
        """
        r = self[item]
        to_delete = PreferenceStructure(r)._relations
        self._relations = [rr for rr in self._relations if rr not in to_delete]

    def __contains__(self, item: Any) -> bool:
        """Check whether a relation is already in the preference structure.

        :param item: relation
        :return: check result

        .. warning:: Does not check for a relation's validity!
        """
        for r in self._relations:
            if r == item:
                return True
        return False

    def __add__(self, other: Any) -> "PreferenceStructure":
        """Create new preference structure with appended relations.

        :param other:
            * :class:`Relation`: relation is appended into new object
            * :class:`PreferenceStructure`: all relations are appended into new
            object
        :return:
        """
        if hasattr(other, "__iter__"):
            return self.__class__(self._relations + [r for r in other])
        return self.__class__(self._relations + [other])

    def __iter__(self) -> Iterator[Relation]:
        """Return iterator over relations

        :return:
        """
        return iter(self._relations)

    @property
    def transitive_closure(self) -> "PreferenceStructure":
        """Apply transitive closure to preference structure and return result.

        .. warning:: Does not check for a valid preference structure!
        """
        return PreferenceStructure.cast_from(
            cast(
                OutrankingMatrix,
                OutrankingMatrix.cast_from(self).transitive_closure,
            )
        )

    @property
    def transitive_reduction(self) -> "PreferenceStructure":
        """Apply transitive reduction to preference structure and return result

        .. warning:: Does not check for a valid preference structure!

        .. warning:: This function may bundle together multiple elements
        """
        return PreferenceStructure.cast_from(
            cast(
                OutrankingMatrix,
                OutrankingMatrix.cast_from(self).transitive_reduction,
            )
        )

    def plot(self, relation_types: list | None = None) -> Digraph:
        """Create a graph for list of relation.

        This function creates a Graph using graphviz and display it.
        """
        relation_types = relation_types or [PreferenceRelation, IndifferenceRelation]
        relation_graph = Digraph("relations", strict=True)
        relation_graph.attr("node", shape="box")
        for e in self.elements:
            relation_graph.node(str(e))
        for r in self._relations:
            for c in relation_types:
                if isinstance(r, c):
                    r._draw(relation_graph)
                    continue
        relation_graph.render()
        return relation_graph

    def copy(self) -> "PreferenceStructure":
        """Copy preference structure into new object.

        :return: copy
        """
        return PreferenceStructure(self)


class OutrankingMatrix(BinaryAdjacencyMatrix, IPreferenceStructure):
    """This class implements an outranking matrix as an adjacency matrix.

    The outranking matrix is represented internally by a
    :class:`pandas.DataFrame` with vertices as the indexes and columns.

    :param data: adjacency matrix in an array-like or dict-structure
    :param vertices:

    :raise ValueError:
        * if non-binary values are in the matrix
        * if columns and rows have different sets of labels
    :raise KeyError:
        * if some indexes are duplicated
        * if some columns are duplicated
    """

    @property
    def preference_structure(self) -> PreferenceStructure:
        """Return corresponding preference structure."""
        relations: list[Relation] = list()
        elements = self.data.index
        matrix = self.data.to_numpy()
        matrix = matrix + 2 * matrix.T
        indices = triu_indices(len(elements))
        for i, j in zip(indices[0], indices[1]):
            match matrix[i, j]:
                case 0:
                    relations.append(IncomparableRelation(elements[i], elements[j]))
                case 1:
                    relations.append(PreferenceRelation(elements[i], elements[j]))
                case 2:
                    relations.append(PreferenceRelation(elements[j], elements[i]))
                case 3:
                    relations.append(IndifferenceRelation(elements[i], elements[j]))
        # for ii, i in enumerate(self.data.index):
        #     for j in self.data.index[ii + 1 :]:
        #         if self.data.loc[i, j]:
        #             if self.data.loc[j, i]:
        #                 relations.append(IndifferenceRelation(i, j))
        #             else:
        #                 relations.append(PreferenceRelation(i, j))
        #         elif self.data.loc[j, i]:
        #             relations.append(PreferenceRelation(j, i))
        #         else:
        #             relations.append(IncomparableRelation(i, j))
        return PreferenceStructure(relations)

    @classmethod
    def _from_preference_structure(
        cls, preference_structure: PreferenceStructure
    ) -> "OutrankingMatrix":
        """Transform a preference structure into an outranking matrix.

        :param preference_structure: the matrix of relations
        :return: outranking matrix
        """
        elements = preference_structure.elements
        N = len(elements)
        element_to_index = dict(zip(elements, range(N)))
        matrix = identity(N)
        indices: tuple[list[Any], list[Any]] = ([], [])
        for r in preference_structure:
            a, b = r.elements
            match r:
                case PreferenceRelation():
                    indices[0].append(element_to_index[a])
                    indices[1].append(element_to_index[b])
                case IndifferenceRelation():
                    indices[0].extend([element_to_index[a], element_to_index[b]])
                    indices[1].extend([element_to_index[b], element_to_index[a]])
        matrix[indices[0], indices[1]] = 1
        return OutrankingMatrix(DataFrame(matrix, index=elements, columns=elements))
        # elements = preference_structure.elements
        # matrix = DataFrame(0, index=elements, columns=elements)
        # fill_diagonal(matrix.values, 1)
        # for r in preference_structure:
        #     a, b = r.elements
        #     match r:
        #         case PreferenceRelation():
        #             matrix.loc[a, b] = 1
        #         case IndifferenceRelation():
        #             matrix.loc[a, b] = 1
        #             matrix.loc[b, a] = 1
        # return OutrankingMatrix(matrix)

    @classmethod
    def cast_from(cls, data: "IPreferenceStructure") -> "OutrankingMatrix":
        """Convert any preference structure subclass instance to current type.

        :param data: instance of a preference structure subclass
        :return:
        """
        return cls._from_preference_structure(data.preference_structure)

    @classmethod
    def from_ordered_alternatives_groups(
        cls, categories: list[list[Any]]
    ) -> "OutrankingMatrix":
        """Convert a ranking of categories of alternatives into an outranking
        matrix.

        :param categories:
            the ranked categories (each category is a list of alternatives)
        :return: outranking matrix
        """

        alternatives = [a for ll in categories for a in ll]
        res = cls(0, vertices=alternatives)
        for category in categories:
            res.data.loc[category, category] = 1
            res.data.loc[
                category, alternatives[alternatives.index(category[-1]) + 1 :]
            ] = 1
        return res
