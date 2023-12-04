from typing import Any, Iterator, cast

from pandas import Series
from pandas.api.types import is_numeric_dtype

from .relations import (
    IndifferenceRelation,
    IPreferenceStructure,
    PreferenceRelation,
    PreferenceStructure,
    Relation,
)
from .scales import (
    NominalScale,
    NormalScale,
    PreferenceDirection,
    QuantitativeScale,
    Scale,
)


def series_equals(s1: Series, s2: Series) -> bool:
    """Check if two series have the same values.

    It will realign the indexes if they are ordered differently.

    :param s1:
    :param s2:
    :return:
    """
    return dict(s1) == dict(s2)


class Values:
    """This class is a wrapper around :class:`pandas.Series`.

    It is intended to be used for all classes across the package that uses
    a Series as their internal data representation.

    :param data: series containing the data
    :raise KeyError: if some labels are duplicated

    :attr data: internal representation of data
    """

    def __init__(self, data: Series):
        self.data = Series(data)
        if self.data.index.has_duplicates:
            raise KeyError(
                "some labels are duplicated: "
                f"{self.data.index[self.data.index.duplicated()].tolist()}"
            )

    def __eq__(self, other: Any) -> bool:
        """Check if both values have the same data

        :return:

        .. note:: values order does not matter
        """
        if type(other) is not type(self):
            return False
        return series_equals(self.data, other.data)

    @property
    def name(self) -> Any:
        """Return the name of the :attr:`data` attribute."""
        return self.data.name

    @property
    def labels(self) -> list[Any]:
        """Return the data labels."""
        return self.data.index.tolist()

    @property
    def is_numeric(self) -> bool:
        """Check whether values are numeric.

        :return:
        :rtype: bool
        """
        return is_numeric_dtype(self.data)

    def sum(self) -> float:
        """Return the sum of the data.

        :return:

        .. warning::
            it will raise a :class:`TypeError` if data contains numeric
            and non-numeric values
        """
        return sum(self.data)

    def copy(self) -> "Values":
        """Return a copy of the object"""
        return Values(self.data.copy())

    def __iter__(self) -> Iterator:
        """Return an iterator over the data."""
        return iter(self.data)

    def __getitem__(self, item: Any) -> Any:
        """Return the value of the data at a specific label.

        :return:
        """
        return self.data[item]

    def __setitem__(self, key: Any, value: Any):
        """Set the value of the data at a specific label."""
        self.data[key] = value


class ScaleValues(Values):
    """This class associates a data :class:`pandas.Series` with their
    multiple :class:`mcda.core.Scale`.

    :param data: series containing the data
    :param scales:
        data scale(s) (one per value or one shared, will be inferred from data
        if absent using :meth:`ScaleValues.bounds`)
    :param preference_direction: (ignored is `scales` supplied)
    :raise KeyError:
        * if some labels are duplicated
        * if `scales` keys and `data` indexes mismatch

    :attr data: internal representation of data
    :attr scales: scales of the data (one per value)
    """

    def __init__(
        self,
        data: Series,
        scales: Scale | dict[Any, Scale] | None = None,
        preference_direction: PreferenceDirection = PreferenceDirection.MAX,
    ):
        super().__init__(data)
        scales = scales or self._bounds(preference_direction)
        if isinstance(scales, Scale):
            scales = {k: scales for k in self.labels}
        self.scales = scales
        if set(self.labels) != set(self.scales.keys()):
            raise KeyError("data and scales must have the same labels")

    def _bounds(
        self,
        preference_direction: PreferenceDirection = PreferenceDirection.MAX,
    ) -> Scale:
        """Infer one common scale from the data.

        It returns a :class:`mcda.core.scales.QuantitativeScale` for numeric
        data, a :class:`mcda.core.scales.NominalScale` otherwise.

        :param preference_direction:
        :return: inferred scale
        """
        if self.is_numeric:
            return QuantitativeScale(
                self.data.min(), self.data.max(), preference_direction
            )
        return NominalScale(cast(list[Any], list(set(self.data.values))))

    def __eq__(self, other: Any) -> bool:
        """Check equality of scale values.

        Equality is defines as having the same set of scales, and having the
        same data.

        :return: ``True`` if both are equal
        """
        if not isinstance(other, ScaleValues):
            return False
        _values = cast(ScaleValues, other)
        if self.scales == _values.scales:
            return super().__eq__(_values)
        return False

    @property
    def bounds(self) -> Scale:
        """Infer one common scale from the data.

        It returns a :class:`mcda.core.scales.QuantitativeScale` with maximize
        preference direction for numeric data, a
        :class:`mcda.core.scales.NominalScale` otherwise.

        :return: inferred scale
        """
        return self._bounds()

    @property
    def within_scales(self) -> "ScaleValues":
        """Return a series indicating which values are within their
        respective scale.

        :return:
        """

        return self.__class__(
            Series({str(k): v in self.scales[k] for k, v in self.data.items()}),
            self.scales,
        )

    @property
    def is_within_scales(self) -> bool:
        """Check whether all values are within their respective scales.

        :return:
        """
        return self.within_scales.data.all()

    def dominate(self, other: "ScaleValues") -> bool:
        """Check whether the scale values dominates an other one.

        :param other:
        :return:
            ``True`` if this object dominates ``other``, ``False`` otherwise

        .. note:: if :attr:`scales` are not quantitative, ``False`` is returned
        """
        _other = other.transform(self.scales)
        strict_dominance = False
        for criterion, scale in self.scales.items():
            if not isinstance(scale, QuantitativeScale):
                return False
            _scale = cast(QuantitativeScale, scale)
            if _scale.is_better(_other.data[criterion], self.data[criterion]):
                return False
            if _scale.is_better(self.data[criterion], _other.data[criterion]):
                strict_dominance = True
        return strict_dominance

    def dominate_strongly(self, other: "ScaleValues") -> bool:
        """Check whether the scale values dominates strongly an other one.

        :param other:
        :return:
            ``True`` if this object dominates strongly ``other``, ``False``
            otherwise

        .. note:: if :attr:`scales` are not quantitative, ``False`` is returned
        """
        _other = other.transform(self.scales)
        for criterion, scale in self.scales.items():
            if not isinstance(scale, QuantitativeScale):
                return False
            _scale = cast(QuantitativeScale, scale)
            if not _scale.is_better(self.data[criterion], _other.data[criterion]):
                return False
        return True

    def transform(self, out_scales: dict[Any, Scale] | Scale) -> "ScaleValues":
        """Return data transformed to the target scales.

        :return:
        """
        out_scales = (
            {k: out_scales for k in self.labels}
            if isinstance(out_scales, Scale)
            else out_scales
        )
        if out_scales == self.scales:
            return self
        return ScaleValues(
            Series(
                {
                    cast(int, k): self.scales[k].transform(v, out_scales[k])
                    for k, v in self.data.items()
                }
            ),
            out_scales,
        )

    def normalize(self) -> "ScaleValues":
        """Return normalized data.

        :return:
        """
        if all(scale == NormalScale() for scale in self.scales.values()):
            return self
        return self.transform(NormalScale())

    def sort(self, reverse: bool = False) -> "ScaleValues":
        """Return sorted data in new instance.

        Normalized data are used to determine the sorting order.

        :param reverse: if ``True``, will sort in ascending order
        :return:
        """
        normalized = self.normalize()
        copy = self.copy()
        copy.data = self.data.reindex(
            normalized.data.sort_values(ascending=reverse).index
        )
        return copy

    def copy(self) -> "ScaleValues":
        """Return a copy of the object"""
        return ScaleValues(self.data.copy(), self.scales.copy())


class Ranking(ScaleValues, IPreferenceStructure):
    """This class describes a ranking as a :class:`ScaleValues`.

    It is intended as a shorthand to create rankings.

    :param data:
    :param preference_direction:
    :raise ValueError: if `data` contains non-numeric values
    :raise KeyError:
        * if some labels are duplicated
        * if `scales` keys and `data` indexes mismatch

    :attr data: internal representation of data
    :attr scales:
        scales of the data (one per value, all equals to the same
        :class:`mcda.core.scales.QuantitativeScale` with bounds inferred from
        data)
    """

    def __init__(
        self,
        data: Series,
        preference_direction: PreferenceDirection = PreferenceDirection.MAX,
    ):
        self.preference_direction = preference_direction
        if not is_numeric_dtype(data):
            raise ValueError(f"{self.__class__} only supports numeric values")
        super().__init__(data=data, preference_direction=preference_direction)

    @property
    def preference_structure(self) -> "PreferenceStructure":
        """Convert ranking into preference structure.

        :return:

        .. note::
            The minimum number of relations representing the scores is returned
            (w.r.t transitivity of preference and indifference relations)
        """
        res: list[Relation] = []
        sorted_scores = self.sort()
        for a, b in zip(sorted_scores.labels[:-1], sorted_scores.labels[1:]):
            if sorted_scores[a] == sorted_scores[b]:
                res.append(IndifferenceRelation(a, b))
            else:
                res.append(PreferenceRelation(a, b))
        return PreferenceStructure(res)

    @classmethod
    def _from_preference_structure(
        cls, preference_structure: "PreferenceStructure"
    ) -> "Ranking":
        """Convert preference structure to ranking.

        :param preference_structure:
        :raises ValueError: if `preference_structure` is not a total pre-order
        :return:

        .. note:: returned ranking goes for 1 to n (with 1 the best rank)
        """
        if not preference_structure.is_total_preorder:
            raise ValueError("only total pre-order can be represented as Ranking")
        s = Series(1, index=preference_structure.elements)
        pref_copy = preference_structure.transitive_closure
        while len(pref_copy.elements) > 0:
            bad_alternatives = set()
            for r in PreferenceStructure(pref_copy[PreferenceRelation]):
                bad_alternatives.add(r.b)
            s[[*bad_alternatives]] += 1
            for a in set(pref_copy.elements) - bad_alternatives:
                del pref_copy[a]
        return Ranking(s, preference_direction=PreferenceDirection.MIN)

    @classmethod
    def cast_from(cls, data: "IPreferenceStructure") -> "Ranking":
        """Convert any preference structure subclass instance to current type.

        :param data: instance of a preference structure subclass
        :return:
        """
        return cls._from_preference_structure(data.preference_structure)

    def copy(self) -> "Ranking":
        """Return a copy of the object"""
        return Ranking(self.data.copy(), self.preference_direction)

    def sort(self, reverse: bool = False) -> "Ranking":
        """Return sorted data in new instance.

        :param reverse: if ``True``, will sort in ascending order
        :return:
        """
        copy = self.copy()
        copy.data = copy.data.sort_values(
            ascending=(
                reverse
                != (
                    cast(
                        QuantitativeScale, next(iter(self.scales.values()))
                    ).preference_direction
                    == PreferenceDirection.MIN
                )
            )
        )
        return copy
