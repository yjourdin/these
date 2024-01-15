from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, cast

from numpy import linspace
from mcda.core.functions import FuzzyNumber, Interval
from pandas import Series
from pandas.api.types import is_numeric_dtype


class PreferenceDirection(Enum):
    """Enumeration of MCDA preference directions."""

    MIN = "MIN"
    MAX = "MAX"

    @classmethod
    def content_message(cls) -> str:
        """Return list of items and their values.

        :return:
        """
        s = ", ".join(f"{item}: {item.value}" for item in cls)
        return "PreferenceDirection only has following values " + s


class Scale(ABC):
    """Basic abstract class for MCDA scale."""

    @abstractmethod
    def __eq__(self, other) -> bool:  # pragma: nocover
        """Test equality of objects.

        :param other:
        :return:
        """
        pass

    @abstractmethod
    def __contains__(self, x: Any) -> bool:  # pragma: nocover
        """Check if values are inside scale.

        :param x:
        :return:
        """
        pass

    @abstractmethod
    def range(self, nb: int | None = None) -> List[Any]:  # pragma: nocover
        """Return range of value from scale.

        :param nb: number of values to return
        :return:
        """
        pass

    @abstractmethod
    def transform(
        self, x: Any, target_scale: "Scale | None" = None
    ) -> Any:  # pragma: nocover
        """Transform value from this scale to target scale.

        :param x:
        :param target_scale:
        :return:
        """
        pass

    def normalize(self, x: Any) -> float:
        """Normalize value.

        :param x:
        :return: normalized value
        """
        return cast(float, self.transform(x, NormalScale()))

    def denormalize(self, x: float) -> Any:
        """Denormalize value.

        :param x: normalized value
        :return: denormalized value
        """
        return NormalScale().transform(x, self)


class NominalScale(Scale):
    """This class implements a MCDA nominal scale.

    :param labels:
    """

    def __init__(self, labels: List[Any]):
        """Constructor method"""
        Scale.__init__(self)
        self.labels = labels

    def __repr__(self) -> str:  # pragma: nocover
        """Return string representation of scale.

        :return:
        """
        return f"{self.__class__.__name__}(labels={str(self.labels)})"

    def __eq__(self, other) -> bool:
        """Test equality of nominal scales.

        Equality is defined as being the same scale types, and having
        the same set of :attr:`labels`.

        :param other:
        :return:
        """
        if type(other) is not type(self):
            return False
        return set(self.labels) == set(cast(NominalScale, other).labels)

    def __contains__(self, x: Any) -> bool:
        """Check if values are inside scale.

        :param x:
        :return:
        """
        return x in self.labels

    def range(self, nb: int | None = None) -> List[Any]:
        """Return range of value from scale.

        :param nb: number of values to return (always ignored here)
        :return:
        """
        return self.labels

    def transform(self, x: Any, target_scale: Scale | None = None) -> Any:
        """Transform value from this scale to target scale.

        :param x:
        :param target_scale:
        :return:
        :raise ValueError:
            * if value `x` is outside this scale
            * if `target_scale` is not set
        :raise TypeError:
            if `target_scale` is neither :class:`QualitativeScale` nor
            :class:`NominalScale`
        """
        if x not in self:
            raise ValueError(f"label outside scale: {x}")
        if not target_scale:
            raise ValueError("non-specified target scale")
        if target_scale == self:
            return x
        if isinstance(target_scale, NominalScale):
            return target_scale.labels[target_scale.labels.index(x)]
        raise TypeError("cannot transform from nominal to quantitative scale")


class QuantitativeScale(Scale, Interval):
    """Class for quantitative scale.

    :param dmin: min boundary of scale
    :param dmax: max boundary of scale
    :param preference_direction: scale preference direction
    :raises ValueError:
        * if `dmax` smaller than `dmin`
        * if `preference_direction` is unknown
    """

    def __init__(
        self,
        dmin: float,
        dmax: float,
        preference_direction: PreferenceDirection = PreferenceDirection.MAX,
    ):
        """Constructor method"""
        Interval.__init__(self, dmin, dmax)
        if preference_direction not in PreferenceDirection:
            raise ValueError(PreferenceDirection.content_message())
        self.preference_direction = preference_direction

    def __repr__(self) -> str:  # pragma: nocover
        """Return string representation of interval.

        :return:
        """
        return (
            f"{self.__class__.__name__}(dmin={self.dmin}, dmax={self.dmax},"
            f"preference_direction={self.preference_direction})"
        )

    def __eq__(self, other) -> bool:
        """Test equality of quantitative scales.

        Equality is defined as being the same scale types, having the same
        interval and :attr:`preference_direction`.

        :param other:
        :return:
        """
        if type(other) is not type(self):
            return False
        _scale = cast(QuantitativeScale, other)
        if self.preference_direction != _scale.preference_direction:
            return False
        return Interval.__eq__(self, _scale)

    def __contains__(self, x: Any) -> bool:
        """Check if values are inside scale.

        :param x:
        :return:
        """
        return self.inside(cast(float, x))

    def range(self, nb: int | None = None) -> List[Any]:
        """Return range of value from scale.

        :param nb: number of values to return
        :return:
        """
        nb = nb or 2
        return cast(List[Any], linspace(self.dmin, self.dmax, nb).tolist())

    def _normalize_value(self, x: float) -> float:
        """Normalize numeric value.

        :param x:
        :return:

        .. note::
            `preference_direction` is taken into account, so preferred
            value is always bigger.
        """
        if self.preference_direction == PreferenceDirection.MIN:
            return 1 - Interval.normalize(self, x)
        return Interval.normalize(self, x)

    def _denormalize_value(self, x: float) -> float:
        """Denormalize normalized numeric value.

        :param x:
        :return:

        .. note::
            `preference_direction` is taken into account, so preferred
            normalized value must always be bigger.
        """
        if self.preference_direction == PreferenceDirection.MIN:
            return cast(float, Interval.denormalize(self, 1 - x))
        return cast(float, Interval.denormalize(self, x))

    def transform(self, x: Any, target_scale: Scale | None = None) -> Any:
        """Transform value from this scale to target scale.

        :param x:
        :param target_scale:
        :return:
        :raise ValueError:
            * if value `x` is outside this scale
            * if `target_scale` is not set
        :raise TypeError:
            if target_scale is neither :class:`QuantitativeScale` nor
            :class:`QualitativeScale`

        .. note:: `preference_direction` attributes are taken into account
        """
        if x not in self:
            raise ValueError(f"value outside scale: {x}")
        if not target_scale:
            raise ValueError("non-specified target scale")
        if target_scale == self:
            return x
        _x = cast(float, x)
        match target_scale:
            case QualitativeScale():
                return target_scale.label_from_value(
                    target_scale._denormalize_value(self._normalize_value(_x))
                )
            case QuantitativeScale():
                return target_scale._denormalize_value(self._normalize_value(_x))
        raise TypeError("cannot transform from quantitative to nominal scale")

    def normalize(self, x: Any) -> float:
        """Normalize value.

        :param x:
        :return: normalized value

        .. note:: `preference_direction` attributes is taken into account
        """
        return Scale.normalize(self, x)

    def denormalize(self, x: float) -> Any:
        """Denormalize value.

        :param x: normalized value
        :return: denormalized value

        .. note:: `preference_direction` attributes is taken into account
        """
        return Scale.denormalize(self, x)

    def is_better(self, x: Any, y: Any) -> bool:
        """Check if x is better than y according to this scale.

        :param x:
        :param y:
        :return:
        """
        _x, _y = cast(float, x), cast(float, y)
        return (
            _x > _y if self.preference_direction == PreferenceDirection.MAX else _x < _y
        )


class NormalScale(QuantitativeScale):
    def __init__(self):
        super().__init__(0, 1)

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__)

    def _normalize_value(self, x: float) -> float:
        return x

    def _denormalize_value(self, x: float) -> float:
        return x


class QualitativeScale(QuantitativeScale, NominalScale):
    """This class implements a MCDA qualitative scale.

    :param values: numeric series with labels as index
    :param preference_direction: scale preference direction
    :param dmin: min boundary of scale (inferred from `values` if not set)
    :param dmax: max boundary of scale (inferred from `values` if not set)
    :raises ValueError:
        * if at least one value is outside the bounds
    :raises TypeError:
        * if `values` contains non-numeric values

    .. warning::
        This scale contains `labels` not `values`. `values` are only here to
        define a corresponding quantitative scale for default scale
        transformation. After calling :meth:`transform_to` with no associated
        scale, the data is no longer considered inside the qualitative scale.

    .. todo::
        this scale is quantitative but has a quantitative scale as an
        attribute. Find out how to best type cast it.
    """

    def __init__(
        self,
        values: Series,
        preference_direction: PreferenceDirection = PreferenceDirection.MAX,
        dmin: float | None = None,
        dmax: float | None = None,
    ):
        """Constructor method"""
        values = Series(values)
        if not is_numeric_dtype(values):
            raise TypeError("QualitativeScale must have numeric values")
        labels = values.index.tolist()
        NominalScale.__init__(self, labels)
        dmin = dmin or values.min()
        dmax = dmax or values.max()
        QuantitativeScale.__init__(
            self,
            dmin,
            dmax,
            preference_direction,
        )
        self.__quantitative = QuantitativeScale(
            dmin,
            dmax,
            preference_direction,
        )
        out_of_bounds = Series(
            {str(k): v not in self.__quantitative for k, v in values.items()}
        )
        if out_of_bounds.any():
            raise ValueError(
                f"values '{values[out_of_bounds]}' are outside the defined "
                f"bounds [{dmin}, {dmax}]"
            )
        self.values = values

    @property
    def quantitative(
        self,
    ) -> QuantitativeScale:  # Property used to hide setter
        """Quantitative scale extracted from current qualitative scale."""
        return self.__quantitative

    def __repr__(self) -> str:  # pragma: nocover
        """Return string representation of interval.

        :return:
        """
        return (
            f"{self.__class__.__name__}(values={self.values},"
            f"preference_direction={self.preference_direction},"
            f"dmin={self.dmin}, dmax={self.dmax})"
        )

    def __eq__(self, other) -> bool:
        """Test equality of qualitative scales.

        Equality is defined as having the same types, having the same set of
        :attr`labels` and corresponding :attr:`values`, and having the same
        interval and :attr:`preference_direction`.

        :param other:
        :return:
        """
        if type(other) is not type(self):
            return False
        _scale = cast(QualitativeScale, other)
        if not QuantitativeScale.__eq__(self, _scale):
            return False
        if not NominalScale.__eq__(self, _scale):
            return False
        if (self.values == other.values[self.labels]).all():
            return True
        return False

    def __contains__(self, x: Any) -> bool:
        """Check if label is inside scale.

        :param x:
        :return:
        """
        return x in self.values

    def range(self, nb: int | None = None) -> List[Any]:
        """Return range of value from scale.

        :param nb: number of values to return (always ignored here)
        :return:
        """
        return NominalScale.range(self, nb)

    def _denormalize_value(self, x: float) -> float:
        denormalized_x = cast(
            float,
            NormalScale().transform(x, self.quantitative),
        )
        closest_prefered_value = (
            min([value for value in self.values if value >= denormalized_x])
            if self.preference_direction == PreferenceDirection.MAX
            else max([value for value in self.values if value <= denormalized_x])
        )
        return closest_prefered_value

    def transform(self, x: Any, target_scale: Scale | None = None) -> Any:
        """Transform value from this scale to target scale.

        :param x:
        :param target_scale:
        :return:
        :raise ValueError: if value `x` is outside this scale
        :raise TypeError: if `target_scale` has unknown type

        .. note::
            `preference_direction` attributes are taken into account when
            rescaling to a :class:`QuantitativeScale`
        """
        if x not in self:
            raise ValueError(f"label outside scale: {x}")
        if target_scale == self:
            return x
        target_scale = target_scale or self.quantitative
        match target_scale:
            case NominalScale():
                return target_scale.labels[target_scale.labels.index(x)]
            case QuantitativeScale():
                value = cast(float, self.values[x])
                return target_scale._denormalize_value(self._normalize_value(value))
            case _:
                raise TypeError(
                    f"unrecognized scale type for scale: {target_scale}"
                )  # pragma: nocover

    def label_from_value(self, x: Any) -> Any:
        """Transform value to this scale.

        :param x:
        :raises ValueError: if `x` corresponds to no label
        :return: label associated to given value
        """
        if x not in self.values.values:
            raise ValueError(f"value outside scale: {x}")
        return self.values[self.values == x].index[0]

    def is_better(self, x: Any, y: Any) -> bool:
        """Check if x is better than y according to this scale.

        :param x:
        :param y:
        :return:
        """
        return self.quantitative.is_better(self.values[x], self.values[y])


class FuzzyScale(QualitativeScale):
    """This class implements a MCDA fuzzy qualitative scale.

    :param labels:
    :param fuzzy:
    :param preference_direction: scale preference direction
    :param dmin: min boundary of scale (inferred from `values` if not set)
    :param dmax: max boundary of scale (inferred from `values` if not set)
    :raises ValueError:
        * if number of `labels` and `fuzzy` differs
        * if `preference_direction` is unknown
        * if at least one fuzzy number is outside the bounds
    :raises TypeError:
        * if `fuzzy` contains non-fuzzy numbers
    """

    def __init__(
        self,
        fuzzy: Series,
        preference_direction: PreferenceDirection = PreferenceDirection.MAX,
        dmin: float | None = None,
        dmax: float | None = None,
        defuzzify_method: str = "centre_of_gravity",
    ):
        fuzzy = Series(fuzzy)
        for fz in fuzzy.values:
            if type(fz) is not FuzzyNumber:
                raise TypeError("fuzzy scales can only contains fuzzy numbers")
        dmin = dmin or min(fz.abscissa[0] for fz in fuzzy)
        dmax = dmax or max(fz.abscissa[-1] for fz in fuzzy)
        for fz in fuzzy:
            if fz.abscissa[0] < dmin or fz.abscissa[-1] > dmax:
                raise ValueError(
                    "fuzzy number sets must be within the defined bounds "
                    f"[{dmin}, {dmax}]"
                )
        self.fuzzy = fuzzy
        self.defuzzify_method = defuzzify_method
        QualitativeScale.__init__(
            self,
            self.defuzzify(),
            preference_direction,
            dmin,
            dmax,
        )

    def __eq__(self, other) -> bool:
        """Test equality of fuzzy scales.

        Equality is defined as having the same types, having the same set of
        :attr`labels` and corresponding :attr:`fuzzy`, and having the same
        interval and :attr:`preference_direction`.

        :param other:
        :return:
        """
        if type(other) is not type(self):
            return False
        _scale = cast(FuzzyScale, other)
        if not QualitativeScale.__eq__(self, _scale):
            return False
        for k, f in zip(self.labels, self.fuzzy):
            if f != _scale.fuzzy[k]:
                return False
        return True

    def defuzzify(self, method: str | None = None) -> Series:
        """Defuzzify all fuzzy numbers using given method.

        :param method:
            method used to defuzzify
            (from :class:`mcda.core.functions.FuzzyNumber` numeric methods)
        """
        method = method or self.defuzzify_method
        return self.fuzzy.apply(lambda x, m=method: getattr(x, m))

    def is_fuzzy_partition(self) -> bool:
        """Test whether the scale define a fuzzy partition.

        :return:
        """
        indexes = self.values.sort_values().index
        fuzzy_sets = [self.fuzzy[i] for i in indexes]
        for i in range(len(fuzzy_sets) - 1):
            for j in range(2):
                if fuzzy_sets[i].abscissa[j + 2] != fuzzy_sets[i + 1].abscissa[j]:
                    return False
        return True

    def similarity(self, fuzzy1: FuzzyNumber, fuzzy2: FuzzyNumber) -> float:
        """Returns similarity between both fuzzy numbers w.r.t this scale.

        :param fuzzy1:
        :param fuzzy2:
        :return:

        .. note:: implementation based on :cite:p:`isern2010ulowa`
        """
        a = [self.quantitative.normalize(v) for v in fuzzy1.abscissa]
        b = [self.quantitative.normalize(v) for v in fuzzy2.abscissa]
        res = [2 - abs(aa - bb) for aa, bb in zip(a, b)]
        prod = 1.0
        for r in res:
            prod *= r
        return prod ** (1 / 4) - 1

    def fuzziness(self, fuzzy: FuzzyNumber) -> float:
        """Returns the fuzziness of given fuzzy number w.r.t this scale.

        :param fuzzy:
        :return:
        """
        return self.quantitative.normalize(
            (
                fuzzy.abscissa[1]
                + fuzzy.abscissa[3]
                - fuzzy.abscissa[0]
                - fuzzy.abscissa[2]
            )
            / 2
        )

    def specificity(self, fuzzy: FuzzyNumber) -> float:
        """Returns the specificity of given fuzzy number w.r.t this scale.

        :param fuzzy:
        :return:
        """
        return 1 - self.quantitative.normalize(fuzzy.area)

    def ordinal_distance(self, a: Any, b: Any) -> float:
        """Returns the ordinal distance between the labels
        (sorted by defuzzified values).

        :param a:
        :param b:
        :return:
        :raises ValueError: if `a` or `b` is not inside the scale
        """
        if a not in self or b not in self:
            raise ValueError("both labels must be inside the fuzzy scale")
        labels = sorted(self.labels, key=lambda v: self.transform(v))
        return abs(labels.index(a) - labels.index(b))


def is_better(x: Any, y: Any, scale: QuantitativeScale) -> bool:
    """Check if x is better than y according to this scale

    :param x:
    :param y:
    :param scale:
    :return:
    """
    return scale.is_better(x, y)


def is_better_or_equal(x: Any, y: Any, scale: QuantitativeScale) -> bool:
    """Check if x is better or equal to y according to this scale

    :param x:
    :param y:
    :param scale:
    :return:
    """
    return x == y or is_better(x, y, scale)
