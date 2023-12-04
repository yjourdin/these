from typing import TYPE_CHECKING, Any, Literal, cast

from mcda.core.aliases import Function
from pandas import DataFrame, Series, concat
from pandas.api.types import is_numeric_dtype

from .matrices import Matrix
from .scales import NormalScale, Scale
from .values import ScaleValues

if TYPE_CHECKING:
    from .criteria_functions import CriteriaFunctions


class PerformanceTable(Matrix):
    """This class is used to represent performance tables.

    :param data: performance table in an array-like or dict structure
    :param scales: criteria scales (scales are inferred from data if not set)
    :param alternatives:
    :param criteria:
    :raise KeyError:
        * if some alternatives are duplicated
        * if some criteria are duplicated
        * if `scales` keys and `data` columns mismatch

    :attr data: dataframe containing the performances
    :attr scales: criteria scales

    .. note::
        when applying pandas methods to modify the performance table, do it
        this way: `table.data = table.data.method()` (for a method called
        `method`)

        Also you may want to modify the criteria scales depending on such
        modifications.
    """

    def __init__(
        self,
        data,
        scales: dict[Any, Scale] | None = None,
        alternatives: list[Any] | None = None,
        criteria: list[Any] | None = None,
    ):
        df = DataFrame(data, index=alternatives, columns=criteria)
        if df.index.has_duplicates:
            raise KeyError(
                "some alternatives are duplicated: "
                f"{df.index[df.index.duplicated()].tolist()}"
            )
        if df.columns.has_duplicates:
            raise KeyError(
                "some criteria are duplicated: "
                f"{df.columns[df.columns.duplicated()].tolist()}"
            )
        super().__init__(df)
        self.scales = scales or self.bounds
        if set(self.criteria) != set(self.scales.keys()):
            raise KeyError("data and scales must have the same criteria")

    def __eq__(self, other) -> bool:
        """Check equality of performance tables.

        Equality is defines as having the same set of scales, and having the
        same dataframe.

        :return: ``True`` if both are equal
        """
        if not isinstance(other, PerformanceTable):
            return False
        _table = cast(PerformanceTable, other)
        if self.scales == _table.scales:
            return super().__eq__(_table)
        return False

    @property
    def criteria(self) -> list[Any]:
        """Return performance table criteria"""
        return self.data.columns.tolist()

    @property
    def alternatives(self) -> list[Any]:
        """Return performance table alternatives"""
        return self.data.index.tolist()

    @property
    def alternatives_values(self) -> dict[Any, ScaleValues]:
        """Iterator on the table alternatives values"""
        return {
            a: ScaleValues(self.data.loc[a], self.scales) for a in self.alternatives
        }

    @property
    def criteria_values(self) -> dict[Any, ScaleValues]:
        """Iterator on the table criteria values"""
        return {c: ScaleValues(self.data[c], self.scales[c]) for c in self.criteria}

    @property
    def is_numeric(self) -> bool:
        """Check whether performance table is numeric.

        :return:
        :rtype: bool
        """
        for col in self.data.columns:
            if not is_numeric_dtype(self.data[col]):
                return False
        return True

    @property
    def bounds(self) -> dict[Any, Scale]:
        """Return criteria scales inferred from performance table values.

        .. note::
            will always assume maximizable quantitative scales for numeric
            criteria and nominal scales for others
        """
        return {
            criterion: ScaleValues(self.data[criterion]).bounds
            for criterion in self.criteria
        }

    @property
    def efficients(self) -> list:
        """Return efficient alternatives.

        This is the list of alternatives that are not strongly dominated by
        another one.

        :return:
        """
        res = set(self.alternatives)
        for avalues in self.alternatives_values.values():
            dominated = set()
            for b in res:
                if avalues.name == b:
                    continue
                if avalues.dominate_strongly(self.alternatives_values[b]):
                    dominated.add(b)
            res -= dominated
        return sorted(res, key=lambda a: self.alternatives.index(a))

    def _apply_criteria_functions(
        self, functions: dict[Any, Function]
    ) -> "PerformanceTable":
        """Apply criteria functions to performance table and return result.

        :param functions: functions identified by their criterion
        :return:
        """
        return PerformanceTable(
            self.data.apply(
                lambda col: col.apply(functions.get(col.name, lambda x: x))
            ),
            self.scales,
        )

    def apply(self, functions: "CriteriaFunctions") -> "PerformanceTable":
        """Apply criteria functions to performance table and return result.

        :param functions:
        :return:
        """
        return functions(self)

    @property
    def within_criteria_scales(self) -> "PerformanceTable":
        """Return a table indicating which performances are within their
        respective criterion scale.

        :return:
        """
        return self._apply_criteria_functions(
            {
                criterion: cast(Function, lambda x, c=criterion: x in self.scales[c])
                for criterion in self.scales.keys()
            },
        )

    @property
    def is_within_criteria_scales(self) -> bool:
        """Check whether all cells are within their respective criteria scales.

        :return:
        """
        return self.within_criteria_scales.data.all(None)

    def transform(
        self,
        out_scales: dict[Any, Scale],
    ) -> "PerformanceTable":
        """Transform performances table between scales.

        :param out_scales: target criteria scales
        :return: transformed performance table
        """
        if out_scales == self.scales:
            return self
        functions = {
            criterion: (
                cast(
                    Function,
                    lambda x, c=criterion: self.scales[c].transform(x, out_scales[c]),
                )
            )  # https://bugs.python.org/issue13652
            for criterion in self.scales.keys()
        }
        return PerformanceTable(
            self._apply_criteria_functions(functions).data, out_scales
        )

    def normalize_without_scales(self) -> "PerformanceTable":
        """Normalize performance table using criteria values bounds.

        :return:
        :raise TypeError: if performance table is not numeric
        """
        return PerformanceTable(self.data).normalize()

    def normalize(self) -> "PerformanceTable":
        """Normalize performance table using criteria scales.

        :return:
        """
        if all(scale == NormalScale() for scale in self.scales.values()):
            return self
        return self.transform({criterion: NormalScale() for criterion in self.criteria})

    def sum(self, axis: Literal[0] | Literal[1] | None = None) -> Series | float:
        """Sum performances.

        Behaviour depends on `axis` value:

        * ``0``: returns column sums as a list
        * ``1``: returns row sums as a list
        * else: returns sum on both dimension as a numeric value

        :param axis: axis on which the sum is made
        :return:

        .. note::
            Non-numeric values are simply ignored as well as non-numeric sums
        """
        if axis:
            return self.data.sum(axis=axis, numeric_only=True)
        return self.data.sum(numeric_only=True).sum()

    def subtable(
        self, alternatives: list[Any] | None = None, criteria: list[Any] | None = None
    ) -> "PerformanceTable":
        """Return the subtable containing given alternatives and criteria.

        :param alternatives:
        :param criteria:
        :return:
        """
        alternatives = alternatives or self.alternatives
        criteria = criteria or self.criteria
        return self.__class__(
            self.data.loc[alternatives, criteria],
            {criterion: self.scales[criterion] for criterion in criteria},
        )

    @classmethod
    def concat(
        cls,
        performance_tables: list["PerformanceTable"],
        scales: dict[Any, Scale] | None = None,
        axis: Literal[0] | Literal[1] = 0,
    ) -> "PerformanceTable":
        """Concatenate multiple performance tables.

        :param performance_tables:
        :param scales:
            scales used in concatenated result (taken from `performance_tables`
            scales if not set, with first occurence taken in case of duplicates
            )
        :param axis:
            axis along which to concatenate
            (0: add alternatives, 1: add criteria)
        :return: concatenated performance table

        .. warning::
            `performance_tables` objects are concatenated as is, no
            transformation of scales is applied.
        """
        scales = scales or {}
        dataframes: list[DataFrame] = []
        for t in performance_tables:
            dataframes.append(t.data)
            for c, s in t.scales.items():
                scales.setdefault(c, s)
        df = concat(dataframes, axis=axis)
        return cls(df, scales=scales)


class NormalPerformanceTable(PerformanceTable):
    def __init__(
        self,
        data,
        alternatives: list[Any] | None = None,
        criteria: list[Any] | None = None,
    ):
        df = DataFrame(data, index=alternatives, columns=criteria)
        if df.index.has_duplicates:
            raise KeyError(
                "some alternatives are duplicated: "
                f"{df.index[df.index.duplicated()].tolist()}"
            )
        if df.columns.has_duplicates:
            raise KeyError(
                "some criteria are duplicated: "
                f"{df.columns[df.columns.duplicated()].tolist()}"
            )
        Matrix.__init__(self, df)
        self.scales = {crit: NormalScale() for crit in df.columns}
        if set(self.criteria) != set(self.scales.keys()):
            raise KeyError("data and scales must have the same criteria")

    @property
    def is_numeric(self) -> bool:
        return True

    def normalize(self) -> PerformanceTable:
        return self
