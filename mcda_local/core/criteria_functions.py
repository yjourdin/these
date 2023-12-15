from typing import Any, cast

from mcda.core.aliases import Function
from mcda.plot.plot import Figure, LinePlot
from pandas import DataFrame, Series

from .performance_table import PerformanceTable
from .scales import Scale
from .values import ScaleValues


class CriteriaFunctions:
    """This class represents a multi-attribute criteria functions.

    :param criteria_functions:
    :param in_scales:
        scales used for inputs (if not provided, it will be inferred from
        inputs)
    :param out_scales:
        scales used for outputs (if not provided, it will be inferred from
        outputs)

    .. note::
        `criteria_functions` are defined for the provided scales, so when
        applying criteria functions you must provide correctly scaled values
        (only :class:`mcda.api.core/performance_table.PerformanceTable`
        and :class:`mcda.api.core.performance_table.ScaleValues` are
        automatically transformed)
    """

    def __init__(
        self,
        criteria_functions: dict[Any, Function],
        in_scales: dict[Any, Scale] | None = None,
        out_scales: dict[Any, Scale] | None = None,
        **kwargs,
    ):
        self.criteria_functions = criteria_functions
        self.in_scales = in_scales
        self.out_scales = out_scales
        super().__init__(**kwargs)

    def _apply_series(self, series: Series, *args, **kwargs) -> Series:
        """Apply criteria functions to a pandas Series.

        :param series:
        """
        return Series(
            {
                criterion: self.criteria_functions[criterion](value)
                for criterion, value in dict(series).items()
            }
        )

    def _apply_scale_values(
        self, scale_values: ScaleValues, *args, **kwargs
    ) -> ScaleValues:
        """Apply criteria functions to a scale values object.

        :param scale_values:
        :return: criteria values

        .. note::
            `scale_values` values are transformed to value functions
            :attr:`in_scales`
        """
        _values = (
            scale_values.transform(cast(dict[Any, Scale], self.in_scales))
            if self.in_scales
            else scale_values
        )
        return ScaleValues(
            self._apply_series(_values.data),
            self.out_scales,
        )

    def _apply_dataframe(self, df: DataFrame, *args, **kwargs) -> DataFrame:
        """Apply criteria functions to a pandas DataFrame.

        :param df:
        :return:
        """
        return cast(
            DataFrame,
            df.apply(lambda col: col.apply(self.criteria_functions.get(col.name))),
        )

    def _apply_performance_table(
        self,
        performance_table: PerformanceTable,
        *args,
        **kwargs,
    ) -> PerformanceTable:
        """Apply criteria functions to a performance table object.

        :param performance_table:
        :return: result in new performance table

        .. note::
            `performance_table` values are transformed to criteria functions
            :attr:`in_scales`
        """
        _table = (
            performance_table.transform(cast(dict[Any, Scale], self.in_scales))
            if self.in_scales
            else performance_table
        )
        return PerformanceTable(
            self._apply_dataframe(_table.data),
            scales=self.out_scales,
        )

    def __call__(
        self,
        data: PerformanceTable | DataFrame | Series | ScaleValues,
        *args,
        **kwargs,
    ) -> Any:
        """Apply criteria functions to input data.

        :param data:
        :return:
        """
        match data:
            case PerformanceTable():
                return self._apply_performance_table(data, *args, **kwargs)
            case DataFrame():
                return self._apply_dataframe(data, *args, **kwargs)
            case ScaleValues():
                return self._apply_scale_values(data, *args, **kwargs)
            case _:
                return self._apply_series(data, *args, **kwargs)

    def plot(self, nb_points=500) -> Figure:  # pragma: nocover
        """Plot each criterion function on its respective scale.

        :return: created figure
        :raise ValueError: if :attr:`in_scales` is not set
        """
        if not self.in_scales:
            raise ValueError("need 'in_scales' attribute to generate samples to plot")
        fig = Figure(ncols=2)
        for c, scale in self.in_scales.items():
            x = cast(list[float], scale.range(nb_points))
            y = [self.criteria_functions[c](xx) for xx in x]
            ax = fig.create_add_axis()
            ax.title = c
            ax.add_plot(LinePlot(x, y))
        fig.draw()
        return fig


class CriteriaWeights(CriteriaFunctions):
    """This class represents a multi-attribute criteria functions which applies
    weights to each criterion values.

    :param criteria_weights:
    :param in_scales:
        scales used for inputs (if not provided, it will be inferred from
        inputs)
    :param out_scales:
        scales used for outputs (if not provided, it will be inferred from
        outputs)

    .. note::
        `criteria_weights` are defined for the provided scales, so when
        applying criteria weights you must provide correctly scaled values
        (only :class:`mcda.api.core/performance_table.PerformanceTable`
        and :class:`mcda.api.core.performance_table.ScaleValues` are
        automatically transformed)
    """

    def __init__(
        self,
        criteria_weights: dict[Any, float],
        in_scales: dict[Any, Scale] | None = None,
        out_scales: dict[Any, Scale] | None = None,
        **kwargs,
    ):
        self.criteria_weights = criteria_weights
        criteria_functions: dict[Any, Function] = {
            c: cast(Function, lambda x, w=weight: w * x)
            for c, weight in criteria_weights.items()
        }
        super().__init__(
            criteria_functions,
            in_scales=in_scales,
            out_scales=out_scales,
            **kwargs,
        )
