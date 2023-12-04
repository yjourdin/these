"""This module contains all functions related to matrices.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Callable

from graphviz import Digraph
from mcda.core.set_functions import HashableSet
from pandas import DataFrame
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, floyd_warshall


def dataframe_equals(df1: DataFrame, df2: DataFrame) -> bool:
    """Check if two dataframes have the same values.

    It will realign the indexes and columns if they are ordered differently.

    :param df1:
    :param df2:
    :return:

    .. todo:: integrate into :class:`mcda/core.adjacency_matrix.Matrix`
    """
    return df1.to_dict() == df2.to_dict()


class Matrix:
    """This class implements a wrapper on :class:`pandas.DataFrame`.

    It adds a method to check if two such objects are equals.
    It is meant to be use for any class that needs a DataFrame as its
    internal data representation in this package.

    :param data: dataframe containing the matrix
    :raise KeyError:
        * if some indexes are duplicated
        * if some columns are duplicated
    """

    def __init__(self, data):
        self.data = DataFrame(data)
        if self.data.index.has_duplicates:
            raise KeyError(
                "some indexes are duplicated: "
                f"{self.data.index[self.data.index.duplicated()].tolist()}"
            )
        if self.data.columns.has_duplicates:
            raise KeyError(
                "some columns are duplicated: "
                f"{self.data.columns[self.data.columns.duplicated()].tolist()}"
            )

    def __mul__(self, other: "Matrix | float") -> "Matrix":
        """Return product.

        :param other:
        :return:
        """
        coeff = other.data if isinstance(other, Matrix) else other
        return self.__class__(self.data * coeff)

    def __add__(self, other: Any) -> "Matrix":
        """Return addition.

        :param other:
        :return:
        """
        added = other.data if isinstance(other, Matrix) else other
        return self.__class__(self.data + added)

    def __eq__(self, other) -> bool:
        """Check if both matrices have the same dataframe

        :return:

        .. note:: vertices order does not matter
        """
        if type(other) is not type(self):
            return False
        return dataframe_equals(self.data, other.data)


class AdjacencyMatrix(Matrix):
    """This class implements graphs as an adjacency matrix.

    The adjacency matrix is represented internally by a
    :class:`pandas.DataFrame` with vertices as the indexes and columns.

    :param data: adjacency matrix in an array-like or dict-structure
    :param vertices:

    :raise ValueError: if columns and rows have different sets of labels
    :raise KeyError:
        * if some indexes are duplicated
        * if some columns are duplicated

    .. note:: the cells of the matrix can be of any type (not just numerics)
    """

    def __init__(self, data, vertices: list | None = None):
        df = DataFrame(
            data.values if isinstance(data, DataFrame) and vertices else data,
            index=vertices,
            columns=vertices,
        )
        if df.columns.tolist() != df.index.tolist():
            raise ValueError(
                f"{self.__class__} supports only same labelled" "index and columns"
            )

        super().__init__(df)

    @property
    def vertices(self) -> list:
        """Return list of vertices"""
        return self.data.index.tolist()

    def plot(
        self,
        edge_label: bool = False,
        self_loop: bool = False,
        cut: float | Callable[[Any], bool] = -float("inf"),
    ) -> Digraph:
        """Create a graph for adjacency matrix.

        This function creates a Graph using graphviz and display it.

        :param edge_label: (optional) parameter to display the value of edges
        :param self_loop: (optional) parameter to display self looping edges
        :param cut:
            either a numeric threshold under which edges are pruned, or a
            filtering function taking one cell and returning if edge must be
            pruned (boolean). Default cuts no edge.
        """
        graph = Digraph("graph", strict=True)
        graph.attr("node", shape="box")

        for v in self.vertices:
            graph.node(str(v))
        for a in self.data.index:
            for b in self.data.columns:
                if not self_loop and a == b:
                    continue
                elif self.data.at[a, b] == 0:
                    continue
                if isinstance(cut, float) and self.data.at[a, b] <= cut:
                    continue
                elif callable(cut) and cut(self.data.at[a, b]):
                    continue

                graph.edge(
                    str(a),
                    str(b),
                    label=str(self.data.at[a, b]) if edge_label else "",
                )
        graph.render()
        return graph


class BinaryAdjacencyMatrix(AdjacencyMatrix):
    """This class implements graphs as a binary adjacency matrix.

    The adjacency matrix is represented internally by a
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

    def __init__(self, data, vertices: list | None = None):
        super().__init__(data, vertices)
        # if ((self.data != 1) & (self.data != 0)).any(axis=None):
        #     raise ValueError("AdjacencyMatrix objects must contain binary values")

    @property
    def transitive_closure(self) -> "BinaryAdjacencyMatrix":
        """Return transitive closure of matrix"""
        _m = floyd_warshall(csr_matrix(self.data.to_numpy())) < float("inf")
        return self.__class__(_m.astype(int))
        # m = DataFrame(
        #     _m,
        #     index=self.vertices,
        #     columns=self.vertices,
        # )
        # res = DataFrame(
        #     0,
        #     index=self.vertices,
        #     columns=self.vertices,
        # )
        # res[m] = 1
        # return self.__class__(res)

    @property
    def transitive_reduction(self) -> "BinaryAdjacencyMatrix":
        """Return transitive reduction of matrix.

        .. note:: this function can change the matrix shape
        """
        matrix = self.graph_condensation
        path_matrix = floyd_warshall(csr_matrix(matrix.data.to_numpy())) == 1
        nodes = range(len(matrix.data))
        for u in nodes:
            for v in nodes:
                if path_matrix[u][v]:
                    for w in nodes:
                        if path_matrix[v][w]:
                            matrix.data.iloc[u, w] = 0
        return matrix

    @property
    def graph_condensation(self) -> "BinaryAdjacencyMatrix":
        """Return the condensation graph

        .. note:: the matrix output by this function is acyclic

        .. warning:: this function changes the matrix shape
        """

        n_components, labels = connected_components(
            self.data.to_numpy(), connection="strong"
        )
        # Return input matrix if no cycle found
        if n_components == len(self.data):
            return self.__class__(self.data)
        # Create new matrix with appropriate names for components
        components = []
        for component_index in range(n_components):
            component = HashableSet(self.data.index[labels == component_index].tolist())
            components.append(component)
        new_matrix = DataFrame(0, index=components, columns=components)
        for component_a, component_b in product(
            range(n_components), range(n_components)
        ):
            if component_a != component_b:
                new_matrix.iloc[component_a, component_b] = (
                    self.data.iloc[labels == component_a, labels == component_b]
                    .to_numpy()  # type: ignore
                    .any()
                )

        return self.__class__(new_matrix.astype(int))

    @property
    def cycle_reduction_matrix(self) -> "BinaryAdjacencyMatrix":
        """Return matrix with cycles removed."""
        n_components, labels = connected_components(
            self.data.to_numpy(), connection="strong"
        )
        components = range(n_components)
        new_matrix = DataFrame(0, index=self.vertices, columns=self.vertices)
        for component_a, component_b in product(components, components):
            if component_a != component_b:
                new_matrix.loc[
                    labels == component_a, labels == component_b
                ] = (  # type: ignore
                    self.data.loc[labels == component_a, labels == component_b]
                    .to_numpy()
                    .any()
                )
        return self.__class__(new_matrix.astype(int))

    @property
    def kernel(self) -> list:
        """Return the kernel of the graph if existing.

        The kernel is a *stable* and *dominant* set of nodes.
        Dominant nodes are the origin of edges, dominated ones are the target.

        :return: the kernel (if existing), else an empty list
        """
        graph = self.data.copy()
        # We remove self loops
        for v in self.vertices:
            graph.at[v, v] = 0
        kernel: set = set()
        outsiders: set = set()
        while not graph.empty:
            domination = (graph == 0).all(axis=0)
            dominators = domination[domination].index.tolist()
            if len(dominators) == 0:
                return []

            dominated = (graph == 1).loc[dominators].any(axis=0)
            neighbours = dominated[dominated].index.tolist()

            to_remove = dominators + neighbours
            graph = graph.drop(index=to_remove, columns=to_remove)
            kernel = kernel.union(dominators)
            outsiders = outsiders.union(neighbours)
        return list(kernel)
