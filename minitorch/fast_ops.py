from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Converts the function passed in, along with any arguments given, into a jit function"""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# create jit'ed versions of the previous tensor functions
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(
                1, a.shape[0], a.shape[1]
            )  # make continuous in memory and reshape dims to be 1, a.shape[0], a.shape[1]
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        x = np.array_equal(in_strides, out_strides) and np.array_equal(
            in_shape, out_shape
        )
        for i in prange(len(out)):
            out_index: Index = np.empty(MAX_DIMS, np.int32)
            in_index: Index = np.empty(
                MAX_DIMS, np.int32
            )  # these are the numpy buffers
            if x:  # strides and shape aligned
                out[i] = fn(in_storage[i])  # save value
            else:
                to_index(
                    i, out_shape, out_index
                )  # get index of the i of the thread we're in (prange makes as many threads as indicated by the param)
                broadcast_index(
                    out_index, out_shape, in_shape, in_index
                )  # convert the index in the out shape into the equivalent index in the in shape
                o = index_to_position(
                    out_index, out_strides
                )  # get the storage indices in the respective tensors
                j = index_to_position(in_index, in_strides)
                out[o] = fn(in_storage[j])  # save value

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        x = (
            np.array_equal(a_strides, b_strides)
            and np.array_equal(a_strides, out_strides)
            and np.array_equal(b_strides, out_strides)
            and np.array_equal(a_shape, b_shape)
            and np.array_equal(a_shape, out_shape)
            and np.array_equal(b_shape, out_shape)
        )
        for i in prange(len(out)):
            out_index: Index = np.empty(MAX_DIMS, np.int32)
            a_index: Index = np.empty(MAX_DIMS, np.int32)
            b_index: Index = np.empty(MAX_DIMS, np.int32)
            if x:
                out[i] = fn(
                    a_storage[i], b_storage[i]
                )  # if aligned, directly fill in out at i
            else:
                to_index(i, out_shape, out_index)
                o = index_to_position(out_index, out_strides)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                j = index_to_position(a_index, a_strides)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                k = index_to_position(b_index, b_strides)
                out[o] = fn(a_storage[j], b_storage[k])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
        reduce_size = a_shape[reduce_dim]
        for i in prange(len(out)):
            out_index: Index = np.empty(MAX_DIMS, np.int32)
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(out_index, a_strides)
            for s in range(reduce_size):
                out_index[reduce_dim] = s
                out[o] = fn(out[o], a_storage[j])
                j += a_strides[reduce_dim]

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = (
        a_strides[0] if a_shape[0] > 1 else 0
    )  # if the tensor was already a 3d tensor before view adjustment
    b_batch_stride = (
        b_strides[0] if b_shape[0] > 1 else 0
    )  # the number to get to the next depth in storage
    # TODO: Implement for Task 3.2.
    for out_pos in prange(len(out)):
        batch_idx = (out_pos // out_strides[0]) % out_shape[0]  # this is from chatgpt
        row_idx = (out_pos // out_strides[1]) % out_shape[1]
        col_idx = (out_pos // out_strides[2]) % out_shape[2]
        # ..., depth, row, col
        dot_prod = 0  # accumulator
        for k in range(a_shape[-1]):  # Shared dimension (the col of A and the row of B)
            a_idx = (
                batch_idx * a_batch_stride + row_idx * a_strides[-2] + k * a_strides[-1]
            )
            # needs the current depth * the number of units to move to get there
            # the row that the current index is in * the stride to get to the next layer
            # the column/row and the strides to get there
            b_idx = (
                batch_idx * b_batch_stride + k * b_strides[-2] + col_idx * b_strides[-1]
            )
            dot_prod += a_storage[a_idx] * b_storage[b_idx]  # accumulate

        # Write to output storage
        out[out_pos] = dot_prod


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
