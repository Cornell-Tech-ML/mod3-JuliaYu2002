# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

3.1 parallel analysis script output:
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
C:\Users\rubyj\Documents\Cornell_Tech\MLE\mod3-JuliaYu2002\minitorch\fast_ops.py
 (166)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, C:\Users\rubyj\Documents\Cornell_Tech\MLE\mod3-JuliaYu2002\minitorch\fast_ops.py (166)
------------------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID      
    def _map(                                                                                                                                               |
        out: Storage,                                                                                                                                       |
        out_shape: Shape,                                                                                                                                   |
        out_strides: Strides,                                                                                                                               |
        in_storage: Storage,                                                                                                                                |
        in_shape: Shape,                                                                                                                                    |
        in_strides: Strides,                                                                                                                                |
    ) -> None:                                                                                                                                              |
        # TODO: Implement for Task 3.1.                                                                                                                     |
        x = np.array_equal(in_strides, out_strides) and np.array_equal(in_shape, out_shape)                                                                 |
        for i in prange(len(out)):--------------------------------------------------------------------------------------------------------------------------| #0
            out_index = np.empty(MAX_DIMS, np.int32)                                                                                                        |
            in_index = np.empty(MAX_DIMS, np.int32) # these are the numpy buffers                                                                           |
            if x: # strides and shape aligned                                                                                                               |
                out[i] = fn(in_storage[i]) # save value                                                                                                     |
            else:                                                                                                                                           |
                to_index(i, out_shape, out_index) # get index of the i of the thread we're in (prange makes as many threads as indicated by the param)      |
                broadcast_index(out_index, out_shape, in_shape, in_index) # convert the index in the out shape into the equivalent index in the in shape    |
                o = index_to_position(out_index, out_strides) # get the storage indices in the respective tensors                                           |
                j = index_to_position(in_index, in_strides)                                                                                                 |
                out[o] = fn(in_storage[j]) # save value                                                                                                     |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #0).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\rubyj\Documents\Cornell_Tech\MLE\mod3-JuliaYu2002\minitorch\fast_ops.py
 (177) is hoisted out of the parallel loop labelled #0 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\rubyj\Documents\Cornell_Tech\MLE\mod3-JuliaYu2002\minitorch\fast_ops.py
 (178) is hoisted out of the parallel loop labelled #0 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, np.int32) # these are the numpy
buffers
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
C:\Users\rubyj\Documents\Cornell_Tech\MLE\mod3-JuliaYu2002\minitorch\fast_ops.py
 (214)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, C:\Users\rubyj\Documents\Cornell_Tech\MLE\mod3-JuliaYu2002\minitorch\fast_ops.py (214)
---------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                      |
        out: Storage,                                                                              |
        out_shape: Shape,                                                                          |
        out_strides: Strides,                                                                      |
        a_storage: Storage,                                                                        |
        a_shape: Shape,                                                                            |
        a_strides: Strides,                                                                        |
        b_storage: Storage,                                                                        |
        b_shape: Shape,                                                                            |
        b_strides: Strides,                                                                        |
    ) -> None:                                                                                     |
        # TODO: Implement for Task 3.1.                                                            |
        x = np.array_equal(a_strides, b_strides) and np.array_equal(a_strides, out_strides) \      |
            and np.array_equal(b_strides, out_strides) and np.array_equal(a_shape, b_shape) \      |
                and np.array_equal(a_shape, out_shape) and np.array_equal(b_shape, out_shape)      |
        for i in prange(len(out)):-----------------------------------------------------------------| #1
            out_index = np.empty(MAX_DIMS, np.int32)                                               |
            a_index = np.empty(MAX_DIMS, np.int32)                                                 |
            b_index = np.empty(MAX_DIMS, np.int32)                                                 |
            if x:                                                                                  |
                out[i] = fn(a_storage[i], b_storage[i]) # if aligned, directly fill in out at i    |
            else:                                                                                  |
                to_index(i, out_shape, out_index)                                                  |
                o = index_to_position(out_index, out_strides)                                      |
                broadcast_index(out_index, out_shape, a_shape, a_index)                            |
                j = index_to_position(a_index, a_strides)                                          |
                broadcast_index(out_index, out_shape, b_shape, b_index)                            |
                k = index_to_position(b_index, b_strides)                                          |
                out[o] = fn(a_storage[j], b_storage[k])                                            |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\rubyj\Documents\Cornell_Tech\MLE\mod3-JuliaYu2002\minitorch\fast_ops.py
 (230) is hoisted out of the parallel loop labelled #1 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\rubyj\Documents\Cornell_Tech\MLE\mod3-JuliaYu2002\minitorch\fast_ops.py
 (231) is hoisted out of the parallel loop labelled #1 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\rubyj\Documents\Cornell_Tech\MLE\mod3-JuliaYu2002\minitorch\fast_ops.py
 (232) is hoisted out of the parallel loop labelled #1 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
C:\Users\rubyj\Documents\Cornell_Tech\MLE\mod3-JuliaYu2002\minitorch\fast_ops.py
 (268)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, C:\Users\rubyj\Documents\Cornell_Tech\MLE\mod3-JuliaYu2002\minitorch\fast_ops.py (268)
-------------------------------------------------------------|loop #ID
    def _reduce(                                             |
        out: Storage,                                        |
        out_shape: Shape,                                    |
        out_strides: Strides,                                |
        a_storage: Storage,                                  |
        a_shape: Shape,                                      |
        a_strides: Strides,                                  |
        reduce_dim: int,                                     |
    ) -> None:                                               |
        # TODO: Implement for Task 3.1.                      |
        reduce_size = a_shape[reduce_dim]                    |
        for i in prange(len(out)):---------------------------| #2
            out_index = np.empty(MAX_DIMS, np.int32)         |
            to_index(i, out_shape, out_index)                |
            o = index_to_position(out_index, out_strides)    |
            j = index_to_position(out_index, a_strides)      |
            for s in range(reduce_size):                     |
                out_index[reduce_dim] = s                    |
                out[o] = fn(out[o], a_storage[j])            |
                j += a_strides[reduce_dim]                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #2).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\rubyj\Documents\Cornell_Tech\MLE\mod3-JuliaYu2002\minitorch\fast_ops.py
 (280) is hoisted out of the parallel loop labelled #2 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None