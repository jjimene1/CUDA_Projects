# Implementation Details

The following program was developed in WSL 2.0 running Ubuntu 22.04. All testing was done on an NVIDIA RTX 3060. Code can be found in the file titled matrixMul_new.cu and can be compiled using the makefile. The makefile is setup in such a way that it will create a bin directory above where you are corrently located as well as linking some files in the included "common" directory which was take from the cuda-samples NVIDIA repo. Make sure to have this "common" directory a level above the working directory to make sure the makefile compiles properly. Debug symbols can be included by using: 

```
make dbg=1
```

To run the compiled code use the following format:

```
./matrixMul_new -wA=Columns_MatrixA -hA=Rows_MatrixA -wB=Columns_MatrixB
```

Where you replace Columns_MatrixA, Rows_MatrixA, and Columns_MatrixB with the numerical value of your desired input matrices. The code is written to fill the first matrix with constants of 1.0 and the second matrix with constants of 0.01 when allocating memory.  