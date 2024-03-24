# Implementation Details

The following program was developed in WSL 2.0 running Ubuntu 22.04. All testing was done on an NVIDIA RTX 3060. Code can be found in the file titled histogram.cu and can be compiled using the makefile. The makefile is setup in such a way that it will create a bin directory above where you are corrently located as well as linking some files in the included "common" directory which was take from the cuda-samples NVIDIA repo. Make sure to have this "common" directory a level above the working directory to make sure the makefile compiles properly. Debug symbols can be included by using: 

```
make dbg=1
```

To run the compiled code use the following format:

```
./histogram VecDim=Desired_input_vector_dimensions BinNum= Desired_bin_dimensions
```

Where the specified VecDim will be filled with random integers from 0-1023 and where BinNum can be any integer that can be written as 2^k where k can be any integer from 2 to 8. Included are three different kernels which can be used to calculate the histogram on the device:

histo_kernel: naive histogram application with atomic functions and stride

histogram_privatized_kernel: atomic histogram kernel with shared privatized memory

histogram_privatized_aggregation_kernel: atomic histogram kernel with shared privatized memory as well as aggregation optimization code. 

Simply comment or uncomment lines 154 and 153 to try the two most optimal kernels.