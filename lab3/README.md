# Implementation Details

The following program was developed in WSL 2.0 running Ubuntu 22.04. All testing was done on an NVIDIA RTX 3060. Code can be found in the file titled convolution2D.cu and can be compiled using the makefile. The makefile is setup in such a way that it will create a bin directory above where you are corrently located as well as linking some files in the included "common" directory which was take from the cuda-samples NVIDIA repo. Make sure to have this "common" directory a level above the working directory to make sure the makefile compiles properly. Debug symbols can be included by using: 

```
make dbg=1
```

To run the compiled code use the following format:

```
./convolution2D dimX=desired_image_width dimY=desired_image_height dimK=desired_mask_dimensions
```

Where the specified image dimensions will be filled with random floating point numbers from 0~15 and each have three channels per pixel. Channels can be changed by changing lines 116 and 120 of the code to have the variables equal the desired number of channels.