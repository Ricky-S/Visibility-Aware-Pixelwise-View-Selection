# Visibility-Aware Pixelwise View Selection for Multi-View Stereo Matching

Source code for the paper:

**Visibility-Aware Pixelwise View Selection for Multi-View Stereo Matching**, ICPR 2024

## About

**Visibility-Aware Pixelwise View Selection for Multi-View Stereo Matching** is an effective multi-view stereo algorithm.

 If you find this project useful for your research, please cite:

```text
@article{2024_Z_AMBC,  
  title={Visibility-Aware Pixelwise View Selection for Multi-View Stereo Matching}, 
  author={Huang, Zhentao and Shi, Yukun and Gong, Minglun}, 
  journal={International Conference on Pattern Recognition (ICPR)},
  year={2024}
}
```


## Dependencies

The code has been tested on Ubuntu 20.04 with RTX 4090.

- Cuda >= 6.0
- OpenCV >= 2.4
- cmake

## How to use

- Compile

```bash
cd build
cmake ..
make
cd ..
```

- Test

```bash
# dtu dataset
./scripts/dtu/dtu_multiscans.sh

# tanksandtemples dataset
# A sample set is shown in the following file.
./scripts/tnt/tntAll.sh
```


- Important parameters
```text
int FoodNumber = 5; //The number of food sources
---
if (c % 2 == 1) // Note: In case of not able to process all input images
```


**tnt dataset:** Please put your data in `./data` folder. The folder structure should be like this:

```text
.
├── advanced
│   ├── Auditorium
│   │   ├── cams_1
│   │   ├── images
│   │   ├── Auditorium.log
│   │   └── pair.txt
│   ├── Ballroom
│   ├── Courtroom
│   ├── Museum
│   ├── Palace
│   └── Temple
├── intermediate
    ├── Family
    ├── Francis
    ├── Horse
    ├── Lighthouse
    ├── M60
    ├── Panther
    ├── Playground
    └── Train

```

## Acknowledgements

The code largely benefits from [Gipuma](https://github.com/kysucix/gipuma). Thanks to their authors.
