# ResNet50v2 Image Classification with ONNX Runtime

This project demonstrates how to perform image classification using the ResNet50v2 model with ONNX Runtime in C++ and python programming languages. The program takes an input image, preprocesses it, runs inference using the model, and outputs the top predicted labels along with their probabilities.

## Prerequisites

Before running the program, ensure you have the following dependencies installed:

- C++
  - C++ compiler supporting C++17 or later
  - CMake version 3.28 or later
  - OpenCV library (version 4.x)
  - ONNX Runtime C++ API
- Python
  - python 3.10 or later
  - Jupyter notebook

## Building and running in C++

1.Clone this repository to your local machine:

  ```bash
  git clone https://github.com/mbicl/onnx_inference.git
  ```

2.Build the program using CMake and run the compiled executable:

  ```bash
  # building
  mkdir build && cd build
  cmake ../
  make
  #running
  cd ..
  ./build/onnx_inference
  ```

3.Follow the on-screen instructions to enter the path to the input image.

4.View the top predicted labels and their probabilities printed on the console.

## Configuration

- **Model Path**: By default, the program expects the ResNet50v2 ONNX model file (`resnet50v2.onnx`) to be located in the project directory. You can modify the `model_path` variable in the source code if the model file is located elsewhere.

- **Label File**: The program loads label names from a JSON file (`imagenet-simple-labels.json`). Ensure this file is present in the specified path. You can change the path by modifying the `load_json` function call in the source code.

## Troubleshooting

- If you encounter any issues during compilation or execution, ensure that all dependencies are properly installed and paths are correctly specified.
