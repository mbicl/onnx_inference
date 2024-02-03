# ONNX Runtime

## Installing ONNX Runtime CXX API - Linux

- Download latest release (cpu or gpu version): [link](https://github.com/microsoft/onnxruntime/releases)
- Extract archive file, move include files to `/usr/include/onnxruntime` folder and lib files to `/usr/lib/onnxruntime` folder.
- Compiling:

```bash
g++ program.cpp -o program -I/usr/include/onnxruntime -L/usr/lib/onnxruntime -lonnxruntime
```

- Header file: `onnxruntime_cxx_api.h`
- Namespace: `Ort`

## Ort::Env

```cpp
Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,"Default");
```

`Ort::Env` holds the logging state used by all other objects. One Env must be created before using any other Onnxruntime functionality.
We mainly use `Ort::Env` for creating `Ort::Session`. [Documentation](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_env.html)

## Ort::SessionOptions

```cpp
   Ort::SessionOptions session_options;
```

Used for creating `Ort::Session` and defines `Ort::Session` options. [Documentation](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_session_options.html) \
Example:

```cpp
   session_options.SetInterOpNumThreads(1);
   session_options.SetIntraOpNumThreads(1);
   session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
```

## Ort::Session

```cpp
Ort::Session session = Ort::Session(env, “model_path”, session_options);
```

`Ort::Session` represents inference session. \
`Ort::Session` class is used to load, initialize and run inference on ONNX models. [Documentation](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_session.html)

## Ort::MemoryInfo

```cpp
Ort::MemoryInfo memory_info{nullptr};
memory_info = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault));
```

`Ort::MemoryInfo` represents information about memory allocation, such as the device type, device ID, and memory type. It is used for specifying memory-related properties when creating tensors or executing operations ([doc](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_memory_info.html)). `CreateCpu` function used to create MemoryInfo object representing CPU memory.

## Working with model

We need input node names, input node dimensions and output node names for working with model. These represents input and output of model. For doing this we use `session.GetInputCount()`, `session.GetInputNameAllocated()`, `type_info.GetTensorTypeAndShapeInfo()`, `tensor_info.GetShape()`, `session.GetOutputCount()`, `session.GetOutputNameAllocated()` functions.

```cpp
   std::vector<const char*>* input_node_names = nullptr;
   std::vector<std::vector<int64_t>> input_node_dims;
   std::vector<const char*>* output_node_names = nullptr;
```

## Getting input/output node count and names of model

`session.GetInputCount()` - returns number of model inputs. \
`session.GetOutputCount()` - returns number of model outputs. \
`session.GetInputNameAllocated(index,allocator)` - returns a copy of input name at the specified index. \
`session.GetOutputNameAllocated(index,allocator)` - returns a copy of output name at the specified index. \
We need to push all input and output names to `input_node_names` and `output_node_names` vectors.

## Additional functions and data types for input and output

- `ONNXTensorElementDataType` - enum of tensor element data types ([doc](https://onnxruntime.ai/docs/api/c/group___global.html#gaec63cdda46c29b8183997f38930ce38e)).
- `Ort::TypeInfo` - type information that may contain either `TensorTypeAndShapeInfo` or the information about contained sequence or map depending on the `ONNXType` ([doc](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_type_info.html));
- `Ort::AllocatorWithDefaultOptions` - allocator with default options, used for allocating memory for tensors or other data structures within the Onnxruntime ([doc](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_allocator_with_default_options.html)).
- `session.GetInputTypeInfo()` - returns input type information Ort::TypeInfo ([doc](https://onnxruntime.ai/docs/api/c/struct_ort_1_1detail_1_1_const_session_impl.html#a0a863a0ed3831b3d6a7f16f6fcb80c97)).
- `type_info.GetTensorTypeAndShapeInfo()` - returns information about tensor’s type and shape (tensor_info), method of `Ort::TypeInfo` class.
- `tensor_info.GetElementType()` - returns `ONNXTensorElementDataType` ([doc](https://onnxruntime.ai/docs/api/c/group___global.html#gaec63cdda46c29b8183997f38930ce38e)).
- `tensor_info.GetShape()` - uses `GetDimensionsCount` & `GetDimensions` to return `std::vector` of the shape.

## Ort::Value

`Ort::Value` is a class used to represent generic value that can hold various types of data, including tensors, sparse tensors, sequences, maps, and opaque data. It’s a fundamental type used for passing to and receiving outputs from ONNX runtime sessions. \
`Ort::Value::CreateTensor(MemoryInfo,p_data,p_data_count,shape,shape_len)` - creates a tensor with user supplied buffer.
p_data - pointer to the data buffer (void *).
shape - pointer to the tensor shape dimensions.

## session.Run()

```cpp
session.Run(Ort::RunOptions{nullptr},input_node_names->data(),input_tensor.data(), input_tensor.size(), output_node_names->data(), output_node_names->size());
```

Runs the model and returns results in vector of `Ort::Value`.

## value.GetTensorMutableData()

```cpp
Type* Ort::Value::Base::GetTensorMutableData<Type>()
```

Returns a non-const typed pointer to an OrtValue/Tensor contained buffer.
No Type checking is performed, the caller must ensure that type matches the OrtValue/Tensor type.
Returns non-const pointer to data, no copies made.
