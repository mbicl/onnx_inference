#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "utils.hpp"

void load_json(char *filename, char **json_data){
    std::ifstream json_file(filename);
    std::string line;
    int i=0;
    while (std::getline(json_file,line)){
        json_data[i] = new char[line.length()+1];
        strncpy(json_data[i],line.c_str(),line.length()+1);
        ++i;
    }
}

int main(int argc, char *argv[]) {
    if (argc!=2){
        std::cout << "Usage: " << argv[0] << " <image_path>\n";
        return -1;
    }
    // Load the model
    std::string model_path = "resnet50v2/resnet50v2.onnx";
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,"Default");
    Ort::SessionOptions session_options;

    session_options.SetInterOpNumThreads(1);
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    Ort::Session session = Ort::Session(env, model_path.c_str(), session_options);

    // memory allocator
    Ort::MemoryInfo memory_info{nullptr};
    try {
        memory_info = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault));
    }
    catch (const Ort::Exception& e){
        std::cout << "ONNX exception caught: " << e.what() << '\n';
        return -1;
    }

    // input and output node names
    // set input node names
    size_t num_input_nodes = 0; 
    std::vector<const char*>* input_node_names = nullptr;
    std::vector<std::vector<int64_t>> input_node_dims;
    ONNXTensorElementDataType type;
    Ort::TypeInfo* type_info = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;
    num_input_nodes = session.GetInputCount();
    input_node_names = new std::vector<const char*>();
    for (int i=0; i<num_input_nodes; ++i){
        char *temp_string = new char[strlen(session.GetInputNameAllocated(i,allocator).get())+1];
        strncpy(temp_string,session.GetInputNameAllocated(i,allocator).get(),strlen(session.GetInputNameAllocated(i,allocator).get())+1);
        input_node_names->push_back(temp_string);
        printf("Input %d : name=%s\n",i,input_node_names->at(i));

        type_info = new Ort::TypeInfo(session.GetInputTypeInfo(i));
        auto tensor_info = type_info->GetTensorTypeAndShapeInfo();
        type = tensor_info.GetElementType();
        input_node_dims.push_back(tensor_info.GetShape());

        delete type_info;

        // debug
        #ifdef DEBUG
        printf("Input %d : name=%s, type=%d\n",i,input_node_names->at(i),type);
        printf("Input %d : dims=[",i);
        for (int j=0; j<input_node_dims[i].size(); ++j){
            printf("%ld",input_node_dims[i][j]);
            if (j<input_node_dims[i].size()-1) printf(",");
        }
        printf("]\n");
        #endif
    }

    // set output node names
    std::vector<const char*>* output_node_names = nullptr;
    size_t num_output_nodes = session.GetOutputCount();
    output_node_names = new std::vector<const char*>();
    for (int i=0; i<num_output_nodes; ++i){
        char *temp_string = new char[strlen(session.GetOutputNameAllocated(i,allocator).get())+1];
        strncpy(temp_string,session.GetOutputNameAllocated(i,allocator).get(),strlen(session.GetOutputNameAllocated(i,allocator).get())+1);
        output_node_names->push_back(temp_string);
        printf("Output %d : name=%s\n",i,output_node_names->at(i));
    }

    // preprocess the input
    std::vector<float> *input_tensor_values = nullptr;
    std::vector<Ort::Value> input_tensor;
    std::string image_path = argv[1];
    cv::Mat image = cv::imread(image_path.c_str(),cv::IMREAD_COLOR);
    // this will make the input into 1x3x224x224
    cv::Mat blob = cv::dnn::blobFromImage(image,1/255.0,cv::Size(224,224),cv::Scalar(0,0,0),false,false);
    size_t input_tensor_size = blob.total();
    input_tensor_values = new std::vector<float>((float*)blob.data, (float*)blob.data+input_tensor_size);
    input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, input_tensor_values->data(), input_tensor_size, input_node_dims.at(0).data(), input_node_dims.at(0).size()));


    // run inference
    Utils::StopWatch<> timer;
    auto output_tensor = session.Run(Ort::RunOptions{nullptr},input_node_names->data(),input_tensor.data(), input_tensor.size(), output_node_names->data(), output_node_names->size());
    auto elapsed_time = timer.elapsed<float, std::chrono::milliseconds>();
    std::cout << "Inference time: " << elapsed_time << " ms\n";

    // softmax the output tensor
    if (output_tensor.size()>0){
        float* floatarr = output_tensor.at(0).GetTensorMutableData<float>();
        float sum = 0.0;
        for (int i=0; i<output_tensor.at(0).GetTensorTypeAndShapeInfo().GetElementCount(); ++i){
            floatarr[i] = exp(floatarr[i]);
            sum += floatarr[i];
        }
        std::vector<std::pair<float,int>> output;
        for (int i=0; i<output_tensor.at(0).GetTensorTypeAndShapeInfo().GetElementCount(); ++i){
            floatarr[i] /= sum;
            output.push_back(std::make_pair(floatarr[i],i));
        }
        std::sort(output.rbegin(),output.rend());

        char *json_data[1<<10];
        load_json("imagenet-simple-labels.json",json_data);
        for (int i=0; i<5; ++i){
            printf("%s : %f\n",json_data[output[i].second],output[i].first);
        }
    }
}