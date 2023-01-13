#include <iostream>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"


namespace {


class Sonar {
public:
    Sonar() {
        std::string pipeline_file = "pipeline.onnx";
        sess_ = std::make_unique<Ort::Session>(Ort::Env{ORT_LOGGING_LEVEL_WARNING}, pipeline_file.c_str());

        auto input_names = sess_->GetInputNames();
        input_name_ = input_names[0];

        auto output_names = sess_->GetOutputNames();
        prob_name_ = output_names[1];
    }

    void ping(const std::string& text) {
        std::vector<const char*> input{ text.c_str() };
        Ort::Value proba = sess_->Run(Ort::RunOptions{nullptr},
                                      {input_name_},
                                      {Ort::Value::CreateTensor<char>(Ort::AllocatorWithDefaultOptions(), input.data(), input.size(), input_shape_, 1)},
                                      {prob_name_})[0];
        std::cout << text << " " << proba << std::endl;
    }

private:
    std::unique_ptr<Ort::Session> sess_;
    std::string input_name_;
    std::string prob_name_;
    std::vector<int64_t> input_shape_{ 1 };
};


}  // namespace


int main() {
    Sonar sonar;
    sonar.ping("今日 は いい 天気 です ね");

    return 0;
}