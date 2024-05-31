#pragma once

#include <memory>
#include <string>
#include <vector>

class Tensor {
private:
    std::vector<float> data;
    std::vector<int> strides;
    std::vector<int> shape;
    int ndim;
    int size;
    std::string device;

public:
    Tensor(std::vector<float>& data, std::vector<int>& shape, int ndim, std::string device);
    const float get_item(std::vector<int> indices);
    const float get_item(int index); 
    void set_item(int index, float value);
    const std::vector<int> get_strides() const;
    const std::vector<int> get_shape() const;
    int get_ndim() const;
    int get_size() const;
    std::string get_device() const;
};