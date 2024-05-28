#include <iostream>
#include "Tensor.h"

Tensor::Tensor(std::vector<float>& data, std::vector<int>& shape, int ndim, std::string device) 
:data(data), shape(shape), ndim(ndim), device(device)
{
    size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape.at(i);
    }

    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides.insert(strides.begin(), stride);
        stride *= shape.at(i);
    }
}

const float Tensor::get_item(std::vector<int> indices) {
    int index = 0;
    for (int i = 0; i < ndim; i++)
    {
        index += indices.at(i) * strides.at(i);
    }
    
    return data[index];
}

const float Tensor::get_item(int index) {
    return data[index];
}

const std::vector<int> Tensor::get_strides() const {
    return strides;
}

const std::vector<int> Tensor::get_shape() const {
    return shape;
}

int Tensor::get_ndim() const {
    return ndim;
}

int Tensor::get_size() const {
    return size;
}

std::string Tensor::get_device() const {
    return device;
}
