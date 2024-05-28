#pragma once

#include <memory>
#include "Tensor.h"

std::shared_ptr<float[]> add_tensor_cpu(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2);
std::shared_ptr<float[]> sub_tensor_cpu(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2);
std::shared_ptr<float[]> elementwise_div_tensor_cpu(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2);
std::shared_ptr<float[]> scalar_div_tensor_cpu(float scalar, std::shared_ptr<Tensor> t1);
std::shared_ptr<float[]> tensor_div_scalar_cpu(std::shared_ptr<Tensor> t1, float scalar);
std::shared_ptr<float[]> elementwise_mul_tensor_cpu(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2);
std::shared_ptr<float[]> eq_tensor_cpu(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2);
