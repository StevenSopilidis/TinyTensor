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
std::shared_ptr<float[]> zero_tensor_cpu(std::shared_ptr<Tensor> t1); 
std::shared_ptr<float[]> one_tensor_cpu(std::shared_ptr<Tensor> t1); 
std::shared_ptr<float[]> sin_tensor_cpu(std::shared_ptr<Tensor> t1);
std::shared_ptr<float[]> cos_tensor_cpu(std::shared_ptr<Tensor> t1);
std::shared_ptr<float[]> log_tensor_cpu(std::shared_ptr<Tensor> t1);
std::shared_ptr<float[]> exp_tensor_cpu(std::shared_ptr<Tensor> t1);
std::shared_ptr<float[]> sigmoid_tensor_cpu(std::shared_ptr<Tensor> t1);
std::shared_ptr<float[]> scalar_pow_tensor_cpu(float base, std::shared_ptr<Tensor> t1);
std::shared_ptr<float[]> tensor_pow_scalar(std::shared_ptr<Tensor> t1, float exp);
std::shared_ptr<float[]> max_tensor_cpu(std::shared_ptr<Tensor> t1, int size, std::vector<int>& result_shape, int axis);
std::shared_ptr<float[]> min_tensor_cpu(std::shared_ptr<Tensor> t1, int size, std::vector<int>& result_shape, int axis);
std::shared_ptr<float[]> transpose_1d_vector_cpu(std::shared_ptr<Tensor> t1);
std::shared_ptr<float[]> transpose_2d_vector_cpu(std::shared_ptr<Tensor> t1);
std::shared_ptr<float[]> transpose_3d_vector_cpu(std::shared_ptr<Tensor> t1);
void assign_tensor_cpu(std::shared_ptr<Tensor> t1, std::shared_ptr<float[]> data);