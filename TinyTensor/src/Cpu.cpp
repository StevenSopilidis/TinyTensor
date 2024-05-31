#include <exception>
#include <iostream>
#include "Cpu.h"

std::shared_ptr<float[]> add_tensor_cpu(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2) {
    auto elements = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++) {
        elements[i] = t1->get_item(i) + t2->get_item(i);
    }

    return elements;
}

std::shared_ptr<float[]> sub_tensor_cpu(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2) {
    auto elements = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++) {
        elements[i] = t1->get_item(i) - t2->get_item(i);
    }

    return elements;
}

std::shared_ptr<float[]> elementwise_mul_tensor_cpu(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2) {
    auto elements = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++) {
        elements[i] = t1->get_item(i) * t2->get_item(i);
    }

    return elements;
}

std::shared_ptr<float[]> elementwise_div_tensor_cpu(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2) {
    auto elements = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++) {
        elements[i] = t1->get_item(i) / t2->get_item(i);
    }

    return elements;
}

std::shared_ptr<float[]> scalar_div_tensor_cpu(float scalar, std::shared_ptr<Tensor> t1) {
    auto elements = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++) {
        elements[i] = scalar / t1->get_item(i);
    }

    return elements;
}

std::shared_ptr<float[]> tensor_div_scalar_cpu(std::shared_ptr<Tensor> t1, float scalar) {
    auto elements = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++) {
        elements[i] = t1->get_item(i) / scalar;
    }

    return elements;
}


std::shared_ptr<float[]> eq_tensor_cpu(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2) {
    auto elements = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++) {
        elements[i] = (t1->get_item(i) == t2->get_item(i))? 1.0f : 0.0f;
    }

    return elements;
}

void zero_tensor_cpu(std::shared_ptr<Tensor> t1) {
    for (int i = 0; i < t1->get_size(); i++) {
        t1->set_item(i, 0.f);
    }
}

void one_tensor_cpu(std::shared_ptr<Tensor> t1) {
    for (int i = 0; i < t1->get_size(); i++) {
        t1->set_item(i, 1.f);
    }
}

void assign_tensor_cpu(std::shared_ptr<Tensor> t1, std::shared_ptr<float[]> data) {
    for (int i = 0; i < t1->get_size(); i++)
    {
        t1->set_item(i, data[i]);
    }
}