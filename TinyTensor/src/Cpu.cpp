#include <exception>
#include <iostream>
#include <math.h>
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

std::shared_ptr<float[]> zero_tensor_cpu(std::shared_ptr<Tensor> t1) {
    auto elements = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++) {
        elements[i] = 0.0f;
    }

    return elements;
}

std::shared_ptr<float[]> one_tensor_cpu(std::shared_ptr<Tensor> t1) {
    auto elements = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++) {
        elements[i] = 1.0f;
    }

    return elements;
}

std::shared_ptr<float[]> sin_tensor_cpu(std::shared_ptr<Tensor> t1) {
    auto elements = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++) {
        elements[i] = sinf(t1->get_item(i));
    }

    return elements;
}

std::shared_ptr<float[]> cos_tensor_cpu(std::shared_ptr<Tensor> t1) {
    auto elements = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++) {
        elements[i] = cosf(t1->get_item(i));
    }

    return elements;
}

std::shared_ptr<float[]> log_tensor_cpu(std::shared_ptr<Tensor> t1) {
    auto elements = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++) {
        elements[i] = logf(t1->get_item(i));
    }

    return elements;
}

std::shared_ptr<float[]> exp_tensor_cpu(std::shared_ptr<Tensor> t1) {
    auto elements = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++) {
        elements[i] = expf(t1->get_item(i));
    }

    return elements;
}


std::shared_ptr<float[]> sigmoid_tensor_cpu(std::shared_ptr<Tensor> t1) {
    auto elements = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++) {
        auto data = t1->get_item(i);

        if (data >= 0) {
            elements[i] = 1 / (1 + expf(-data));
        } else {
            elements[i] = 1 / (1 + expf(data));
        }
    }

    return elements;
}

std::shared_ptr<float[]> scalar_pow_tensor_cpu(float base, std::shared_ptr<Tensor> t1) {
    auto elements = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++) {
        elements[i] = pow(base, t1->get_item(i));
    }

    return elements;
}

std::shared_ptr<float[]> tensor_pow_scalar(std::shared_ptr<Tensor> t1, float exp) {
    auto elements = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++) {
        elements[i] = pow(t1->get_item(i), exp);
    }

    return elements;
}


void assign_tensor_cpu(std::shared_ptr<Tensor> t1, std::shared_ptr<float[]> data) {
    for (int i = 0; i < t1->get_size(); i++)
    {
        t1->set_item(i, data[i]);
    }
}