#include <gtest/gtest.h>
#include <math.h>
#include "Tensor.h"
#include "Cpu.h"

TEST(CPU, AddTensorsCPU) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {2, 2};
    auto tensor1 = std::make_shared<Tensor>(data, shape, 2, device);
    auto tensor2 = std::make_shared<Tensor>(data, shape, 2, device);

    float expected_outputs[] = {648, 68, 108, 24};
    auto out = add_tensor_cpu(tensor1, tensor2);
    for (int i = 0; i < tensor1->get_size(); i++)
    {
        EXPECT_EQ(expected_outputs[i], out[i]);
    }
}

TEST(CPU, SubTensorsCPU) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {2, 2};
    auto tensor1 = std::make_shared<Tensor>(data, shape, 2, device);
    auto tensor2 = std::make_shared<Tensor>(data, shape, 2, device);

    float expected_outputs[] = {0, 0, 0, 0};
    auto out = sub_tensor_cpu(tensor1, tensor2);
    for (int i = 0; i < tensor1->get_size(); i++)
    {
        EXPECT_EQ(expected_outputs[i], out[i]);
    }
}

TEST(CPU, ElementWiseMultiplicationTensorsCPU) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {2, 2};
    auto tensor1 = std::make_shared<Tensor>(data, shape, 2, device);
    auto tensor2 = std::make_shared<Tensor>(data, shape, 2, device);

    float expected_outputs[] = {data[0] * data[0], data[1] * data[1], data[2] * data[2], data[3] * data[3]};
    auto out = elementwise_mul_tensor_cpu(tensor1, tensor2);
    for (int i = 0; i < tensor1->get_size(); i++)
    {
        EXPECT_EQ(expected_outputs[i], out[i]);
    }
}

TEST(CPU, ElementWiseDivTensorsCPU) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {2, 2};
    auto tensor1 = std::make_shared<Tensor>(data, shape, 2, device);
    auto tensor2 = std::make_shared<Tensor>(data, shape, 2, device);

    float expected_outputs[] = {1, 1, 1, 1};
    auto out = elementwise_div_tensor_cpu(tensor1, tensor2);
    for (int i = 0; i < tensor1->get_size(); i++)
    {
        EXPECT_EQ(expected_outputs[i], out[i]);
    }
}

TEST(CPU, TensorDivScalarCPU) {
    std::string device = "cpu";
    std::vector<float> data = {4, 8, 16, 20};
    std::vector<int> shape = {2, 2};
    auto tensor1 = std::make_shared<Tensor>(data, shape, 2, device);
    float scalar = 2;

    float expected_outputs[] = {2, 4, 8, 10};
    auto out = tensor_div_scalar_cpu(tensor1, scalar);
    for (int i = 0; i < tensor1->get_size(); i++)
    {
        EXPECT_EQ(expected_outputs[i], out[i]);
    }
}

TEST(CPU, ScalarDivTensorCPU) {
    std::string device = "cpu";
    std::vector<float> data = {4, 8, 16, 20};
    std::vector<int> shape = {2, 2};
    auto tensor1 = std::make_shared<Tensor>(data, shape, 2, device);
    float scalar = 2;

    float expected_outputs[] = {0.5, 0.25, 0.125, 0.1};
    auto out = scalar_div_tensor_cpu(scalar, tensor1);
    for (int i = 0; i < tensor1->get_size(); i++)
    {
        EXPECT_EQ(expected_outputs[i], out[i]);
    }
}

TEST(CPU, EqualTensorsCPU) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<float> data2 = {324, 213, 12, 12};
    std::vector<int> shape = {2, 2};
    auto tensor1 = std::make_shared<Tensor>(data, shape, 2, device);
    auto tensor2 = std::make_shared<Tensor>(data2, shape, 2, device);

    float expected_outputs[] = {1.0f, 0.0f, 0.0f, 1.0f};
    auto out = eq_tensor_cpu(tensor1, tensor2);
    for (int i = 0; i < tensor1->get_size(); i++)
    {
        EXPECT_EQ(expected_outputs[i], out[i]);
    }
}

TEST(CPU, ZeroTensorCPU) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {2, 2};
    auto tensor = std::make_shared<Tensor>(data, shape, 2, device);

    auto result = zero_tensor_cpu(tensor);
    for (int i = 0; i < tensor->get_size(); i++)
    {
        EXPECT_EQ(result[i], 0.f);
    }
}

TEST(CPU, OneTensorCPU) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {2, 2};
    auto tensor = std::make_shared<Tensor>(data, shape, 2, device);

    auto result = one_tensor_cpu(tensor);
    for (int i = 0; i < tensor->get_size(); i++)
    {
        EXPECT_EQ(result[i], 1.f);
    }
}

TEST(CPU, SinTensorCPU) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {2, 2};
    auto tensor = std::make_shared<Tensor>(data, shape, 2, device);

    auto result = sin_tensor_cpu(tensor);
    for (int i = 0; i < tensor->get_size(); i++)
    {
        EXPECT_EQ(result[i], sinf(tensor->get_item(i)));
    }
}

TEST(CPU, LogTensorCpu) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {2, 2};
    auto tensor = std::make_shared<Tensor>(data, shape, 2, device);

    auto result = log_tensor_cpu(tensor);
    for (int i = 0; i < tensor->get_size(); i++)
    {
        EXPECT_EQ(result[i], logf(tensor->get_item(i)));
    }
}

TEST(CPU, ExpTensorCpu) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {2, 2};
    auto tensor = std::make_shared<Tensor>(data, shape, 2, device);

    auto result = exp_tensor_cpu(tensor);
    for (int i = 0; i < tensor->get_size(); i++)
    {
        EXPECT_EQ(result[i], expf(tensor->get_item(i)));
    }
}

TEST(CPU, SigmoidTensorCpu) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {2, 2};
    auto tensor = std::make_shared<Tensor>(data, shape, 2, device);

    auto result = sigmoid_tensor_cpu(tensor);
    for (int i = 0; i < tensor->get_size(); i++)
    {
        auto data = tensor->get_item(i);

        if (data >= 0) {
            EXPECT_EQ(result[i], 1/ (1 + expf(-data)));
        } else {
            EXPECT_EQ(result[i], 1/ (1 + expf(data)));
        }
    }
}

TEST(CPU, ScalarPowTensor) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {2, 2};
    auto tensor = std::make_shared<Tensor>(data, shape, 2, device);

    auto result = scalar_pow_tensor_cpu(2.0f, tensor);
    for (int i = 0; i < tensor->get_size(); i++)
    {
        EXPECT_EQ(result[i], pow(2.0f, tensor->get_item(i)));
    }
}

TEST(CPU, TensorPowScalar) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {2, 2};
    auto tensor = std::make_shared<Tensor>(data, shape, 2, device);

    auto result = tensor_pow_scalar(tensor, 2.0f);
    for (int i = 0; i < tensor->get_size(); i++)
    {
        EXPECT_EQ(result[i], pow(tensor->get_item(i), 2.0f));
    }
}


TEST(CPU, CosTensorCPU) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {2, 2};
    auto tensor = std::make_shared<Tensor>(data, shape, 2, device);

    auto result = cos_tensor_cpu(tensor);
    for (int i = 0; i < tensor->get_size(); i++)
    {
        EXPECT_EQ(result[i], cosf(tensor->get_item(i)));
    }
}

TEST(CPU, AssignTensorCPU) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {2, 2};
    auto tensor = std::make_shared<Tensor>(data, shape, 2, device);
    auto values = std::shared_ptr<float[]>(new float[4]{1, 2, 3, 4}, std::default_delete<float[]>());


    assign_tensor_cpu(tensor, values);
    for (int i = 0; i < tensor->get_size(); i++)
    {
        EXPECT_EQ(tensor->get_item(i), values[i]);
    }
}

TEST(CPU, Transpose1DTensorCPU) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {4};
    auto tensor = std::make_shared<Tensor>(data, shape, 1, device);
    auto result = transpose_1d_vector_cpu(tensor);

    for (int i = 0; i < tensor->get_size(); i++)
    {
        EXPECT_EQ(tensor->get_item(i), result[i]);
    }
    
}


TEST(CPU, Transpose2DTensorCPU) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {2, 2};
    auto tensor = std::make_shared<Tensor>(data, shape, 2, device);
    auto result = transpose_2d_vector_cpu(tensor);
    std::vector<float> expected = {324, 54, 34, 12};

    for (int i = 0; i < tensor->get_size(); i++)
    {
        EXPECT_EQ(expected.at(i), result[i]);
    }
}

TEST(CPU, Transpose3DTensorCPU) {
    std::string device = "cpu";
    std::vector<float> data = {1,2,3,4,5,6,7,8};
    std::vector<int> shape = {2, 2, 2};
    auto tensor = std::make_shared<Tensor>(data, shape, 3, device);
    auto result = transpose_3d_vector_cpu(tensor);
    std::vector<float> expected = {1, 5, 3, 7, 2, 6, 4, 8};

    for (int i = 0; i < tensor->get_size(); i++)
    {
        EXPECT_EQ(expected.at(i), result[i]);
    }
}

