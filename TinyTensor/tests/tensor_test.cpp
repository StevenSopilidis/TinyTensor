#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "Tensor.h"

TEST(TENSOR, TestCreateTensor) {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {2, 2};
    auto tensor = new Tensor(data, shape, 2, device);
    std::vector<int> expected_strides = {2, 1};

    auto size = tensor->get_size();
    auto ndim = tensor->get_ndim();
    auto tensor_device = tensor->get_device();
    auto tensor_shape = tensor->get_shape();
    auto tensor_strides = tensor->get_strides();
    EXPECT_TRUE(size == 4);
    EXPECT_TRUE(ndim == 2);
    EXPECT_TRUE(tensor_device == device);

    for (int i = 0; i < shape.size(); i++)
    {
        EXPECT_EQ(shape.at(i), tensor_shape.at(i));
    }

    for (int i = 0; i < expected_strides.size(); i++)
    {
        EXPECT_EQ(expected_strides.at(i), tensor_strides.at(i));
    }
}

TEST(TENSOR, TestGetItem) {
    std::string device = "cpu";
    std::vector<float> data = {324.23, 34,34, 54.24, 12.34, 12.234, 5.34};
    std::vector<int> shape = {2, 3};
    auto tensor = new Tensor(data, shape, 2, device);
    auto item1 = tensor->get_item(std::vector<int>{1, 2});
    auto item2 = tensor->get_item(std::vector<int>{0, 0});
    float expected_value1 = 12.234;
    float expected_value2 = 324.23;

    EXPECT_EQ(expected_value1, item1);
    EXPECT_EQ(expected_value2, item2);
}