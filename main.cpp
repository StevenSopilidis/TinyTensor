#include <iostream>
#include "Tensor.h"

int main() {
    std::string device = "cpu";
    std::vector<float> data = {324.23, 34,34, 54.24, 12.34, 12.234, 5.34};
    std::vector<int> shape = {2, 3};
    auto tensor = new Tensor(data, shape, 2, device);
    auto item = tensor->get_item(std::vector<int>{1, 2});

    std::cout << item << std::endl;
}