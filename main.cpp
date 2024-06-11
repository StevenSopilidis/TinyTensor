#include <iostream>
#include "Tensor.h"
#include "Cpu.h"
#include <vector>

int main() {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12, 
                                234, 234, 5, 1};
    std::vector<int> shape = {2, 4};
    auto tensor = std::make_shared<Tensor>(data, shape, 2, device);

    std::vector<int> result_shape = {4};

    auto maxes = max_tensor_cpu(tensor, 4, result_shape, 0);

    std::cout << "Max values along axis 0: ";
    for (int i = 0; i < 4; i++) {
        std::cout << maxes[i] << " ";
    }
    std::cout << std::endl;
}