#include <iostream>
#include "Tensor.h"
#include "Cpu.h"
#include <vector>

int main() {
    std::string device = "cpu";
    std::vector<float> data = {324, 34, 54, 12};
    std::vector<int> shape = {2, 2};
    auto tensor1 = std::make_shared<Tensor>(data, shape, 2, device);
    data = {324, 34};
    shape = {2};
    auto tensor2 = std::make_shared<Tensor>(data, shape, 1, device);

    auto out = add_tensor_broadcasted_cpu(tensor1, tensor2, std::vector<int> {2, 2}, 4);
    for (int i = 0; i < 4; i++)
    {   
        std::cout << out[i] << std::endl;
    }
    
}   