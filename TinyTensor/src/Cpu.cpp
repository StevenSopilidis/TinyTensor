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

bool can_broadcast_cpu(const std::vector<int>& shape1, const std::vector<int>& shape2) {
    int size1 = shape1.size();
    int size2 = shape2.size();
    int max_size = std::max(size1, size2);

    for (int i = 0; i < max_size; i++)
    {
        int dim1 = (i < size1)? shape1[size1 - 1 - i]: 1;
        int dim2 = (i < size2)? shape2[size2 - 1 - i]: 1;

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<float[]> add_tensor_broadcasted_cpu(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2,  std::vector<int> broadcasted_shape, int broadcasted_size) {
    if(!can_broadcast_cpu(t1->get_shape(), t2->get_shape()))
        throw std::invalid_argument("Shapes cannot be brodcasted");
    
    int max_ndim = std::max(t1->get_ndim(), t2->get_ndim());
    std::vector<int> strides1(max_ndim, 0);
    std::vector<int> strides2(max_ndim, 0);

    int stride1 = 1, stride2 = 1;
    for (int i = max_ndim - 1; i >= 0; --i) {
        int dim1 = (i >= max_ndim - t1->get_ndim()) ? t1->get_shape()[i - (max_ndim - t1->get_ndim())] : 1;
        int dim2 = (i >= max_ndim - t2->get_ndim()) ? t2->get_shape()[i - (max_ndim - t2->get_ndim())] : 1;

        strides1[i] = dim1 == broadcasted_shape[i]? stride1 : 0;
        strides2[i] = dim2 == broadcasted_shape[i]? stride2 : 0;

        stride1 *= (dim1 == broadcasted_shape[i])? dim1 : 1;
        stride2 *= (dim2 == broadcasted_shape[i])? dim2 : 1;

    }

    auto result = std::shared_ptr<float[]>(new float[broadcasted_size]);

    // iterate through all items
    for (int i = 0; i < broadcasted_size; i++) {
        int idx1 = 0;
        int idx2 = 0;
        int temp = i;

        // iterate through dimensions of broadcasted_shape
        for (int j = max_ndim - 1; j >= 0; --j) {
            // index that j-th dimension
            int index = temp % broadcasted_shape[j];
            // go up one dimension
            temp /= broadcasted_shape[j];
            
            if (strides1[j] != 0) 
                idx1 += index * strides1[j];
            if (strides2[j] != 0) 
                idx2 += index * strides2[j];
        }

        result[i] = t1->get_item(idx1) + t2->get_item(idx2);
    }

    return result;
}   

std::shared_ptr<float[]> sub_tensor_cpu(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2) {
    auto elements = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++) {
        elements[i] = t1->get_item(i) - t2->get_item(i);
    }

    return elements;
}

std::shared_ptr<float[]> sub_tensor_broadcasted_cpu(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2,  std::vector<int> broadcasted_shape, int broadcasted_size) {
    if(!can_broadcast_cpu(t1->get_shape(), t2->get_shape()))
        throw std::invalid_argument("Shapes cannot be brodcasted");
    
    int max_ndim = std::max(t1->get_ndim(), t2->get_ndim());
    std::vector<int> strides1(max_ndim, 0);
    std::vector<int> strides2(max_ndim, 0);

    int stride1 = 1, stride2 = 1;
    for (int i = max_ndim - 1; i >= 0; --i) {
        int dim1 = (i >= max_ndim - t1->get_ndim()) ? t1->get_shape()[i - (max_ndim - t1->get_ndim())] : 1;
        int dim2 = (i >= max_ndim - t2->get_ndim()) ? t2->get_shape()[i - (max_ndim - t2->get_ndim())] : 1;

        strides1[i] = dim1 == broadcasted_shape[i]? stride1 : 0;
        strides2[i] = dim2 == broadcasted_shape[i]? stride2 : 0;

        stride1 *= (dim1 == broadcasted_shape[i])? dim1 : 1;
        stride2 *= (dim2 == broadcasted_shape[i])? dim2 : 1;

    }

    auto result = std::shared_ptr<float[]>(new float[broadcasted_size]);

    // iterate through all items
    for (int i = 0; i < broadcasted_size; i++) {
        int idx1 = 0;
        int idx2 = 0;
        int temp = i;

        // iterate through dimensions of broadcasted_shape
        for (int j = max_ndim - 1; j >= 0; --j) {
            // index that j-th dimension
            int index = temp % broadcasted_shape[j];
            // go up one dimension
            temp /= broadcasted_shape[j];
            
            if (strides1[j] != 0) 
                idx1 += index * strides1[j];
            if (strides2[j] != 0) 
                idx2 += index * strides2[j];
        }

        result[i] = t1->get_item(idx1) - t2->get_item(idx2);
    }

    return result;
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

std::shared_ptr<float[]> eq_tensor_broadcasted_cpu(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, std::vector<int> broadcasted_shape, int broadcasted_size) {
    if(!can_broadcast_cpu(t1->get_shape(), t2->get_shape()))
        throw std::invalid_argument("Shapes cannot be brodcasted");
    
    int max_ndim = std::max(t1->get_ndim(), t2->get_ndim());
    std::vector<int> strides1(max_ndim, 0);
    std::vector<int> strides2(max_ndim, 0);

    int stride1 = 1, stride2 = 1;
    for (int i = max_ndim - 1; i >= 0; --i) {
        int dim1 = (i >= max_ndim - t1->get_ndim()) ? t1->get_shape()[i - (max_ndim - t1->get_ndim())] : 1;
        int dim2 = (i >= max_ndim - t2->get_ndim()) ? t2->get_shape()[i - (max_ndim - t2->get_ndim())] : 1;

        strides1[i] = dim1 == broadcasted_shape[i]? stride1 : 0;
        strides2[i] = dim2 == broadcasted_shape[i]? stride2 : 0;

        stride1 *= (dim1 == broadcasted_shape[i])? dim1 : 1;
        stride2 *= (dim2 == broadcasted_shape[i])? dim2 : 1;

    }

    auto result = std::shared_ptr<float[]>(new float[broadcasted_size]);

    // iterate through all items
    for (int i = 0; i < broadcasted_size; i++) {
        int idx1 = 0;
        int idx2 = 0;
        int temp = i;

        // iterate through dimensions of broadcasted_shape
        for (int j = max_ndim - 1; j >= 0; --j) {
            // index that j-th dimension
            int index = temp % broadcasted_shape[j];
            // go up one dimension
            temp /= broadcasted_shape[j];
            
            if (strides1[j] != 0) 
                idx1 += index * strides1[j];
            if (strides2[j] != 0) 
                idx2 += index * strides2[j];
        }

        result[i] = t1->get_item(idx1) == t2->get_item(idx2)? 1.0f : 0.0f;
    }

    return result;
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

std::shared_ptr<float[]> max_tensor_cpu(std::shared_ptr<Tensor> t1, int size, std::vector<int>& result_shape, int axis) {
    if (axis == -1) {
        auto maxes = std::shared_ptr<float[]>(new float[1]);
        auto max = -INFINITY;

        for (int i = 0; i < t1->get_size(); i++) {
            max = fmax(max, t1->get_item(i));
        }

        maxes[0] = max;
        return maxes;
    } else {
        if (axis < 0 || axis >= t1->get_ndim()) {
            throw new std::invalid_argument("Invalid axis provided");
        }
        
        auto maxes = std::shared_ptr<float[]>(new float[size]);
        std::fill_n(maxes.get(), size, -INFINITY);
        int axis_stride = t1->get_strides()[axis];

        for (int i = 0; i < t1->get_shape()[axis]; i++) {
            for (int j = 0; j < size; j++) {
                int index = 0;
                int remainder = j;
                for (int k = t1->get_ndim() - 2; k >= 0; k--) {
                    index += (remainder % result_shape[k]) * t1->get_strides()[k < axis ? k : k + 1];     
                    remainder /= result_shape[k];
                }
                maxes[j] = fmax(maxes[j], t1->get_item(index + i * axis_stride));
            }
        }
        
        return maxes;
    }
}

std::shared_ptr<float[]> min_tensor_cpu(std::shared_ptr<Tensor> t1, int size, std::vector<int>& result_shape, int axis) {
    if (axis == -1) {
        auto mins = std::shared_ptr<float[]>(new float[1]);
        auto min = INFINITY;

        for (int i = 0; i < t1->get_size(); i++)
        {
            min = fmin(min, t1->get_item(i));
        }

        mins[0] = min;

        return mins;        
    } else {
        if (axis < 0 || axis >= t1->get_ndim()) {
            throw new std::invalid_argument("Invalid axis provided");
        }

        auto mins = std::shared_ptr<float[]>(new float[size]);
        std::fill_n(mins.get(), size, INFINITY);
        auto axis_stride = t1->get_strides()[axis];

        for (int i = 0; i < t1->get_shape()[axis]; i++)
        {
            for (int j = 0; j < size; j++)
            {
                int index = 0;
                int remainder = j;
                for (int k = t1->get_ndim() - 2; k >= 0; k--) {
                    index += (remainder % result_shape[k]) * t1->get_strides()[k < axis ? k : k + 1];     
                    remainder /= result_shape[k];
                }
                mins[j] = fmin(mins[j], t1->get_item(index + i * axis_stride));
            }
        }

        return mins;
    }
}

std::shared_ptr<float[]> transpose_1d_vector_cpu(std::shared_ptr<Tensor> t1) {
    if (t1->get_ndim() != 1)
        throw new std::invalid_argument("Tensor is not 1D");

    auto result = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < t1->get_size(); i++)
    {
        result[i] = t1->get_item(i);
    }
    
    return result;
}

std::shared_ptr<float[]> transpose_2d_vector_cpu(std::shared_ptr<Tensor> t1) {
    if (t1->get_ndim() != 2)
        throw new std::invalid_argument("Tensor is not 2D");

    auto rows = t1->get_shape()[0];
    auto cols = t1->get_shape()[1];

    auto result_data = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result_data[j * cols + i] = t1->get_item(i * cols + j);
        }
    }
    
    return result_data;
}

std::shared_ptr<float[]> transpose_3d_vector_cpu(std::shared_ptr<Tensor> t1) {
    if (t1->get_ndim() != 3)
        throw new std::invalid_argument("Tensor is not 3D");

    auto batch = t1->get_shape()[0];
    auto rows = t1->get_shape()[1];
    auto cols = t1->get_shape()[2];

    auto result_data = std::shared_ptr<float[]>(new float[t1->get_size()]);

    for (int i = 0; i < batch; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            for (int k = 0; k < cols; k++)
            {
                result_data[k * rows * batch + j * batch + i] = t1->get_item(i * rows * cols + j * cols + k);
            }
        }
    }

    return result_data;
}