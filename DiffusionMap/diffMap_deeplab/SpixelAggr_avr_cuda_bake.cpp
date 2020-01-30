#include <torch/torch.h>
#include <vector>

// CUDA forward declarations
std::vector<at::Tensor> SpixelAggr_avr_cuda_forward(
      at::Tensor res,
      at::Tensor Feat,
      at::Tensor SegsIdx,
      at::Tensor nPsPerSegs);

std::vector<at::Tensor> SpixelAggr_avr_cuda_backward(
      at::Tensor grad,
      at::Tensor Feat,
      at::Tensor SegsIdx,
      at::Tensor nPsPerSegs,
      at::Tensor DzDy);

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> SpixelAggr_avr_forward(
    at::Tensor res,
    at::Tensor input,
    at::Tensor segLabels,
    at::Tensor nSegs
    ){
    CHECK_INPUT(res);
    CHECK_INPUT(input);
    CHECK_INPUT(segLabels);
    CHECK_INPUT(nSegs);
    SpixelAggr_avr_cuda_forward(res, input, segLabels, nSegs);

    //return
 }


std::vector<at::Tensor> SpixelAggr_avr_backward(
    at::Tensor grad,
    at::Tensor input,
    at::Tensor segLabels,
    at::Tensor nSegs,
    at::Tensor grad_out
    ){
    CHECK_CUDA(grad);
    CHECK_INPUT(input);
    CHECK_INPUT(segLabels);
    CHECK_INPUT(nSegs);
    CHECK_INPUT(grad_out);

    SpixelAggr_avr_cuda_backward(grad, input, segLabels, nSegs, grad_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &SpixelAggr_avr_forward, "Super-pixel based average feature pooling---forward stage (CUDA)");
    m.def("backard", &SpixelAggr_avr_backward, "Super-pixel based average feature pooling--- backward");
}


