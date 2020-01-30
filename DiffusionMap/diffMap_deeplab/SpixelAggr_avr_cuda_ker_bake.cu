#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace{

template <typename scalar_t>
__global__ void feat_aggr_avr_forward_ker(
    scalar_t* __restrict__ res,
    const scalar_t* __restrict__ d_Feats,
    const scalar_t* __restrict__ d_SegsIdx,
    const int nSegs,
    const int nPixels,
    const int dFeats)
{
    /* how to implement */
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < nSegs * dFeats)
    {
        int idseg = id % nSegs;
        int idFea = id / nSegs;
        float val = 0;
        int count = 0;

        for(int l = 0; l < nPixels; l++ )
        {
            if (d_SegsIdx[l] == idseg+1)
            {
                val = val + d_Feats[idFea * nPixels + l];
                count = count + 1;
            }
        }
        res[id] = val / count;
    }
}


template <typename scalar_t>
__global__ void feat_aggr_avr_backward_ker(
    scalar_t* __restrict__ res,
    const scalar_t* __restrict__ d_dzdy,
    const scalar_t* __restrict__ d_SegsIdx,
    const scalar_t* __restrict__ d_nPsSeg,
    const int nSegs,
    const int nPixels,
    const int dFeats
    )
{
     /* how to implement */
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < nPixels * dFeats)
    {
        int idPixel = id % nPixels;
        int idFea = id / nPixels;
        int idSeg = d_SegsIdx[idPixel];
        if (idSeg > 0)
        {
            res[id] = d_dzdy[idFea * nSegs + idSeg - 1] / d_nPsSeg[idSeg - 1];
        }
    }
}


template <typename scalar_t>
__global__ void conv_cls_ker_mt(
        scalar_t* __restrict__ res,
        const scalar_t* __restrict__ images,
        const scalar_t* __restrict__ indSet,
        const scalar_t* __restrict__ filters,
        const scalar_t* __restrict__ biases,
        const scalar_t imw,
        const scalar_t imh,
        const scalar_t nch,
        const scalar_t ptw,
        const scalar_t pth,
        const scalar_t ncs,
        const scalar_t nfs
        )
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int idpx = blockIdx.x;
    int lb = indSet[idpx];
    int x = idpx % imw;
    int y = (idpx - x) / imw;

    int PH = (2 * pth + 1);
    int PW = (2 * ptw + 1);

    int idth = threadIdx.x;
    //int idco = idth % PH;
    int idft = 0; //(idth - idco) / PH;

    int T = imw * imh * nfs;
    int Reso = imw * imh;
    int dfs = PH * PW * nch;

    int PW2 = 32; 
    int PH2 = 32;
    int dfsSpat = PW2 * PH2;

    __shared__ float res_tmp[3072];  
    __shared__ float filt_tmp[3072];    
        
    // load filter
    for (int d = idth; d < PW2 * PH2 * nch; d=d+blockDim.x) 
    {
        int pos = d % dfsSpat;
        int ic = (d - pos) / dfsSpat;
        
        int px = pos % PW2;
        int py = (pos - px) / PW2;
        
        if (px < PW & py < PH)
        {
            filt_tmp[d] = filters[lb * dfs * nfs + dfs * idft + ic * PW * PH + py * PW + px]; 
        }
        else
        {
            filt_tmp[d] = 0;
        }

    }
    __syncthreads(); 
    
    
    // load image data
    for (int d = idth; d < PW2 * PH2 * nch; d=d+blockDim.x) 
    {
        int pos = d % dfsSpat;
        int ic = (d - pos) / dfsSpat;

        int dx = pos % PW2;
        int dy = (pos - dx) / PW2;
        
        int px = x + dx - ptw;
        int py = y + dy - pth;

        px = (px < 0) ? 0 : (px >= imw ? (imw - 1) : px);
        py = (py < 0) ? 0 : (py >= imh ? (imh - 1) : py);

        res_tmp[d] = images[ic * Reso + py * imw + px]; 
     }
     __syncthreads();   

     // compute the inner product


      if (idth < 512)
      {   
            float val = 0;
            for(int l = idth; l < dfsSpat * nch; l = l + 512)
            {
                val = val +  res_tmp[l] * filt_tmp[l];
            }
            res_tmp[idth] = val;
        }
        
        __syncthreads(); 
    
        if(idth < 64)
        {
            float val = 0;
            for(int l = idth; l < 512; l = l + 64)
            {
                val = val + res_tmp[l];
            }
            res_tmp[idth] = val;
        }
     
        __syncthreads(); 
        if(idth < 8)
        {
            float val = 0;
            for(int l = idth; l < 64; l = l + 8)
            {
                val = val + res_tmp[l];
            }
            res_tmp[idth] = val;
        }
        
        __syncthreads(); 
       if(idth < 1)
        {
            float val = 0;
            for(int l = 0; l < 8; l++)
            {
                val = val + res_tmp[l];
            }
            res[idpx] = val;
        }    

}
} // end namespace

std::vector<at::Tensor> SpixelAggr_avr_cuda_forward(
      at::Tensor Feat,
      at::Tensor SegsIdx,
      at::Tensor nPsPerSegs) {


    int nPixels = Feat.size(0) * Feat.size(1);
    int dFeats = Feat.size(2);
    int nSegs = nPsPerSegs.size(0);

    at::Tensor res = at::zeros(nPsPerSegs.size(0), Feat.size(2));
    printf("%d, %d\n", Feat.size(0), Feat.size(2));

    int const threadsPerBlock = 512;
    int blocksPerGrid = ( nSegs * dFeats + threadsPerBlock - 1) / threadsPerBlock;
    /*AT_DISPATCH_FLOATING_TYPES(Feat.type(), "SpixelAggr_avr_cuda_forw", ([&] {
        feat_aggr_avr_forward_ker<<<blocksPerGrid, threadsPerBlock>>>(
        res.data<scalar_t>(),
        Feat.data<scalar_t>(),
        SegsIdx.data<scalar_t>(),
        nSegs,
        nPixels,
        dFeats);
     })); */

    return {res};
}

std::vector<at::Tensor> SpixelAggr_avr_cuda_backward(
      at::Tensor Feat,
      at::Tensor SegsIdx,
      at::Tensor nPsPerSegs,
      at::Tensor DzDy) {
    auto res = at::zeros_like(Feat);
    const int nSegs = DzDy.size(0);
    const int nPixels = Feat.size(0) * Feat.size(1);
    const int dFeats = Feat.size(2);

    const int threadsPerBlock = 512;
    int blocksPerGrid = (nPixels * dFeats + threadsPerBlock - 1) / threadsPerBlock;
    AT_DISPATCH_FLOATING_TYPES(Feat.type(), "SpixelAggr_avr_cuda_back", ([&] {
        feat_aggr_avr_backward_ker<<<blocksPerGrid, threadsPerBlock>>>(
        res.data<scalar_t>(),
        Feat.data<scalar_t>(),
        SegsIdx.data<scalar_t>(),
        nPsPerSegs.data<scalar_t>(),
        nSegs,
        nPixels,
        dFeats);
     }));

    return {res};
}

