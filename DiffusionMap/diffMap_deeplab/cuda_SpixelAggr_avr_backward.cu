/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"


/*
 * Device code
 */


void __global__ feat_aggr_avr_ker(float *res, const float *d_dzdy, const float *d_SegsIdx, const float *d_nPsSeg,
                        int nSegs, int nPixels, int dFeats)
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




// int const threadsPerBlock = pth * nfs;
// int const blocksPerGrid = imw * imh;
void __global__ conv_cls_ker_mt(float *res, const float *images, const float* indSet,
                            const float* filters, const float* biases,
                            int imw, int imh, int nch, int ptw, int pth, int ncs, int nfs)
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

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/ /*  x, segs, n_segs, coor_idx  */
    mxGPUArray const *Feats, *nPsPerSegs, *SegsIdx,*DzDy;
    float const *d_Feats, *d_nPsPerSegs, *d_SegsIdx, *d_dzdy;

    mxGPUArray *res;    
    float* d_res;    
    
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if ((nrhs!=4) || !(mxIsGPUArray(prhs[0]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    Feats = mxGPUCreateFromMxArray(prhs[0]);    
    SegsIdx = mxGPUCreateFromMxArray(prhs[1]);    
    nPsPerSegs = mxGPUCreateFromMxArray(prhs[2]);
    DzDy = mxGPUCreateFromMxArray(prhs[3]);   
    
    if (mxGPUGetClassID(Feats) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    d_Feats = (float const *)(mxGPUGetDataReadOnly(Feats));
    d_nPsPerSegs = (float const *)(mxGPUGetDataReadOnly(nPsPerSegs));
    d_SegsIdx = (float const *)(mxGPUGetDataReadOnly(SegsIdx));
    d_dzdy = (float const *)(mxGPUGetDataReadOnly(DzDy)); 

    /* Get the dimensions of input data */
    const mwSize *f_dim = mxGPUGetDimensions(Feats);    
    int nPixels = f_dim[0] * f_dim[1];
    int dFeats = f_dim[2];

    const mwSize *s_dim = mxGPUGetDimensions(DzDy); 
    int nSegs = s_dim[0]; /* number of segments */
 
    int ndim = 2;    
    mwSize dims[2];
    dims[0] = nPixels;
    dims[1] = dFeats;    

    /* printf("%d %d %d /n", nPixels, dFeats, nSegs);*/
  
    /* Create a GPUArray to hold the result and get its underlying pointer. */
    res = mxGPUCreateGPUArray(ndim,
                            dims,
                            mxSINGLE_CLASS,
                            mxREAL,
                            MX_GPU_INITIALIZE_VALUES);
    d_res = (float *)(mxGPUGetData(res));
   
    int const threadsPerBlock = 512; 
    int blocksPerGrid = (nPixels * dFeats + threadsPerBlock - 1) / threadsPerBlock;
    feat_aggr_avr_ker<<<blocksPerGrid, threadsPerBlock>>>(d_res, d_dzdy, d_SegsIdx, d_nPsPerSegs, nSegs, nPixels, dFeats);
    
   
    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(res);
   
    mxGPUDestroyGPUArray(Feats);
    mxGPUDestroyGPUArray(res);
    mxGPUDestroyGPUArray(SegsIdx);
    mxGPUDestroyGPUArray(DzDy);
    mxGPUDestroyGPUArray(nPsPerSegs);    
}
