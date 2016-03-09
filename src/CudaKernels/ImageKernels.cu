/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>       // CUDA device initialization helper functions

texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;
texture<float, 2, cudaReadModeNormalizedFloat> grayTex;



__device__ uint rgbaFloatToInt(float4 rgba)
{
	rgba.x = __saturatef(fabs(rgba.x));   // clamp to [0.0, 1.0]
	rgba.y = __saturatef(fabs(rgba.y));
	rgba.z = __saturatef(fabs(rgba.z));
	rgba.w = __saturatef(fabs(rgba.w));
	return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) | (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}


__device__ float4 rgbaIntToFloat(uint c)
{
	float4 rgba;
	rgba.x = (c & 0xff) * 0.003921568627f;       //  /255.0f;
	rgba.y = ((c >> 8) & 0xff) * 0.003921568627f;  //  /255.0f;
	rgba.z = ((c >> 16) & 0xff) * 0.003921568627f; //  /255.0f;
	rgba.w = ((c >> 24) & 0xff) * 0.003921568627f; //  /255.0f;
	return rgba;
}


__global__ void
d_passthrough_texture(uint *od, int w, int h)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
	{
		return;
	}
	float4 pixel = tex2D(rgbaTex, x, y);
	float4 pixel_inverted;
	pixel_inverted.x = pixel.z;
	pixel_inverted.y = pixel.y;
	pixel_inverted.z = pixel.x;
	od[y * w + x] = rgbaFloatToInt(pixel_inverted);
	return;
}




extern "C"
void passthrough_texture(unsigned int *dOutputImage, unsigned int *dInputImage, int width, int height, int pitch)
{
	// Bind the array to the texture
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
	checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dInputImage, desc, width, height, pitch));

	const dim3 threads_per_block(16, 16);
	dim3 num_blocks;
	num_blocks.x = (width + threads_per_block.x - 1) / threads_per_block.x;
	num_blocks.y = (height + threads_per_block.y - 1) / threads_per_block.y;
	
	d_passthrough_texture << <  num_blocks, threads_per_block >> >(dOutputImage, width, height);
}

