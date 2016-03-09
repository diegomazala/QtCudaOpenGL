#include "CudaKernels.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/device_vector.h>

extern "C"
{

	__global__ void update_vb(float *verts, int vertex_count, float timeElapsed)
	{
		const unsigned long long int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadId < vertex_count * 4)
		{
			float valx = verts[threadId * 4 + 0];
			float valy = verts[threadId * 4 + 1];
			float valz = verts[threadId * 4 + 2];


			verts[threadId * 4 + 0] = valx * timeElapsed;
			verts[threadId * 4 + 1] = valy * timeElapsed;
			verts[threadId * 4 + 2] = valz * timeElapsed;
		}
	}

	void cuda_kernel(float *verts, int vertex_count, float timeElapsed)
	{
		if (vertex_count > 1024)
			update_vb << <vertex_count / 1024 + 1, 1024 >> >(verts, vertex_count, timeElapsed);
		else
			update_vb << <1, vertex_count >> >(verts, vertex_count, timeElapsed);
	}

};