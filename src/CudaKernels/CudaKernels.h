#ifndef _CUDA_KERNELS_H_
#define _CUDA_KERNELS_H_


#if _MSC_VER // this is defined when compiling with Visual Studio
#define CUDA_KERNELS_API __declspec(dllexport) // Visual Studio needs annotating exported functions with this
#else
#define CUDA_KERNELS_API // XCode does not need annotating exported functions, so define is empty
#endif

#ifdef _WIN32
#include <windows.h>
#endif



typedef unsigned char	uchar;
typedef unsigned int	uint;
typedef unsigned short	ushort;


extern "C"
{
	void cuda_kernel(float *verts, int vertex_count, float timeElapsed);
	
	void passthrough_texture_uint(
		uint* dOutputImage,
		uint* dInputImage,
		int width, 
		int height, 
		size_t pitch,
		bool invert_channel);

	void passthrough_texture_uchar(
		uchar* dOutputImage,
		uchar* dInputImage,
		int width,
		int height,
		size_t pitch, 
		bool invert_channel);
};





#endif // _CUDA_KERNELS_H_