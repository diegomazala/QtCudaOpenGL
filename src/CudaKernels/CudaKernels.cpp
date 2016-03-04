#include "CudaKernels.h"
#include <cmath>





template<typename Type>
static void perspective_matrix(Type out[16], Type fovy, Type aspect_ratio, Type near_plane, Type far_plane)
{
	std::memset(out, 0, 16 * sizeof(Type));

	const Type y_scale = (Type)(1.0 / tan((fovy / 2.0)*(M_PI / 180.0)));
	const Type x_scale = y_scale / aspect_ratio;
	const Type depth_length = far_plane - near_plane;

	out[0] = x_scale;
	out[5] = y_scale;
	out[10] = -((far_plane + near_plane) / depth_length);
	out[14] = -1.0;
	out[11] = -((2 * near_plane * far_plane) / depth_length);

}

