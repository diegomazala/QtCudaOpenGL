
#include "GLPointCloud.h"
#include <iostream>

#include "CudaKernels/CudaKernels.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


struct cudaGraphicsResource *cuda_vb_resource;
static int accum = 0;
static bool direction = true;


GLPointCloud::GLPointCloud() : GLModel()
{
}


GLPointCloud::~GLPointCloud()
{
	cleanupGL();
}


GLuint GLPointCloud::vertexBufferId() const
{
	return vertexBuf.bufferId();
}


void GLPointCloud::initGL()
{
	initializeOpenGLFunctions();

	vertexBuf.create();
}


void GLPointCloud::cleanupGL()
{
	vertexBuf.destroy();
}


void GLPointCloud::setVertices(const float* vertices, uint count, uint tuple_size)
{
	vertexCount = count;
	tupleSize = tuple_size;
	stride = sizeof(float) * tuple_size;


	vertexBuf.bind();
	vertexBuf.allocate(vertices, static_cast<int>(count * stride));
	
	cudaGraphicsGLRegisterBuffer(&cuda_vb_resource, vertexBuf.bufferId(), cudaGraphicsMapFlagsWriteDiscard);

}




void GLPointCloud::render(QOpenGLShaderProgram *program)
{
	if (!vertexBuf.isCreated())
		return;

	//////////////////////////////////////////
	//
	// begin Cuda code
	//
	float* verts;
	size_t num_bytes;
	cudaGraphicsMapResources(1, &cuda_vb_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&verts, &num_bytes, cuda_vb_resource);
	float dir = direction ? 1.01f : 0.99f;
		
	if (++accum % 10 == 0)
		direction = !direction;

	cuda_kernel(verts, vertexCount, dir);
	cudaGraphicsUnmapResources(1, &cuda_vb_resource, 0);
	//
	// end Cuda code
	//
	//////////////////////////////////////////

    vertexBuf.bind();
	int vertexLocation = program->attributeLocation("in_position");
	program->enableAttributeArray(vertexLocation);
	program->setAttributeBuffer(vertexLocation, GL_FLOAT, 0, tupleSize, stride);


    // Draw geometry 
	glDrawArrays(GL_POINTS, 0, static_cast<int>(vertexCount * tupleSize));
}

