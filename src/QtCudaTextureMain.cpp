
#include <QApplication>
#include <QKeyEvent>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "helper_image.h"
#include "CudaKernels/CudaKernels.h"
#include "QImageWidget.h"



int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: QtCudaTexture.exe ../../data/input.jpg ../../data/output.jpg" << std::endl;
		return EXIT_FAILURE;
	}


	QApplication app(argc, argv);
	app.setApplicationName("Qt Point Cloud OpenGL View");
	
	QImage inputImage, outputImage;
	if (!inputImage.load(argv[1]))
	{
		std::cout << "Error: Could not load file image: " << argv[1] << std::endl;
		return EXIT_FAILURE;
	}

	outputImage = inputImage;



	unsigned int* hImage = (unsigned int*)inputImage.bits();
	unsigned int width = inputImage.width();
	unsigned int height = inputImage.height();

	size_t pitch;

	unsigned int *dInputImage = nullptr;
	// copy image data to array
	checkCudaErrors(cudaMallocPitch(&dInputImage, &pitch, sizeof(uint)*width, height));
	checkCudaErrors(cudaMemcpy2D(
		dInputImage, 
		pitch, 
		hImage, 
		sizeof(unsigned int) * width,
		sizeof(unsigned int) * width,
		height, 
		cudaMemcpyHostToDevice));

	

	unsigned int *dOutputImage;
	checkCudaErrors(cudaMallocPitch(
		&dOutputImage, 
		&pitch, 
		width * sizeof(unsigned int), 
		height));


	passthrough_texture(dOutputImage, dInputImage, width, height, pitch);


	cudaMemcpy2D(
		outputImage.bits(), 
		sizeof(unsigned int) * width, 
		dOutputImage,
		pitch,
		sizeof(unsigned int) * width, 
		height, 
		cudaMemcpyDeviceToHost);


	checkCudaErrors(cudaFree(dInputImage));
	checkCudaErrors(cudaFree(dOutputImage));


	QImageWidget inputWidget;
	inputWidget.setImage(inputImage);
	inputWidget.move(0, 0);
	inputWidget.setWindowTitle(argv[1]);
	inputWidget.show();


	QImageWidget outputWidget;
	outputWidget.setImage(outputImage);
	outputWidget.move(inputWidget.width(), 0);
	outputWidget.setWindowTitle(argv[2]);
	outputWidget.show();


	

	return app.exec();
}
