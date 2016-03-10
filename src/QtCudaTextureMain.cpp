
#include <QApplication>
#include <QKeyEvent>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "helper_image.h"
#include "CudaKernels/CudaKernels.h"
#include "QImageWidget.h"


template<typename PixelType>
void run_test(const QImage& inputImage, QImage& outputPassImage, QImage& outputInvImage)
{
	PixelType* hImage = (PixelType*)(inputImage.bits());
	unsigned int width = inputImage.width();
	unsigned int height = inputImage.height();

	size_t pitch;

	PixelType* dInputImage = nullptr;
	// copy image data to array
	checkCudaErrors(cudaMallocPitch(&dInputImage, &pitch, sizeof(PixelType) * width, height));
	checkCudaErrors(cudaMemcpy2D(
		dInputImage,
		pitch,
		hImage,
		sizeof(PixelType) * width,
		sizeof(PixelType) * width,
		height,
		cudaMemcpyHostToDevice));



	uchar* dOutputImage;
	checkCudaErrors(cudaMallocPitch(
		&dOutputImage,
		&pitch,
		width * sizeof(PixelType),
		height));


	if (sizeof(PixelType) == 1)
		passthrough_texture_uchar((uchar*)dOutputImage, (uchar*)dInputImage, width, height, pitch, false);
	else
		passthrough_texture_uint((uint*)dOutputImage, (uint*)dInputImage, width, height, pitch, false);


	cudaMemcpy2D(
		outputPassImage.bits(),
		sizeof(PixelType) * width,
		dOutputImage,
		pitch,
		sizeof(PixelType) * width,
		height,
		cudaMemcpyDeviceToHost);


	if (sizeof(PixelType) == 1)
		passthrough_texture_uchar((uchar*)dOutputImage, (uchar*)dInputImage, width, height, pitch, true);
	else
		passthrough_texture_uint((uint*)dOutputImage, (uint*)dInputImage, width, height, pitch, true);


	cudaMemcpy2D(
		outputInvImage.bits(),
		sizeof(PixelType) * width,
		dOutputImage,
		pitch,
		sizeof(PixelType) * width,
		height,
		cudaMemcpyDeviceToHost);


	checkCudaErrors(cudaFree(dInputImage));
	checkCudaErrors(cudaFree(dOutputImage));
}



int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: QtCudaTexture.exe ../../data/input.jpg ../../data/output.jpg" << std::endl;
		return EXIT_FAILURE;
	}


	QApplication app(argc, argv);
	app.setApplicationName("Qt Point Cloud OpenGL View");
	
	QImage inputImage, outputPassthroughImage, outputInvertImage;
	if (!inputImage.load(argv[1]))
	{
		std::cout << "Error: Could not load file image: " << argv[1] << std::endl;
		return EXIT_FAILURE;
	}
	
	outputInvertImage = outputPassthroughImage = inputImage;


	if (inputImage.format() == QImage::Format_Indexed8)
		run_test<uchar>(inputImage, outputPassthroughImage, outputInvertImage);
	else
		run_test<uint>(inputImage, outputPassthroughImage, outputInvertImage);



	QImageWidget inputWidget;
	inputWidget.setImage(inputImage);
	inputWidget.move(0, 0);
	inputWidget.setWindowTitle("Input");
	inputWidget.show();


	QImageWidget outputPassWidget;
	outputPassWidget.setImage(outputPassthroughImage);
	outputPassWidget.move(0, inputWidget.height());
	outputPassWidget.setWindowTitle("Output Passthrough");
	outputPassWidget.show();

	QImageWidget outputInvWidget;
	outputInvWidget.setImage(outputInvertImage);
	outputInvWidget.move(inputWidget.width(), inputWidget.height());
	outputInvWidget.setWindowTitle("Output Inverted");
	outputInvWidget.show();

	

	return app.exec();
}
