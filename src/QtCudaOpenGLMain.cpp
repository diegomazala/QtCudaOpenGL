
#include <QApplication>
#include <QKeyEvent>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "CudaKernels/CudaKernels.h"

#include "GLBaseWidget.h"
#include "GLPointCloud.h"




template<typename T>
static void vector_write(std::ostream& out_file, const std::vector<T>& data)
{
	const std::size_t count = data.size();
	out_file.write(reinterpret_cast<const char*>(&count), sizeof(std::size_t));
	out_file.write(reinterpret_cast<const char*>(&data[0]), count * sizeof(T));
}

template<typename T>
static void vector_read(std::istream& in_file, std::vector<T>& data)
{
	std::size_t count;
	in_file.read(reinterpret_cast<char*>(&count), sizeof(std::size_t));
	data.resize(count);
	in_file.read(reinterpret_cast<char*>(&data[0]), count * sizeof(T));
}


template<typename T>
bool load_buffer(const std::string& filename, std::vector<T>& data)
{
	std::ifstream in_file;
	in_file.open(filename, std::ifstream::binary);
	if (in_file.is_open())
	{
		vector_read(in_file, data);
		in_file.close();
		return true;
	}
	return false;
}


template<typename T>
void save_buffer(const std::string& filename, const std::vector<T>& data)
{
	std::ofstream out;
	out.open(filename, std::ofstream::binary);
	vector_write(out, data);
	out.close();
}




int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: QtCudaOpenGL.exe ../../data/vertices_4f.buf" << std::endl;
		return EXIT_FAILURE;
	}

	std::string filename = argv[1];

	std::vector<float4> src_pts, dst_pts;
	if (!load_buffer(filename, src_pts))
	{
		std::cerr << "Error: Could not load buffer from file " << filename << std::endl;
		return EXIT_FAILURE;
	}


	QApplication app(argc, argv);
	app.setApplicationName("Qt Point Cloud OpenGL View");
	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	QSurfaceFormat::setDefaultFormat(format);


	GLBaseWidget glwidget;
	glwidget.setMinimumSize(720, 433);
	glwidget.setWindowTitle("Point Cloud OpenGL View");
	glwidget.show();


	GLPointCloud pointCloud;
	pointCloud.initGL();
	pointCloud.setVertices(&src_pts[0].x, static_cast<uint>(src_pts.size()), static_cast<uint>(4));
	glwidget.setModel(&pointCloud);


	return app.exec();
}
