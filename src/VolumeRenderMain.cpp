
#include <QApplication>
#include <QKeyEvent>
#include <iostream>
#include <fstream>
#include <sstream>


#include "VolumeRenderWidget.h"



int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: QtCudaOpenGL.exe ../../data/monkey_tsdf_float2_33.raw 33" << std::endl;
		return EXIT_FAILURE;
	}

	std::string filename = argv[1];

	size_t vol_width = 0;
	size_t vol_height = 0;
	size_t vol_depth = 0;

	if (argc > 4)
	{
		vol_width = atoi(argv[2]);
		vol_height = atoi(argv[3]);
		vol_depth = atoi(argv[4]);
	}
	else if (argc > 3)
	{
		vol_width = atoi(argv[2]);
		vol_height = atoi(argv[3]);
		vol_height = 1;
	}
	else
	{
		vol_width = vol_height = vol_depth = atoi(argv[2]);;
	}
	

	QApplication app(argc, argv);
	app.setApplicationName("Qt Point Cloud OpenGL View");
	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	//format.setMajorVersion(3);
	//format.setMinorVersion(3);
	format.setProfile(QSurfaceFormat::CoreProfile);
	format.setOption(QSurfaceFormat::DebugContext);
	QSurfaceFormat::setDefaultFormat(format);


	VolumeRenderWidget widget;
	widget.setup(filename, vol_width, vol_height, vol_depth);
	widget.setFixedSize(512, 512);
	widget.setWindowTitle("Volume Render Widget");
	widget.show();


	


	return app.exec();
}
