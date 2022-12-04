// cvtest.cpp : Defines the entry point for the application.
//

#include "cvtest.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include  "opencv2/features2d/features2d.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudawarping.hpp"
#include <opencv2/core/opengl.hpp>
#include "stitching/EstimateRigid.h"


using namespace cv;
using namespace std;

int main()
{
	// Read images
	Mat image1 = imread("d:/1.png");
	Mat image2 = imread("d:/2.png");
	cv::cuda::GpuMat stitchingGpuMat = stitchingTwoImagesByEstimateRigid(cv::cuda::GpuMat(image1), cv::cuda::GpuMat(image2));
	cv::Mat result_(stitchingGpuMat.size(), stitchingGpuMat.type());
	stitchingGpuMat.download(result_);
	imshow("Stitching Result", result_);
	//imshow("Stitching Result", cv::ogl::Texture2D(stitchingGpuMat));
	waitKey(0);
	//Mat gray = imread("../lena.jpg", IMREAD_GRAYSCALE);

	//if (!color.data) // Check for invalid input
	//{
	//	cout << "Could not open or find the image" << std::endl;
	//	return -1;
	//}

	//// Write images
	//imwrite("lenaGray.jpg", gray);

	//// Get same pixel with opencv function
	//int myRow = color.rows - 1;
	//int myCol = color.cols - 1;
	//auto pixel = color.at<Vec3b>(myRow, myCol);
	//cout << "Pixel value (B,G,R): (" << (int)pixel[0] << "," << (int)pixel[1] << "," << (int)pixel[2] << ")" << endl;

	//// show images
	//imshow("Lena BGR", color);
	//imshow("Lena Gray", gray);
	//// wait for any key press
	//waitKey(0);
	//cout << "Hello CMake." << endl;
	return 0;
}
