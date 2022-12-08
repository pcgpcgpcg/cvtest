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
#include "stitching/Homography.h"


using namespace cv;
using namespace std;

int main()
{
	// Read videos
	cv::VideoCapture cap_1("d:/videoplayback1920x1080right.mp4");
	cv::VideoCapture cap_2("d:/videoplayback1920x1080left.mp4");
	for (int i = 0; i < 2000; i++)
	{
		auto start = std::chrono::high_resolution_clock::now();
		cv::UMat seq_1, seq_2;
		cap_1 >> seq_1;
		cap_2 >> seq_2;
		cv::cuda::GpuMat src1, src2;
		src1.upload(seq_1);
		src2.upload(seq_2);
		cv::cuda::GpuMat stitchingGpuMat = stitchingTwoImagesByHomography(src1, src2);
		//show result
		if (stitchingGpuMat.rows == 0) {
			continue;
		}
		cv::Mat result_(stitchingGpuMat.size(), stitchingGpuMat.type());
		stitchingGpuMat.download(result_);
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		std::cout << "*************Elapsed time new: " << elapsed.count() << " s\n";
		imshow("Stitching Result", result_);
		int key = waitKey(33);
		if (key == 27) {
			break;
		}
	}
	cap_1.release();
	cap_2.release();

	//// Read images
	//Mat image1 = imread("d:/road2.png");
	//Mat image2 = imread("d:/road1.png");
	//cv::cuda::GpuMat stitchingGpuMat = stitchingTwoImagesByHomography(cv::cuda::GpuMat(image1), cv::cuda::GpuMat(image2));
	////cv::cuda::GpuMat stitchingGpuMat = stitchingTwoImagesByHomography(cv::cuda::GpuMat(image1), cv::cuda::GpuMat(image2));
	//cv::Mat result_(stitchingGpuMat.size(), stitchingGpuMat.type());
	//stitchingGpuMat.download(result_);
	//imshow("Stitching Result", result_);
	waitKey(0);

	return 0;
}
