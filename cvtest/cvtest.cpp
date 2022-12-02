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

using namespace cv;
using namespace std;

cv::cuda::GpuMat stitchingTwoImagesByEstimateRigid() {
	cv::cuda::GpuMat img1_gray_gpu, img2_gray_gpu;
	cv::cuda::cvtColor(img1_gpu, img1_gray_gpu, cv::COLOR_BGR2GRAY);
	cv::cuda::cvtColor(img2_gpu, img2_gray_gpu, cv::COLOR_BGR2GRAY);
	cv::cuda::GpuMat mask1;
	cv::cuda::GpuMat mask2;

	cv::cuda::threshold(img1_gray_gpu, mask1, 1, 255, cv::THRESH_BINARY);
	cv::cuda::threshold(img2_gray_gpu, mask2, 1, 255, cv::THRESH_BINARY);

	cv::cuda::SURF_CUDA detector;
	cv::cuda::GpuMat keypoints1_gpu, descriptors1_gpu;
	detector(img1_gray_gpu, mask1, keypoints1_gpu, descriptors1_gpu);

	std::vector<cv::KeyPoint> keypoints1;
	detector.downloadKeypoints(keypoints1_gpu, keypoints1);

	cv::cuda::GpuMat keypoints2_gpu, descriptors2_gpu;
	detector(img2_gray_gpu, mask2, keypoints2_gpu, descriptors2_gpu);

	std::vector<cv::KeyPoint> keypoints2;
	detector.downloadKeypoints(keypoints2_gpu, keypoints2);

	cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
		cv::cuda::DescriptorMatcher::createBFMatcher();

	std::vector<std::vector<cv::DMatch>> knn_matches;
	matcher->knnMatch(descriptors2_gpu, descriptors1_gpu, knn_matches, 2);

	std::vector<cv::DMatch> matches;
	std::vector<std::vector<cv::DMatch>>::const_iterator it;
	for (it = knn_matches.begin(); it != knn_matches.end(); ++it) {
		if (it->size() > 1 && (*it)[0].distance / (*it)[1].distance < 0.55) {
			matches.push_back((*it)[0]);
		}
	}

	std::vector<cv::Point2f> src_pts;
	std::vector<cv::Point2f> dst_pts;
	for (auto m : matches) {
		src_pts.push_back(keypoints2[m.queryIdx].pt);
		dst_pts.push_back(keypoints1[m.trainIdx].pt);
	}

	cv::Mat A = cv::estimateRigidTransform(src_pts, dst_pts, false);
	int height1 = img1.rows, width1 = img1.cols;
	int height2 = img2.rows, width2 = img2.cols;

	std::vector<std::vector<float>> corners1{ {0,0},{0,height1},{width1,height1},{width1,0} };
	std::vector<std::vector<float>> corners2{ {0,0},{0,height2},{width2,height2},{width2,0} };

	std::vector<std::vector<float>> warpedCorners2(4, std::vector<float>(2));
	std::vector<std::vector<float>> allCorners = corners1;

	for (int i = 0; i < 4; i++) {
		float cornerX = corners2[i][0];
		float cornerY = corners2[i][1];
		warpedCorners2[i][0] = A.at<double>(0, 0) * cornerX +
			A.at<double>(0, 1) * cornerY + A.at<double>(0, 2);
		warpedCorners2[i][1] = A.at<double>(1, 0) * cornerX +
			A.at<double>(1, 1) * cornerY + A.at<double>(1, 2);
		allCorners.push_back(warpedCorners2[i]);
	}

	float xMin = 1e9, xMax = -1e9;
	float yMin = 1e9, yMax = -1e9;
	for (int i = 0; i < 7; i++) {
		xMin = (xMin > allCorners[i][0]) ? allCorners[i][0] : xMin;
		xMax = (xMax < allCorners[i][0]) ? allCorners[i][0] : xMax;
		yMin = (yMin > allCorners[i][1]) ? allCorners[i][1] : yMin;
		yMax = (yMax < allCorners[i][1]) ? allCorners[i][1] : yMax;
	}
	int xMin_ = (xMin - 0.5);
	int xMax_ = (xMax + 0.5);
	int yMin_ = (yMin - 0.5);
	int yMax_ = (yMax + 0.5);

	cv::Mat translation = (cv::Mat_<double>(3, 3) << 1, 0, -xMin_, 0, 1, -yMin_, 0, 0, 1);

	cv::cuda::GpuMat warpedResImg;
	cv::cuda::warpPerspective(img1_gpu, warpedResImg, translation,
		cv::Size(xMax_ - xMin_, yMax_ - yMin_));

	cv::cuda::GpuMat warpedImageTemp;
	cv::cuda::warpPerspective(img2_gpu, warpedImageTemp, translation,
		cv::Size(xMax_ - xMin_, yMax_ - yMin_));
	cv::cuda::GpuMat warpedImage2;
	cv::cuda::warpAffine(warpedImageTemp, warpedImage2, A,
		cv::Size(xMax_ - xMin_, yMax_ - yMin_));

	cv::cuda::GpuMat mask;
	cv::cuda::threshold(warpedImage2, mask, 1, 255, cv::THRESH_BINARY);
	int type = warpedResImg.type();

	warpedResImg.convertTo(warpedResImg, CV_32FC3);
	warpedImage2.convertTo(warpedImage2, CV_32FC3);
	mask.convertTo(mask, CV_32FC3, 1.0 / 255);
	cv::Mat mask_;
	mask.download(mask_);

	cv::cuda::GpuMat dst(warpedImage2.size(), warpedImage2.type());
	cv::cuda::multiply(mask, warpedImage2, warpedImage2);

	cv::Mat diff_ = cv::Scalar::all(1.0) - mask_;
	cv::cuda::GpuMat diff(diff_);
	cv::cuda::multiply(diff, warpedResImg, warpedResImg);
	cv::cuda::add(warpedResImg, warpedImage2, dst);
	dst.convertTo(dst, type);

	return dst;
}


cv::cuda::GpuMat stitchingTwoImagesByHomography() {
}

int main()
{
	cv::namedWindow("example3", cv::WINDOW_AUTOSIZE);
	cv::VideoCapture cap;
	cap.open(0);
	
	cv::Mat frame;
	for (;;) {
		cap>>frame;
		if(frame.empty()) break;
		cv::imshow("example3", frame);
		if( cv::waitKey(33)>=0) break;
	}

	// Read images
	//Mat color = imread("../lena.jpg");
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
