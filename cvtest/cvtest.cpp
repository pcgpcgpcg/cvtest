// cvtest.cpp : Defines the entry point for the application.
//

#include "cvtest.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

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
