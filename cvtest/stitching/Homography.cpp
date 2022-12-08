#include "Homography.h"
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::cuda;

cv::cuda::GpuMat testStitchingTwoImagesByHomography() {
	cv::VideoCapture cap_1("d:/1.mp4");
	cv::VideoCapture cap_2("d:/2.mp4");
	for (;;)
	{
		cv::UMat seq_1, seq_2, hasil;
		cap_1 >> seq_1;
		cap_2 >> seq_2;
		cv::cuda::GpuMat src1, src2;
		src1.upload(seq_1);
		src2.upload(seq_2);
		cv::cuda::GpuMat result = stitchingTwoImagesByHomography(src1, src2);
		//show result
	}
	cap_1.release();
	cap_2.release();

}

cv::cuda::GpuMat stitchingTwoImagesByHomography2(cv::cuda::GpuMat& img1_gpu, cv::cuda::GpuMat& img2_gpu) {

	cv::cuda::GpuMat temp1, temp2;
	temp1 = img1_gpu.clone();
	temp2 = img2_gpu.clone();
	cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(9000);
	cv::cuda::GpuMat keypoints1GPU, keypoints2GPU;
	cv::cuda::GpuMat descriptors1GPU, descriptors2GPU;
	std::vector< cv::KeyPoint > keypoints_scene, keypoints_object;
	cv::Ptr< cv::cuda::DescriptorMatcher > matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
	std::vector< std::vector< cv::DMatch> > matches;

	cv::cuda::cvtColor(img1_gpu, img1_gpu, cv::COLOR_BGR2GRAY);
	cv::cuda::cvtColor(img2_gpu, img2_gpu, cv::COLOR_BGR2GRAY);

	orb->detectAndComputeAsync(img1_gpu, cv::noArray(), keypoints1GPU, descriptors1GPU, false);
	orb->detectAndComputeAsync(img2_gpu, cv::noArray(), keypoints2GPU, descriptors2GPU, false);
	orb->convert(keypoints1GPU, keypoints_object);
	orb->convert(keypoints2GPU, keypoints_scene);

	//cout << "KPTS = " << keypoints_scene.size() << endl;

	matcher->knnMatch(descriptors1GPU, descriptors2GPU, matches, 2);

	std::vector< cv::DMatch > good_matches;

	for (int z = 0; z < std::min(keypoints_object.size() - 1, matches.size()); z++)
	{
		if (matches[z][0].distance < 0.75 * (matches[z][1].distance))
		{
			good_matches.push_back(matches[z][0]);
		}
	}

	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;
	for (int y = 0; y < good_matches.size(); y++)
	{
		obj.push_back(keypoints_object[good_matches[y].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[y].trainIdx].pt);
	}

	//cout << "Match points = " << good_matches.size() << endl;

	cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC);

	cv::cuda::GpuMat result, H_gpu, store;
	H_gpu.upload(H);
	cv::UMat result_mat, cek_mat, dst;

	// cv::cuda::warpPerspective(temp1, result, H, cv::Size(temp2.cols + temp1.cols, temp2.rows));
	//cv::cuda::warpPerspective(temp1, result, H, cv::Size(1400, temp2.rows));
	cv::cuda::warpPerspective(temp1, result, H, cv::Size(1500, temp2.rows));
	//result.copyTo(store);
	//cv::cuda::GpuMat half(result, cv::Rect(0, 0, temp2.cols, temp2.rows));
	//temp2.copyTo(half);

	keypoints_object.clear();
	keypoints_scene.clear();
	matcher.release();

	return result;

}


bool first = true;
Mat homo;
GpuMat stitchingTwoImagesByHomography(GpuMat& img1_gpu, GpuMat& img2_gpu) {

	GpuMat gray_image1;
	GpuMat gray_image2;

	//Covert to Grayscale
	cv::cuda::cvtColor(img1_gpu, gray_image1, cv::COLOR_BGR2GRAY);
	cv::cuda::cvtColor(img2_gpu, gray_image2, cv::COLOR_BGR2GRAY);

	//Find the Homography Matrix
	if (1) {
#if 1
		//--Step 1 : Detect the keypoints using SURF Detector
		auto start = std::chrono::high_resolution_clock::now();
		SURF_CUDA surf = cv::cuda::SURF_CUDA::SURF_CUDA(400, 4, 3, false, 0.01f, false);
		//GpuMat img1, img2;
		//img1.upload(gray_image1);
		//img2.upload(gray_image2);
		// detecting keypoints & computing descriptors
		GpuMat keypoints1GPU, keypoints2GPU;
		GpuMat descriptors1GPU, descriptors2GPU;
		surf(gray_image1, GpuMat(), keypoints1GPU, descriptors1GPU);
		surf(gray_image2, GpuMat(), keypoints2GPU, descriptors2GPU);

		// matching descriptors
		Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
		vector<DMatch> matches;
		matcher->match(descriptors1GPU, descriptors2GPU, matches);

		// downloading results
		vector<KeyPoint> keypoints1, keypoints2;
		vector<float> descriptors1, descriptors2;
		surf.downloadKeypoints(keypoints1GPU, keypoints1);
		surf.downloadKeypoints(keypoints2GPU, keypoints2);
		surf.downloadDescriptors(descriptors1GPU, descriptors1);
		surf.downloadDescriptors(descriptors2GPU, descriptors2);
		//// drawing the results
		//Mat img_matches;
		//drawMatches(Mat(img1), keypoints1, Mat(img2), keypoints2, matches, img_matches);
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		std::cout << "*************Surf algho time is: " << elapsed.count() << " s\n";

		double max_dist = 0;
		double min_dist = 100;

		//--Quick calculation of min-max distances between keypoints
		for (int i = 0; i < descriptors1GPU.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		printf("-- Max dist : %f \n", max_dist);
		printf("-- Min dist : %f \n", min_dist);

		//--Use only "good" matches (i.e. whose distance is less than 3 X min_dist )
		std::vector< DMatch > good_matches;

		for (int i = 0; i < descriptors1GPU.rows; i++)
		{
			if (matches[i].distance < 3 * min_dist)
			{
				good_matches.push_back(matches[i]);
			}
		}
		std::vector< Point2f > obj;
		std::vector< Point2f > scene;
		if (good_matches.size() < 4) {
			return GpuMat();
		}
		for (int i = 0; i < good_matches.size(); i++)
		{
			//--Get the keypoints from the good matches
			obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
		}

		//-- Draw matches
		//Mat img_matches2;
		//drawMatches(img1_gpu, keypoints1, img2_gpu, keypoints2, good_matches, img_matches2, Scalar::all(-1),
		//	Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		Mat H = findHomography(obj, scene, RANSAC);
		if (H.rows == 0) {
			return GpuMat();
		}
		homo = H;
		first = false;
#else
		cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(9000);
		cv::cuda::GpuMat keypoints1GPU, keypoints2GPU;
		cv::cuda::GpuMat descriptors1GPU, descriptors2GPU;
		std::vector< cv::KeyPoint > keypoints_scene, keypoints_object;
		cv::Ptr< cv::cuda::DescriptorMatcher > matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
		std::vector< std::vector< cv::DMatch> > matches;

		orb->detectAndComputeAsync(gray_image1, cv::noArray(), keypoints1GPU, descriptors1GPU, false);
		orb->detectAndComputeAsync(gray_image2, cv::noArray(), keypoints2GPU, descriptors2GPU, false);
		orb->convert(keypoints1GPU, keypoints_object);
		orb->convert(keypoints2GPU, keypoints_scene);

		//cout << "KPTS = " << keypoints_scene.size() << endl;

		matcher->knnMatch(descriptors1GPU, descriptors2GPU, matches, 2);

		std::vector< cv::DMatch > good_matches;

		for (int z = 0; z < std::min(keypoints_object.size() - 1, matches.size()); z++)
		{
			if (matches[z][0].distance < 0.75 * (matches[z][1].distance))
			{
				good_matches.push_back(matches[z][0]);
			}
		}

		std::vector<cv::Point2f> obj;
		std::vector<cv::Point2f> scene;
		for (int y = 0; y < good_matches.size(); y++)
		{
			obj.push_back(keypoints_object[good_matches[y].queryIdx].pt);
			scene.push_back(keypoints_scene[good_matches[y].trainIdx].pt);
		}
		//cout << "Match points = " << good_matches.size() << endl;
		homo = cv::findHomography(obj, scene, cv::RANSAC);
		first = false;
#endif
	}



	// Use the homography Matrix to warp the images
	cv::cuda::GpuMat result;
	//cv::cuda::GpuMat gpuInput = cv::cuda::GpuMat(image1);
	cv::cuda::warpPerspective(img1_gpu, result, homo, cv::Size(img1_gpu.cols + img2_gpu.cols, img1_gpu.rows));
	cv::cuda::GpuMat half(result, cv::Rect(0, 0, img2_gpu.cols, img2_gpu.rows));
	img2_gpu.copyTo(half);

	//cv::Mat half(result, cv::Rect(0, 0, image2.cols, image2.rows));
	//image2.copyTo(half);

	//GpuMat imgResult(img1_gpu.rows, img1_gpu.cols + img2_gpu.cols, img1_gpu.type());

	//GpuMat roiImgResult_Left = imgResult(Rect(0, 0, img1_gpu.cols, img1_gpu.rows));
	//GpuMat roiImgResult_Right = imgResult(Rect(img1_gpu.cols, 0, img1_gpu.cols, img2_gpu.rows));

	//GpuMat roiImg1 = result(Rect(img2_gpu.cols, 0, img2_gpu.cols, img2_gpu.rows));
	//GpuMat roiImg2 = img2_gpu(Rect(0, 0, img2_gpu.cols, img2_gpu.rows));

	//roiImg2.copyTo(roiImgResult_Left); //Img2 will be on the left of imgResult
	//roiImg1.copyTo(roiImgResult_Right); //Img1 will be on the right of imgResult

	//									/* To remove the black portion after stitching, and confine in a rectangular region*/
	//Mat fresult(imgResult);
	//// vector with all non-black point positions
	//std::vector<cv::Point> nonBlackList;
	//nonBlackList.reserve(fresult.rows * fresult.cols);

	//// add all non-black points to the vector
	//// there are more efficient ways to iterate through the image
	//for (int j = 0; j < fresult.rows; ++j)
	//	for (int i = 0; i < fresult.cols; ++i)
	//	{
	//		// if not black: add to the list
	//		if (fresult.at<cv::Vec3b>(j, i) != cv::Vec3b(0, 0, 0))
	//		{
	//			nonBlackList.push_back(cv::Point(i, j));
	//		}
	//	}


	//// create bounding rect around those points
	//cv::Rect bb = cv::boundingRect(nonBlackList);

	//return fresult(bb);
	return result;
}
