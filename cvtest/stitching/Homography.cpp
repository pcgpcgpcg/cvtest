#include "Homography.h"

cv::cuda::GpuMat stitchingTwoImagesByHomography(cv::cuda::GpuMat& img1_gpu, cv::cuda::GpuMat& img2_gpu) {

    //cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(9000);
    //cv::cuda::GpuMat keypoints1GPU, keypoints2GPU;
    //cv::cuda::GpuMat descriptors1GPU, descriptors2GPU;
    //std::vector< KeyPoint > keypoints_scene, keypoints_object;
    //Ptr< cuda::DescriptorMatcher > matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    //vector< vector< DMatch> > matches;

    //cv::cuda::cvtColor(src2, src2, COLOR_BGR2GRAY);
    //cv::cuda::cvtColor(src1, src1, COLOR_BGR2GRAY);

    //orb->detectAndComputeAsync(src1, noArray(), keypoints1GPU, descriptors1GPU, false);
    //orb->detectAndComputeAsync(src2, noArray(), keypoints2GPU, descriptors2GPU, false);
    //orb->convert(keypoints1GPU, keypoints_object);
    //orb->convert(keypoints2GPU, keypoints_scene);

    ////cout << "KPTS = " << keypoints_scene.size() << endl;

    //matcher->knnMatch(descriptors1GPU, descriptors2GPU, matches, 2);

    //std::vector< DMatch > good_matches;

    //for (int z = 0; z < std::min(keypoints_object.size() - 1, matches.size()); z++)
    //{
    //    if (matches[z][0].distance < 0.75 * (matches[z][1].distance))
    //    {
    //        good_matches.push_back(matches[z][0]);
    //    }
    //}

    //std::vector<Point2f> obj;
    //std::vector<Point2f> scene;
    //for (int y = 0; y < good_matches.size(); y++)
    //{
    //    obj.push_back(keypoints_object[good_matches[y].queryIdx].pt);
    //    scene.push_back(keypoints_scene[good_matches[y].trainIdx].pt);
    //}

    ////cout << "Match points = " << good_matches.size() << endl;

    //Mat H = findHomography(obj, scene, RANSAC);

    //GpuMat result, H_gpu, store;
    //H_gpu.upload(H);
    //UMat result_mat, cek_mat, dst;

    //// cv::cuda::warpPerspective(temp1, result, H, cv::Size(temp2.cols + temp1.cols, temp2.rows));
    ////cv::cuda::warpPerspective(temp1, result, H, cv::Size(1400, temp2.rows));
    //cv::cuda::warpPerspective(temp1, result, H, cv::Size(1400, temp2.rows));
    ////result.copyTo(store);
    //GpuMat half(result, cv::Rect(0, 0, temp2.cols, temp2.rows));
    //temp2.copyTo(half);

    //result.download(result_mat);

    ////cout << result_mat.size() << endl;

    //imshow("Result Image", result_mat);
    const int rows = 16 * 50;
    const int cols = 16 * 60;
    const int type = CV_8UC3;

    return cv::cuda::GpuMat(rows, cols, type, cv::Scalar(0, 0, 0));
}
