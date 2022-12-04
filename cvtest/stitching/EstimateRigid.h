#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include  "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudawarping.hpp"

cv::cuda::GpuMat stitchingTwoImagesByEstimateRigid(cv::cuda::GpuMat& img1_gpu, cv::cuda::GpuMat& img2_gpu);