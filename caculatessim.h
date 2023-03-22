#ifndef _CACULATE_SSIM_H_
#define _CACULATE_SSIM_H_
#include <opencv2/opencv.hpp>

double SSIM(
	cv::Mat src1, 
	cv::Mat src2
);

double MSSIM(
	cv::Mat src1,
	cv::Mat src2,
	int patchSize
);

#endif // !_CACULATE_SSIM_H_
