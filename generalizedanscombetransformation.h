#ifndef _GENERALIZED_ANSCOMBE_TRANSFORMATION_H_
#define _GENERALIZED_ANSCOMBE_TRANSFORMATION_H_
#include <opencv2/opencv.hpp>

void GeneralizedAnscombeTransformation_Forward(
	cv::Mat src,
	cv::Mat& dst,
	double sigma,
	double alpha = 1.0,
	double mu = 0.0
);

void GeneralizedAnscombeTransformation_Forward(
	double& pixelValue,
	double sigma,
	double alpha = 1.0,
	double mu = 0.0
);

void GeneralizedAnscombeTransformation_ClosedFormInverse(
	cv::Mat src,
	cv::Mat& dst,
	double sigma,
	double alpha = 1.0,
	double mu = 0.0
);

void GeneralizedAnscombeTransformation_ClosedFormInverse(
	double& pixelValue,
	double sigma,
	double alpha = 1.0,
	double mu = 0.0
);

void AnscombeTransformation_Forward(
	cv::Mat src,
	cv::Mat& dst
);

void AnscombeTransformation_Forward(
	double& pixelValue
);

void AnscombeTransformation_Inverse(
	cv::Mat src,
	cv::Mat& dst
);

void AnscombeTransformation_Inverse(
	double& pixelValue
);
#endif // !_GENERALIZED_ANSCOMBE_TRANSFORMATION_H_
