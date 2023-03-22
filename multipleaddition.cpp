#include "multipleaddition.h"
#include "readdcm.h"

cv::Mat ImageAddition(
	cv::Mat& src1, 
	cv::Mat& src2
){
	if (src1.size() != src2.size() || src1.type() != src2.type())
		return cv::Mat();


	cv::Mat result = cv::Mat::zeros(src1.size(), CV_64FC1);

	for (int i = 0; i < result.rows; i++)
	{
		for (int j = 0; j < result.cols; j++)
		{
			result.at<double>(i, j) = (src1.at<double>(i, j) + src2.at<double>(i, j)) / 2;
		}
	}

	return result;
}

cv::Mat MultipleAddtion(
	std::string filefolder
){
	std::vector<std::string> fileNames;
	cv::String patternDcm = filefolder + "/*.dcm";
	cv::glob(patternDcm, fileNames, false);

	cv::Mat result = cv::Mat::zeros(ReadDcm(fileNames[0]).size(), CV_64FC1);

	for (auto it = fileNames.begin(); it != fileNames.end(); it++)
	{
		cv::Mat dcm = ReadDcm(*it);
		dcm.convertTo(dcm, CV_64FC1);
		result = ImageAddition(result, dcm);
	}

	result.convertTo(result, CV_16UC1);

	return result;
}