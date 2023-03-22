#pragma execution_character_set("gbk")
#include <opencv2/opencv.hpp>
#include <highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <xphoto/bm3d_image_denoising.hpp>
#include <xphoto/dct_image_denoising.hpp>
#include "readdcm.h"
#include "savedcm.h"
#include "multipleaddition.h"
#include "caculatessim.h"
#include "generalizedanscombetransformation.h"
#include "getnoiselevel.h"
#include <iostream>
#include <ctime>
#include <fstream>

clock_t start, end;


double timeuse;
const std::string resultPath = "C:\\Users\\EDY\\Desktop\\result.csv";

/*float sigma = 1000.0f;
int templateWindowSize = 8;
int searchWindowSize = 32;
int blockMatchingStep1 = 25000;
int blockMatchingStep2 = 4000;
int groupSize = 32;
int slidingStep = 1;
float beta = 2.0f;
int normType = cv::NORM_L1;
int step = cv::xphoto::BM3D_STEPALL;
int transformType = cv::xphoto::HAAR;*/

void mynormalize(
	cv::Mat src,
	cv::Mat dst,
	double alpha = 0.0,
	double beta = 65535.0
);

cv::Mat logTransform(
	cv::Mat src,
	double c
);

void addGaussionNoise(
	cv::Mat src,
	cv::Mat& dst,
	double mean,
	double sigma
) {
	cv::RNG rng;
	rng.fill(dst, cv::RNG::NORMAL, mean, sigma);
	dst = dst + src;
}

template<class T>
void SaveCsv(
	cv::Mat m,
	std::string fileName
) {
	std::fstream ofs;
	ofs.open(fileName);
	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			ofs << m.at<T>(cv::Point(j, i)) << ",";
		}
		ofs << "\n";
	}
	ofs.close();
}

void MouseEvent(
	int event,
	int x,
	int y,
	int flag,
	void* params
) {
	cv::Point* ptr = (cv::Point*)params;

	if (event == cv::EVENT_LBUTTONDOWN && ptr[0].x == -1 && ptr[0].y == -1)
	{
		ptr[0].x = x;
		ptr[0].y = y;
	}
	if (event == cv::EVENT_FLAG_LBUTTON || event == cv::EVENT_MOUSEMOVE)
	{
		ptr[1].x = x;
		ptr[1].y = y;
	}
	if (event == cv::EVENT_LBUTTONUP && ptr[2].x == -1 && ptr[2].y == -1)
	{
		ptr[2].x = x;
		ptr[2].y = y;
	}
}

void ROISelect(
	cv::Mat& src,
	cv::Mat& roi
) {
	cv::Point* Corners = new cv::Point[3];
	Corners[0].x = Corners[0].y = -1;
	Corners[1].x = Corners[1].y = -1;
	Corners[2].x = Corners[2].y = -1;

	cv::namedWindow("选取执行算法的区域", cv::WINDOW_FREERATIO);
	cv::imshow("选取执行算法的区域", src);

	bool downFlag = false, upFlag = false;
	while (cv::waitKey(1) != 13)
	{
		cv::setMouseCallback("选取执行算法的区域", MouseEvent, Corners);

		if (Corners[0].x != -1 && Corners[0].y != -1)
			downFlag = true;

		if (Corners[2].x != -1 && Corners[2].y != -1) 
			upFlag = true; 

		if (downFlag && !upFlag && Corners[1].x != -1)
		{
			cv::Mat LocalImage = src.clone();
			cv::rectangle(LocalImage, cv::Rect(Corners[0], Corners[1]), cv::Scalar(65535, 0, 0), 3, cv::LINE_AA);
			cv::imshow("选取执行算法的区域", LocalImage);
		}

		cv::Rect ROIBox;

		if (downFlag && upFlag)
		{
			ROIBox.width = abs(Corners[0].x - Corners[2].x);
			ROIBox.height = abs(Corners[0].y - Corners[2].y);

			if (ROIBox.width < 32 && ROIBox.height < 32)
			{
				std::cerr << "选取的ROI区域过小，请重新选择！" << std::endl;
			}


			ROIBox.x = Corners[0].x < Corners[1].x ? Corners[0].x : Corners[1].x;
			ROIBox.y = Corners[0].y < Corners[1].y ? Corners[0].y : Corners[1].y;

			roi = src(ROIBox);
			downFlag = upFlag = false;

			Corners[0].x = Corners[0].y = -1;
			Corners[1].x = Corners[1].y = -1;
			Corners[2].x = Corners[2].y = -1;

		}
	}
	cv::destroyWindow("选取执行算法的区域");
	delete[] Corners;
}

void ROISelect(
	cv::Mat& src,
	cv::Mat& gt,
	cv::Mat& src_roi,
	cv::Mat& gt_roi
) {
	cv::Point* Corners = new cv::Point[3];
	Corners[0].x = Corners[0].y = -1;
	Corners[1].x = Corners[1].y = -1;
	Corners[2].x = Corners[2].y = -1;

	cv::namedWindow("选取执行算法的区域", cv::WINDOW_FREERATIO);
	cv::imshow("选取执行算法的区域", src);

	bool downFlag = false, upFlag = false;
	while (cv::waitKey(1) != 13)
	{
		cv::setMouseCallback("选取执行算法的区域", MouseEvent, Corners);

		if (Corners[0].x != -1 && Corners[0].y != -1)
			downFlag = true;

		if (Corners[2].x != -1 && Corners[2].y != -1)
			upFlag = true;

		if (downFlag && !upFlag && Corners[1].x != -1)
		{
			cv::Mat LocalImage = src.clone();
			cv::rectangle(LocalImage, cv::Rect(Corners[0], Corners[1]), cv::Scalar(65535, 0, 0), 3, cv::LINE_AA);
			cv::imshow("选取执行算法的区域", LocalImage);
		}

		cv::Rect ROIBox;

		if (downFlag && upFlag)
		{
			ROIBox.width = abs(Corners[0].x - Corners[2].x);
			ROIBox.height = abs(Corners[0].y - Corners[2].y);

			if (ROIBox.width < 32 && ROIBox.height < 32)
			{
				std::cerr << "选取的ROI区域过小，请重新选择！" << std::endl;
			}


			ROIBox.x = Corners[0].x < Corners[1].x ? Corners[0].x : Corners[1].x;
			ROIBox.y = Corners[0].y < Corners[1].y ? Corners[0].y : Corners[1].y;

			src_roi = src(ROIBox);
			gt_roi = gt(ROIBox);
			downFlag = upFlag = false;

			Corners[0].x = Corners[0].y = -1;
			Corners[1].x = Corners[1].y = -1;
			Corners[2].x = Corners[2].y = -1;

		}
	}
	cv::destroyWindow("选取执行算法的区域");
	delete[] Corners;
}

void BM3D(
	const cv::Mat groundTruth,
	const cv::Mat src,
	std::string outputFileName,
	float sigma = 2000.0f,
	int templateWindowSize = 8,
	int searchWindowSize = 16,
	int blockMatchingStep1 = 45000,
	int blockMatchingStep2 = 800,
	int groupSize = 24,
	int slidingStep = 1,
	float beta = 2.0f,
	int normType = cv::NORM_L1,
	int step = cv::xphoto::BM3D_STEPALL,
	int transformType = cv::xphoto::HAAR
){
	cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
	start = clock();
	cv::xphoto::bm3dDenoising(src, dst, sigma, templateWindowSize, searchWindowSize,
		blockMatchingStep1, blockMatchingStep2, groupSize, slidingStep,
		beta, normType, step, transformType);

	end = clock();
	
	double timeuse = (double)(end - start) / CLOCKS_PER_SEC;
	std::cout << "BM3D Algorithm Running Time = " << timeuse << "s" << std::endl;
	SaveDcm(dst, outputFileName);

	double psnr = cv::PSNR(groundTruth, dst, 65535.0);
	std::cout << "PSNR = " << psnr << "\n";

	double ssim = SSIM(groundTruth, dst);
	std::cout << "SSIM = " << ssim << "\n";

	std::fstream ofs;
	ofs.open(resultPath, std::ios::app);
	ofs << sigma << ","
		<< templateWindowSize << ","
		<< searchWindowSize << ","
		<< blockMatchingStep1 << ","
		<< blockMatchingStep2 << ","
		<< groupSize << ","
		<< slidingStep << ","
		<< psnr << ","
		<< ssim << ","
		<< timeuse << "," << "\n";
	ofs.close();
}

void BM3Dtest()
{
	std::string fileFolder = "D:/资料/reference/FILESOFME/Thy";
	/*cv::Mat gt = MultipleAddtion(fileFolder);
	SaveDcm(gt, fileFolder + "\\gt.dcm");*/
	std::string gtName = fileFolder + "/手gt.dcm";
	cv::Mat gt = ReadDcm(gtName);

	std::string filePath = fileFolder + "/手.dcm";
	//std::string filePath = fileFolder + "\\5020手表\\0001.dcm";
	cv::Mat src = ReadDcm(filePath);
	cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
	
	//cv::Mat src, dst, gt;
	//ROISelect(srcr, gtr, src, gt);

	std::cout << SSIM(gt, src) << "\n";
	
	std::fstream ofs;
	ofs.open(resultPath, std::ios::app);
	ofs << "sigma" << ","
		<< "templateWindowSize" << ","
		<< "searchWindowSize" << ","
		<< "blockMatchingStep1" << ","
		<< "blockMatchingStep2" << ","
		<< "groupSize" << ","
		<< "slidingStep" << ","
		<< "PSNR" << ","
		<< "SSIM" << ","
		<< "TIME" << "," << "\n";
	ofs.close();


	filePath = "D:/资料/reference/FILESOFME/Thy/手/手.dcm";
	for (double sigma = 0.f; sigma <= 20000.f;)
	{
		std::cout << "----------------------------------------h = " << sigma << "----------------------------------------" << "\n";
		std::string outputFileName = filePath;
		outputFileName.insert(filePath.find(".dcm"), "_sigma_" + std::to_string(sigma));
		BM3D(gt, src, outputFileName, sigma);
		std::cout << outputFileName << std::endl;
		std::cout << "\n";
		if (sigma < 500)
			sigma += 10;
		else if (sigma < 5000)
			sigma += 100;
		else
			sigma += 1000;
	}

	for (int templateWindowSize = 2; templateWindowSize <= 16; templateWindowSize *= 2)
	{
		std::cout << "----------------------------------------templateWindowSize = " << templateWindowSize << "----------------------------------------" << "\n";
		std::string outputFileName = filePath;
		outputFileName.insert(filePath.find(".dcm"), "_templateWindowSize_" + std::to_string(templateWindowSize));
		if (templateWindowSize < 16)
			BM3D(gt, src, outputFileName, 200.f, templateWindowSize);
		else if (templateWindowSize == 16)
			BM3D(gt, src, outputFileName, 200.f, templateWindowSize, 32);
		std::cout << outputFileName << std::endl;
		std::cout << "\n";

	}

	for (int searchWindowSize = 16; searchWindowSize <= 96; searchWindowSize += 8)
	{
		std::cout << "----------------------------------------searchWindowSize = " << searchWindowSize << "----------------------------------------" << "\n";
		std::string outputFileName = filePath;
		outputFileName.insert(filePath.find(".dcm"), "_searchWindowSize_" + std::to_string(searchWindowSize));
		BM3D(gt, src, outputFileName, 200.f, 8, searchWindowSize);
		std::cout << outputFileName << std::endl;
		std::cout << "\n";

	}

	for (int blockMatchingStep1 = 500; blockMatchingStep1 <= 100000;)
	{
		std::cout << "----------------------------------------blockMatchingStep1 = " << blockMatchingStep1 << "----------------------------------------" << "\n";
		std::string outputFileName = filePath;
		outputFileName.insert(filePath.find(".dcm"), "_blockMatchingStep1_" + std::to_string(blockMatchingStep1));
		BM3D(gt, src, outputFileName, 200.f, 8, 16, blockMatchingStep1);
		std::cout << outputFileName << std::endl;
		std::cout << "\n";
		if (blockMatchingStep1 < 5000)
			blockMatchingStep1 += 500;
		else if (blockMatchingStep1 < 50000)
			blockMatchingStep1 += 5000;
		else
			blockMatchingStep1 += 10000;
	}

	for (int blockMatchingStep2 = 50; blockMatchingStep2 <= 50000;)
	{
		std::cout << "----------------------------------------blockMatchingStep2 = " << blockMatchingStep2 << "----------------------------------------" << "\n";
		std::string outputFileName = filePath;
		outputFileName.insert(filePath.find(".dcm"), "_blockMatchingStep2_" + std::to_string(blockMatchingStep2));
		BM3D(gt, src, outputFileName, 200.f, 8, 16, 45000, blockMatchingStep2);
		std::cout << outputFileName << std::endl;
		std::cout << "\n";
		if (blockMatchingStep2 < 500)
			blockMatchingStep2 += 50;
		else if (blockMatchingStep2 < 5000)
			blockMatchingStep2 += 500;
		else
			blockMatchingStep2 += 5000;
	}

	for (int groupSize = 8; groupSize <= 32; groupSize += 4)
	{
		std::cout << "----------------------------------------groupSize = " << groupSize << "----------------------------------------" << "\n";
		std::string outputFileName = filePath;
		outputFileName.insert(filePath.find(".dcm"), "_groupSize_" + std::to_string(groupSize));
		BM3D(gt, src, outputFileName, 200.f, 8, 16, 45000, 800, groupSize);
		std::cout << outputFileName << std::endl;
		std::cout << "\n";
	}

	for (int slidingStep = 1; slidingStep <= 4; slidingStep++)
	{
		std::cout << "----------------------------------------slidingStep = " << slidingStep << "----------------------------------------" << "\n";
		std::string outputFileName = filePath;
		outputFileName.insert(filePath.find(".dcm"), "_slidingStep_" + std::to_string(slidingStep));
		BM3D(gt, src, outputFileName, 200.f, 8, 16, 45000, 800, 24, slidingStep);
		std::cout << outputFileName << std::endl;
		std::cout << "\n";
	}
}

int main3()
{
	//BM3Dtest();
	std::string filePath = "D:/资料/reference/FILESOFME/Thy/手.dcm";
	//std::string filePath = "D:/资料/reference/FILESOFME/Thy/5020芯片/0001.dcm";
	cv::Mat src = ReadDcm(filePath);
	cv::Mat gt = ReadDcm("D:/资料/reference/FILESOFME/Thy/手gt.dcm");
	//cv::Mat gt = ReadDcm("D:/资料/reference/FILESOFME/Thy/5020芯片gt.dcm");
	cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
	
	//GeneralizedAnscombeTransformation_Forward(src, src, 50);
	
	/*std::fstream ofs;
	ofs.open(resultPath, std::ios::app);
	ofs << "sigma" << ","
		<< "patchSize" << ","
		<< "psnr" << ","
		<< "ssim" << ","
		<< "timeuse" << "," << "\n";
	ofs.close();*/
	
		start = clock();
		//cv::xphoto::dctDenoising(src, dst, 200);


		/*std::vector<float> h;
		h.push_back(100.0);
		cv::fastNlMeansDenoising(src, dst, h, 7, 21, cv::NORM_L1);*/


		/*src.convertTo(src, CV_32FC1);
		dst.convertTo(dst, CV_32FC1);
		cv::bilateralFilter(src, dst, 5, 100, 100);*/
		cv::boxFilter(src, dst, -1, cv::Size(13, 13));

		end = clock();
		double timeuse = (double)(end - start) / CLOCKS_PER_SEC;
		std::cout << "DCT Algorithm Running Time = " << timeuse << "s" << std::endl;

		dst.convertTo(dst, CV_16UC1);
		double psnr = cv::PSNR(gt, dst, 65535.0);
		std::cout << "PSNR = " << psnr << "\n";

		double ssim = SSIM(gt, dst);
		std::cout << "SSIM = " << ssim << "\n";

		/*std::fstream ofs;
		ofs.open(resultPath, std::ios::app);
		ofs << 1000 << ","
			<< 16 << ","
			<< psnr << ","
			<< ssim << ","
			<< timeuse << "," << "\n";
		ofs.close();*/
		
		//SaveDcm(dst, "C:/Users/EDY/Desktop/box.dcm");
		
	
	//GeneralizedAnscombeTransformation_ClosedFormInverse(dst, dst, 50);
		return 0;

}

int main()
{
	/*cv::Mat src = cv::Mat::ones(cv::Size(60, 10), CV_16UC1);
	for (int i = 0; i < 10; i++)
		for (int j = 0; j < 60; j++)
			src.at<ushort>(i, j) = rand() % 99 + 1;*/
	cv::Mat src = ReadDcm("P0606A 40kV 70uA 20SOD 320SID High Gain 10Fps 10Avg 1x1 CSI 芯片 bongdwire.dcm");
	float sigma = GetNoiseLevel(src, 7, 3);
	std::cout << sigma << "\n";



	/*cv::Mat src = cv::Mat::ones(cv::Size(20, 1), CV_32FC1);
	for (int i = 0; i < 1; i++)
	for (int j = 0; j < 20; j++)
		src.at<float>(i, j) = ((float)(rand() % 999 + 1)) / 10;*/
	/*std::cout << src << "\n";

	float tau = 50;
	cv::threshold(src, src, tau, 1, cv::THRESH_TOZERO);
	std::vector<cv::Point> index;
	cv::findNonZero(src, index);

	cv::Mat Xtrtemp = cv::Mat::zeros(cv::Size(index.size(), src.rows), src.type());
	for (int j = 0; j < index.size(); ++j)
		src.col(index[j].x).copyTo(Xtrtemp.col(j));
	src = Xtrtemp;
	Xtrtemp.release();
	std::cout << src << "\n";*/
	
	return 0;
}

int main2()
{
	std::string workSpace = "D:/资料/reference/FILESOFME/Thy/5020芯片";
	std::string srcPath = workSpace + "/0001.dcm";
	cv::Mat src = ReadDcm(srcPath);
	cv::Mat src_GAT = cv::Mat::zeros(src.size(), src.type());
	cv::Mat dst_GAT = cv::Mat::zeros(src.size(), src.type());
	cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
	cv::Mat src_roi, dst_roi;
	
	ROISelect(src, src_roi);

	//GeneralizedAnscombeTransformation_Forward(src, src_GAT, 100.0);

	float sigma = 100.0f;
	int templateWindowSize = 8;
	int searchWindowSize = 16;
	int blockMatchingStep1 = 10000;
	int blockMatchingStep2 = 800;
	int groupSize = 24;
	int slidingStep = 1;
	float beta = 2.0f;
	int normType = cv::NORM_L1;
	int step = cv::xphoto::BM3D_STEPALL;
	int transformType = cv::xphoto::HAAR;

	start = clock();
	cv::xphoto::bm3dDenoising(src_roi, dst_roi, sigma, templateWindowSize, searchWindowSize,
		blockMatchingStep1, blockMatchingStep2, groupSize, slidingStep,
		beta, normType, step, transformType);
	end = clock();
	timeuse = (double)(end - start) / CLOCKS_PER_SEC;
	std::cout << "BM3D: " << timeuse << "s" << "\n";
	
	
	//GeneralizedAnscombeTransformation_ClosedFormInverse(dst_GAT, dst, 100.0);
	
	//SaveCsv<UINT16>(dst, "dst.csv");

	std::string dstPath = workSpace + "/roi.dcm";
	SaveDcm(dst_roi, dstPath);
	
	
	cv::waitKey(0);
	return 0;
}




int main1()
{
	cv::Mat src = ReadDcm("P1613D  50kV 160uA SID SOD Gain0 10.00Fps 20Avg 1x1(1).dcm");
	cv::Mat dst;
	//dst.create(src.size(), CV_64F);
	//伽马变换
	/*double gamma = 0.05;
	src.convertTo(src, CV_64F, 1.0 / 65535.0, 0);
	int height = src.rows;
	int width = src.cols;
	dst.create(src.size(), CV_64F);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			dst.at<double>(i, j) = pow(src.at<double>(i, j), gamma);

		};
	};
	dst.convertTo(dst, CV_16U, 65535.0, 0);*/
	double c = 1;
	dst = logTransform(src, c);
	SaveDcm(dst, "dst_" + std::to_string(c) + ".dcm");
	return 0;
}

void mynormalize(
	cv::Mat src,
	cv::Mat dst,
	double alpha,
	double beta 
) {
	dst.convertTo(dst, CV_64FC1);
	double* min = new double;
	double* max = new double;
	cv::minMaxIdx(src, min, max);
	/*std::cout << *min << std::endl;
	std::cout << *max << std::endl;*/

	dst = (beta - alpha) / (*max - *min) * (dst - *min) + alpha;
	delete min;
	delete max;
	dst.convertTo(dst, CV_16UC1);
}

cv::Mat logTransform(
	cv::Mat src,
	double c
) {
	src.convertTo(src, CV_64F);
	cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
	cv::add(src, cv::Scalar(1.0), src);
	cv::log(src, dst);
	dst = c * dst;

	mynormalize(dst, dst);
	dst.convertTo(dst, CV_16UC1);
	return dst;
}