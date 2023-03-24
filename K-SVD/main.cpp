#include <opencv2/opencv.hpp>
#include "readdcm.h"
#include "savedcm.h"
#include "utilities.h"
#include "ksvd.h"
#include "omp.h"


int main()
{
	//读取图像
	cv::Mat src = ReadDcm("");
	std::vector<float> srcVec = Mat2Vector(src);

	float sigma = 30.0;
	int imgWidth = src.cols;
	int imgHeight = src.rows;
	int patchWidth = 8;
	int patchHeight = 8;
	int slidingStep = 3;
	int interationOfKSVD = 10;
	int numberOfAtomsOfDictionary = 512;
	int featureSize = patchWidth * patchHeight;
	int numberOfCoefficients = 10;

	//把分割图像成块
	std::vector<float> srcPatches = Img2Patches(srcVec, imgWidth, imgHeight, patchWidth, patchHeight, slidingStep);

	//设置KSVD的参数
	ksvdPara kPara;
	kPara.interation = interationOfKSVD;
	kPara.atoms = numberOfAtomsOfDictionary;
	kPara.featureSize = featureSize;
	kPara.sparsityThreshold = 5;

	//初始化字典
	std::vector<float> dictionary = ksvdInitializeDictionary(srcPatches, kPara);

	//展示字典
	showDictionary(dictionary, kPara.featureSize, kPara.atoms);

	//开始KSVD的字典学习
	std::vector<float> sparseCode(kPara.atoms * srcPatches.size() / featureSize);
	KSVD(sparseCode, dictionary, srcPatches, kPara);

	showDictionary(dictionary, kPara.featureSize, kPara.atoms);

	//稀疏编码
	//设置稀疏编码OMP的参数
	ompPara oPara;
	oPara.atoms = kPara.atoms;
	oPara.sparsityLevel = numberOfCoefficients;
	oPara.featureSize = featureSize;
	sparseCode = omp(srcPatches, dictionary, oPara);

	//重建图像
	std::vector<float> dstPatches = matrixMultiply(dictionary, sparseCode, featureSize, oPara.atoms, srcPatches.size() / oPara.featureSize);

	//转换成图像
	std::vector<float> weight(imgWidth * imgHeight, 0);
	std::vector<float> dstVec = Patch2Img(dstPatches, weight, patchWidth, patchWidth, sqrt(dstPatches.size() / featureSize), sqrt(dstPatches.size() / featureSize), slidingStep);

	//优化 ||Y - X||
	for (int i = 0; i < dstVec.size(); ++i)
		dstVec[i] = (srcVec[i] + 0.034 * sigma * dstVec[i]) / (1 + 0.034 * sigma * weight[i]);

	//转换成灰度图
	cv::Mat dst = Vector2Mat(dstVec, src.rows, src.cols);

	system("pause");
	return 0;
}