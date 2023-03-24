#ifndef _READ_DCM_H
#define _READ_DCM_H
#include <stdlib.h>
#include <iostream>


#include <opencv2/opencv.hpp>
#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmimgle/dcmimage.h"

#pragma comment(lib, "Netapi32.lib")
#pragma comment(lib, "ws2_32.lib")

cv::Mat ReadDcm(std::string filepath);
#endif // !_READ_DCM_H
