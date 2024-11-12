//
// Copyright 2014 Mitsubishi Electric Research Laboratories All
// Rights Reserved.
//
// Permission to use, copy and modify this software and its
// documentation without fee for educational, research and non-profit
// purposes, is hereby granted, provided that the above copyright
// notice, this paragraph, and the following three paragraphs appear
// in all copies.
//
// To request permission to incorporate this software into commercial
// products contact: Director; Mitsubishi Electric Research
// Laboratories (MERL); 201 Broadway; Cambridge, MA 02139.
//
// IN NO EVENT SHALL MERL BE LIABLE TO ANY PARTY FOR DIRECT,
// INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
// LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
// DOCUMENTATION, EVEN IF MERL HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES.
//
// MERL SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN
// "AS IS" BASIS, AND MERL HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE,
// SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
//
#pragma warning(disable: 4996)
#pragma warning(disable: 4819)
#define _CRT_SECURE_NO_WARNINGS

#include <map>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <pcl/common/transforms.h>
#include <filesystem>

#include "opencv2/opencv.hpp"
#include <opencv2/core/eigen.hpp>

#include "AHCPlaneFitter.hpp"
using ahc::utils::Timer;

namespace fs=std::filesystem;

// pcl::PointCloud interface for our ahc::PlaneFitter
template<class PointT>
struct OrganizedImage3D {
	const pcl::PointCloud<PointT>& cloud;
	//note: ahc::PlaneFitter assumes mm as unit!!!
	const double unitScaleFactor;

	OrganizedImage3D(const pcl::PointCloud<PointT>& c) : cloud(c), unitScaleFactor(1) {}
	OrganizedImage3D(const OrganizedImage3D& other) : cloud(other.cloud), unitScaleFactor(other.unitScaleFactor) {}

	inline int width() const { return cloud.width; }
	inline int height() const { return cloud.height; }
	inline bool get(const int row, const int col, double& x, double& y, double& z) const {
		const PointT& pt=cloud.at(col,row);
		x=pt.x*unitScaleFactor; y=pt.y*unitScaleFactor; z=pt.z*unitScaleFactor; //TODO: will this slowdown the speed?
		return std::isnan(z)==0; //return false if current depth is NaN
	}
};
typedef OrganizedImage3D<pcl::PointXYZ> ImageXYZ;
typedef ahc::PlaneFitter< ImageXYZ > PlaneFitter;
typedef pcl::PointCloud<pcl::PointXYZRGB> CloudXYZRGB;

namespace global {
std::map<std::string, std::string> ini;
PlaneFitter pf;

bool showWindow = true;

#ifdef _WIN32
const char filesep = '\\';
#else
const char filesep = '/';
#endif

// similar to matlab's fileparts
// if in=parent/child/file.txt
// then path=parent/child
// name=file, ext=txt
void fileparts(const std::string& str, std::string* pPath=0,
	std::string* pName=0, std::string* pExt=0)
{
	std::string::size_type last_sep = str.find_last_of(filesep);
	std::string::size_type last_dot = str.find_last_of('.');
	if (last_dot<last_sep) // "D:\parent\child.folderA\file", "D:\parent\child.folderA\"
		last_dot = std::string::npos;

	std::string path, name, ext;

	if (last_sep==std::string::npos) {
		path = ".";
		if(last_dot==std::string::npos) { // "test"
			name = str;
			ext = "";
		} else { // "test.txt"
			name = str.substr(0, last_dot);
			ext = str.substr(last_dot+1);
		}
	} else {
		path = str.substr(0, last_sep);
		if(last_dot==std::string::npos) { // "d:/parent/test", "d:/parent/child/"
			name = str.substr(last_sep+1);
			ext = "";
		} else { // "d:/parent/test.txt"
			name = str.substr(last_sep+1, last_dot-last_sep-1);
			ext = str.substr(last_dot+1);
		}
	}
	
	if(pPath!=0) {
		*pPath = path;
	}
	if(pName!=0) {
		*pName = name;
	}
	if(pExt!=0) {
		*pExt = ext;
	}
}

//"D:/test/test.txt" -> "D:/test/"
std::string getFileDir(const std::string &fileName)
{
	std::string path;
	fileparts(fileName, &path);
	return path;
}

//"D:/parent/test.txt" -> "test"
//"D:/parent/test" -> "test"
std::string getNameNoExtension(const std::string &fileName)
{
	std::string name;
	fileparts(fileName, 0, &name);
	return name;
}

void iniLoad(std::string iniFileName) {
	std::ifstream in(iniFileName);
	if(!in.is_open()) {
		std::cout<<"[iniLoad] "<<iniFileName<<" not found, use default parameters!"<<std::endl;
		return;
	}
	while(in) {
		std::string line;
		std::getline(in, line);
		if(line.empty() || line[0]=='#') continue;
		std::string key, value;
		size_t eqPos = line.find_first_of("=");
		if(eqPos == std::string::npos || eqPos == 0) {
			std::cout<<"[iniLoad] ignore line:"<<line<<std::endl;
			continue;
		}
		key = line.substr(0,eqPos);
		value = line.substr(eqPos+1);
		std::cout<<"[iniLoad] "<<key<<"=>"<<value<<std::endl;
		ini[key]=value;
	}
}

template<class T>
T iniGet(std::string key, T default_value) {
	std::map<std::string, std::string>::const_iterator itr=ini.find(key);
	if(itr!=ini.end()) {
		std::stringstream ss;
		ss<<itr->second;
		T ret;
		ss>>ret;
		return ret;
	}
	return default_value;
}

template<> std::string iniGet(std::string key, std::string default_value) {
	std::map<std::string, std::string>::const_iterator itr=ini.find(key);
	if(itr!=ini.end()) {
		return itr->second;
	}
	return default_value;
}
}//global

void processOneFrame(pcl::PointCloud<pcl::PointXYZ>& cloud, const std::string& outputFilePrefix)
{
	using global::pf;
	cv::Mat seg(cloud.height, cloud.width, CV_8UC3);

	//run PlaneFitter on the current frame of point cloud
	ImageXYZ Ixyz(cloud);
	Timer timer(1000);
	timer.tic();
	pf.run(&Ixyz, 0, &seg);
	double process_ms=timer.toc();
	std::cout<<process_ms<<" ms"<<std::endl;

	//save seg image
	cv::cvtColor(seg,seg,cv::COLOR_RGB2BGR);
	cv::imwrite(outputFilePrefix+".seg.png", seg);
	std::cout<<"output: "<<outputFilePrefix<<".seg.png"<<std::endl;

	//save seg cloud
	CloudXYZRGB xyzrgb(cloud.width, cloud.height);
	for(int r=0; r<(int)xyzrgb.height; ++r) {
		for(int c=0; c<(int)xyzrgb.width; ++c) {
			pcl::PointXYZRGB& pix = xyzrgb.at(c, r);
			const pcl::PointXYZ& pxyz = cloud.at(c, r);
			const cv::Vec3b& prgb = seg.at<cv::Vec3b>(r,c);;
			pix.x=pxyz.x;
			pix.y=pxyz.y;
			pix.z=pxyz.z;
			pix.r=prgb(2);
			pix.g=prgb(1);
			pix.b=prgb(0);
		}
	}
	pcl::io::savePCDFileBinary(outputFilePrefix+".seg.pcd", xyzrgb);
	
	if(global::showWindow) {
		//show frame rate
		std::stringstream stext;
		stext<<"Frame Rate: "<<(1000.0/process_ms)<<"Hz";
		cv::putText(seg, stext.str(), cv::Point(15,15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255,1));

		cv::imshow("seg", seg);
		cv::waitKey(10);
	}
}

int process(fs::path cloud_path) {

	const double unitScaleFactor = global::iniGet<double>("unitScaleFactor", 1.0f);
	const fs::path outputDir = global::iniGet<fs::path>("outputDir", "output");
	
	if (!fs::exists(outputDir)) {
		fs::create_directories(outputDir);
		#ifdef DEBUG_INIT
			std::cout << "# Created output directory in: " << outputDir << std::endl;
		#endif
	}

	using global::pf;
	//setup fitter
	pf.minSupport = global::iniGet<int>("minSupport", 3000);
	pf.windowWidth = global::iniGet<int>("windowWidth", 10);
	pf.windowHeight = global::iniGet<int>("windowHeight", 10);
	pf.doRefine = global::iniGet<int>("doRefine", 1) != 0;

	pf.params.initType = (ahc::InitType)global::iniGet("initType", (int)pf.params.initType);

	//T_mse
	pf.params.stdTol_merge = global::iniGet("stdTol_merge", pf.params.stdTol_merge);
	pf.params.stdTol_init = global::iniGet("stdTol_init", pf.params.stdTol_init);
	pf.params.depthSigma = global::iniGet("depthSigma", pf.params.depthSigma);

	//T_dz
	pf.params.depthAlpha = global::iniGet("depthAlpha", pf.params.depthAlpha);
	pf.params.depthChangeTol = global::iniGet("depthChangeTol", pf.params.depthChangeTol);

	//T_ang
	pf.params.z_near = global::iniGet("z_near", pf.params.z_near);
	pf.params.z_far = global::iniGet("z_far", pf.params.z_far);
	pf.params.angle_near = MACRO_DEG2RAD(global::iniGet("angleDegree_near", MACRO_RAD2DEG(pf.params.angle_near)));
	pf.params.angle_far = MACRO_DEG2RAD(global::iniGet("angleDegree_far", MACRO_RAD2DEG(pf.params.angle_far)));
	pf.params.similarityTh_merge = std::cos(MACRO_DEG2RAD(global::iniGet("similarityDegreeTh_merge", MACRO_RAD2DEG(pf.params.similarityTh_merge))));
	pf.params.similarityTh_refine = std::cos(MACRO_DEG2RAD(global::iniGet("similarityDegreeTh_refine", MACRO_RAD2DEG(pf.params.similarityTh_refine))));


#if defined(DEBUG_INIT) || defined(DEBUG_CLUSTER)
	pf.saveDir = outputDir.string();
	if(!fs::exists(outputDir / "output"))
		fs::create_directories(outputDir / "output");
	std::filesystem::create_directories(pf.saveDir + "/output");
	std::cout << "create:" << (pf.saveDir + "\\output") << std::endl;
#endif


	bool is_debug = global::iniGet<bool>("debug", 0);
	int loop_cnt = global::iniGet<int>("loop", 0); //0: no loop; -1: infinite loop; >0: n loops

	using global::showWindow;
	showWindow = global::iniGet("showWindow", true);
	if (showWindow)
		cv::namedWindow("seg");

	
	// Read PointCloud
	pcl::PointCloud<pcl::PointXYZ> cloud;
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(cloud_path, cloud)){
		PCL_ERROR ("ERROR: Could not read input point cloud %s.\n", cloud_path.stem().c_str());
		return 3;
	} 
	else {
		pcl::transformPointCloud<pcl::PointXYZ>(cloud, cloud, Eigen::Affine3f(Eigen::UniformScaling<float>((float)unitScaleFactor)));
		fs::path outputFilePrefix = outputDir / cloud_path.stem();
		processOneFrame(cloud, outputFilePrefix.string());
	}

	std::cout << " \033[32m##-->process function done!\33[0m" << std::endl;
	return 0;
}

int main(const int argc, const char** argv) {
    if(argc==2) {
		global::iniLoad("/home/fran/workspaces/isec/jazzy_ws/src/rgbd_plane_mapping/include/segmentation/peac/src/plane_fitter_pcd.ini");
		return process(argv[1]);
	}
	else if (argc==3) {
		global::iniLoad(argv[2]);
		return process(argv[1]);
	}
	else {
		std::cout << "Usage : " << argv[0] << " <cloud_path> (optional)<ini_file_path>" << std::endl;
		return 1;
	}
}