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
#pragma once

#include <set>					//PlaneSeg::NbSet
#include <vector>				//mseseq
#include <limits>				//quiet_NaN
#include <Eigen/Core>
#include <opencv4/opencv2/core/eigen.hpp>
#include <algorithm>

#include "AHCTypes.hpp"		//shared_ptr
#include "eig33sym.hpp"		//PlaneSeg::Stats::compute
#include "AHCParamSet.hpp"		//depthDisContinuous
#include "DisjointSet.hpp"	//PlaneSeg::mergeNbsFrom


namespace ahc {

//return true if d0 and d1 is discontinuous
inline static bool depthDisContinuous(const double d0, const double d1, const ParamSet& params)
{
	return std::fabs(d0-d1) > params.T_dz(d0);
}

/**
 *  \brief PlaneSeg is a struct representing a Plane Segment as a node of a graph
 *  
 *  \details It is usually dynamically allocated and garbage collected by boost::shared_ptr
 */
struct PlaneSeg {
	typedef PlaneSeg* Ptr;
	typedef ahc::shared_ptr<PlaneSeg> shared_ptr;



	/**
	 * \brief An internal struct holding this PlaneSeg's member points' 1st and 2nd order statistics
	  \details It is usually dynamically allocated and garbage collected by boost::shared_ptr
	  */
	struct Stats {
		double sx, sy, sz, //sum of x/y/z
			sxx, syy, szz, //sum of xx/yy/zz
			sxy, syz, sxz; //sum of xy/yz/xz
		int N; //#points in this PlaneSeg
		// double ev[3][3]; //eigen vectors

		Stats() : sx(0), sy(0), sz(0),
			sxx(0), syy(0), szz(0),
			sxy(0), syz(0), sxz(0), N(0) {
				// std::fill(&ev[0][0], &ev[0][0] + 3 * 3, 0.0);
			}

		//merge from two other Stats
		Stats(const Stats& a, const Stats& b) :
		sx(a.sx+b.sx), sy(a.sy+b.sy), sz(a.sz+b.sz),
			sxx(a.sxx+b.sxx), syy(a.syy+b.syy), szz(a.szz+b.szz),
			sxy(a.sxy+b.sxy), syz(a.syz+b.syz), sxz(a.sxz+b.sxz), N(a.N+b.N) {
			// std::copy(&a.ev[0][0], &a.ev[0][0] + 3 * 3, &ev[0][0]);
		}

		inline void clear() {
			sx=sy=sz=sxx=syy=szz=sxy=syz=sxz=0;
			N=0;
			// std::fill(&ev[0][0], &ev[0][0] + 3 * 3, 0.0);
		}

		//push a new point (x,y,z) into this Stats
		inline void push(const double x, const double y, const double z) {
			sx+=x; sy+=y; sz+=z;
			sxx+=x*x; syy+=y*y; szz+=z*z;
			sxy+=x*y; syz+=y*z; sxz+=x*z;
			++N;
		}

		//push a new Stats into this Stats
		inline void push(const Stats& other) {
			sx+=other.sx; sy+=other.sy; sz+=other.sz;
			sxx+=other.sxx; syy+=other.syy; szz+=other.szz;
			sxy+=other.sxy; syz+=other.syz; sxz+=other.sxz;
			N+=other.N;
		}

		//caller is responsible to ensure (x,y,z) was collected in this stats
		inline void pop(const double x, const double y, const double z) {
			sx-=x; sy-=y; sz-=z;
			sxx-=x*x; syy-=y*y; szz-=z*z;
			sxy-=x*y; syz-=y*z; sxz-=x*z;
			--N;

			assert(N>=0);
		}

		//caller is responsible to ensure {other} were collected in this stats
		inline void pop(const Stats& other) {
			sx-=other.sx; sy-=other.sy; sz-=other.sz;
			sxx-=other.sxx; syy-=other.syy; szz-=other.szz;
			sxy-=other.sxy; syz-=other.syz; sxz-=other.sxz;
			N-=other.N;

			assert(N>=0);
		}

		/**
		*  \brief PCA-based plane fitting
		*  
		*  \param [out] center center of mass of the PlaneSeg
		*  \param [out] normal unit normal vector of the PlaneSeg (ensure normal.z>=0)
		*  \param [out] mse mean-square-error of the plane fitting
		*  \param [out] curvature defined as in pcl
		*/
		inline void compute(double center[3], double normal[3],
			double& mse, double& curvature, double eigenvectors[3][3]) const
		{
			#ifdef STATS_TIMER
			auto start = std::chrono::high_resolution_clock::now();
			#endif

			assert(N>=4);

			const double sc=((double)1.0)/this->N;//this->ids.size();
			//calc plane equation: center, normal and mse
			center[0]=sx*sc;
			center[1]=sy*sc;
			center[2]=sz*sc;
			double K[3][3] = {
				{sxx-sx*sx*sc,sxy-sx*sy*sc,sxz-sx*sz*sc},
				{           0,syy-sy*sy*sc,syz-sy*sz*sc},
				{           0,           0,szz-sz*sz*sc}
			};
			K[1][0]=K[0][1]; K[2][0]=K[0][2]; K[2][1]=K[1][2]; //symmetrice matrix
			double sv[3]={0,0,0};
			double V[3][3]={0};
			LA::eig33sym(K, sv, V); //!!! first eval is the least one
			//LA.svd33(K, sv, V);
			if(V[0][0]*center[0]+V[1][0]*center[1]+V[2][0]*center[2]<=0) {//enforce dot(normal,center)<00 so normal always points towards camera
				normal[0]=V[0][0];
				normal[1]=V[1][0];
				normal[2]=V[2][0];
			} else {
				normal[0]=-V[0][0];
				normal[1]=-V[1][0];
				normal[2]=-V[2][0];
			}

			eigenvectors[0][0] = V[0][0];
			eigenvectors[1][0] = V[1][0];
			eigenvectors[2][0] = V[2][0];

			eigenvectors[0][1] = V[0][1];
			eigenvectors[1][1] = V[1][1];
			eigenvectors[2][1] = V[2][1];

			eigenvectors[0][2] = V[0][2];
			eigenvectors[1][2] = V[1][2];
			eigenvectors[2][2] = V[2][2];

			mse = sv[0]*sc;
			curvature=sv[0]/(sv[0]+sv[1]+sv[2]);

			#ifdef STATS_TIMER
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			std::cout << "S duration:" <<  duration.count() << std::endl;
			#endif
		}
	};					//member points' 1st & 2nd order statistics


#ifdef USE_NEIGHBOR_STATS
	/**
	 * \brief Struct to store the plane statistics, as the center, the normal, curvature an mse
	 * It employs a method described in: T. -k. Lee, S. Lim, S. Lee, S. An and S. -y. Oh, "Indoor mapping using planes extracted from noisy RGB-D sensors,"
	 * \details It only estimate aproximate parameters. It uses only the central point and the edge points of the window to calculate the plane statistics
	 */
	struct NeighborStats {
		double normal[3];
		double curvature;

		/**
		*  \brief PCA-based plane fitting
		*  \param [in] window_block block of points
		*  \param [out] center center of mass of the PlaneSeg
		*  \param [out] normal unit normal vector of the PlaneSeg (ensure normal.z>=0)
		*  \param [out] mse mean-square-error of the plane fitting
		*  \param [out] curvature defined as in pcl
		*/
		inline void compute(const cv::Mat& window_block, double center[3], double normal[3], double& mse, double& curvature) const
		{

#ifdef TIMER
			auto start = std::chrono::high_resolution_clock::now();
#endif

			// cv::Vec3f centroid;
			// std::vector<cv::Vec3f> points;
			// for(int i=0; i<window_block.rows; i++) {
			// 	for(int j=0; j<window_block.cols; j++) {
			// 		points.push_back(window_block.at<cv::Vec3f>(i,j));
			// 		centroid += window_block.at<cv::Vec3f>(i,j);
			// 	}
			// }
			// centroid = centroid / (window_block.rows * window_block.cols);
			// center[0] = centroid(0);
			// center[1] = centroid(1);
			// center[2] = centroid(2);

			// Eigen::Vector3f centroid_eigen {centroid(0),centroid(1),centroid(2)};

			// Check if rows and cols are even
			if (window_block.rows % 2 == 0 && window_block.cols % 2 == 0) {
				// Calculate the virtual center point using the four central neighbors
				int mid_row = window_block.rows / 2; // redondeo hacia arriba
				int mid_col = window_block.cols / 2; // redondeo hacia arriba

				// Compute the virtual center from the four central points
				cv::Vec3f virtual_center;
				cv::Vec3f center_tl = window_block.at<cv::Vec3f>(mid_row -1, mid_col - 1) ;
				cv::Vec3f center_tr = window_block.at<cv::Vec3f>(mid_row -1, mid_col);
				cv::Vec3f center_bl = window_block.at<cv::Vec3f>(mid_row, mid_col - 1);
				cv::Vec3f center_br = window_block.at<cv::Vec3f>(mid_row, mid_col);

				std::vector<cv::Vec3f> center_points;

				if(!std::isnan(center_tl(2))) 
					center_points.push_back(center_tl);
				if(!std::isnan(center_tr(2)))
					center_points.push_back(center_tr);
				if(!std::isnan(center_bl(2)))
					center_points.push_back(center_bl);
				if(!std::isnan(center_br(2)))
					center_points.push_back(center_br);

				// Calculate the virtual center
				virtual_center = std::accumulate(center_points.begin(), center_points.end(), cv::Vec3f(0,0,0)) / static_cast<float>(center_points.size());

				// GET THE EDGES OF THE WINDOW
				// TOP LEFT
				std::uint8_t padding = 0;
				cv::Vec3f edge_tl = window_block.at<cv::Vec3f>(0, 0);
				while (std::isnan(edge_tl(2))) {
					if (padding > window_block.rows*0.3) {
						std::cout << "Todos los puntos de la ventana son NaN" << std::endl;
						break;
					}
					padding++;
					edge_tl = window_block.at<cv::Vec3f>(0 + padding, 0 + padding);
				}
				
				// TOP RIGHT
				padding = 0;
				cv::Vec3f edge_tr = window_block.at<cv::Vec3f>(0, window_block.cols - 1);
				while (std::isnan(edge_tr(2))) {
					if (padding > window_block.rows*0.3) {
						std::cout << "Todos los puntos de la ventana son NaN" << std::endl;
						break;
					}
					padding++;
					edge_tr = window_block.at<cv::Vec3f>(0 + padding, window_block.cols - 1 - padding);
				}
				
				// BOTTOM LEFT
				padding = 0;
				cv::Vec3f edge_bl = window_block.at<cv::Vec3f>(window_block.rows - 1, 0);
				while (std::isnan(edge_bl(2))) {
					if (padding > window_block.rows*0.3) {
						std::cout << "Todos los puntos de la ventana son NaN" << std::endl;
						break;
					}
					padding++;
					edge_bl = window_block.at<cv::Vec3f>(window_block.rows - 1 - padding, 0 + padding);
				}

				// BOTTOM RIGHT
				padding = 0;
				cv::Vec3f edge_br = window_block.at<cv::Vec3f>(window_block.rows - 1, window_block.cols - 1);
				while (std::isnan(edge_br(2))) {
					if (padding > window_block.rows*0.3) {
						std::cout << "Todos los puntos de la ventana son NaN" << std::endl;
						break;
					}
					padding++;
					edge_br = window_block.at<cv::Vec3f>(window_block.rows - 1 - padding, window_block.cols - 1 - padding);
				}

				// Calculate the centroid of the plane
				cv::Vec3f centroid_cv = (virtual_center + edge_tl + edge_tr + edge_bl + edge_br) / 5;
				Eigen::Vector3f centroid_eigen {centroid_cv(0),centroid_cv(1),centroid_cv(2)};
				center[0] = centroid_cv(0);
				center[1] = centroid_cv(1);
				center[2] = centroid_cv(2);

				// Vector from the center to the edge
				cv::Vec3f vec_tl =  edge_tl - center_tl;
				cv::Vec3f vec_tr =  edge_tr - center_tr;
				cv::Vec3f vec_bl =  edge_bl - center_bl;
				cv::Vec3f vec_br =  edge_br - center_br;

				// TRANSFORM TO EIGEN
				Eigen::Vector3f vec_tl_eigen {vec_tl(0),vec_tl(1),vec_tl(2)};
				Eigen::Vector3f vec_tr_eigen {vec_tr(0),vec_tr(1),vec_tr(2)};
				Eigen::Vector3f vec_bl_eigen {vec_bl(0),vec_bl(1),vec_bl(2)};
				Eigen::Vector3f vec_br_eigen {vec_br(0),vec_br(1),vec_br(2)};	

				// COMPUTE THE NORMAL OF EACH QUADRANT
				Eigen::Vector3f n_left = vec_tl_eigen.cross(vec_bl_eigen);
				Eigen::Vector3f n_right = vec_tr_eigen.cross(vec_br_eigen);
				Eigen::Vector3f n_top = vec_tl_eigen.cross(vec_tr_eigen);
				Eigen::Vector3f n_bottom = vec_bl_eigen.cross(vec_br_eigen);

				Eigen::Vector3f n = (n_left + n_right + n_top + n_bottom) / 4;

				n.normalize();

				normal[0] = n[0];
				normal[1] = n[1];
				normal[2] = n[2];

				// COMPUTE THE CURVATURE
				double A = n_left.norm() + n_right.norm() + n_top.norm() + n_bottom.norm();
				double B = (n_left + n_right + n_top + n_bottom).norm();

				double PI= 3.1416;

				curvature = (std::sqrt( 8 * PI * (A-B))) / A;


				// COMPUTE THE MSE OF THE PLANE
				for (int i = 0; i < window_block.rows; i++)
				{
					for(int j=0; j<window_block.cols; j++) {
						cv::Vec3f point = window_block.at<cv::Vec3f>(i,j);
						Eigen::Vector3f p {point(0),point(1),point(2)};
						mse += std::pow(p.dot(n) - n.dot(centroid_eigen), 2);
					}
				}


				mse = mse / (window_block.rows * window_block.cols);

				#ifdef TIMER
				auto end = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
				std::cout << "N duration:" <<  duration.count() << std::endl;
				#endif

			}

			else {
				std::cout << "El tamaño de la ventana no es par." << std::endl;
				std::cout << "Ancho: " << window_block.cols << " Alto: " << window_block.rows << std::endl;
			}

		}


	}; 

	NeighborStats stats;

#else
	Stats stats;
#endif


	int rid;					//root block id
	double mse;					//mean square error
	double center[3]; 			//q: plane center (center of mass)
	double normal[3]; 			//n: plane equation n'p=q
	int N;						//#member points, same as stats.N
	double curvature;
	bool nouse;					//this PlaneSeg will be marked as nouse after merged with others to produce a new PlaneSeg node in the graph
	double eigenvectors[3][3];

#ifdef DEBUG_INIT
	enum Type {
		TYPE_NORMAL=0,				//default value
		TYPE_MISSING_DATA=1,
		TYPE_DEPTH_DISCONTINUE=2
	} type;
#endif

#if defined(DEBUG_INIT) || defined(DEBUG_CLUSTER)
	cv::Vec3b clr;
	cv::Vec3b normalClr;
	cv::Vec3b& getColor(bool useNormal=true) {
		if(useNormal) return normalClr;
		return clr;
	}
#endif

#ifdef DEBUG_CALC
	std::vector<cv::Vec2d> mseseq;
#endif

	typedef std::set<typename PlaneSeg::Ptr> NbSet; //no ownership of its content
	NbSet nbs;			//neighbors, i.e. adjacency list for a graph structure

	inline void update() {
		this->stats.compute(this->center, this->normal, this->mse, this->curvature, this->eigenvectors);
	}

	PlaneSeg(const int init_block_id, const double mse, const double center[3], const double normal[3], const double curvature, const Stats& stats)
	{
		this->stats = stats;
		this->rid = init_block_id;
		this->mse = mse;
		std::copy(center, center + 3, this->center);
		std::copy(normal, normal + 3, this->normal);
		this->N = stats.N;
		this->curvature = curvature;
		if (this->N >= 4) {
			nouse = false;
		} else {
			nouse = true;
		}
	}

	/**
	*  \brief construct a PlaneSeg during graph initialization
	*  
	*  \param [in] points organized point cloud adapter, see NullImage3D
	*  \param [in] root_block_id initial window/block's id
	*  \param [in] seed_row row index of the upper left pixel of the initial window/block
	*  \param [in] seed_col row index of the upper left pixel of the initial window/block
	*  \param [in] imgWidth width of the organized point cloud
	*  \param [in] imgHeight height of the organized point cloud
	*  \param [in] winWidth width of the initial window/block
	*  \param [in] winHeight height of the initial window/block
	*  \param [in] depthChangeFactor parameter to determine depth discontinuity
	*  
	*  \details if exist depth discontinuity in this initial PlaneSeg, nouse will be set true and N 0.
	*/
	template<class Image3D>
	PlaneSeg(const Image3D& points, const int root_block_id,
		const int seed_row, const int seed_col,
		const int imgWidth, const int imgHeight,
		const int winWidth, const int winHeight,
		const ParamSet& params)
	{
		//assert(0<=seed_row && seed_row<height && 0<=seed_col && seed_col<width && winW>0 && winH>0);
		this->rid = root_block_id;

		// INIT STATS
		bool windowValid=true;
		int nanCnt=0;
		int nanCntTh= std::floor(winHeight*winWidth*(1-params.nanTh));

		#ifdef USE_NEIGHBOR_STATS
		cv::Mat window_block(winHeight, winWidth, CV_32FC3);
		window_block.setTo(cv::Vec3f(0,0,0));
		#endif

		// VALIDA CADA VENTANA A PARTIR DE SUS PIXELES/PUNTOS (windowValid) (Con haber 1 pixel con depth discontinuity, se rechaza la ventana)
		for(int i=seed_row, icnt=0; icnt<winHeight && i<imgHeight; ++i, ++icnt) {
			for(int j=seed_col, jcnt=0; jcnt<winWidth && j<imgWidth; ++j, ++jcnt) {
				
				// Filter NaN points
				double x=0,y=0,z=10000;
				if(!points.get(i,j,x,y,z)) {
					if(params.initType==INIT_LOOSE) {
						++nanCnt;
						if(nanCnt<nanCntTh) 
							continue;
					}
#ifdef DEBUG_INIT
					this->type=TYPE_MISSING_DATA;
#endif
					windowValid=false;
					break;
				}

				// Filter Horizontal depth discontinuity 
				double xn=0,yn=0,zn=10000;
				if(j+1<imgWidth && (points.get(i,j+1,xn,yn,zn)
					&& depthDisContinuous(z,zn,params))) {
				
					windowValid=false; 
#ifdef DEBUG_INIT
					this->type=TYPE_DEPTH_DISCONTINUE;
#endif
					break;
				}

				// Filter Vertical depth discontinuity 
				if(i+1<imgHeight && (points.get(i+1,j,xn,yn,zn)
					&& depthDisContinuous(z,zn,params))) {
					
					windowValid=false;
#ifdef DEBUG_INIT
					this->type=TYPE_DEPTH_DISCONTINUE;
#endif
					break;
				}
				
				// Point is valid
				this->stats.push(x,y,z);

				#ifdef USE_NEIGHBOR_STATS
				window_block.at<cv::Vec3f>(icnt,jcnt)=cv::Vec3f(x,y,z);
				#endif
			}

			if(!windowValid) 
				break;
		}

		// VENTANA VALIDA
		if(windowValid) {//if nan or depth-discontinuity shows, this obj will be rejected
			this->nouse=false;
			this->N=this->stats.N;
			
			#ifdef DEBUG_INIT
			this->type=TYPE_NORMAL;
			#endif
		} 
		
		// VENTANA NO VALIDA
		else {
			this->N=0;
			this->stats.clear();
			this->nouse=true;
		}

		// NO SUFICIENTE SPUTNOS PARA CALCULAR EL PLANO
		if(this->N < 4) {
			this->mse = std::numeric_limits<double>::quiet_NaN();
			this->curvature = std::numeric_limits<double>::quiet_NaN();
		} 
		
		// SUFICIENTES PUNTOS PARA CALCULAR EL PLANO
		else {
			#ifdef USE_NEIGHBOR_STATS
			this->stats.compute(window_block, this->center, this->normal, this->mse, this->curvature);
			#else
			this->stats.compute(this->center, this->normal, this->mse, this->curvature, this->eigenvectors);
			#endif

#ifdef DEBUG_CALC
			this->mseseq.push_back(cv::Vec2d(this->N,this->mse));
#endif
			//nbs information to be maintained outside the class typically when initializing the graph structure
		}
		
#if defined(DEBUG_INIT) || defined(DEBUG_CLUSTER)
		// ADD COLOR DEPENDING ON THE NORMAL
		const uchar clx=uchar((this->normal[0]+1.0)*0.5*255.0);
		const uchar cly=uchar((this->normal[1]+1.0)*0.5*255.0);
		const uchar clz=uchar((this->normal[2]+1.0)*0.5*255.0);
		this->normalClr=cv::Vec3b(clx,cly,clz);

		// ADD RANDOM COLOR
		this->clr=cv::Vec3b(rand()%255,rand()%255,rand()%255);
#endif
	} //end of Plane constructor

	/**
	*  \brief construct a new PlaneSeg from two PlaneSeg pa and pb when trying to merge
	*  
	*  \param [in] pa a PlaneSeg
	*  \param [in] pb a PlaneSeg
	*/
	PlaneSeg(const PlaneSeg& pa, const PlaneSeg& pb) : stats(pa.stats, pb.stats)
	{
#ifdef DEBUG_INIT
		this->type=TYPE_NORMAL;
#endif
		this->nouse=false;
		this->rid = pa.N>=pb.N ? pa.rid : pb.rid;
		this->N=this->stats.N;

		//ds.union(pa.rid, pb.rid) will be called later
		//in mergeNbsFrom(pa,pb) function, since
		//this object might not be accepted into the graph structure

		this->stats.compute(this->center, this->normal, this->mse, this->curvature, this->eigenvectors);

#if defined(DEBUG_CLUSTER)
		const uchar clx=uchar((this->normal[0]+1.0)*0.5*255.0);
		const uchar cly=uchar((this->normal[1]+1.0)*0.5*255.0);
		const uchar clz=uchar((this->normal[2]+1.0)*0.5*255.0);
		this->normalClr=cv::Vec3b(clx,cly,clz);
		this->clr=cv::Vec3b(rand()%255,rand()%255,rand()%255);
#endif
		//nbs information to be maintained later if this node is accepted
	}

	/**
	*  \brief similarity of two plane normals
	*  
	*  \param [in] p another PlaneSeg
	*  \return abs(dot(this->normal, p->normal))
	*  
	*  \details 1 means identical, 0 means perpendicular
	*/
	inline double normalSimilarity(const PlaneSeg& p) const {
		return std::abs(normal[0]*p.normal[0]+
			normal[1]*p.normal[1]+
			normal[2]*p.normal[2]);
	}

	/**
	*  \brief signed distance between this plane and the point pt[3]
	*/
	inline double signedDist(const double pt[3]) const {
		return normal[0]*(pt[0]-center[0])+
			normal[1]*(pt[1]-center[1])+
			normal[2]*(pt[2]-center[2]);
	}

	/**
	*  \brief connect this PlaneSeg to another PlaneSeg p in the graph
	*  
	*  \param [in] p the other PlaneSeg
	*/
	inline void connect(PlaneSeg::Ptr p) {
		if(p) {
			this->nbs.insert(p);
			p->nbs.insert(this);
		}
	}

	/**
	*  \brief disconnect this PlaneSeg with all its neighbors
	*  
	*  \details after this call, this->nbs.nbs should not contain this, and this->nbs should be empty i.e. after this call this PlaneSeg node should be isolated in the graph
	*/
	inline void disconnectAllNbs() {
		NbSet::iterator itr = this->nbs.begin();
		for(; itr!=this->nbs.end(); ++itr) {
			PlaneSeg::Ptr nb = (*itr);
			if(!nb->nbs.erase(this)) {
				std::cout<<"[PlaneSeg warn] this->nbs.nbs"
					" should have contained this!"<<std::endl;
			}
		}
		this->nbs.clear();
	}

	/**
	*  \brief finish merging PlaneSeg pa and pb to this
	*  
	*  \param [in] pa a parent PlaneSeg of this
	*  \param [in] pb another parent PlaneSeg of this
	*  \param [in] ds the disjoint set of initial window/block membership to be updated
	*  
	*  \details Only call this if this obj is accepted to be added to the graph of PlaneSeg pa and pb should not exist after this function is called, i.e. after this call this PlaneSeg node will be representing a merged node of pa and pb, and pa/pb will be isolated (and thus Garbage Collected) in the graph
	*/
	inline void mergeNbsFrom(PlaneSeg& pa, PlaneSeg& pb, DisjointSet& ds) {
		//now we are sure that merging pa and pb is accepted
		this->rid = ds.Union(pa.rid, pb.rid);

		//the new neighbors should be pa.nbs+pb.nbs-pa-pb
		this->nbs.insert(pa.nbs.begin(), pa.nbs.end());
		this->nbs.insert(pb.nbs.begin(), pb.nbs.end());
		this->nbs.erase(&pa);
		this->nbs.erase(&pb);

		//pa and pb should be GC later after the following two steps
		pa.disconnectAllNbs();
		pb.disconnectAllNbs();

		//complete the neighborhood from the other side
		NbSet::iterator itr = this->nbs.begin();
		for(; itr!=this->nbs.end(); ++itr) {
			PlaneSeg::Ptr nb = (*itr);
			nb->nbs.insert(this);
		}

		pa.nouse=pb.nouse=true;
#ifdef DEBUG_CALC
		if(pa.N>=pb.N) {
			this->mseseq.swap(pa.mseseq);
		} else {
			this->mseseq.swap(pb.mseseq);
		}
		this->mseseq.push_back(cv::Vec2d(this->N,this->mse));
#endif
	}
};//PlaneSeg

}//ahc