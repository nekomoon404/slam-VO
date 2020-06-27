#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/video/tracking.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>
#include <iostream>
#include <ctype.h>
#include <algorithm> 
#include <iterator>
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>
using namespace cv;
using namespace std;

///////////////////////////////////////////
///////////////特征提取////////////////////
//////////////////////////////////////////

void featureDetection(Mat img_1, vector<Point2f>& points1)	{   
  vector<KeyPoint> kps;
  int fast_threshold = 10;
  Ptr<FastFeatureDetector> detector=FastFeatureDetector::create();
  detector->detect(img_1,kps);
  KeyPoint::convert(kps, points1, vector<int>());
}


//////////////////////////////////////////
////////////////光流法特征跟踪////////////
/////////////////////////////////////////

void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)	{ 
  vector<float>err;
																											
  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err);//调用LK跟踪特征点
  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)//删除跟丢点
     {  Point2f pt = points2.at(i- indexCorrection);
     	if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
     		  if((pt.x<0)||(pt.y<0))	{
     		  	status.at(i) = 0;
     		  }
     		  points1.erase (points1.begin() + (i - indexCorrection));
     		  points2.erase (points2.begin() + (i - indexCorrection));
     		  indexCorrection++;
     	}

     }

}

////////////////像素坐标转化///////////

Point2f pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2f
    (
        ( p.x - K.at<double>(0,2) ) / K.at<double>(0,0), 
        ( p.y - K.at<double>(1,2) ) / K.at<double>(1,1) 
    );
}
///


/////////////////////////////////////////////////////
/////////////三角测量得到深度信息再转为归一化坐标/////
///////////////////////////////////////////////////

void triangulation ( 
    const vector< Point2f>& points_1, 
    const vector< Point2f>& points_2, 
    const Mat& R, const Mat& t, 
    vector< Point3f >& points)
   
{
    Mat T1 = (Mat_<float> (3,4) <<
        1,0,0,0,
        0,1,0,0,
        0,0,1,0);
    Mat T2 = (Mat_<float> (3,4) <<
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0)
    );
    
    
    Mat K = ( Mat_<double> ( 3,3 ) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1 );
    vector<Point2f> pts_1, pts_2;
    for ( int i=0;i<points_1.size();i++ )
    {
        // 将像素坐标转换至相机坐标
        pts_1.push_back ( pixel2cam( points_1[i], K) );
        pts_2.push_back ( pixel2cam( points_2[i], K) );
    }

    Mat pts_4d;
    cv::triangulatePoints( T1, T2, pts_1, pts_2, pts_4d );
    
    // 转换成非齐次坐标
    for ( int i=0; i<pts_4d.cols; i++ )
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0); // 归一化
        Point3f p (
            x.at<float>(0,0), 
            x.at<float>(1,0), 
            x.at<float>(2,0) 
        );
        points.push_back( p );
    }
}


////////////////绝对尺度//////////////

double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)
{
  
  string line;
  int i = 0;
  ifstream myfile ("/home/pty/slambook/homework/yxc/00.txt");
  double x =0, y=0, z = 0;
  double x_prev, y_prev, z_prev;
  if (myfile.is_open())
  {
    while (( getline (myfile,line) ) && (i<=frame_id))
    {
      z_prev = z;
      x_prev = x;
      y_prev = y;
      std::istringstream in(line);
      for (int j=0; j<12; j++)  {
        in >> z ;
        if (j==7) y=z;
        if (j==3)  x=z;
      }
      
      i++;
    }
    myfile.close();
  }

  else {
    cout << "Unable to open file";
    return 0;
  }

  return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;

}




///////////////////////////////////////
//////////////////////BA优化///////////
//////////////////////////////////////

void bundleAdjustment (
    vector< Point3f > points_3d,
    vector< Point2f > points_2d,
    Mat K,
    Mat R, Mat t,
    Mat& RR, Mat& tt )
{
    // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6, landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr  = new Block ( linearSolver);    // 矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg (solver_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    Eigen::Matrix3d R_mat;
    R_mat <<
               R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
               R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
               R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
                            R_mat,
                            Eigen::Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )
                        ) );
    optimizer.addVertex ( pose );

    int index = 1;
    for ( const Point3f p:points_3d )   // landmarks
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId ( index++ );
        point->setEstimate ( Eigen::Vector3d ( p.x, p.y, p.z ) );
        point->setMarginalized ( true ); 
        optimizer.addVertex ( point );
    }

    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters (
        K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0
    );
    camera->setId ( 0 );
    optimizer.addParameter ( camera );

    // edges
    index = 1;
    for ( const Point2f p:points_2d )
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId ( index );
        edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ) );
        edge->setVertex ( 1, pose );
        edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );
        edge->setParameterId ( 0,0 );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
	edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge ( edge );
        index++;
    }

    optimizer.setVerbose ( false );
    optimizer.initializeOptimization();
    optimizer.optimize ( 5);
    
    Eigen::Isometry3d T = Eigen::Isometry3d ( pose->estimate() );
    //RR、t为优化后的R和t
    RR=(Mat_<double>(3,3)<<
    T(0,0),T(0,1),T(0,2),
    T(1,0),T(1,1),T(1,2),
    T(2,0),T(2,1),T(2,2));

    tt=(Mat_<double>(3,1)<<
    T(0,3),T(1,3),T(2,3));
   
}


//////获得Ground Truth/////////////
vector<vector<float>>get_Pose(const std::string& path)
{

  vector<vector<float>> poses;
  ifstream myfile(path);
  string line;
  if (myfile.is_open())
  {
    while ( getline (myfile,line) )
    {
          char * dup = strdup(line.c_str());
   		  char * token = strtok(dup, " ");
   		  std::vector<float> v;
	   	  while(token != NULL){
	        	v.push_back(atof(token));
	        	token = strtok(NULL, " ");
	    	}
	    	poses.push_back(v);
	    	free(dup);
    }
    myfile.close();
  } else {
  	cout << "Unable to open file"; 
  }	

  return poses;

}

