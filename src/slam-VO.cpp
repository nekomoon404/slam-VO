
#include "slam-VO.h"

using namespace std;
using namespace cv;


#define MAX_FRAME 2000
#define MIN_NUM_FEAT 1000


int main( int argc, char** argv )
{

  Mat img_1, img_2;
  
  ofstream myfile ("position.txt");//用于保存优化轨迹
  double scale = 1.00;//两帧之间的平移距离作为scale
  
  //图片读取
  char filename1[200];
  char filename2[200];
  sprintf(filename1, "/home/pty/slambook/homework/yxc/image/%06d.png", 0);
  sprintf(filename2, "/home/pty/slambook/homework/yxc/image/%06d.png", 1);
  string pose_path =  "/home/pty/slambook/homework/yxc/00.txt";
  img_1 = imread(filename1);
  img_2 = imread(filename2);

  if ( !img_1.data || !img_2.data ) { 
    cout<< "Error reading images " << endl; return -1;
  }


  //特征提取，检测Fast角点
  vector<Point2f> points1, points2; 
  featureDetection(img_1, points1);
  
  //特征跟踪，光流法
  vector<uchar> status;
  featureTracking(img_1,img_2,points1,points2, status);
  
  //对极约束求解相机运动
  double focal_length= 718.856;//焦距
  cv::Point2d principal_point(607.1928, 185.2157);//光心
  Mat E, R, t, mask;
  E = findEssentialMat(points1, points2, focal_length, principal_point, RANSAC, 0.999, 1.0, mask);//计算本质矩阵，可信度0.999，阈值1
  recoverPose(E, points1, points2, R, t, focal_length, principal_point, mask);//恢复第二帧旋转平移信息
  
  //三角测量得到三维信息
  vector< Point3f > points;
  triangulation (points1,points2,R,t,points);  
  Mat K = ( Mat_<double> ( 3,3 ) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1 );//相机内参标定 
 
  //Bundle Adjustment优化
  Mat RR,tt,R_E,t_E;
  Mat R_f, t_f; //存储最终优化位姿
  bundleAdjustment ( points,points2, K,R,t, RR,tt );

  //绘制窗口
  char text[100];
  int fontFace = FONT_HERSHEY_SIMPLEX;//定义字体
  double fontScale = 0.5;//字体大小 
  int thickness = 1;  //线宽
  namedWindow( "CAMERA IMAGE", WINDOW_AUTOSIZE);//建立窗口
  namedWindow( "Trajectory Drawing", WINDOW_AUTOSIZE );// 建立窗口
  Mat trace = Mat::zeros(800, 800, CV_8UC3); //创建空白图用于绘制轨迹
  trace.setTo(Scalar(255,255,255));//设置窗口背景颜色
       
  
  string text1 = "Black--Ground Truth";//添加标签
  string text2 = "Green--Not optimized";
  string text3 = "Red--BA optimized";     
       
  putText(trace, text1, Point2f(10,50), fontFace, fontScale, Scalar(0,0,0), thickness, 8);
  putText(trace, text2, Point2f(10,70), fontFace, fontScale, Scalar(0,255,0), thickness, 8);
  putText(trace, text3, Point2f(10,90), fontFace, fontScale, Scalar(0,0,255), thickness, 8);
       


  
  //以第二张图为重新检测的第一张 
  Mat prevImage = img_2;
  Mat currImage;
  vector<Point2f> prevFeatures = points2;
  vector<Point2f> currFeatures;
  R_f = R.clone();
  t_f = t.clone();
  R_E = R.clone();
  t_E = t.clone();
  Mat pre_t=t.clone();
  char filename[100];
  
  //循环之前的光流跟踪特征点、三角测量、BA优化等过程
  for(int numFrame=2; numFrame < MAX_FRAME; numFrame++)
  {
        points.clear();
        points1.clear();
        points2.clear();
  	sprintf(filename, "/home/pty/slambook/homework/yxc/image/%06d.png", numFrame);
        cout << "Current frame number:"<<numFrame << endl;
  	currImage = imread(filename);
  	
        //光流法跟踪特征点
  	vector<uchar> status;
        featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
        //本质矩阵,恢复R,t
  	E = findEssentialMat(prevFeatures, currFeatures, focal_length, principal_point, RANSAC, 0.999, 1.0, mask);
  	recoverPose(E, prevFeatures,currFeatures,  R, t, focal_length, principal_point, mask);
        //三角测量
        triangulation (prevFeatures,currFeatures,R,t,points);
	cout<<"id<points.size():"<<points.size()<<endl;        	
  	for ( int id=0; id<points.size(); id++ )
  	{
	    points1.push_back(prevFeatures[id]);
		points2.push_back(currFeatures[id]);
  	} 
        
       
        scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));//平移的距离     
        if ((scale>0.1)&&(-t.at<double>(2) > -t.at<double>(0)) && (-t.at<double>(2) > -t.at<double>(1)))
       	//确保有一定程度的平移而不是纯旋转以保证三角测量的精度
        {
	 	t_f = t_f + scale*(R_f*(-t));
      		R_f = R.inv()*R_f;			
                pre_t=t.clone();
		int x = int(t_f.at<double>(0)) + 400;
        	int y = int(t_f.at<double>(2)) + 150; 
                Point2f trace2 = Point2f(x,y) ;     
              	circle(trace, trace2 ,1, Scalar(0,255,0), 1);//绘制初步估计的轨迹
		
	}
        else
	{
     		cout << "scale below 0.1, or incorrect translation" << endl;
        }
        
        
	
      vector<vector<float>> poses = get_Pose(pose_path);
      Point2f trace1 = Point2f(int(poses[numFrame][3]) + 400, int(poses[numFrame][11]) + 150); //绘制Ground Truth      
      circle(trace, trace1, 1, Scalar(0,0,0), 1); 
  
	        
	
        //BA优化
        bundleAdjustment ( points,points2, K,R,t, RR,tt );
        if ((scale>0.1)&&(-tt.at<double>(2) > -tt.at<double>(0)) && (-tt.at<double>(2) > -tt.at<double>(1)))
        {		
		if(abs(tt.at<double>(2)-t.at<double>(2))<0.05)
		{
			t_E = t_E + scale*(R_E*(-tt));
      			R_E = RR.inv()*R_E;
                        pre_t=tt.clone();								
		}
	        	
		else
                {	
                        cout<<"优化失败"<<endl;                        	
			t_E = t_E + scale*(R_E*(-t));
      		        R_E = R.inv()*R_E;			
                        pre_t=t.clone();						
		}								     		       	
	}
  	
        else
	{
     		cout << "scale below 0.1, or incorrect translation" << endl;
        }
        Point2f trace3 = Point2f(int(t_E.at<double>(0)) + 400, int(t_E.at<double>(2)) + 150);
        circle(trace, trace3, 1, Scalar(0,0,255), 1);//绘制BA优化后的轨迹

        //保存优化后的轨迹数据
        myfile << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << endl;
       
        // 当特征点数目过小的时候重新检测
        if (points1.size() < MIN_NUM_FEAT)
        {
      		cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
 		featureDetection(prevImage, prevFeatures);
      		featureTracking(prevImage,currImage,prevFeatures,currFeatures,status);
                prevImage = currImage.clone();
       		prevFeatures = currFeatures;
       }
       else
       {
		prevImage = currImage.clone();
       		prevFeatures = points2;
       }
       
       imshow( "CAMERA IMAGE", currImage);
       imshow( "Trajectory Drawing", trace );
 
     
       waitKey(1);

  }

  
  
  return 0;
}

