#include "ros/ros.h"
#include "visualization.h"
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <pcl/common/common_headers.h>
#include <thread>


#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>

using namespace std::chrono_literals;




pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Listener Viewer"));
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);



void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud2) // (const boost::shared_ptr<const sensor_msgs::PointCloud2>& cloud2)
{
  // ROS_INFO("I heard: [%s]", msg->data.c_str());
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*cloud2, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, *cloud);

}

int main(int argc, char **argv)
{

    ros::init(argc, argv, "pointcloud2_listener");
    ros::NodeHandle n;

    ros::Subscriber sub = n.subscribe<sensor_msgs::PointCloud2>("/realsense/depth/points", 1, cloudCallback);

    int c1[3] = {255, 0, 255};
    int c2[3] = {255, 0, 0};
    int c3[3] = {255, 255, 0};

    int v1(0);
    // viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    // viewer->setBackgroundColor (0, 0, 0, v1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(cloud, c1[0], c1[1], c1[2]);
    viewer->addCoordinateSystem (0.1);



  while (!viewer->wasStopped())
  {
      ros::spinOnce();
      viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_color, "sample cloud1", v1);
      // viewer->spinOnce(100);
      viewer->spinOnce();
      viewer->removePointCloud("sample cloud1", v1);

      std::this_thread::sleep_for(50ms);
      // cv::waitKey(1);
  }

  return 0;
}
