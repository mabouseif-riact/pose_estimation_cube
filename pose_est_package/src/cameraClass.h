#include "ros/ros.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl/common/common_headers.h>
#include <pcl/point_cloud.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>


class CameraStream
{

public:
    CameraStream();
    // virtual pcl::PointCloud<pcl::PointXYZ>::Ptr getCloud() = 0;


};




class CameraStreamROS: CameraStream
{
    std::string topic_name;
    ros::Subscriber sub;
    // pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::PCLPointCloud2 pcl_cloud_PointCloud2;
    sensor_msgs::PointCloud2 ros_PointCloud2;
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud2);

public:
    CameraStreamROS(ros::NodeHandle node_handle, std::string topic_name);
    sensor_msgs::PointCloud2 getCloud();
    pcl::PointCloud<pcl::PointXYZ> toPCLPointCloud(sensor_msgs::PointCloud2& ros_PointCloud2);

};
