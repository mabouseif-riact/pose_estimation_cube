#include "cameraClass.h"



// CameraStream Definitions

CameraStream::CameraStream()
{
    // TODO
}





// CameraStreamROS Definitions

CameraStreamROS::CameraStreamROS(ros::NodeHandle node_handle, std::string topic) : topic_name{topic}
{
    std::cout << "Constructing object.." << std::endl;
    this->sub = node_handle.subscribe<sensor_msgs::PointCloud2>(this->topic_name, 1, &CameraStreamROS::cloudCallback, this);
    std::cout << "Object constructed" << std::endl;
}

sensor_msgs::PointCloud2 CameraStreamROS::getCloud()
{
    return this->ros_PointCloud2;
}

void CameraStreamROS::cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud2) // (const boost::shared_ptr<const sensor_msgs::PointCloud2>& cloud2)
{
  // ROS_INFO("I heard: [%s]", msg->data.c_str());
    this->ros_PointCloud2 = *cloud2;

}

pcl::PointCloud<pcl::PointXYZ> CameraStreamROS::toPCLPointCloud(sensor_msgs::PointCloud2& ros_PointCloud2)
{
    pcl_conversions::toPCL(ros_PointCloud2, this->pcl_cloud_PointCloud2);
    pcl::fromPCLPointCloud2(this->pcl_cloud_PointCloud2, (this->pcl_cloud));

    return this->pcl_cloud;
}
