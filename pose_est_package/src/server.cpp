#include "ros/ros.h"
// #include <eigen_conversions>
#include <eigen_conversions/eigen_msg.h>
#include "PoseEstimatorClass.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>

#include "pose_est_package/PoseEstimationAction.h"  // Note: "Action" is appended
#include <actionlib/server/simple_action_server.h>

typedef actionlib::SimpleActionServer<pose_est_package::PoseEstimationAction> Server;

// class PoseEstimationServer
// {
// protected:
//     Server server_;
//     std::string action_name_;
//     pose_est_package::PoseEstimationResult result_;
//     pose_est_package::PoseEstimationFeedback feedback_;
//     PoseEstimator est_;
    

// public:
//     PoseEstimationServer(ros::NodeHandle nh, std::string action_name);
//     ~PoseEstimationServer(void);
//     void executeCB(const pose_est_package::PoseEstimationGoalConstPtr &goal);


// };

// PoseEstimationServer::PoseEstimationServer(ros::NodeHandle nh, std::string action_name): 
//   server_(nh, action_name, boost::bind(&PoseEstimationServer::executeCB, this, _1), false), action_name_{action_name}
// {
//     this->server_.start();
// }

// PoseEstimationServer::~PoseEstimationServer(void) {}

// // void PoseEstimationServer::executeCB(const pose_est_package::PoseEstimationGoalConstPtr &goal)
// // {

// //     pcl::PointCloud<pcl::PointXYZ> *pcl_cloud;
// //     pcl::PCLPointCloud2 pcl_cloud_PointCloud2;
// //     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_workingCopy(new pcl::PointCloud<pcl::PointXYZ>);
// //     Eigen::Matrix4f final_transform;
// //     std_msgs::Float64MultiArray m;

// //     ros::Rate r(1);
// //     bool success = true;

// //     ROS_INFO("%s: Executing, Recognizing and estimating object in scene..", this->action_name_);

// //     pcl_conversions::toPCL(goal->point_cloud, pcl_cloud_PointCloud2);
// //     pcl::fromPCLPointCloud2(pcl_cloud_PointCloud2, *pcl_cloud);
// //     copyPointCloud(*pcl_cloud, *cloud_workingCopy);

// //     final_transform = this->est_.estimate(cloud_workingCopy);

// //     tf::matrixEigenToMsg(final_transform,  m);

// //     result_.pose = m;

// //     ROS_INFO("%s: Succeeded", this->action_name_.c_str());
// //     // set the action state to succeeded
// //     server_.setSucceeded(result_);

// // }



// // int main(int argc, char** argv)
// // {
// //   ros::init(argc, argv, "pose_estimation_server_node");
// //   ros::NodeHandle n;
// //   PoseEstimationServer(n, "pose_estimation_server");
// //   ROS_INFO("Pose Estimation Server Started!");
// //   ros::spin();
// //   return 0;
// // }








class PoseEstimationServer
{
protected:

  ros::NodeHandle nh_;
  actionlib::SimpleActionServer<pose_est_package::PoseEstimationAction> as_;
  std::string action_name_;
  // create messages that are used to published feedback/result
  pose_est_package::PoseEstimationResult result_;
  pose_est_package::PoseEstimationFeedback feedback_;
  PoseEstimator est_;

public:

  PoseEstimationServer(std::string name) :
    as_(nh_, name, boost::bind(&PoseEstimationServer::executeCB, this, _1), false),
    action_name_(name)
  {
    as_.start();
  }

  ~PoseEstimationServer(void)
  {
  }

  void executeCB(const pose_est_package::PoseEstimationGoalConstPtr &goal)
  {
      pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PCLPointCloud2 pcl_cloud_PointCloud2;
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_workingCopy(new pcl::PointCloud<pcl::PointXYZ>);
      Eigen::Matrix4f final_transform;
      std_msgs::Float64MultiArray m;

      ros::Rate r(1);
      bool success = true;

      ROS_INFO("%s: Executing, Recognizing and estimating object in scene..", (this->action_name_).c_str());

      size_t cloud_size = goal->point_cloud.width * goal->point_cloud.height;
      std::cout << "Cloud size from callback: " << cloud_size << std::endl;

      pcl_conversions::toPCL(goal->point_cloud, pcl_cloud_PointCloud2);
      std::cout << "Done converting from ros_pointcloud_2 to pcl_cloud_PointCloud2" << std::endl;
      pcl::fromPCLPointCloud2(pcl_cloud_PointCloud2, *pcl_cloud);
      std::cout << "Done converting from pcl_cloud_PointCloud2 to pcl_pointcloud" << std::endl;
      copyPointCloud(*pcl_cloud, *cloud_workingCopy);

      std::cout << "Before estimation" << std::endl;

      final_transform = this->est_.estimate(cloud_workingCopy);

      std::cout << "After estimation" << std::endl;

      tf::matrixEigenToMsg(final_transform,  m);

      result_.pose = m;

      ROS_INFO("%s: Succeeded", this->action_name_.c_str());
      // set the action state to succeeded
      as_.setSucceeded(result_);
  }


};


int main(int argc, char** argv)
{
  ros::init(argc, argv, "pe_server");

  PoseEstimationServer pe_server("pe_server");
  ros::spin();

  return 0;
}