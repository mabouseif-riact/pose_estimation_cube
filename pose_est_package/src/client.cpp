#include "ros/ros.h"
#include "cameraClass.h"
#include <thread>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include "pose_est_package/PoseEstimationAction.h"
#include "visualization.h"

using namespace std::chrono_literals;


pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Client_Viewer"));


int main(int argc, char* argv[])
{

    // auto old_buffer = std::cout.rdbuf(nullptr);

    ros::init(argc, argv, "pose_estimation_client_node");
    ros::NodeHandle n;

    // create the action client
    // true causes the client to spin its own thread
    actionlib::SimpleActionClient<pose_est_package::PoseEstimationAction> Client("pe_server", true);

    ROS_INFO("Waiting for action server to start.");
    // wait for the action server to start
    Client.waitForServer(); //will wait for infinite time

    ROS_INFO("Action server started, sending goal.");

    

    CameraStreamROS streamObj(n, "/realsense/depth/points");
    sensor_msgs::PointCloud2 cloud_PointCloud2;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_workingCopy(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    int c1[3] = {255, 0, 255};
    int c3[3] = {255, 255, 0};
    int v1(0);
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(cloud, c1[0], c1[1], c1[2]);
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_cloud_color(aligned_cloud, c3[0], c3[1], c3[2]);

    // viewer->addCoordinateSystem (0.1);


    size_t cloud_size = 0;


    while (cloud_size == 0)
    {
        ros::spinOnce();
        cloud_PointCloud2 = streamObj.getCloud();
        *cloud = streamObj.toPCLPointCloud(cloud_PointCloud2);
        cloud_size = cloud->width * cloud->height;
        ROS_INFO("No cloud!");
    }

    ROS_INFO("Cloud received!");
    ROS_INFO("Cloud size: %i!", cloud_size);
    
    while (!viewer->wasStopped())
        {
            
            ros::spinOnce();
            cloud_PointCloud2 = streamObj.getCloud();
            *cloud = streamObj.toPCLPointCloud(cloud_PointCloud2);
            // viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_color, "scene cloud", v1);
            viewer->addPointCloud<pcl::PointXYZ>(cloud, "scene cloud", v1);
            viewer->spinOnce();
            std::this_thread::sleep_for(5ms);
            viewer->removePointCloud("scene cloud", v1);
            // cv::waitKey(1);
            
        }
    viewer->removePointCloud("scene cloud", v1);


    ROS_INFO("Action server started, sending goal.");

    pose_est_package::PoseEstimationGoal goal;
    goal.point_cloud = cloud_PointCloud2;
    Client.sendGoal(goal);

    //wait for the action to return
    bool finished_before_timeout = Client.waitForResult(ros::Duration(30.0));

    if (finished_before_timeout)
    {
        actionlib::SimpleClientGoalState state = Client.getState();
        ROS_INFO("Action finished: %s",state.toString().c_str());
    }
    else
        ROS_INFO("Action did not finish before the time out.");



    return 1;
}
