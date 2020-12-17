#include "ros/ros.h"
#include "cameraClass.h"
#include "PoseEstimatorClass.h"
#include <thread>



using namespace std::chrono_literals;


pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Client Viewer"));


int main(int argc, char* argv[])
{

    auto old_buffer = std::cout.rdbuf(nullptr);


    ros::init(argc, argv, "PE_Client");
    ros::NodeHandle n;

    CameraStreamROS streamObj(n, "/realsense/depth/points");
    sensor_msgs::PointCloud2 cloud_PointCloud2;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_workingCopy(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    PoseEstimator est;



    int c1[3] = {255, 0, 255};
    int c3[3] = {255, 255, 0};
    int v1(0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(cloud, c1[0], c1[1], c1[2]);
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_cloud_color(aligned_cloud, c3[0], c3[1], c3[2]);

    viewer->addCoordinateSystem (0.1);


    size_t cloud_size = 0;


    while (cloud_size == 0)
    {
        ros::spinOnce();
        cloud_PointCloud2 = streamObj.getCloud();
        *cloud = streamObj.toPCLPointCloud(cloud_PointCloud2);
        cloud_size = cloud->width * cloud->height;

    }


    // passthroughFilter(cloud, "x", -0.1, 0.1); // -0.075, 0.075   -0.3, 0.3 -0.15, 0.15
    // passthroughFilter(cloud, "z", -0.4, 0.4); // -0.7, 0.7        -0.4, 0.4 -0.5, -0.1

    // viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_color, "scene cloud", v1);

    copyPointCloud(*cloud, *cloud_workingCopy);

    aligned_cloud = est.estimate(cloud_workingCopy);

    std::cout << "Aligned cloud size: " << aligned_cloud->width * aligned_cloud->height << std::endl;

    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_color, "scene cloud", v1);
    // viewer->addPointCloud<pcl::PointXYZ> (aligned_cloud, aligned_cloud_color, "aligned cloud", v1);
    viewer->addPointCloud<pcl::PointXYZ> (aligned_cloud, "aligned cloud", v1);

    while (!viewer->wasStopped())
        {
            viewer->spinOnce(100);
            std::this_thread::sleep_for(50ms);
            cv::waitKey(1);
        }

    // while(cv::waitKey(0))
    // while (!viewer->wasStopped())
    // {
        // ros::spinOnce();
        // cloud_PointCloud2 = streamObj.getCloud();
        // *cloud = streamObj.toPCLPointCloud(cloud_PointCloud2);



        // viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_color, "scene cloud", v1);
        // viewer->spinOnce();
        // cv::waitKey(1);
        // viewer->removePointCloud("scene cloud", v1);
        // cloud_size = cloud->width * cloud->height;

        // if (!(cloud_size == 0))
            // aligned_cloud = est.estimate(cloud);
            // viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_color, "scene cloud", v1);
            // viewer->addPointCloud<pcl::PointXYZ> (aligned_cloud, aligned_cloud_color, "aligned cloud", v1);
            // viewer->spinOnce(100);
            // viewer->spinOnce();

            // viewer->removePointCloud("scene cloud", v1);
            // viewer->removePointCloud("aligned cloud", v1);

        // std::this_thread::sleep_for(10ms);
        // cv::waitKey(1);
    // }



    return 1;
}
