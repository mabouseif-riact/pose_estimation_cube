#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common_headers.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <thread>




pcl::visualization::PCLVisualizer::Ptr  simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);

pcl::visualization::PCLVisualizer::Ptr customColorVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);

pcl::visualization::PCLVisualizer::Ptr normalVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                                 pcl::PointCloud<pcl::Normal>::ConstPtr normals);

void addCloudToVisualizer(pcl::visualization::PCLVisualizer::Ptr viewer, pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,                                                            pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                            bool display_normals,
                                                            int color_vec[3]);

void viewCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string viewer_name);
