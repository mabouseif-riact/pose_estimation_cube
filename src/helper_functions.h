// #include <Eigen/Core>
#include <eigen3/Eigen/Core>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/console/time.h>
#include <pcl/common/common_headers.h>
#include <experimental/filesystem>
#include <fstream>      // std::ofstream



void passthroughFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, char* field, double min_val, double max_val);

void scaleCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double scale=0.008);

void moveCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const char axis, double dist);

void SORFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

std::vector<pcl::PointIndices> clustering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

int PCDIndex(std::string path);

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> openData(std::string fileToOpen);

std::vector<std::vector<float>> readCRH(std::string file_name);
