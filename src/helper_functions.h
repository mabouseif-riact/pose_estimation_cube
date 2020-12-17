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
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/mls.h>
#include <string.h>
#include <algorithm>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <pcl/registration/correspondence_estimation.h>

typedef std::pair<int, std::vector<float>> vfh_model;

struct candidateDictionary
{
    std::string cloud_name;
    Eigen::Matrix4f transform;
    Eigen::Matrix4f view_frame_transform;
};


struct sceneResult
{
    std::string cloud_name_fitness, cloud_name_inliers, scene_name;
    double score, inliers;

    void print()
    {
        std::cout << this->scene_name << " "
                  << this->score << " "
                  << this->inliers << " "
                  << this->cloud_name_inliers << " "
                  << this->cloud_name_fitness << " " << std::endl;
    }

};

void passthroughFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, char* field, double min_val, double max_val);

void scaleCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double scale=0.008);

void moveCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const char axis, double dist);

void SORFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

std::vector<pcl::PointIndices> clustering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int min_clust_points, int max_clust_points, float clust_tolerance);

int PCDIndex(std::string path);

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> openData(std::string fileToOpen);

std::vector<std::vector<float>> readCRH(std::string file_name);

double computeCloudResolution (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud);

pcl::PointCloud<pcl::PointXYZ>::Ptr downsampleCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size);

pcl::PointCloud<pcl::PointXYZ>::Ptr upsampleCloudMLS(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float radius);

Eigen::Matrix4f alignCloudAlongZ(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

int countInliers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr view);

void deleteDirectoryContents(const std::string& dir_path);

std::vector<int> readVectorFromFile(std::string filename);

void writeVectorToFile(std::string filename, const std::vector<int>& myVector);

void populateFeatureVector(const pcl::PointCloud<pcl::VFHSignature308>::ConstPtr descriptor_cloud,
       std::vector<vfh_model>& all_models,
       int pose_idx);

void convertToFLANN(std::vector<vfh_model> m,
                    std::string training_data_h5_file_name,
                    std::string kdtree_idx_file_name,
                    std::string view_names_vec_file);

int countInliersCE(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr view);

Eigen::Matrix4f getCloudTransform(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
