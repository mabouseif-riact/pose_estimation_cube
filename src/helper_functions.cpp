#include "helper_functions.h"



void passthroughFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, char* field, double min_val, double max_val)
{
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName(field);
    pass.setFilterLimits(min_val, max_val);
    pass.filter(*cloud);
}


void scaleCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double scale)
{
    // Something wrong here. It should be 3x3
    Eigen::Matrix4f affine_mat = Eigen::Matrix4f::Identity(4, 4);
    affine_mat *= scale;
    pcl::transformPointCloud(*cloud, *cloud, affine_mat);
}


void moveCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const char axis, double dist)
{
    // Something wrong here. It should be 3x3
    Eigen::Matrix4f affine_mat = Eigen::Matrix4f::Identity(4, 4);
    switch(axis)
    {
        case 'x':
            affine_mat(3, 0) = dist;
            break;
        case 'y':
            affine_mat(3, 1) = dist;
            break;
        case 'z':
            affine_mat(3, 2) = dist;
            break;
        default:
            PCL_ERROR("INVALID AXIS!");
            break;
    }

    pcl::transformPointCloud(*cloud, *cloud, affine_mat);
}


void SORFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    // // Create the filtering object
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (200);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cloud);
}



std::vector<pcl::PointIndices> clustering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    std::vector<pcl::PointIndices> cluster_indices;
    {

    pcl::ScopeTime t("Clustering");

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.02); // 2cm
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud);
    ec.extract (cluster_indices);

    }

    return cluster_indices;
}
