#include "registration.h"


pcl::PointCloud<pcl::PointXYZ>::Ptr segmentPlane(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, double dist_thresh, int max_iterations)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    *cloud_filtered = *cloud;

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations (max_iterations);
    seg.setDistanceThreshold(dist_thresh);


    int i=0, nr_points = (int) cloud_filtered->size ();
    while (cloud_filtered->size () > 0.5 * nr_points)
    {
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size () == 0)
        {
          std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
          break;
        }

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud (cloud_filtered);
        extract.setIndices (inliers);
        extract.setNegative (true);
        extract.filter (*cloud_filtered);
    }

    return cloud_filtered;
}



void ICP(pcl::PointCloud<pcl::PointXYZ>::ConstPtr object_aligned, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster)
{
  // The Iterative Closest Point algorithm
    int iterations = 200;
    // time.tic ();
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setMaximumIterations (iterations);
    icp.setInputSource (cloud_cluster);
    icp.setInputTarget (object_aligned);
    {
        pcl::ScopeTime t("ICP Alignment");
        icp.align (*cloud_cluster);
    }

    icp.setMaximumIterations (1);  // We set this variable to 1 for the next time we will call .align () function
    // std::cout << "Applied " << iterations << " ICP iteration(s) in " << time.toc () << " ms" << std::endl;

    if (icp.hasConverged ())
    {
        std::cout << "\nICP has converged, score is " << icp.getFitnessScore () << std::endl;
        std::cout << "\nICP transformation " << iterations << " : Scene -> Object" << std::endl;
        auto transformation_matrix = icp.getFinalTransformation (); // .cast<double>();
        // pcl::transformPointCloud(*object_aligned, *object_aligned, transformation_matrix);
        // print4x4Matrix (transformation_matrix);
    }
    else
    {
        PCL_ERROR ("\nICP has not converged.\n");
        // return (-1);
    }

}
