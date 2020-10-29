#include <iostream>
#include <thread>
#include <Eigen/Core>

#include <pcl/console/time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/registration/icp.h>
#include <pcl/segmentation/extract_clusters.h>

#include "src/config.h"


using namespace std::chrono_literals;

pcl::visualization::PCLVisualizer::Ptr  simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, "Sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Sample cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    return viewer;
}


pcl::visualization::PCLVisualizer::Ptr customColorVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer with custom color"));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 255, 0, 0);
    viewer->addPointCloud(cloud, single_color, "Sample cloud with color");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "Sample cloud with color");
    // viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    return viewer;
}



pcl::visualization::PCLVisualizer::Ptr normalVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                                 pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer with color and normals"));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 255, 0, 0);
    viewer->addPointCloud(cloud, single_color, "Sample cloud with color and normals");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "Sample cloud with color and normals");
    viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (cloud, normals, 10, 0.005, "normals");
    // viewer->addCoordinateSystem(0.5);
    viewer->initCameraParameters ();
    return viewer;
}


void addCloudToVisualizer(pcl::visualization::PCLVisualizer::Ptr viewer,
                                                            pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                                            pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                                            bool display_normals,
                                                            int color_vec[3])
{
    static int count = 1;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, color_vec[0], color_vec[1], color_vec[2]);
    viewer->addPointCloud(cloud, single_color, "Sample cloud with color and normals " + count);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "Sample cloud with color and normals " + count);
    if (display_normals)
        viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (cloud, normals, 10, 0.005, "normals " + count);

    ++count;
}



pcl::PointCloud<pcl::Normal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, bool flip, double radius=0.03)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;

    // if (viewpoint)
    //     ne.setViewPoint(0, 0, 0);

    ne.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

    ne.setSearchMethod(tree);

    ne.setRadiusSearch(radius);

    ne.compute(*normals);

    if (flip)
        for (auto &normal: *normals)
        {
            normal.normal_x *=-1;
            normal.normal_y *=-1;
            normal.normal_z *=-1;
        }

    return normals;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr segmentPlane(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, double dist_thresh=0.01, int max_iterations=1000)
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


pcl::PointCloud<pcl::PointXYZ>::Ptr RANSAC(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                           pcl::PointCloud<pcl::PointXYZ>::ConstPtr target,
                                           pcl::PointCloud<pcl::Normal>::ConstPtr cloud_features,
                                           pcl::PointCloud<pcl::Normal>::ConstPtr target_featries)
{
    // Perform alignment
    pcl::console::print_highlight ("Starting alignment...\n");
    const float leaf = 0.005f;
    pcl::PointCloud<pcl::PointXYZ>::Ptr object_aligned(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::Normal> align;
    align.setInputSource (cloud);
    align.setSourceFeatures (cloud_features);
    align.setInputTarget (target);
    align.setTargetFeatures (target_featries);
    align.setMaximumIterations (10000); // Number of RANSAC iterations
    align.setNumberOfSamples (3); // Number of points to sample for generating/prerejecting a pose
    align.setCorrespondenceRandomness (20); // Number of nearest features to use
    align.setSimilarityThreshold (0.9f); // Polygonal edge length similarity threshold
    align.setMaxCorrespondenceDistance (2.5f * leaf); // Inlier threshold
    align.setInlierFraction (0.3f); // Required inlier fraction for accepting a pose hypothesis
    {
      pcl::ScopeTime t("Alignment");
      align.align (*object_aligned);
    }

    std::cout << "RANSAC Convergence: " << align.hasConverged() << std::endl;

    return object_aligned;

}





void passthroughFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, char* field, double min_val, double max_val)
{
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName(field);
    pass.setFilterLimits(min_val, max_val);
    pass.filter(*cloud);
}


void scaleCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double scale=0.008)
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





    ///////////////////////////////////////////////////////
    ////////////////////// Main /////////////////
    ///////////////////////////////////////////////////////

int main(int argc, char* argv[])
{

    if (argc < 2)
    {
    // report version
    std::cout << argv[0] << " Version " << PoseEstimation_VERSION_MAJOR << "."
              << PoseEstimation_VERSION_MINOR << std::endl;
    std::cout << "Usage: " << argv[0] << " number" << std::endl;
    return 1;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr object(new pcl::PointCloud<pcl::PointXYZ>);


    // if (pcl::io::loadOBJFile("/home/mohamed/drive/pointclouds/1.obj", *scene) == -1)
    if (pcl::io::loadOBJFile("/home/mohamed/Downloads/3.obj", *scene) == -1)
    {
        PCL_ERROR("Could not load scene file! \n");
        return -1;
    }

    if (pcl::io::loadOBJFile("/home/mohamed/riact_ws/src/skiros2_examples/src/skiros2_examples/turtle_test/cube_edited.obj", *object) == -1)
    {
        PCL_ERROR("Could not load object file! \n");
        return -1;
    }

    std::cout << "Scene cloud loaded with "
              << scene->width * scene->height
              << " data points from cube.pcd with the following fields: "
              << std::endl;

    ///////////////////////////////////////////////////////
    ////////////////////// SceneFiltering /////////////////
    ///////////////////////////////////////////////////////

    // // // Passthrough filter for scene
    passthroughFilter(scene, "x", -0.15, 0.15); // -0.075, 0.075   -0.3, 0.3
    passthroughFilter(scene, "z", -0.5, -0.1); // -0.7, 0.7        -0.4, 0.4


    // // // Plane segmentation and removal
    // for (int i = 0; i < 3; ++i)
    scene = segmentPlane(scene);

    // // Create the filtering object
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (scene);
    sor.setMeanK (200);
    sor.setStddevMulThresh (1.0);
    sor.filter (*scene);

    ///////////////////////////////////////////////////////
    ////////////////////// Object scaling /////////////////
    ///////////////////////////////////////////////////////


    // Scaling object model
    scaleCloud(object, 0.0079);

    // Move object model
    moveCloud(object, 'x', 1);


    // Normals computation
    // pcl::PointCloud<pcl::Normal>::Ptr scene_normals = computeNormals(scene, false, 0.01);
    pcl::PointCloud<pcl::Normal>::Ptr object_normals = computeNormals(object, true, 0.015);

    std::cout << "Scene cloud size: " << scene->size() << std::endl;
    // std::cout << "Scene normals cloud size: " << scene_normals->size() << std::endl;


    ///////////////////////////////////////////////////////
    ////////////////////// Clustering /////////////////////
    ///////////////////////////////////////////////////////

    std::vector<pcl::PointIndices> cluster_indices;
    {

    pcl::ScopeTime t("Clustering");

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (scene);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.02); // 2cm
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (scene);
    ec.extract (cluster_indices);

    }

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (scene);

    pcl::visualization::PCLVisualizer::Ptr viewer = customColorVis(object);
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
            cloud_cluster->push_back ((*scene)[*pit]); //*

        cloud_cluster->width = cloud_cluster->size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        pcl::PointCloud<pcl::Normal>::Ptr cluster_normals = computeNormals(cloud_cluster, false, 0.01);

        // pcl::PointCloud<pcl::PointXYZ>::Ptr object_aligned = RANSAC(object, cloud_cluster, object_normals, cluster_normals);

        pcl::PointCloud<pcl::PointXYZ>::Ptr object_aligned(new pcl::PointCloud<pcl::PointXYZ>);
        *object_aligned = *object;

           // The Iterative Closest Point algorithm
          int iterations = 200;
          // time.tic ();
          pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
          icp.setMaximumIterations (iterations);
          icp.setInputSource (cloud_cluster);
          icp.setInputTarget (object_aligned);
          {
            pcl::ScopeTime t("Alignment");
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
            return (-1);
        }



        int c1[3] = {255, 255, 0};
        int c2[3] = {255, 0, 255};
        addCloudToVisualizer(viewer, cloud_cluster, object_normals, false, c1);
        addCloudToVisualizer(viewer, object_aligned, object_normals, false, c2);
    }


    ///////////////////////////////////////////////////////
    ////////////////////// ICP ////////////////////////////
    ///////////////////////////////////////////////////////


    //    // The Iterative Closest Point algorithm
    //   int iterations = 200;
    //   // time.tic ();
    //   pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    //   icp.setMaximumIterations (iterations);
    //   icp.setInputSource (cloud_cluster);
    //   icp.setInputTarget (object_aligned);
    //   {
    //     pcl::ScopeTime t("Alignment");
    //     icp.align (*cloud_cluster);
    //   }

    //   icp.setMaximumIterations (1);  // We set this variable to 1 for the next time we will call .align () function
    //   // std::cout << "Applied " << iterations << " ICP iteration(s) in " << time.toc () << " ms" << std::endl;

    //   if (icp.hasConverged ())
    // {
    //     std::cout << "\nICP has converged, score is " << icp.getFitnessScore () << std::endl;
    //     std::cout << "\nICP transformation " << iterations << " : Scene -> Object" << std::endl;
    //     auto transformation_matrix = icp.getFinalTransformation (); // .cast<double>();
    //     pcl::transformPointCloud(*object_aligned, *object_aligned, transformation_matrix);
    //     // print4x4Matrix (transformation_matrix);
    // }
    // else
    // {
    //     PCL_ERROR ("\nICP has not converged.\n");
    //     return (-1);
    // }





    ///////////////////////////////////////////////////////
    ////////////////////// Visualization //////////////////
    ///////////////////////////////////////////////////////


    // pcl::visualization::PCLVisualizer::Ptr viewer = simpleVis(scene);
    // pcl::visualization::PCLVisualizer::Ptr viewer = customColorVis(scene);
    // pcl::visualization::PCLVisualizer::Ptr viewer = normalVis(scene, scene_normals);
    // int color[3] = {0, 0, 255};
    // addCloudToVisualizer(viewer, object, object_normals, true, color);
    // addCloudToVisualizer(viewer, object_aligned, object_normals, false, color);


    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(50ms);
    }



    return 0;
}
