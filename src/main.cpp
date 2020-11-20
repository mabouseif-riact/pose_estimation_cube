#include <iostream>
#include <thread>

#include <pcl/console/time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/common_headers.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/recognition/hv/hv_go.h>


#include <flann/algorithms/dist.h>








#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "visualization.h"
#include "features.h"
#include "registration.h"
#include "helper_functions.h"

#include "src/config.h"


using namespace std::chrono_literals;




///////////////////////////////////////////////////////
////////////////////// Main ///////////////////////////
///////////////////////////////////////////////////////

int main(int argc, char* argv[])
{

    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);

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




    if (pcl::io::loadPLYFile("../models/cube.ply", *object) == -1)
    {
        PCL_ERROR("Could not load PLY file! \n");
        return -1;
    }


    // Scaling object model
    scaleCloud(object, 0.08);


    std::string dst = "../models/cube_scaled_down.ply";
    pcl::io::savePLYFileBinary(dst, *object);
    return -1;


    // if (pcl::io::loadOBJFile("/home/mohamed/drive/pointclouds/1.obj", *scene) == -1)
    if (pcl::io::loadOBJFile("../models/scene_1.obj", *scene) == -1)
    {
        PCL_ERROR("Could not load scene file! \n");
        return -1;
    }

    // if (pcl::io::loadOBJFile("../models/cube_points_1000.obj", *object) == -1)
    // if (pcl::io::loadPLYFile("../models/cube_points_1000_new.ply", *object) == -1)
    if (pcl::io::loadPCDFile("../data/views_/17.pcd", *object) == -1)
    {
        PCL_ERROR("Could not load object file! \n");
        return -1;
    }

    std::cout << "Scene cloud loaded with "
              << scene->width * scene->height
              << " data points from cube.pcd with the following fields: "
              << std::endl;

    std::cout << "Object cloud loaded with "
              << object->width * object->height
              << " data points from cube.pcd with the following fields: "
              << std::endl;


    // Create visualizer
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer with custom color"));
    viewer->setBackgroundColor(0, 0, 0);

    // Passthrough filter for scene
    passthroughFilter(scene, "x", -0.15, 0.15); // -0.075, 0.075   -0.3, 0.3 -0.15, 0.15
    passthroughFilter(scene, "z", -0.5, -0.1); // -0.7, 0.7        -0.4, 0.4 -0.5, -0.1

    // Plane segmentation and removal
    scene = segmentPlane(scene);

    // Statistical Outlier Removal filter
    SORFilter(scene);

    // Scaling object model
    scaleCloud(object, 1); // 0.079

    // Move object model
    moveCloud(object, 'x', 1);

    // Clustering
    std::vector<pcl::PointIndices> cluster_indices = clustering(scene);

    std::cout << "cluster_indices size: " << cluster_indices.size() << std::endl;


    // Normals computation
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals = computeNormals(scene, false, 0.01);
    pcl::PointCloud<pcl::Normal>::Ptr object_normals = computeNormals(object, true, 0.015);

    // // FPFH computation
    // // pcl::PointCloud<pcl::FPFHSignature33>::Ptr scene_FPFH = computeFPFH  (scene, scene_normals);
    // // pcl::PointCloud<pcl::FPFHSignature33>::Ptr object_FPFH = computeFPFH (object, object_normals);

    // // RoPs computation
    // // pcl::PointCloud<pcl::Histogram<135>>::Ptr scene_RoPs = computeROPS(scene);
    // // pcl::PointCloud<pcl::Histogram<135>>::Ptr object_RoPs = computeROPS(object);

    // // VFH computation
    // pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs_scene = computeVFH(scene, scene_normals);
    // pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs_object = computeVFH(object, object_normals);

    // std::cout << "vfhs_scene size: " << vfhs_scene->size() << std::endl;
    // std::cout << "vfhs_object size: " << vfhs_object->size() << std::endl;

    // // CVFH computation
    // pcl::PointCloud<pcl::VFHSignature308>::Ptr CVFHS_scene = computeCVFH(scene, scene_normals);
    // pcl::PointCloud<pcl::VFHSignature308>::Ptr CVFHS_object = computeCVFH(object, object_normals);

    // std::cout << "CVFHS_scene size: " << CVFHS_scene->size() << std::endl;
    // std::cout << "CVFHS_object size: " << CVFHS_object->size() << std::endl;

    // // OUR-CVFH computation
    // pcl::PointCloud<pcl::VFHSignature308>::Ptr OURCVFHS_scene = computeOURCVFH(scene, scene_normals);
    // pcl::PointCloud<pcl::VFHSignature308>::Ptr OURCVFHS_object = computeOURCVFH(object, object_normals);

    // std::cout << "OURCVFHS_scene size: " << OURCVFHS_scene->size() << std::endl;
    // std::cout << "OURCVFHS_object size: " << OURCVFHS_object->size() << std::endl;



    // std::cout << "Scene cloud size: " << scene->size() << std::endl;
    // std::cout << "Object cloud size: " << object->size() << std::endl;


    // double th = 100;
    // pcl::Correspondences corr;
    // pcl::registration::CorrespondenceEstimation<pcl::VFHSignature308,pcl::VFHSignature308> estim;
    // estim.setInputSource(vfhs_scene);
    // estim.setInputTarget(vfhs_object);
    // estim.determineCorrespondences(corr,th);

    // int c1[3] = {255, 255, 0};
    // int c2[3] = {255, 0, 255};
    // addCloudToVisualizer(viewer, scene, object_normals, false, c1);
    // addCloudToVisualizer(viewer, object, object_normals, false, c2);


    // for (pcl::PointIndices indices: cluster_indices)
    // {

    //     pcl::PointIndices::Ptr cluster_point_indices(new  pcl::PointIndices(indices)); // This conversion is because of PCL's shared boost ptr
    //     std::cout << "Cluster indices: " << cluster_point_indices->indices.size() << std::endl;

    //     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    //     pcl::PointCloud<pcl::VFHSignature308>::Ptr cloud_cluster_features (new pcl::PointCloud<pcl::VFHSignature308>);

    //     {
    //     pcl::ExtractIndices<pcl::PointXYZ> extract(false);
    //     extract.setInputCloud(scene);
    //     extract.setIndices(cluster_point_indices);
    //     extract.filter(*cloud_cluster);
    //     }


    //     auto object_aligned =  RANSACPrerejective(cloud_cluster,
    //                                        object,
    //                                        scene_normals,
    //                                        object_normals);

    //     int c1[3] = {255, 255, 0};
    //     int c2[3] = {255, 0, 255};
    //     addCloudToVisualizer(viewer, cloud_cluster, object_normals, false, c1);
    //     addCloudToVisualizer(viewer, object_aligned, object_normals, false, c2);

    // }




    for (pcl::PointIndices indices: cluster_indices)
    {

        pcl::PointIndices::Ptr cluster_point_indices(new  pcl::PointIndices(indices)); // This conversion is because of PCL's shared boost ptr
        std::cout << "Cluster indices: " << cluster_point_indices->indices.size() << std::endl;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_cluster_features (new pcl::PointCloud<pcl::Normal>);

        {
        pcl::ExtractIndices<pcl::PointXYZ> extract(false);
        extract.setInputCloud(scene);
        extract.setIndices(cluster_point_indices);
        extract.filter(*cloud_cluster);
        }

        std::cout << "Cloud cluster has " << cloud_cluster->size() << " points" << std::endl;

        {
        pcl::ExtractIndices<pcl::Normal> extract(false);
        extract.setInputCloud(scene_normals);
        extract.setIndices(cluster_point_indices);
        extract.filter(*cloud_cluster_features);
        }
        std::cout << "cloud_cluster_features has " << cloud_cluster_features->size() << " points" << std::endl;


        // auto object_aligned =  RANSACPrerejective(cloud_cluster,
        //                            object,
        //                            cloud_cluster_features,
        //                            object_normals);
        pcl::PointCloud<pcl::PointXYZ>::Ptr object_aligned(new pcl::PointCloud<pcl::PointXYZ>);
        *object_aligned = *object;
        ICP(object_aligned, cloud_cluster);


        double th = 100;
        pcl::Correspondences corr;
        pcl::registration::CorrespondenceEstimation<pcl::Normal,pcl::Normal> estim;
        estim.setInputSource(cloud_cluster_features);
        estim.setInputTarget(object_normals);
        estim.determineCorrespondences(corr,th);

        int c1[3] = {255, 255, 0};
        int c2[3] = {255, 0, 255};
        addCloudToVisualizer(viewer, cloud_cluster, object_normals, false, c1);
        addCloudToVisualizer(viewer, object, object_normals, false, c2);
        addCloudToVisualizer(viewer, object_aligned, object_normals, false, c2);

        viewer->addCorrespondences<pcl::PointXYZ> (scene, object, corr, "Correspondences");

    }


    int a = 0;

    cv::namedWindow( "picture", 1);

    // Creating trackbars uisng opencv to control the pcl filter limits
    cv::createTrackbar("X_limit", "picture", &a, 30, NULL);


    // cv::waitKey();

    while (!viewer->wasStopped())
    {
        // passthroughFilter(scene, "x", static_cast<double>(a) / 10.0, 0.15);
        viewer->spinOnce(100);
        std::this_thread::sleep_for(50ms);
        cv::waitKey(1);
    }



    return 0;
}
