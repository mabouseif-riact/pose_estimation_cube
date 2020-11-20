#include <pcl/io/vtk_lib_io.h>
#include <vtkPolyDataMapper.h>
#include <pcl/apps/render_views_tesselated_sphere.h>
#include <fstream>      // std::ofstream
#include <experimental/filesystem>
#include "features.h"
#include "helper_functions.h"
#include "visualization.h"
#include "registration.h"
#include "features.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <thread>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/console/parse.h>



using namespace std::chrono_literals;



void alignCloudAlongZ(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    Eigen::Vector4f cloudCentroidVector;
    pcl::compute3DCentroid(*cloud, cloudCentroidVector);
    Eigen::Vector3f vec1 = cloudCentroidVector.head<3>();
    cloudCentroidVector.normalize();
    Eigen::Vector3f viewPointUnitVector;
    viewPointUnitVector << 0.0, 0.0, -1.0;
    std::cout << "Cloud centeroid: " << cloudCentroidVector << std::endl;
    // float angle = std::acos(cloudCentroidVector.dot(viewPointUnitVector));
    // Eigen::Vector3f cross_vec = vec1.cross(viewPointUnitVector); // or switch vectors
    Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(vec1, viewPointUnitVector);
    Eigen::Vector3f offset_vec;
    offset_vec << 0, 0, 0;
    pcl::transformPointCloud(*cloud, *cloud, offset_vec, q);

}







void viewCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string viewer_name)
{
    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud(new pcl::PointCloud<pcl::Normal>);
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer (viewer_name));
    viewer->setBackgroundColor(0, 0, 0);
    int c1[3] = {255, 0, 255};
    int c2[3] = {255, 0, 0};
    int c3[3] = {255, 255, 0};
    addCloudToVisualizer(viewer, cloud, normal_cloud, false, c1);
    // addCloudToVisualizer(viewer, cloud_cluster, cluster_normals, false, c2);
    // addCloudToVisualizer(viewer, aligned_cloud, cluster_normals, false, c3);
    viewer->addCoordinateSystem (0.1, "global", 0);

    while (!viewer->wasStopped())
    {
    // passthroughFilter(scene, "x", static_cast<double>(a) / 10.0, 0.15);
    viewer->spinOnce(100);
    std::this_thread::sleep_for(50ms);
    cv::waitKey(1);
    }
}


int main(int argc, char* argv[])
{


    pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>);
    double x_low;
    double x_high;
    double z_low;
    double z_high;
    int min_clust_points;
    int max_clust_points;
    float clust_tolerance;
    bool PLANE_SEG = 0;
    bool SOR = 0;

    if (argc < 11)
        std::cerr << "Usage: " << argv[0] << " [FILE_NAME] [PASSTHROUGH_X_LOW] [PASSTHROUGH_X_HIGH] [PASSTHROUGH_Z_LOW] [PASSTHROUGH_Z_HIGH]" << std::endl;
    else
    {

        // if (pcl::io::loadPLYFile("../data/scene_1_test.ply", *scene) == -1)
        // if (pcl::io::loadPLYFile("../data/test_model.ply", *scene) == -1)
        if (pcl::io::loadPLYFile(argv[1], *scene) == -1)
        {
            PCL_ERROR("Could not load scene file! \n");
            return -1;
        }


        std::cout << "Resolution of Scene cloud is " << computeCloudResolution(scene) << std::endl;


        x_low = std::atof(argv[2]);
        x_high = std::atof(argv[3]);
        z_low = std::atof(argv[4]);
        z_high = std::atof(argv[5]);
        min_clust_points = std::atof(argv[6]);
        max_clust_points = std::atof(argv[7]);
        clust_tolerance = std::atof(argv[8]);
        PLANE_SEG = std::atoi(argv[9]);
        SOR = std::atoi(argv[10]);
    }

    // Paths
    std::string base_dir = "/home/mohamed/riact_ws/src/skiros2_examples/src/skiros2_examples/turtle_test/pose_estimation";
    std::string pcd_dir_name = base_dir + "/data/views_";
    std::string poses_dir_name = base_dir + "/data/poses";
    std::string CRH_dir_name = base_dir + "/data/CRH";
    std::string training_data_h5_file_name = base_dir + "/data/training_data.h5";
    std::string kdtree_idx_file_name = base_dir + "/data/kdtree.idx";
    std::string training_data_list_file_name = "training_data.list";


    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> poses_new = openData(poses_dir_name + "/poses.txt");
    std::vector<std::vector<float>> crh_vecs = readCRH(CRH_dir_name + "/CRH.txt");
    int descriptor_size = 308; // VFH
    size_t n_train = poses_new.size();
    flann::Matrix<float> data (new float[n_train * descriptor_size], n_train, descriptor_size);
    // flann::Matrix<int> k_indices;
    // flann::Matrix<float> k_distances;
    std::cout << "data rows: " << data.rows << std::endl;
    std::cout << "data cols: " << data.cols << std::endl;
    int k = 7;
    // Check if the data has already been saved to disk
    if (!boost::filesystem::exists (training_data_h5_file_name))
    {
        pcl::console::print_error ("Could not find training data models files %s!\n",
        training_data_h5_file_name.c_str ());
        return (-1);
    }
    else
    {
        flann::load_from_file (data, training_data_h5_file_name, "training_data");
        pcl::console::print_highlight ("Training data found. Loaded %d VFH models from %s.\n",
             (int)data.rows, training_data_h5_file_name.c_str ());
    }

    // Check if the tree index has already been saved to disk
    if (!boost::filesystem::exists (kdtree_idx_file_name))
    {
        pcl::console::print_error ("Could not find kd-tree index in file %s!", kdtree_idx_file_name.c_str ());
        return (-1);
    }
    else
    {
        flann::Index<flann::ChiSquareDistance<float> > index_loaded (data, flann::SavedIndexParams (kdtree_idx_file_name));
        index_loaded.buildIndex ();


        // pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>);
        // if (pcl::io::loadOBJFile("../models/scene_1.obj", *scene) == -1)
        // {
        //     PCL_ERROR("Could not load scene file! \n");
        //     return -1;
        // }



        // std::cout << "Scene contains " << scene->width * scene->height << " points" << std::endl;


        // Passthrough filter for scene
        passthroughFilter(scene, "x", x_low, x_high); // -0.075, 0.075   -0.3, 0.3 -0.15, 0.15
        passthroughFilter(scene, "z", z_low, z_high); // -0.7, 0.7        -0.4, 0.4 -0.5, -0.1

        // Plane segmentation and removal
        if (PLANE_SEG)
            scene = segmentPlane(scene);

        // Statistical Outlier Removal filter
        if (SOR)
            SORFilter(scene);

        viewCloud(scene, "plane segmentation and SOR");

        // Upsample
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_upsampled(new pcl::PointCloud<pcl::PointXYZ>);
        cloud_upsampled = upsampleCloudMLS(scene, 0.003);

        std::cout << "Resolution of Upsampled Scene cloud is " << computeCloudResolution(cloud_upsampled) << std::endl;

        viewCloud(cloud_upsampled, "Upsampled scene");

        // Downsample
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
        cloud_downsampled = downsampleCloud(cloud_upsampled, 0.005);

        // Filter scene from NaNs
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*scene, *scene, indices);

        std::cout << "Resolution of Downsampled Scene cloud is " << computeCloudResolution(cloud_downsampled) << std::endl;
        viewCloud(cloud_downsampled, "Downsampled scene");




        std::cout << "Scene contains " << cloud_downsampled->width * cloud_downsampled->height << " points" << std::endl;

        // Clustering
        std::vector<pcl::PointIndices> cluster_indices = clustering(cloud_downsampled, min_clust_points, max_clust_points, clust_tolerance);

        std::cout << "cluster_indices size: " << cluster_indices.size() << std::endl;




        pcl::PointCloud<pcl::VFHSignature308>::Ptr cloud_cluster_features (new pcl::PointCloud<pcl::VFHSignature308>);
        for (pcl::PointIndices indices: cluster_indices)
        {

            pcl::PointIndices::Ptr cluster_point_indices(new  pcl::PointIndices(indices)); // This conversion is because of PCL's shared boost ptr
            std::cout << "Cluster indices: " << cluster_point_indices->indices.size() << std::endl;

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);

            {
            pcl::ExtractIndices<pcl::PointXYZ> extract(false);
            extract.setInputCloud(cloud_downsampled);
            extract.setIndices(cluster_point_indices);
            extract.filter(*cloud_cluster);
            }


            alignCloudAlongZ(cloud_cluster);
            // exit(-1);

            // Normals computation
            pcl::PointCloud<pcl::Normal>::Ptr cluster_normals = computeNormals(cloud_cluster, false, 0.01);
            // Descriptor computation
            pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs_cluster = computeVFH(cloud_cluster, cluster_normals);
            // CRH computation
            pcl::PointCloud<CRH90>::Ptr cluster_CRH = computeCRH(cloud_cluster, cluster_normals);

            // Query point
            // float test_model_histogram[308] = cloud_cluster_features->points[0].histogram; //  = vfhs_object->points[0].histogram
            int histogram_size = 308;
            flann::Matrix<float> p = flann::Matrix<float>(new float[descriptor_size], 1, descriptor_size);
            // memcpy (&p.ptr ()[0], &vfhs_cluster->points[0].histogram[0], p.cols * p.rows * sizeof (float));
            for (int i = 0; i < histogram_size; ++i)
                p[0][i] = vfhs_cluster->points[0].histogram[i];


            // pcl::ScopeTime t("FLANN matching");
            flann::Matrix<int> k_indices = flann::Matrix<int>(new int[k], 1, k);
            flann::Matrix<float> k_distances = flann::Matrix<float>(new float[k], 1, k);
            index_loaded.knnSearch (p, k_indices, k_distances, k, flann::SearchParams (512));
            delete[] p.ptr ();

            std::cout << "Distances: " << std::endl;
            for (int i = 0; i < k; ++i)
                std::cout << "Distance: "  << k_distances[0][i] << ", Index: " << k_indices[0][i] << std::endl;


            // Output the results on screen
            pcl::console::print_highlight ("The closest neighbor are:\n");
            for (int i = 0; i < k; ++i)
                pcl::console::print_info (" %d - (%d) with a distance of: %f\n",
                i, k_indices[0][i], k_distances[0][i]);


            pcl::console::print_highlight ("Query performed.\n");

              // Load the results
              pcl::visualization::PCLVisualizer v ("VFH Cluster Classifier");
              int y_s = (int)std::floor (sqrt ((double)k));
              int x_s = y_s + (int)std::ceil ((k / (double)y_s) - y_s);
              double x_step = (double)(1 / (double)x_s);
              double y_step = (double)(1 / (double)y_s);
              pcl::console::print_highlight ("Preparing to load ");
              pcl::console::print_value ("%d", k);
              pcl::console::print_info (" files (");
              pcl::console::print_value ("%d", x_s);
              pcl::console::print_info ("x");
              pcl::console::print_value ("%d", y_s);
              pcl::console::print_info (" / ");
              pcl::console::print_value ("%f", x_step);
              pcl::console::print_info ("x");
              pcl::console::print_value ("%f", y_step);
              pcl::console::print_info (")\n");


              int viewport = 0, l = 0, m = 0;
              for (int i = 0; i < k; ++i)
              {
                std::string cloud_name = std::to_string(k_indices[0][i] + 1); // NOTICE + 1
                boost::replace_last (cloud_name, "_vfh", "");

                v.createViewPort (l * x_step, m * y_step, (l + 1) * x_step, (m + 1) * y_step, viewport);
                l++;
                if (l >= x_s)
                {
                  l = 0;
                  m++;
                }

                pcl::PointCloud<pcl::PointXYZ> cloud_xyz;
                pcl::console::print_highlight ( stderr, "Loading "); pcl::console::print_value (stderr, "%s ", cloud_name.c_str ());
                if (pcl::io::loadPCDFile (pcd_dir_name + "/" + cloud_name + ".pcd", cloud_xyz) == -1)
                  break;



                // Convert from blob to PointCloud
                // pcl::PointCloud<pcl::PointXYZ> cloud_xyz;
                // pcl::fromPCLPointCloud2 (cloud, cloud_xyz);

                if (cloud_xyz.size() == 0)
                  break;

                pcl::console::print_info ("[done, ");
                pcl::console::print_value ("%zu", static_cast<std::size_t>(cloud_xyz.size ()));
                pcl::console::print_info (" points]\n");
                // pcl::console::print_info ("Available dimensions: ");
                // pcl::console::print_value ("%s\n", pcl::getFieldsList (cloud).c_str ());


                // Get matching view pose
                Eigen::Matrix4f matching_pose = poses_new.at(k_indices[0][0]); // NOTICE - 1
                std::cout << "Matching Pose: \n" << matching_pose << std::endl;

                // Get matching view CRH
                std::vector<float> matching_crh = crh_vecs.at(k_indices[0][0]); // NOTICE - 1

                // Match CRH between view and cluster
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_ptr(new pcl::PointCloud<pcl::PointXYZ>);
                if (pcl::io::loadPCDFile (pcd_dir_name + "/" + cloud_name + ".pcd", *cloud_xyz_ptr) == -1)
                  break;

                std::cout << "Resolution of candidate cloud is " << computeCloudResolution(cloud_xyz_ptr) << std::endl;

                pcl::PointCloud<pcl::Normal>::Ptr view_normals = computeNormals(cloud_xyz_ptr, false, 0.01);
                // std::vector<float> angles = alignCRHAngles(cloud_xyz_ptr, cloud_cluster, view_normals, cluster_normals);
                // float best_roll_angle = angles.at(0);

                // Eigen::Matrix4f roll_transform;
                // float sin_roll = std::sin(best_roll_angle);
                // float cos_roll = std::cos(best_roll_angle);
                // roll_transform << cos_roll, -sin_roll, 0, 0,
                //                   sin_roll, cos_roll,  0, 0,
                //                   0,        0,         1, 0,
                //                   0,        0,         0, 1;


                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> transforms = alignCRHTransforms(cloud_xyz_ptr, cloud_cluster, view_normals, cluster_normals);

                matching_pose = transforms.at(0);

                Eigen::Matrix4f final_transform = matching_pose;
                // final_transform(3, 3) = -10;

                pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                pcl::transformPointCloud(*cloud_xyz_ptr, *aligned_cloud, final_transform);

                {
                    pcl::ScopeTime("ICP");
                    ICP(cloud_cluster, aligned_cloud);
                }


                // Visualizer
                pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("Aligned"));
                viewer->setBackgroundColor(0, 0, 0);
                int c1[3] = {255, 0, 255};
                int c2[3] = {255, 0, 0};
                int c3[3] = {255, 255, 0};
                addCloudToVisualizer(viewer, cloud_xyz_ptr, view_normals, false, c1);
                addCloudToVisualizer(viewer, cloud_cluster, cluster_normals, false, c2);
                addCloudToVisualizer(viewer, aligned_cloud, cluster_normals, false, c3);
                viewer->addCoordinateSystem (0.1, "global", 0);

                while (!viewer->wasStopped())
                {
                // passthroughFilter(scene, "x", static_cast<double>(a) / 10.0, 0.15);
                viewer->spinOnce(100);
                std::this_thread::sleep_for(50ms);
                cv::waitKey(1);
                }


                // Demean the cloud
                Eigen::Vector4f centroid;
                pcl::compute3DCentroid (cloud_xyz, centroid);
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_demean (new pcl::PointCloud<pcl::PointXYZ>);
                pcl::demeanPointCloud<pcl::PointXYZ> (cloud_xyz, centroid, *cloud_xyz_demean);
                // Add to renderer*
                v.addPointCloud (cloud_xyz_demean, cloud_name, viewport);

                // Check if the model found is within our inlier tolerance
                double thresh = DBL_MAX; // 50
                std::stringstream ss;
                ss << k_distances[0][i];
                if (k_distances[0][i] > thresh)
                {
                  v.addText (ss.str (), 20, 30, 1, 0, 0, ss.str (), viewport);  // display the text with red

                  // Create a red line
                  pcl::PointXYZ min_p, max_p;
                  pcl::getMinMax3D (*cloud_xyz_demean, min_p, max_p);
                  std::stringstream line_name;
                  line_name << "line_" << i;
                  v.addLine (min_p, max_p, 1, 0, 0, line_name.str (), viewport);
                  v.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, line_name.str (), viewport);
                }
                else
                  v.addText (ss.str (), 20, 30, 0, 1, 0, ss.str (), viewport);

                // Increase the font size for the score*
                v.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_FONT_SIZE, 18, ss.str (), viewport);

                // Add the cluster name
                v.addText (cloud_name, 20, 10, cloud_name, viewport);
              }


              // Add coordianate systems to all viewports
              v.addCoordinateSystem (0.1, "global", 0);

              v.spin ();




        }




    }


    return 0;
}
