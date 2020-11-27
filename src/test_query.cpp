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
#include <map>


using namespace std::chrono_literals;



pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>);
std::string scene_file;
std::string descriptor_name("");
double x_low = -DBL_MAX;
double x_high = DBL_MAX;
double z_low = -DBL_MAX;
double z_high = DBL_MAX;
int min_clust_points = 50;
int max_clust_points = 25000;
float clust_tolerance = 0.02;
bool upsample = false;
bool downsample = false;
bool plane_seg = false;
bool sor = false;
bool icp = false;



void
showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*                        Usage Guide                                      *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " [File_name] [Descriptor] [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                     Show help" << std::endl;
  std::cout << "     --x_low:                Passthrough X low" << std::endl;
  std::cout << "     --x_high:               Passthrough X high" << std::endl;
  std::cout << "     --z_low:                Passthrough z low" << std::endl;
  std::cout << "     --z_high:               Passthrough z high" << std::endl;
  std::cout << "     --min_clust_points:     min points in a cluster" << std::endl;
  std::cout << "     --max_clust_points:     max points in a cluster" << std::endl;
  std::cout << "     --clust_tolerance:      distance between two points to be considered in same cluster" << std::endl;
  std::cout << "     --icp:                  Perform ICP" << std::endl;
  std::cout << "     --upsample:             Upsampling of scene cloud" << std::endl;
  std::cout << "     --downsample:           Downsampling of scene cloud" << std::endl;
  std::cout << "     --plane_seg:            Perform plane segmentation" << std::endl;
  std::cout << "     --sor:                  Use Statistical Outlier Filter" << std::endl;
}



void parseCommandLine(int argc, char *argv[])
{

    //Show help
    if (pcl::console::find_switch (argc, argv, "-h"))
    {
        showHelp (argv[0]);
        exit (0);
    }

    std::vector<int> filenames;
    filenames = pcl::console::parse_file_extension_argument (argc, argv, ".ply");
    if (filenames.size () != 1)
    {
        std::cout << "Filenames missing.\n";
        showHelp (argv[0]);
        exit (-1);
    }

    scene_file = argv[filenames[0]];
    std::cout << "Scene file name: " << scene_file << std::endl;


    //General parameters
    pcl::console::parse_argument (argc, argv, "--x_low", x_low);
    pcl::console::parse_argument (argc, argv, "--x_high", x_high);
    pcl::console::parse_argument (argc, argv, "--z_low", z_low);
    pcl::console::parse_argument (argc, argv, "--z_high", z_high);
    pcl::console::parse_argument (argc, argv, "--min_clust_points", min_clust_points);
    pcl::console::parse_argument (argc, argv, "--max_clust_points", max_clust_points);
    pcl::console::parse_argument (argc, argv, "--clust_tolerance", clust_tolerance);

    //Program behavior
    if (pcl::console::find_switch (argc, argv, "--upsample"))
        upsample = true;
    if (pcl::console::find_switch (argc, argv, "--downsample"))
        downsample = true;
    if (pcl::console::find_switch (argc, argv, "--plane_seg"))
        plane_seg = true;
    if (pcl::console::find_switch (argc, argv, "--sor"))
        sor = true;
    if (pcl::console::find_switch (argc, argv, "--icp"))
        icp = true;
    if (pcl::console::find_switch (argc, argv, "--vfh"))
        descriptor_name = "VFH";
    if (pcl::console::find_switch (argc, argv, "--cvfh"))
        descriptor_name = "CVFH";
    if (pcl::console::find_switch (argc, argv, "--ourcvfh"))
        descriptor_name = "OURCVFH";

    if (descriptor_name == "")
    {
        std::cerr << "No descriptor chosen!" << std::endl;
        exit(-1);
    }


    std::cout << "x_low: " << x_low << std::endl;
    std::cout << "x_high: " << x_high << std::endl;
    std::cout << "z_low: " << z_low << std::endl;
    std::cout << "z_high: " << z_high << std::endl;
    std::cout << "min_clust_points: " << min_clust_points << std::endl;
    std::cout << "max_clust_points: " << max_clust_points << std::endl;
    std::cout << "clust_tolerance: " << clust_tolerance << std::endl;
    std::cout << "upsample: " << upsample << std::endl;
    std::cout << "downsample: " << downsample << std::endl;
    std::cout << "plane_seg: " << plane_seg << std::endl;
    std::cout << "sor: " << sor << std::endl;
    std::cout << "icp: " << icp << std::endl;

}




int main(int argc, char* argv[])
{

    parseCommandLine (argc, argv);


    if (pcl::io::loadPLYFile(scene_file, *scene) == -1)
    {
        PCL_ERROR("Could not load scene file! \n");
        return -1;
    }

    double cloud_res = computeCloudResolution(scene);

    std::cout << "Resolution of Scene cloud is " << cloud_res << std::endl;


    // Paths
    // std::string base_dir = "/home/mohamed/turtle_test_link/pose_estimation_cube";
    std::string base_dir = "/home/mohamed/riact_ws/src/skiros2_examples/src/skiros2_examples/turtle_test/pose_estimation";
    std::string pcd_dir_name = base_dir + "/data/views_";
    std::string poses_dir_name = base_dir + "/data/poses";
    std::string CRH_dir_name = base_dir + "/data/CRH";
    std::string view_names_vec_file= base_dir + "/data/view_names.vec";
    std::string training_data_h5_file_name = base_dir + "/data/training_data.h5";
    std::string kdtree_idx_file_name = base_dir + "/data/kdtree.idx";
    std::string training_data_list_file_name = "training_data.list";


    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> poses_new = openData(poses_dir_name + "/poses.txt");
    std::vector<std::vector<float>> crh_vecs = readCRH(CRH_dir_name + "/CRH.txt");
    std::vector<int> clustered_view_files_vec = readVectorFromFile(view_names_vec_file);
    int descriptor_size = 308; // VFH
    size_t n_train = clustered_view_files_vec.size();
    flann::Matrix<float> data (new float[n_train * descriptor_size], n_train, descriptor_size);
    std::cout << "data rows: " << data.rows << std::endl;
    std::cout << "data cols: " << data.cols << std::endl;
    int k = 11;
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
        // Load FLANN Matrix
        flann::Index<flann::ChiSquareDistance<float> > index_loaded (data, flann::SavedIndexParams (kdtree_idx_file_name));
        index_loaded.buildIndex ();

        // Passthrough filter for scene
        passthroughFilter(scene, "x", x_low, x_high); // -0.075, 0.075   -0.3, 0.3 -0.15, 0.15
        passthroughFilter(scene, "z", z_low, z_high); // -0.7, 0.7        -0.4, 0.4 -0.5, -0.1

        // Plane segmentation and removal
        if (plane_seg)
            scene = segmentPlane(scene);

        // Statistical Outlier Removal filter
        if (sor)
            SORFilter(scene);

        viewCloud(scene, "plane segmentation and SOR");

        // Upsample
        if (upsample && cloud_res > 0.003)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_upsampled(new pcl::PointCloud<pcl::PointXYZ>);
            scene = upsampleCloudMLS(scene, 0.003);
            // Filter scene from NaNs
            std::vector<int> upsampled_indices;
            pcl::removeNaNFromPointCloud(*scene, *scene, upsampled_indices);
            std::cout << "Upsampled cloud size after NaN removal: " << upsampled_indices.size() << std::endl;

            std::cout << "Resolution of Upsampled Scene cloud is " << computeCloudResolution(scene) << std::endl;

            viewCloud(scene, "Upsampled scene");
        }


        if (downsample)
        {
            // Downsample
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
            scene = downsampleCloud(scene, 0.005);

            // Filter scene from NaNs
            std::vector<int> downsampled_indices;
            pcl::removeNaNFromPointCloud(*scene, *scene, downsampled_indices);
            std::cout << "Downsampled cloud size after NaNs size: " << downsampled_indices.size() << std::endl;

            std::cout << "Resolution of Downsampled Scene cloud is " << computeCloudResolution(scene) << std::endl;
            viewCloud(scene, "Downsampled scene");
        }


        std::cout << "Scene contains " << scene->width * scene->height << " points" << std::endl;


        // Clustering
        std::vector<pcl::PointIndices> cluster_indices = clustering(scene, min_clust_points, max_clust_points, clust_tolerance);

        std::cout << "cluster_indices size: " << cluster_indices.size() << std::endl;


        pcl::PointCloud<pcl::VFHSignature308>::Ptr cloud_cluster_features (new pcl::PointCloud<pcl::VFHSignature308>);
        for (pcl::PointIndices indices: cluster_indices)
        {

            pcl::PointIndices::Ptr cluster_point_indices(new  pcl::PointIndices(indices)); // This conversion is because of PCL's shared boost ptr
            std::cout << "Cluster indices: " << cluster_point_indices->indices.size() << std::endl;

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);

            {
            pcl::ExtractIndices<pcl::PointXYZ> extract(false);
            extract.setInputCloud(scene);
            extract.setIndices(cluster_point_indices);
            extract.filter(*cloud_cluster);
            }

            alignCloudAlongZ(cloud_cluster);
            // exit(-1);

            // Normals computation
            pcl::PointCloud<pcl::Normal>::Ptr cluster_normals = computeNormals(cloud_cluster, false, 0.01);
            std::cout << "Cluster normals computed" << std::endl;

            // Compute cluster centroid
            Eigen::Vector4f cloud_cluster_centroid;
            pcl::compute3DCentroid(*cloud_cluster, cloud_cluster_centroid);

            // Descriptor computation
            pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptor_cluster(new pcl::PointCloud<pcl::VFHSignature308>);
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> ourcvfh_transforms;
            std::vector<bool> ourcvfh_valid_roll_transforms;
            std::vector<pcl::PointIndices> ourcvfh_cluster_indices;
            std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> ourcvfh_centroids;
            if (descriptor_name == "VFH")
                descriptor_cluster = computeVFH(cloud_cluster, cluster_normals);
            if (descriptor_name == "CVFH")
                descriptor_cluster = computeCVFH(cloud_cluster, cluster_normals);
            if (descriptor_name == "OURCVFH")
                descriptor_cluster = computeOURCVFH(cloud_cluster, cluster_normals,
                                                    ourcvfh_transforms, ourcvfh_valid_roll_transforms,
                                                    ourcvfh_cluster_indices, ourcvfh_centroids);

            std::cout << "Cluster " << descriptor_name << " signature computed" << std::endl;

            // CRH computation
            if (descriptor_name == "VFH" || descriptor_name == "CVFH")
            {
                pcl::PointCloud<CRH90>::Ptr cluster_CRH = computeCRH(cloud_cluster, cluster_normals);
                std::cout << "Cluster CRH signature computed" << std::endl;
            }

            int n_regions_per_cluster = descriptor_cluster->width * descriptor_cluster->height;

            std::cout << "Cluster " << descriptor_name << " signature has " << n_regions_per_cluster << " regions" << std::endl;
            std::cout << "ourcvfh_transforms size: " << ourcvfh_transforms.size() << std::endl;
            std::cout << "ourcvfh_valid_roll_transforms size: " << ourcvfh_valid_roll_transforms.size() << std::endl;
            std::cout << "ourcvfh_cluster_indices size: " << ourcvfh_cluster_indices.size() << std::endl;
            std::cout << "ourcvfh_centroids size: " << ourcvfh_centroids.size() << std::endl;


            // for (auto &point: *descriptor_cluster)
            for (int j = 0; j < n_regions_per_cluster; ++j)
            {
                std::cout << std::endl;
                std::cout << "***************************************************************************" << std::endl;
                std::cout << "*                                                                         *" << std::endl;
                std::cout << "*                        Cluster Region                                   *" << std::endl;
                std::cout << "*                                                                         *" << std::endl;
                std::cout << "***************************************************************************" << std::endl << std::endl;


                // Cluster region
                auto point = descriptor_cluster->points[j];

                // Query point
                int histogram_size = 308;
                flann::Matrix<float> p = flann::Matrix<float>(new float[descriptor_size], 1, descriptor_size);
                memcpy (&p.ptr ()[0], &point.histogram[0], p.cols * p.rows * sizeof (float));
                // for (int i = 0; i < histogram_size; ++i)
                //     p[0][i] = vfhs_cluster->points[0].histogram[i];

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

                for (int i = 0; i < k; ++i)
                {
                    std::string cloud_name;
                    int flann_index = k_indices[0][i];
                    if (descriptor_name == "VFH")
                        cloud_name = std::to_string(flann_index + 1); // NOTICE the + 1
                    else if (descriptor_name == "CVFH" || descriptor_name == "OURCVFH")
                    {
                        int real_view_index = clustered_view_files_vec.at(flann_index); // view name
                        cloud_name = std::to_string(real_view_index);
                        flann_index = real_view_index - 1;
                    }

                    // boost::replace_last (cloud_name, "_vfh", "");
                    std::cout << "Cloud name: " << pcd_dir_name + "/" + cloud_name + ".pcd" << std::endl;

                    pcl::PointCloud<pcl::PointXYZ> cloud_xyz;
                    pcl::console::print_highlight ( stderr, "Loading "); pcl::console::print_value (stderr, "%s ", cloud_name.c_str ());
                    if (pcl::io::loadPCDFile (pcd_dir_name + "/" + cloud_name + ".pcd", cloud_xyz) == -1)
                      break;

                    if (cloud_xyz.size() == 0)
                      break;

                    pcl::console::print_info ("[done, ");
                    pcl::console::print_value ("%zu", static_cast<std::size_t>(cloud_xyz.size ()));
                    pcl::console::print_info (" points]\n");
                    // pcl::console::print_info ("Available dimensions: ");
                    // pcl::console::print_value ("%s\n", pcl::getFieldsList (cloud).c_str ());


                    // Get matching view pose
                    Eigen::Matrix4f matching_pose = poses_new.at(flann_index);
                    std::cout << "Matching Pose: \n" << matching_pose << std::endl;

                    // Get matching view CRH
                    std::vector<float> matching_crh = crh_vecs.at(flann_index);

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

                    Eigen::Matrix4f final_transform;

                    if (descriptor_name == "OURCVFH")
                    {
                        Eigen::Matrix4f T_kli, T_cen, guess;
                        Eigen::Matrix3f sensor_rotation( cloud_xyz_ptr->sensor_orientation_);
                        Eigen::Vector4f sensor_translation;
                        sensor_translation = cloud_xyz_ptr->sensor_origin_;
                        //Transformation from local object reference frame to kinect frame (as it was during database acquisition)
                        T_kli << sensor_rotation(0,0), sensor_rotation(0,1), sensor_rotation(0,2), sensor_translation(0),
                        sensor_rotation(1,0), sensor_rotation(1,1), sensor_rotation(1,2), sensor_translation(1),
                        sensor_rotation(2,0), sensor_rotation(2,1), sensor_rotation(2,2), sensor_translation(2),
                        0,                    0,                    0,                    1;

                        std::cout << "Here" << std::endl;

                        Eigen::Vector4f viewCentroid;
                        pcl::compute3DCentroid(*cloud_xyz_ptr, viewCentroid);

                        // T_cen <<           1, 0, 0, ourcvfh_centroids.at(j)[0] - viewCentroid[0],
                        //                    0, 1, 0, ourcvfh_centroids.at(j)[1] - viewCentroid[1],
                        //                    0, 0, 1, ourcvfh_centroids.at(j)[2] - viewCentroid[2],
                        //                    0,        0,         0, 1;

                        T_cen <<           1, 0, 0, cloud_cluster_centroid[0] - viewCentroid[0],
                                           0, 1, 0, cloud_cluster_centroid[1] - viewCentroid[1],
                                           0, 0, 1, cloud_cluster_centroid[2] - viewCentroid[2],
                                           0,        0,         0, 1;

                        std::cout << "Here" << std::endl;

                        final_transform = T_cen*T_kli;; // ourcvfh_transforms.at(j).inverse();


                    }

                    if (descriptor_name == "VFH" || descriptor_name == "CVFH" || ourcvfh_cluster_indices.size() == 0)
                    {
                        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> transforms = alignCRHTransforms(cloud_xyz_ptr, cloud_cluster, view_normals, cluster_normals);
                        matching_pose = transforms.at(0);
                        final_transform = matching_pose;
                    }

                    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::transformPointCloud(*cloud_xyz_ptr, *aligned_cloud, final_transform);

                    if (icp)
                    {
                        pcl::ScopeTime("ICP");
                        ICP(cloud_cluster, aligned_cloud);
                    }


                    int view_size = cloud_xyz_ptr->width * cloud_xyz_ptr->height;
                    int inliers =  countInliers(cloud_cluster, aligned_cloud);
                    double inliers_percentage = static_cast<double>(inliers) / static_cast<double>(view_size);
                    std::cout << "View points count: " << inliers << std::endl;
                    std::cout << "Inlier points count: " << inliers << std::endl;
                    std::cout << "Percentage of inliers: " << inliers_percentage << std::endl;

                    // top_pair.second = cloud_name;
                    // top_pair.first = inliers_percentage;
                    // top_candidates_map.insert(top_pair);


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

                }
                std::cout << "Done showing all neighbours" << std::endl;
            }

        }

    }


    return 0;
}
