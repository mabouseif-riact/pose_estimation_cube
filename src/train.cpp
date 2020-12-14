#include <pcl/io/vtk_lib_io.h>
#include <vtkPolyDataMapper.h>
#include <pcl/apps/render_views_tesselated_sphere.h>
#include <fstream>      // std::ofstream
#include <experimental/filesystem>
#include "features.h"
#include "helper_functions.h"
#include "visualization.h"
// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/highgui/highgui.hpp"
#include <thread>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <fstream>
#include <pcl/console/parse.h>
#include <cstdio>
#include <iterator>

using namespace std::chrono_literals;


bool vfh = false;
bool cvfh = false;
bool ourcvfh = false;
std::string descriptor_name;

std::string base_dir = "/home/mohamed/drive/ros_ws/riact_ws/src/skiros2_examples/src/skiros2_examples/turtle_test/pose_estimation_cube";
// std::string base_dir = "/home/mohamed/riact_ws/src/skiros2_examples/src/skiros2_examples/turtle_test/pose_estimation";
std::string pcd_dir_name = base_dir + "/data/views_";
std::string poses_dir_name = base_dir + "/data/poses";
std::string CRH_dir_name = base_dir + "/data/CRH";

std::string view_names_vec_file = base_dir + "/data/view_names"; // .vec";
std::string training_data_h5_file_name = base_dir + "/data/training_data"; // .h5";
std::string kdtree_idx_file_name = base_dir + "/data/kdtree"; // .idx";
std::string training_data_list_file_name = "training_data"; // .list";



void
showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*                        Usage Guide                                      *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " [DESCRIPTOR]" << std::endl << std::endl;
  std::cout << "[DESCRIPTOR]:" << std::endl;
  std::cout << "     -h:                     Show help" << std::endl;
  std::cout << "     --vfh:                  VFH Descriptor" << std::endl;
  std::cout << "     --cvfh:                 CVFH Descriptor" << std::endl;
  std::cout << "     --ourcvfh:              OURCVFH Descriptor" << std::endl;
}


void parseCommandLine(int argc, char *argv[])
{

    //Show help
    if (pcl::console::find_switch (argc, argv, "-h"))
    {
        showHelp (argv[0]);
        exit (0);
    }

    //Program behavior
    if (pcl::console::find_switch (argc, argv, "--vfh"))
    {
        vfh = true;
        descriptor_name = "VFH";
        std::cout << "Chosen descriptor is VFH" << std::endl;
    }
    if (pcl::console::find_switch (argc, argv, "--cvfh"))
    {
        cvfh = true;
        descriptor_name = "CVFH";
        std::cout << "Chosen descriptor is CVFH" << std::endl;
    }
    if (pcl::console::find_switch (argc, argv, "--ourcvfh"))
    {
        ourcvfh = true;
        descriptor_name = "OURCVFH";
        std::cout << "Chosen descriptor is OURCVFH" << std::endl;
    }

    if (!(vfh || cvfh || ourcvfh))
    {
        std::cerr << "\n\nNo descriptors chosen!" << std::endl << std::endl;
        showHelp(argv[0]);
        exit(-1);
    }

}




int main(int argc, char* argv[])
{

    // Parse args
    parseCommandLine(argc, argv);

    // Create visualizer
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer with custom color"));
    viewer->setBackgroundColor(0, 0, 0);

    // Read poses file
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> poses_new = openData(poses_dir_name + "/poses.txt");
    std::cout << "Size of poses vector: " << poses_new.size() << std::endl;

    // Read CRH file
    std::vector<std::vector<float>> crh_vecs = readCRH(CRH_dir_name + "/CRH.txt");
    std::cout << "Hist vecs size: " << crh_vecs.size() << std::endl;

    int count = 0;
    char* descriptor_options_arr[] = {"VFH", "CVFH", "OURCVFH"};
    std::vector<std::string> descriptor_options_vec(descriptor_options_arr, descriptor_options_arr + sizeof(descriptor_options_arr) / sizeof(descriptor_options_arr[0]));
    std::map<std::string, std::vector<vfh_model>> descriptor_to_vfh_model_vec_map;
    std::vector<vfh_model> vfh_vfh_models_vec;
    std::vector<vfh_model> cvfh_vfh_models_vec;
    std::vector<vfh_model> ourvfh_vfh_models_vec;
    descriptor_to_vfh_model_vec_map.insert(std::pair<std::string, std::vector<vfh_model>> (descriptor_options_vec.at(0), vfh_vfh_models_vec));
    descriptor_to_vfh_model_vec_map.insert(std::pair<std::string, std::vector<vfh_model>> (descriptor_options_vec.at(1), cvfh_vfh_models_vec));
    descriptor_to_vfh_model_vec_map.insert(std::pair<std::string, std::vector<vfh_model>> (descriptor_options_vec.at(2), ourvfh_vfh_models_vec));

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    int descriptor_cloud_size;

    for (const auto & entry : std::experimental::filesystem::directory_iterator(pcd_dir_name))
    {
        if (pcl::io::loadPCDFile<pcl::PointXYZ> (entry.path(), *cloud) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
            return (-1);
        }

        // Take path, extract view name (which are numbers), extract that number calling PCDIndex
        // subtract one from that number to get its corresponding index in the CRH and Pose files
        std::string path = entry.path().string();
        int pose_idx = PCDIndex(path) - 1;

        // Compute normals
        pcl::PointCloud<pcl::Normal>::Ptr object_normals = computeNormals(cloud, true, 0.015);

        // Compute Descriptor
        pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptor_cloud(new pcl::PointCloud<pcl::VFHSignature308>);

        for (std::string& descriptor_name: descriptor_options_vec)
        {
            std::cout << "Descriptor name: " << descriptor_name << std::endl;
            descriptor_cloud = computeVFHBasedDescriptor(cloud, object_normals, descriptor_name);

            // descriptor_cloud_size = descriptor_cloud->height * descriptor_cloud->width;
            // if (!(descriptor_cloud_size > 0))
            // {
            //     std::cout << "Could not compute features for cloud " << count << std::endl;
            //     exit(-1);
            // }

            auto idx_it = descriptor_to_vfh_model_vec_map.find(descriptor_name);
            if (idx_it != descriptor_to_vfh_model_vec_map.end())
                populateFeatureVector(descriptor_cloud, idx_it->second, pose_idx);
        }

        ++count;
        std::cout << "Count: " << count << std::endl;
    }


    std::string descriptor_name;
    for (auto it = descriptor_to_vfh_model_vec_map.begin(); it != descriptor_to_vfh_model_vec_map.end(); ++it)
    {
        descriptor_name = it->first;
        std::vector<vfh_model> m = it->second;
        convertToFLANN(m,
                       training_data_h5_file_name + "_" + descriptor_name + ".h5",
                       kdtree_idx_file_name + "_" + descriptor_name + ".idx",
                       view_names_vec_file + "_" + descriptor_name + ".vec");
    }


    return 1;
}
