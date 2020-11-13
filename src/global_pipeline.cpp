#include <pcl/io/vtk_lib_io.h>
#include <vtkPolyDataMapper.h>
#include <pcl/apps/render_views_tesselated_sphere.h>
#include <fstream>      // std::ofstream
#include <experimental/filesystem>
#include "features.h"
#include "helper_functions.h"
#include "visualization.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <thread>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

using namespace std::chrono_literals;



int PCDIndex(std::string path)
{
    // std::string path = entry.path().string();
    std::string delimiter = ".";
    std::string token = path.substr(0, path.find(delimiter));
    std::string num_str = "";
    for (std::string::reverse_iterator rev_it = token.rbegin(); rev_it != token.rbegin() + 2; ++rev_it)
    {
        if ((int)*rev_it >= 48 && (int)*rev_it <= 57)
            num_str += *rev_it;
    }
    std::reverse(num_str.begin(), num_str.end());

    return std::stoi(num_str);
}



std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> openData(std::string fileToOpen)
{

    // the inspiration for creating this function was drawn from here (I did NOT copy and paste the code)
    // https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix

    // the input is the file: "fileToOpen.csv":
    // a,b,c
    // d,e,f
    // This function converts input file data into the Eigen matrix format



    // the matrix entries are stored in this variable row-wise. For example if we have the matrix:
    // M=[a b c
    //    d e f]
    // the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector
    // later on, this vector is mapped into the Eigen matrix format
    std::vector<double> matrixEntries;

    // in this object we store the data from the matrix
    std::ifstream matrixDataFile(fileToOpen);

    // this variable is used to store the row of the matrix that contains commas
    std::string matrixRowString;

    // this variable is used to store the matrix entry;
    std::string matrixEntry;

    // this variable is used to track the number of rows
    int matrixRowNumber = 0;

    double a, b, c, d;
    Eigen::Matrix4f temp_mat;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> poses_new;


    while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    {
        std::istringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.

        matrixRowStringStream >> a >> b >> c >> d;
        temp_mat.row(matrixRowNumber) << a, b, c, d;

        matrixRowNumber++; //update the column

        if (((matrixRowNumber+1) % 5) == 0)
        {
            matrixRowNumber = 0;
            poses_new.push_back(temp_mat);
        }

    }

    // here we convet the vector variable into the matrix and return the resulting object,
    // note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
    // return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);

    return poses_new;
}







int main(int argc, char* argv[])
{

    // Create visualizer
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer with custom color"));
    viewer->setBackgroundColor(0, 0, 0);

    // std::string base_dir = "/home/mohamed/riact_ws/src/skiros2_examples/src/skiros2_examples/turtle_test/pose_estimation";
    std::string base_dir = "/home/mohamed/drive/ros_ws/riact_ws/src/skiros2_examples/src/skiros2_examples/turtle_test/pose_estimation_cube";
    std::string pcd_dir_name = base_dir + "/data/views_";
    std::string poses_dir_name = base_dir + "/data/poses";


    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> poses_new = openData(poses_dir_name + "/poses.txt");
    std::cout << "Size of poses vector: " << poses_new.size() << std::endl;
    // for (auto p: poses_new)
    //     std::cout << p << " \n" << std::endl;


    // Convert data into FLANN format
    size_t n_train = poses_new.size(); // Could also be grabbed from the number of pcd files
    int descriptor_size = 308; // VFH
    flann::Matrix<float> data (new float[n_train * descriptor_size], n_train, descriptor_size);
    int count = 0;
    // flann::Matrix<float> data (new float[models.size () * models[0].second.size ()], models.size (), models[0].second.size ());

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto & entry : std::experimental::filesystem::directory_iterator(pcd_dir_name))
    {
        if (pcl::io::loadPCDFile<pcl::PointXYZ> (entry.path(), *cloud) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
            return (-1);
        }

        std::string path = entry.path().string();
        int pose_idx = PCDIndex(path) - 1;

        // std::cout << path << std::endl;
        // std::cout << "Pose idx: " << pose_idx << std::endl;
        // std::cout << poses_new.at(pose_idx) << std::endl;

        // Scaling object model
        scaleCloud(cloud, 0.079); // 0.079

        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud,*cloud, indices);

        pcl::PointCloud<pcl::Normal>::Ptr object_normals = computeNormals(cloud, true, 0.015);
        pcl::PointCloud<CRH90>::Ptr object_CRH = computeCRH(cloud, object_normals);
        pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs_object = computeVFH(cloud, object_normals);
        // pcl::PointCloud<pcl::VFHSignature308>::Ptr OURCVFHS_object(new pcl::PointCloud<pcl::VFHSignature308>);
        // VFHS_object = computeOURCVFH(cloud, object_normals);

        // pcl::PointCloud <pcl::VFHSignature308> point = *OURCVFHS_object;
        std::cout << "point size: " << cloud->width * cloud->height << std::endl;
        // std::cout << "cloud size: " << OURCVFHS_object->width * OURCVFHS_object->height << std::endl;
        std::cout << "cloud size: " << vfhs_object->width * vfhs_object->height << std::endl;
        std::cout << "normals size: " << object_normals->width * object_normals->height << std::endl;

        vfhs_object->points[0].histogram;

        std::cout << "Histogram size: " << sizeof(vfhs_object->points[0].histogram) / sizeof(vfhs_object->points[0].histogram[0]) << std::endl;

        for (int i = 0; i < 308; ++i)
            data[count][i] = vfhs_object->points[0].histogram[i];

        ++count;
        std::cout << "Count: " << count << std::endl;



        pcl::KdTreeFLANN<pcl::VFHSignature308> match_search;
        match_search.setInputCloud (vfhs_object);
        // int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 
        //                                                 1, 
        //                                                 neigh_indices, 
        //                                                 neigh_sqr_dists);



        // Visualization
        // viewer->addPointCloud<pcl::PointXYZ> (cloud, entry.path());
        // while (!viewer->wasStopped())
        // {
        //     viewer->spinOnce(100);
        //     std::this_thread::sleep_for(50ms);
        //     cv::waitKey(1);
        // }

        // viewer->removePointCloud(entry.path());
        // viewer->resetStoppedFlag();
    }



    // std::cout << poses_new.at(0).size() << std::endl;
    // flann::Matrix<float> data (new float[poses_new.size () * poses_new[0].size()], poses_new.size () * poses_new[0].size());

    // std::cout << CVFHS_object[0].histogram.size() << std::endl;










}
