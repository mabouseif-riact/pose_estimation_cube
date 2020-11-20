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



int main(int argc, char* argv[])
{

    // Create visualizer
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer with custom color"));
    viewer->setBackgroundColor(0, 0, 0);

    // std::string base_dir = "/home/mohamed/drive/ros_ws/riact_ws/src/skiros2_examples/src/skiros2_examples/turtle_test/pose_estimation_cube";
    std::string base_dir = "/home/mohamed/riact_ws/src/skiros2_examples/src/skiros2_examples/turtle_test/pose_estimation";
    std::string pcd_dir_name = base_dir + "/data/views_";
    std::string poses_dir_name = base_dir + "/data/poses";
    std::string CRH_dir_name = base_dir + "/data/CRH";
    std::string training_data_h5_file_name = base_dir + "/data/training_data.h5";
    std::string kdtree_idx_file_name = base_dir + "/data/kdtree.idx";
    std::string training_data_list_file_name = "training_data.list";


    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> poses_new = openData(poses_dir_name + "/poses.txt");
    std::cout << "Size of poses vector: " << poses_new.size() << std::endl;
    // for (auto p: poses_new)
    //     std::cout << p << " \n" << std::endl;

    std::vector<std::vector<float>> crh_vecs = readCRH(CRH_dir_name + "/CRH.txt");
    std::cout << "Hist vecs size: " << crh_vecs.size() << std::endl;


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
        // scaleCloud(cloud, 0.079); // 0.079

        // Compute normals
        pcl::PointCloud<pcl::Normal>::Ptr object_normals = computeNormals(cloud, true, 0.015);
        // Compute CRH
        // pcl::PointCloud<CRH90>::Ptr object_CRH = computeCRH(cloud, object_normals);
        // Compute Descriptor
        pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs_object = computeVFH(cloud, object_normals);
        // pcl::PointCloud<pcl::VFHSignature308>::Ptr OURCVFHS_object(new pcl::PointCloud<pcl::VFHSignature308>);
        // VFHS_object = computeOURCVFH(cloud, object_normals);

        // pcl::PointCloud <pcl::VFHSignature308> point = *OURCVFHS_object;
        std::cout << "point size: " << cloud->width * cloud->height << std::endl;
        // std::cout << "cloud size: " << OURCVFHS_object->width * OURCVFHS_object->height << std::endl;
        std::cout << "descriptor cloud size: " << vfhs_object->width * vfhs_object->height << std::endl;
        std::cout << "normals size: " << object_normals->width * object_normals->height << std::endl;

        int histogram_size = sizeof(vfhs_object->points[0].histogram) / sizeof(vfhs_object->points[0].histogram[0]);
        std::cout << "Histogram size: " << histogram_size << std::endl;

        for (int i = 0; i < histogram_size; ++i)
            data[count][i] = vfhs_object->points[0].histogram[i];

        ++count;
        std::cout << "Count: " << count << std::endl;

        // if (count == 43)
        //     break;


        // Another way..
        // pcl::KdTreeFLANN<pcl::VFHSignature308> match_search;
        // match_search.setInputCloud (vfhs_object);
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

    std::cout << "Here" << std::endl;

    // Save data to disk (list of models)
    flann::save_to_file (data, training_data_h5_file_name, "training_data");
    // Build the tree index and save it to disk
    pcl::console::print_error ("Building the kdtree index (%s) for %d elements...\n", kdtree_idx_file_name.c_str (), (int)data.rows);
    flann::Index<flann::L1<float>> index (data, flann::LinearIndexParams());
    //flann::Index<flann::ChiSquareDistance<float> > index (data, flann::KDTreeIndexParams (4));
    index.buildIndex();
    index.save(kdtree_idx_file_name);
    delete[] data.ptr();


    // flann::load_from_file (data, training_data_h5_file_name, "training_data");
    // flann::Index<flann::ChiSquareDistance<float> > index (data, flann::SavedIndexParams (kdtree_idx_file_name));


    // flann::Matrix<int> k_indices;
    // flann::Matrix<float> k_distances;
    // // flann::Matrix<float> data;
    // int k = 3;
    // // Check if the data has already been saved to disk
    // if (!boost::filesystem::exists (training_data_h5_file_name))
    // {
    //     pcl::console::print_error ("Could not find training data models files %s!\n",
    //     training_data_h5_file_name.c_str ());
    //     return (-1);
    // }
    // else
    // {
    //     flann::load_from_file (data, training_data_h5_file_name, "training_data");
    //     pcl::console::print_highlight ("Training data found. Loaded %d VFH models from %s.\n",
    //          (int)data.rows, training_data_h5_file_name.c_str ());
    // }

    // // Check if the tree index has already been saved to disk
    // if (!boost::filesystem::exists (kdtree_idx_file_name))
    // {
    //     pcl::console::print_error ("Could not find kd-tree index in file %s!", kdtree_idx_file_name.c_str ());
    //     return (-1);
    // }
    // else
    // {
    //     flann::Index<flann::ChiSquareDistance<float> > index_loaded (data, flann::SavedIndexParams (kdtree_idx_file_name));
    //     index_loaded.buildIndex ();
    //     // nearestKSearch (index, histogram, k, k_indices, k_distances);

    //     // Query point
    //     float test_model_histogram[308]; //  = vfhs_object->points[0].histogram
    //     int histogram_size = 308;
    //     flann::Matrix<float> p = flann::Matrix<float>(new float[descriptor_size], 1, descriptor_size);
    //     memcpy (&p.ptr ()[0], &test_model_histogram[0], histogram_size * sizeof (float));


    //     flann::Matrix<int> indices = flann::Matrix<int>(new int[k], 1, k);
    //     flann::Matrix<float> distances = flann::Matrix<float>(new float[k], 1, k);
    //     index_loaded.knnSearch (p, indices, distances, k, flann::SearchParams (512));
    //     delete[] p.ptr ();

    //     pcl::console::print_highlight ("Query performed.\n");

    // }





}
