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
#include <pcl/filters/extract_indices.h>
#include <thread>

using namespace std::chrono_literals;



int
main(int argc, char** argv)
{
    // Load the PLY model from a file.
    vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
    reader->SetFileName(argv[1]);
    reader->Update();

    // VTK is not exactly straightforward...
    vtkSmartPointer < vtkPolyDataMapper > mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    mapper->Update();

    vtkSmartPointer<vtkPolyData> object = mapper->GetInput();


    int resolution = 100; // 150
    int tesselation_level = 1;
    int use_vertices = false;
    // Virtual scanner object.
    pcl::apps::RenderViewsTesselatedSphere render_views;
    render_views.addModelFromPolyData(object);
    // Pixel width of the rendering window, it directly affects the snapshot file size.
    render_views.setResolution(resolution);
    // Horizontal FoV of the virtual camera.
    render_views.setViewAngle(57.0f); //  * 1.1
    // Radius of the sphere where the virtual camera will be placed
    render_views.setRadiusSphere(1.f);
    // If true, the resulting clouds of the snapshots will be organized.
    render_views.setGenOrganized(true);
    // How much to subdivide the icosahedron. Increasing this will result in a lot more snapshots.
    render_views.setTesselationLevel(tesselation_level); // 1
    // If true, the camera will be placed at the vertices of the triangles. If false, at the centers.
    // This will affect the number of snapshots produced (if true, less will be made).
    // True: 42 for level 1, 162 for level 2, 642 for level 3...
    // False: 80 for level 1, 320 for level 2, 1280 for level 3...
    render_views.setUseVertices(use_vertices);
    // If true, the entropies (the amount of occlusions) will be computed for each snapshot (optional).
    render_views.setComputeEntropies(true);

    render_views.generateViews();

    // Object for storing the rendered views.
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> views;
    // Object for storing the poses, as 4x4 transformation matrices.
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses;
    // Object for storing the entropies (optional).
    std::vector<float> entropies;
    // generated pointclouds in camera coordinates
    render_views.getViews(views);
    // 4x4 matrices representing the pose of the cloud relative to the model coordinate system
    render_views.getPoses(poses);
    render_views.getEntropies(entropies);


    // std::cout << views.at(0)->width * views.at(0)->height << std::endl;

    // std::string base_dir = "/home/mohamed/turtle_test_link/pose_estimation_cube";
    std::string base_dir = "/home/mohamed/riact_ws/src/skiros2_examples/src/skiros2_examples/turtle_test/pose_estimation";
    std::string pcd_dir_name = base_dir + "/data/views_";
    std::string poses_dir_name = base_dir + "/data/poses";
    std::string CRH_dir_name = base_dir + "/data/CRH";
    std::ofstream poses_file (poses_dir_name + "/poses.txt");
    std::ofstream CRH_file (CRH_dir_name + "/CRH.txt");


    for (size_t i = 0; i < views.size(); ++i)
    {
        std::cout << "View " << i + 1 << std::endl;
        // Scale view down
        scaleCloud(views.at(i), -0.079); // 0.079

        // Filter view from NaNs
        std::cout << "Points in cloud: " << views.at(i)->width * views.at(i)->height << std::endl;
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*views.at(i),*views.at(i), indices);
        std::cout << "Number of non-NaN indices: " << indices.size() << std::endl;
        std::cout << "Points in cloud after NaN removal: " << views.at(i)->width * views.at(i)->height << std::endl;

        // Compute normals
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = computeNormals(views.at(i), true, 0.015);

        // Filter Normals from NaNs
        std::cout << "Points in Normal cloud: " << cloud_normals->width * cloud_normals->height << std::endl;
        std::vector<int> normal_indices;
        pcl::removeNaNNormalsFromPointCloud(*cloud_normals,*cloud_normals, normal_indices);
        std::cout << "Number of Normal non-NaN indices: " << normal_indices.size() << std::endl;
        std::cout << "Points in Normal cloud after NaN removal: " << cloud_normals->width * cloud_normals->height << std::endl;

        // Filter view from points that have NaN Normals
        pcl::ExtractIndices<pcl::PointXYZ> eifilter (true); // Initializing with true will allow us to extract the removed indices
        boost::shared_ptr<std::vector<int> > indicesptr (new std::vector<int> (normal_indices)); // conversion to boost
        eifilter.setInputCloud (views.at(i));
        eifilter.setIndices (indicesptr);
        eifilter.filter (*views.at(i));
        std::cout << "Points in cloud after Normal NaN removal: " << views.at(i)->width * views.at(i)->height << std::endl;

        // Save view to PCD file
        pcl::io::savePCDFileBinary(pcd_dir_name + "/" + std::to_string(i+1) + ".pcd", *views.at(i));

        // Write poses to file
        std::cout << "Writing poses.." << std::endl;
        poses_file << poses.at(i) << "\n";
        std::cout << "Done writing poses.." << std::endl;

        // Compute CRH
        pcl::PointCloud<CRH90>::Ptr cloud_CRH = computeCRH(views.at(i), cloud_normals);
        std::cout << "cloud_CRH size: " << cloud_CRH->width * cloud_CRH->height << std::endl;

        std::cout << "\n\n\n" << std::endl;

        // Write CRH to file
        std::cout << "Writing CRH.." << std::endl;
        for (int i = 0; i < sizeof(cloud_CRH->points[0].histogram) / sizeof(cloud_CRH->points[0].histogram[0]); ++i)
        {

            std::cout << cloud_CRH->points[0].histogram[i] << " ";
            CRH_file << cloud_CRH->points[0].histogram[i] << " ";
        }
        CRH_file <<"\n";
        std::cout << std::endl;
        // std::cout << "Done writing CRH.." << std::endl;

    }

    // Close file streams
    poses_file.close();
    CRH_file.close();




    // std::vector<std::vector<float>> hist_vecs = readCRH(CRH_dir_name + "/CRH.txt");

    // for (int i = 0; i < hist_vecs[1].size(); ++i)
    //     std::cout << hist_vecs[1][i] << " ";
    // std::cout << std::endl;


}
