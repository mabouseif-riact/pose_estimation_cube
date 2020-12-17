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
#include <pcl/registration/correspondence_estimation.h>
#include <map>



class PoseEstimator
{
    double x_low = -0.2;
    double x_high = 0.2;
    double z_low = -0.7;
    double z_high = 0.7;
    int min_clust_points = 50;
    int max_clust_points = 10000;
    float clust_tolerance = 0.02;
    bool upsample = false;
    bool downsample = false;
    bool plane_seg = true;
    bool sor = true;
    bool icp = true;
    // std::string criterion("inliers");

    void loadParams();

public:
    PoseEstimator();
    pcl::PointCloud<pcl::PointXYZ>::Ptr estimate(pcl::PointCloud<pcl::PointXYZ>::Ptr scene);
};
