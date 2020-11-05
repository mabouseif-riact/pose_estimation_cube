
#include "visualization.h"



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
