#include "PoseEstimatorClass.h"






PoseEstimator::PoseEstimator()
{
    this->loadParams();
}



void PoseEstimator::loadParams()
{
    // Paths
    // std::string base_dir = "/home/mohamed/turtle_test_link/pose_estimation_cube";
    std::string base_dir = "/home/mohamed/riact_ws/src/skiros2_examples/src/skiros2_examples/turtle_test/pose_estimation";
    std::string pcd_dir_name = base_dir + "/data/views_";
    std::string poses_dir_name = base_dir + "/data/poses";
    std::string CRH_dir_name = base_dir + "/data/CRH";

    std::string view_names_vec_file = base_dir + "/data/view_names_OURCVFH.vec";
    std::string training_data_h5_file_name = base_dir + "/data/training_data_OURCVFH.h5";
    std::string kdtree_idx_file_name = base_dir + "/data/kdtree_OURCVFH.idx";
    std::string training_data_list_file_name = "training_data_OURCVFH.list";

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
        exit (-1);
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
        exit (-1);
    }

    std::cout << "Params loaded successfully." << std::endl;


}



Eigen::Matrix4f PoseEstimator::estimate(pcl::PointCloud<pcl::PointXYZ>::Ptr scene)
{

    // pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Client Viewer"));

    // Paths
    // std::string base_dir = "/home/mohamed/turtle_test_link/pose_estimation_cube";
    std::string base_dir = "/home/mohamed/riact_ws/src/skiros2_examples/src/skiros2_examples/turtle_test/pose_estimation";
    std::string pcd_dir_name = base_dir + "/data/views_";
    std::string poses_dir_name = base_dir + "/data/poses";
    std::string CRH_dir_name = base_dir + "/data/CRH";

    std::string view_names_vec_file = base_dir + "/data/view_names_OURCVFH.vec";
    std::string training_data_h5_file_name = base_dir + "/data/training_data_OURCVFH.h5";
    std::string kdtree_idx_file_name = base_dir + "/data/kdtree_OURCVFH.idx";
    std::string training_data_list_file_name = "training_data_OURCVFH.list";

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
        exit (-1);
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
        exit (-1);
    }

    std::cout << "Params loaded successfully." << std::endl;
    std::cout << "Scene cloud size: " << scene->width * scene->height << std::endl;

    sceneResult scene_res;
    std::vector<sceneResult> scene_result_vec;

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

    scene->is_dense = false;

    // Filter scene from NaNs
    std::vector<int> valid_indices;
    pcl::removeNaNFromPointCloud(*scene, *scene, valid_indices);

    // Clustering
    std::vector<pcl::PointIndices> cluster_indices = clustering(scene, min_clust_points, max_clust_points, clust_tolerance);

    std::cout << "N clusers: " << cluster_indices.size() << std::endl;

    // viewCloud(scene, "scene_filtered");

    std::multimap<double, candidateDictionary> transform_lookup_fitness;
    std::multimap<double, candidateDictionary> transform_lookup_inliers;
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

        // Compensation of rotation about Y axis (along Z) to match how original generated views
        Eigen::Matrix4f align_transform = alignCloudAlongZ(cloud_cluster);
        pcl::transformPointCloud(*cloud_cluster, *cloud_cluster, align_transform);

        // Cluster normals computation
        pcl::PointCloud<pcl::Normal>::Ptr cluster_normals = computeNormals(cloud_cluster, false, 0.01);

        // Compute cluster centroid
        Eigen::Vector4f cloud_cluster_centroid;
        pcl::compute3DCentroid(*cloud_cluster, cloud_cluster_centroid);

        // Descriptor computation
        pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptor_cluster(new pcl::PointCloud<pcl::VFHSignature308>);
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> ourcvfh_transforms;
        std::vector<bool> ourcvfh_valid_roll_transforms;
        std::vector<pcl::PointIndices> ourcvfh_cluster_indices;
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> ourcvfh_centroids;

        std::string descriptor_name("OURCVFH");

        if (descriptor_name == "OURCVFH")
            descriptor_cluster = computeOURCVFH(cloud_cluster, cluster_normals,
                                                ourcvfh_transforms, ourcvfh_valid_roll_transforms,
                                                ourcvfh_cluster_indices, ourcvfh_centroids);

        std::cout << "Cluster " << descriptor_name << " signature computed" << std::endl;

        // CRH computation
        if (descriptor_name == "VFH" || descriptor_name == "CVFH")
        {
            pcl::PointCloud<CRH90>::Ptr cluster_CRH = computeCRH(cloud_cluster, cluster_normals);
        }

        int n_regions_per_cluster = descriptor_cluster->width * descriptor_cluster->height;


        // for (auto &point: *descriptor_cluster)
        for (int j = 0; j < n_regions_per_cluster; ++j)
        {

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

                pcl::PointCloud<pcl::PointXYZ> cloud_xyz;
                if (pcl::io::loadPCDFile (pcd_dir_name + "/" + cloud_name + ".pcd", cloud_xyz) == -1)
                  break;

                if (cloud_xyz.size() == 0)
                  break;

                // Get matching view pose
                Eigen::Matrix4f matching_pose = poses_new.at(flann_index);

                // Get matching view CRH
                std::vector<float> matching_crh = crh_vecs.at(flann_index);

                // Match CRH between view and cluster
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_ptr(new pcl::PointCloud<pcl::PointXYZ>);
                if (pcl::io::loadPCDFile (pcd_dir_name + "/" + cloud_name + ".pcd", *cloud_xyz_ptr) == -1)
                  break;

                pcl::PointCloud<pcl::Normal>::Ptr view_normals = computeNormals(cloud_xyz_ptr, false, 0.01);

                Eigen::Matrix4f final_transform;
                candidateDictionary temp_dict;

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

                    Eigen::Vector4f viewCentroid;
                    pcl::compute3DCentroid(*cloud_xyz_ptr, viewCentroid);

                    T_cen <<           1, 0, 0, cloud_cluster_centroid[0] - viewCentroid[0],
                                       0, 1, 0, cloud_cluster_centroid[1] - viewCentroid[1],
                                       0, 0, 1, cloud_cluster_centroid[2] - viewCentroid[2],
                                       0,        0,         0, 1;

                    final_transform = T_cen*T_kli;; // ourcvfh_transforms.at(j).inverse();

                    temp_dict.view_frame_transform = final_transform;


                }

                if (descriptor_name == "VFH" || descriptor_name == "CVFH" || ourcvfh_cluster_indices.size() == 0)
                {
                    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> transforms = alignCRHTransforms(cloud_xyz_ptr, cloud_cluster, view_normals, cluster_normals);
                    matching_pose = transforms.at(0);
                    final_transform = matching_pose;

                    temp_dict.view_frame_transform = getCloudTransform(cloud_xyz_ptr);
                }

                pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                pcl::transformPointCloud(*cloud_xyz_ptr, *aligned_cloud, final_transform);

                Eigen::Matrix4f icp_transform;
                double fitness_score = DBL_MAX;

                if (icp)
                {
                    fitness_score = ICP(cloud_cluster, aligned_cloud, icp_transform);
                }
                else
                {
                    icp_transform = Eigen::Matrix4f::Identity();
                }

                int view_size = cloud_xyz_ptr->width * cloud_xyz_ptr->height;
                int inliers =  countInliers(cloud_cluster, aligned_cloud);
                double inliers_percentage = static_cast<double>(inliers) / static_cast<double>(view_size);

                int inliers_CE = countInliersCE(cloud_cluster, aligned_cloud);
                double inliers_percentage_CE = static_cast<double>(inliers_CE) / static_cast<double>(view_size);

                temp_dict.transform = align_transform.inverse() * icp_transform * final_transform;
                temp_dict.view_frame_transform = temp_dict.transform * temp_dict.view_frame_transform.inverse();
                temp_dict.cloud_name = cloud_name;

                // fitness_score += ((view_size - inliers) - inliers) * 1e-6;
                fitness_score -= (inliers_percentage) * fitness_score;
                // fitness_score += 0.001 * ((view_size - inliers)/(inliers+ 0.0001) ); // best
                // fitness_score += 0.0001 * ((view_size - inliers)/(inliers+ 0.0001) ); // best


                // transform_lookup_inliers.insert(std::pair<double, candidateDictionary>(inliers_percentage, temp_dict));
                transform_lookup_inliers.insert(std::pair<double, candidateDictionary>(inliers_percentage_CE, temp_dict));
                transform_lookup_fitness.insert(std::pair<double, candidateDictionary>(fitness_score, temp_dict));


            }
            std::cout << "Done showing all neighbours" << std::endl;
        }

    }

    scene_res.cloud_name_inliers = transform_lookup_inliers.rbegin()->second.cloud_name;
    scene_res.cloud_name_fitness = transform_lookup_fitness.begin()->second.cloud_name;
    scene_res.inliers = (transform_lookup_inliers.rbegin()->first * 100.0); // std::to_string
    scene_res.score = (transform_lookup_fitness.begin()->first); // std::to_string

    scene_result_vec.push_back(scene_res);


    pcl::PointCloud<pcl::PointXYZ>::Ptr candidate_cloud_inliers_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr candidate_cloud_fitness_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile (pcd_dir_name + "/" + transform_lookup_inliers.rbegin()->second.cloud_name + ".pcd", *candidate_cloud_inliers_ptr) == -1)
    {
        std::cout << "Could not read top inliers candidate. Exit.." << std::endl;
        exit(-1);
    }
    if (pcl::io::loadPCDFile (pcd_dir_name + "/" + transform_lookup_fitness.begin()->second.cloud_name + ".pcd", *candidate_cloud_fitness_ptr) == -1)
    {
        std::cout << "Could not read top fitness candidate. Exit.." << std::endl;
        exit(-1);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud_inliers(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud_fitness(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*candidate_cloud_inliers_ptr, *aligned_cloud_inliers, transform_lookup_inliers.rbegin()->second.transform);
    pcl::transformPointCloud(*candidate_cloud_fitness_ptr, *aligned_cloud_fitness, transform_lookup_fitness.begin()->second.transform);

    std::cout << "Aligned cloud size: " << aligned_cloud_inliers->width * aligned_cloud_inliers->height << std::endl;

    return transform_lookup_inliers.rbegin()->second.transform;
    // return aligned_cloud_inliers;

}
