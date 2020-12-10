#include "helper_functions.h"



void passthroughFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, char* field, double min_val, double max_val)
{
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName(field);
    pass.setFilterLimits(min_val, max_val);
    pass.filter(*cloud);
}


void scaleCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double scale)
{
    // Something wrong here. It should be 3x3
    Eigen::Matrix4f affine_mat = Eigen::Matrix4f::Identity(4, 4);
    affine_mat *= scale;
    pcl::transformPointCloud(*cloud, *cloud, affine_mat);
}


void moveCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const char axis, double dist)
{
    // Something wrong here. It should be 3x3
    Eigen::Matrix4f affine_mat = Eigen::Matrix4f::Identity(4, 4);
    switch(axis)
    {
        case 'x':
            affine_mat(3, 0) = dist;
            break;
        case 'y':
            affine_mat(3, 1) = dist;
            break;
        case 'z':
            affine_mat(3, 2) = dist;
            break;
        default:
            PCL_ERROR("INVALID AXIS!");
            break;
    }

    pcl::transformPointCloud(*cloud, *cloud, affine_mat);
}


void SORFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    // // Create the filtering object
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (200);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cloud);
}



std::vector<pcl::PointIndices> clustering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int min_clust_points, int max_clust_points, float clust_tolerance)
{
    std::vector<pcl::PointIndices> cluster_indices;
    {

    pcl::ScopeTime t("Clustering");

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (clust_tolerance); // 0.02 -> 2cm
    ec.setMinClusterSize (min_clust_points); // 100
    ec.setMaxClusterSize (max_clust_points); // 25000
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud);
    ec.extract (cluster_indices);

    }

    return cluster_indices;
}




int PCDIndex(std::string path)
{
    // std::string path = entry.path().string();
    std::string delimiter = ".";
    std::string token = path.substr(0, path.find(delimiter));
    std::reverse(token.begin(), token.end());
    delimiter = '/';
    std::string num_str = token.substr(0, token.find(delimiter));
    std::reverse(num_str.begin(), num_str.end());

    std::cout << "Filename from PCDIndex: " << num_str << std::endl;

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



std::vector<std::vector<float>> readCRH(std::string file_name)
{
    std::ifstream CRH_file(file_name);
    std::string crh_string;
    float num;
    std::vector<float> temp_hist_vec;
    std::vector<std::vector<float>> hist_vecs;

    std::cout << "\n\n\n" << std::endl;
    while (getline(CRH_file, crh_string)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    {
        std::istringstream matrixRowStringStream(crh_string); //convert matrixRowString that is a string to a stream variable.
        while (matrixRowStringStream >> num)
            temp_hist_vec.push_back(num);
        hist_vecs.push_back(temp_hist_vec);
        temp_hist_vec.clear();
    }

    return hist_vecs;
}



double
computeCloudResolution (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<pcl::PointXYZ> tree;
  tree.setInputCloud (cloud);

  for (std::size_t i = 0; i < cloud->size (); ++i)
  {
    if (! std::isfinite ((*cloud)[i].x))
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }

  return res;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr upsampleCloudMLS(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float radius)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_upsampled(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_upsampled_w_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
    // polynomial fitting could be disabled for speeding up smoothing.
    // Please consult the code API (:pcl:`MovingLeastSquares <pcl::MovingLeastSquares>`)
    // for default values and additional parameters to control the smoothing process.
    mls.setPolynomialOrder (2);
    mls.setInputCloud(cloud);
    mls.setComputeNormals (false);
    // Set parameters
    mls.setSearchMethod (tree);
    mls.setSearchRadius (radius); // original 0.03  // 0.003
    // Reconstruct
    mls.process (*cloud_upsampled_w_normals);
    pcl::copyPointCloud(*cloud_upsampled_w_normals, *cloud_upsampled);

    return cloud_upsampled;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr downsampleCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size)
{
    // Create the filtering object
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setLeafSize (leaf_size, leaf_size, leaf_size); // 0.005f
    sor.filter (*cloud_downsampled);

    return cloud_downsampled;
}




Eigen::Matrix4f alignCloudAlongZ(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
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
    // pcl::transformPointCloud(*cloud, *cloud, offset_vec, q);

    Eigen::Matrix3f rot = q.toRotationMatrix();
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block(0, 0, 3, 3) = rot;
    transform.block(0, 3, 3, 1) = offset_vec;

    return transform;

}



int countInliers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr view)
{
    pcl::ScopeTime("Count Inliers");
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud);

    std::vector<float> distances(1);
    std::vector<int> indices (1);
    int k = 1;
    int k_neighbours_found;
    // double eps = 0.00000000001;
    // tree->setEpsilon(eps);
    double thresh = 0.000005; // 0.00005;



    float total_dist = 0;
    int inliers = 0;


    for (size_t i = 0; i < view->size(); ++i)
    {
        k_neighbours_found = tree->nearestKSearch(view->points[i], k, indices, distances);
        if (k_neighbours_found > 0)
        {
            if (distances.at(0) < thresh)
            {
                total_dist += distances.at(0);
                inliers += 1;
            }

        }
    }
    std::cout << "Total distance error: " << total_dist << std::endl;

    return inliers;
}


void deleteDirectoryContents(const std::string& dir_path)
{
    for (const auto& entry : std::experimental::filesystem::directory_iterator(dir_path))
        std::experimental::filesystem::remove_all(entry.path());
}


void writeVectorToFile(std::string filename, const std::vector<int>& myVector)
{
    if (std::experimental::filesystem::exists(filename))
        std::remove(filename.c_str());

    std::ofstream ofs(filename, std::ios::out | std::ofstream::binary);
    std::ostream_iterator<double> osi{ofs," "};
    std::copy(myVector.begin(), myVector.end(), osi);
}

std::vector<int> readVectorFromFile(std::string filename)
{
    std::vector<int> newVector{};
    std::ifstream ifs(filename, std::ios::in | std::ifstream::binary);
    std::istream_iterator<int> iter{ifs};
    std::istream_iterator<int> end{};
    std::copy(iter, end, std::back_inserter(newVector));
    return newVector;
}


void populateFeatureVector(const pcl::PointCloud<pcl::VFHSignature308>::ConstPtr descriptor_cloud,
       std::vector<vfh_model>& all_models,
       int pose_idx)
{
    int histogram_size = sizeof(descriptor_cloud->points[0].histogram) / sizeof(descriptor_cloud->points[0].histogram[0]);
    std::cout << "Histogram size: " << histogram_size << std::endl;

    for (auto &point: *descriptor_cloud)
    {
        vfh_model m;
        m.first = pose_idx + 1; // By adding 1 to the pose index, we get the file name
        m.second.resize(histogram_size);
        for (int i = 0; i < histogram_size; ++i)
            m.second.at(i) = point.histogram[i];
        all_models.push_back(m);
    }

    std::cout << "all_models size: " << all_models.size() << std::endl;
}


void convertToFLANN(std::vector<vfh_model> m,
                    std::string training_data_h5_file_name,
                    std::string kdtree_idx_file_name,
                    std::string view_names_vec_file)
{
    // Convert data into FLANN format
    int n_train = m.size();
    int descriptor_size = 308; // VFH
    flann::Matrix<float> data (new float[n_train * descriptor_size], n_train, descriptor_size);

    // Populate FLANN matrix with histograms
    std::vector<int> view_files;
    view_files.resize(n_train);
    for (size_t i = 0; i < n_train; ++i)
    {
        view_files.at(i) = m[i].first;
        for (int j = 0; j < descriptor_size; ++j)
            data[i][j] = m[i].second[j];
    }

    writeVectorToFile(view_names_vec_file, view_files);


    if (std::experimental::filesystem::exists(training_data_h5_file_name))
        std::remove(training_data_h5_file_name.c_str());

    if (std::experimental::filesystem::exists(kdtree_idx_file_name))
        std::remove(kdtree_idx_file_name.c_str());

    // Save data to disk (list of models)
    flann::save_to_file (data, training_data_h5_file_name, "training_data");
    // Build the tree index and save it to disk
    pcl::console::print_error ("Building the kdtree index (%s) for %d elements...\n", kdtree_idx_file_name.c_str (), (int)data.rows);
    flann::Index<flann::ChiSquareDistance<float>> index (data, flann::LinearIndexParams());
    //flann::Index<flann::ChiSquareDistance<float> > index (data, flann::KDTreeIndexParams (4));
    index.buildIndex();
    index.save(kdtree_idx_file_name);
    delete[] data.ptr();
}
