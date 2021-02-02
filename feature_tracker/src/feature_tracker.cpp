#include "feature_tracker.h"
#include <fstream>

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
//***************************************************************************//
Database::Database(int image_width,int image_height) :
    m_vignette_estimate(-0.3,0,0,image_width,image_height),
    m_response_estimate()
{
    m_image_width  = image_width;
    m_image_height = image_height;
    num = 0;
}


void Database::visualizeRapidExposureTimeEstimates(double exponent)
{
    // Visualize exposure times 
    // If GT data is available, the estimated exposure times will be aligned 
    // to the GT by computing an optimal alignment factor alignment_alpha
    // If no GT data is available, the estimated exposure is simply scaled between [0,1]
    int nr_frames_to_vis = int(m_tracked_frames.size());
    int exp_image_height = 150;
    int draw_spacing = 8;
    std::vector<double> estimated_exp_times;
    std::vector<double> gt_exp_times;
    double alignment_alpha = 1.0;
    double max_exp = -10000.0;
    double min_exp = 10000.0;
    double top = 0;
    double bot = 0;
    for(int i = 0;i < nr_frames_to_vis;i++)
    {
        // Fetch estimated and GT exposure time data, pow estimates with alignment exponent
        Frame current_frame = m_tracked_frames.at(m_tracked_frames.size()-nr_frames_to_vis+i);
        double frame_exp_time = pow(current_frame.m_exp_time,exponent);
        double frame_time_gt  = current_frame.m_gt_exp_time;

        // Keep track of max and min exposure to scale between [0,1]
        if(frame_exp_time > max_exp)
            max_exp = frame_exp_time;
        if(frame_exp_time < min_exp)
            min_exp = frame_exp_time;

        // Accumulate information for least square fit between GT and estimated exposure
        //top += frame_exp_time*frame_time_gt;
        //bot += frame_exp_time*frame_exp_time;
        
        // Push back estimated exposure values
        estimated_exp_times.push_back(frame_exp_time);

        // Push gt exposure time if available
        if(!(frame_time_gt < 0))
            gt_exp_times.push_back(frame_time_gt);
    }

    // Set alignment factor only if GT exposure is available
    /*if(gt_exp_times.size() == estimated_exp_times.size())
        alignment_alpha = top/bot;
    else
    {
        // Normalize estimated exposures between [0,1]
        for(int k = 0;k < estimated_exp_times.size();k++)
        {
            estimated_exp_times.at(k) = (estimated_exp_times.at(k)-min_exp)/(max_exp-min_exp);
        }
    }*/

     for(int k = 0;k < estimated_exp_times.size();k++)
     {
            estimated_exp_times.at(k) = (estimated_exp_times.at(k)-min_exp)/(max_exp-min_exp);
     }

    // Create exposure time canvas
    cv::Mat exposure_vis_image(exp_image_height,draw_spacing*nr_frames_to_vis,CV_8UC3,cv::Scalar(0,0,0));

    // Draw estimated exposure times as lines to graph
    for(int i = 0;i < nr_frames_to_vis-1;i++)
    {
        int drawing_y_exp_1 = exp_image_height - exp_image_height*(alignment_alpha * estimated_exp_times.at(i));
        drawing_y_exp_1 = int(fmax(0,drawing_y_exp_1));
        drawing_y_exp_1 = int(fmin(exp_image_height-1,drawing_y_exp_1));

        int drawing_y_exp_2 = exp_image_height - exp_image_height*(alignment_alpha * estimated_exp_times.at(i+1));
        drawing_y_exp_2 = int(fmax(0,drawing_y_exp_2));
        drawing_y_exp_2 = int(fmin(exp_image_height-1,drawing_y_exp_2));

        // Draw exposure lines
        cv::line(exposure_vis_image, cv::Point(draw_spacing*i,drawing_y_exp_1), cv::Point(draw_spacing*(i+1),drawing_y_exp_2), cv::Scalar(0,0,255));
     }

    // Draw GT exposure line only if GT exposure data is available
    if(gt_exp_times.size() == estimated_exp_times.size())
    {
        for(int i = 0;i < nr_frames_to_vis-1;i++)
        {
            int drawing_y_gt_exp_1 = exp_image_height - exp_image_height * gt_exp_times.at(i);
            drawing_y_gt_exp_1 = int(fmax(0,drawing_y_gt_exp_1));
            drawing_y_gt_exp_1 = int(fmin(exp_image_height-1,drawing_y_gt_exp_1));

            int drawing_y_gt_exp_2 = exp_image_height - exp_image_height * gt_exp_times.at(i+1);
            drawing_y_gt_exp_2 = int(fmax(0,drawing_y_gt_exp_2));
            drawing_y_gt_exp_2 = int(fmin(exp_image_height-1,drawing_y_gt_exp_2));

            cv::line(exposure_vis_image, cv::Point(draw_spacing*i,drawing_y_gt_exp_1), cv::Point(draw_spacing*(i+1),drawing_y_gt_exp_2), cv::Scalar(255,255,0));
        }
    }   

    /*if(nr_frames_to_vis>1)
    {
	double drawing_y_exp_1 = estimated_exp_times.at(nr_frames_to_vis-1);
	
     	estimated_exp_times.push_back(frame_exp_time);
    }*/

    std::ofstream outFile;
    outFile.open("/home/liupan/output/time.txt",std::ios::app);
    if(nr_frames_to_vis>1)
    {
        //int drawing_y_exp_1 = exp_image_height - exp_image_height*(alignment_alpha * estimated_exp_times.at(nr_frames_to_vis-1));
        //drawing_y_exp_1 = int(fmax(0,drawing_y_exp_1));
        //drawing_y_exp_1 = int(fmin(exp_image_height-1,drawing_y_exp_1));

	double drawing_y_exp_1 = estimated_exp_times.at(nr_frames_to_vis-1);

     	outFile << num  << " "<< drawing_y_exp_1 << " " << std::endl;
	num++;
    }

    cv::imshow("Estimated Exposure (Rapid)", exposure_vis_image);
    cv::moveWindow("Estimated Exposure (Rapid)", 20+20+256,20);

}

vector<cv::Point2f> Database::fetchActiveFeatureLocations()
{
    std::vector<cv::Point2f> point_locations;
    
    Frame last_frame = m_tracked_frames.at(m_tracked_frames.size()-1);
    
    for(int i = 0;i < last_frame.m_features.size();i++)
    {
        point_locations.push_back(last_frame.m_features.at(i)->m_xy_location);
    }
    
    return point_locations;
}

vector<cv::Point2f> Database::fetchActiveFeatureLocationsUndistorted()
{
    std::vector<cv::Point2f> point_locations_undistorted;
    
    Frame last_frame = m_tracked_frames.at(m_tracked_frames.size()-1);
    
    for(int i = 0;i < last_frame.m_features.size();i++)
    {
        point_locations_undistorted.push_back(last_frame.m_features.at(i)->m_xy_location_undistorted);
    }
    
    return point_locations_undistorted;
}

vector<int> Database::fetchActiveFeatureID()
{
    vector<int> point_id;
    
    Frame last_frame = m_tracked_frames.at(m_tracked_frames.size()-1);
    
    for(int i = 0;i < last_frame.m_features.size();i++)
    {
        point_id.push_back(last_frame.m_features.at(i)->id_feature);
    }
    
    return point_id;
}

void Database::removeLastFrame()
{
    // Erase the information about the first frame
    m_tracked_frames.erase(m_tracked_frames.begin());
    
    // Let all pointers of the new first frame point to NULL
    for(int i = 0;i < m_tracked_frames.at(0).m_features.size();i++)
    {
        m_tracked_frames.at(0).m_features.at(i)->m_prev_feature = NULL;
    }
}
//***************************************************************************//
VignetteModel::VignetteModel(double v1,double v2,double v3,int image_width,int image_height)
{
    // Initialize vignette parameters
    m_v1 = v1;
    m_v2 = v2;
    m_v3 = v3;
    
    // Image information is necessary to compute normalized radial information from pixel coordinates
    m_image_width = image_width;
    m_image_height= image_height;
    
    // Max. non normalized image radius (center of image = center of radial vignetting assumption)
    m_max_radius = sqrt((image_width/2)*(image_width/2) + (image_height/2)*(image_height/2));
}

double VignetteModel::getNormalizedRadius(cv::Point2f xy_location)
{
    double x = xy_location.x;
    double y = xy_location.y;
    
    double x_norm = x - m_image_width/2;
    double y_norm = y - m_image_height/2;
    
    double radius = sqrt(x_norm*x_norm + y_norm*y_norm);
    radius /= m_max_radius;
    
    return radius;
}

double VignetteModel::getVignetteFactor(cv::Point2f xy_location)
{
    double r = getNormalizedRadius(xy_location);
    double r2 = r * r;
    double r4 = r2 * r2;
    double v = 1 + m_v1 * r2 + m_v2 * r4 + m_v3 * r2 * r4;
    
    return v;
}

double VignetteModel::getVignetteFactor(double norm_radius)
{
    double r = norm_radius;
    double r2 = r * r;
    double r4 = r2 * r2;
    double v = 1 + m_v1 * r2 + m_v2 * r4 + m_v3 * r2 * r4;
    
    return v;
}

void VignetteModel::setVignetteParameters(vector<double> vignette_model)
{
    m_v1 = vignette_model.at(0);
    m_v2 = vignette_model.at(1);
    m_v3 = vignette_model.at(2);
    //m_v1 = vignette_model[0];
    //m_v2 = vignette_model[1];
    //m_v3 = vignette_model[2];
}
//***************************************************************************//
RapidExposureTimeEstimator::RapidExposureTimeEstimator(int window_size,Database* database)
{
    m_window_size = window_size;
    m_database    = database;
}

double RapidExposureTimeEstimator::estimateExposureTime()
{
    /*
     * A minimum number of m_window_size frames is required for exposure time estimation
     * Not enough frames available yet, fix exposure time to 1.0
     */
    if(m_database->m_tracked_frames.size() < m_window_size)//15
    {
        return 1.0;
    }
    
    /*
     * Iterate all features in the most current frame backwards for m_window_size images
     */
    std::vector<Feature*> features_last_frame = m_database->m_tracked_frames.at(m_database->m_tracked_frames.size()-1).m_features;
    
    double e_estimate = 0.0;
    double nr_estimates = 0;
    
    for(int i = 0;i < features_last_frame.size();i++)
    {
        //skip newly extracted features (no link to previous frame, cannot be used for exp. estimation)
        if(features_last_frame.at(i)->m_prev_feature == NULL)
            continue;
        
        // average the radiance estimate for this feature from the last m_window_size frames
        std::vector<double> radiances = features_last_frame.at(i)->m_prev_feature->m_radiance_estimates;
        
        // count the number of features used to estimate the point radiances
        int nr_features_used  = 1;
        
        // Iterate the rest of the m_window_size features in the other images to accumulate radiance information
        Feature* curr_feature = features_last_frame.at(i)->m_prev_feature;

        // Todo: k should start from 1?
        for(int k = 0;k < m_window_size;k++)
        {
            // Go one feature backwards
            curr_feature = curr_feature->m_prev_feature;
            
            // No tracking information anymore -> break
            if(curr_feature == NULL)
                break;
            
            // Tracking information is available
            nr_features_used++;
            
            // Accumulate radiance information
            std::vector<double> radiances_temp = curr_feature->m_radiance_estimates;
            for(int r = 0;r < radiances_temp.size();r++)
            {
                radiances.at(r) += radiances_temp.at(r);
            }
        }
        
        // Average radiance estimates for this feature
        for(int r = 0;r < radiances.size();r++)
        {
            radiances.at(r) /= nr_features_used;
        }
  
        // Image output values (corrected by vignette + response but NOT yet by exposure) taken from the current frame
        std::vector<double> outputs = features_last_frame.at(i)->m_radiance_estimates;
        
        // Fetch gradient values for the tracked feature
        std::vector<double> grad_values = features_last_frame.at(i)->m_gradient_values;
        
        // Estimate one exposure ratio for each of the tracking patch points
        for(int k = 0;k < radiances.size();k++)
        {
            // Weight the estimate depending on the gradient (high gradient = low confidence)
            double weight = 1.0 - grad_values.at(k)/125;
            if(weight < 0)weight = 0;
            
            // Avoid division by 0 for underexposed points
            if(fabs(radiances.at(k)) < 0.0001)
                continue;
            
            // Estimate the exposure ratio
            double curr_e_estimate = (outputs.at(k) / radiances.at(k));
            
            // Skip spurious implausible estimates (such as radiance = 0)
            if(curr_e_estimate < 0.001 || curr_e_estimate > 100)
                continue;
            
            // Accumulate estimation information
            e_estimate   += weight*curr_e_estimate;
            nr_estimates += weight;
        }
    }
    
    // This should not happen, just for safety
    if(nr_estimates == 0)
        return 1.0;
    
    // Average the exposure information for each patch
    // [TODO] Maybe use a more robust way to select an exposure time in the presence of severe noise
    double final_exp_estimate = e_estimate / nr_estimates;

    // Todo: this part is confusing...
    // Handle exposure time drift
    // If corrected images are highly over/underexposes, push exposure times
    // Todo: change to use ref
    cv::Mat corrected_image = m_database->m_tracked_frames.at(m_database->m_tracked_frames.size()-1).m_image_corrected;
    cv::Mat original_image  = m_database->m_tracked_frames.at(m_database->m_tracked_frames.size()-1).m_image;
    corrected_image /= final_exp_estimate;
    
    // Count the number of pixels that are under/overexposed in the corrected image
    // And not under/overexposed in the original input image
    int nr_underexposed = 0;
    int nr_overexposed  = 0;
    int nr_pixels = corrected_image.rows*corrected_image.cols;
    
    for(int r = 0;r < corrected_image.rows;r++)
    {
        for(int c = 0;c < corrected_image.cols;c++)
        {
            uchar image_value = corrected_image.at<uchar>(r,c);
            uchar orig_value  = original_image.at<uchar>(r,c);
            if(image_value < 15 && orig_value >= 30)
            {
                nr_underexposed++;
            }
            else if(image_value > 240 && orig_value <= 200)
            {
                nr_overexposed++;
            }
        }
    }
    
    double percentage_underexposed = nr_underexposed / (1.0*nr_pixels);
    double percentage_overexposed  = nr_overexposed / (1.0*nr_pixels);
    
    //std::cout << "UNDER: " << percentage_underexposed << " -- OVER: " << percentage_overexposed << std::endl;
    
    // If the amount of over/underexposed images are too large, correct the exposure time drift
    if(percentage_overexposed > 0.05)
    {
        final_exp_estimate += 0.03;
    }
    else if(percentage_underexposed > 0.03)
    {
        final_exp_estimate -= 0.05;
    }
    
    return final_exp_estimate;
}
//***************************************************************************//
NonlinearOptimizer::NonlinearOptimizer(int keyframe_spacing,//15
                                       Database* database,
                                       int safe_zone_size,//20
                                       int min_keyframes_valid,//3
                                       int patch_size)//3
{
    // Initialize object parameters with passed values
    m_keyframe_spacing = keyframe_spacing;//15
    m_database = database;
    m_safe_zone_size = safe_zone_size;//20
    m_min_keyframes_valid = min_keyframes_valid;//3
    m_patch_size = patch_size;//3
    
    // No optimization data yet on initialization
    m_optimization_block = NULL;
    
    // Keep an estimate of the inverse response here for fast plotting
    m_raw_inverse_response = new double[256];
}

bool NonlinearOptimizer::extractOptimizationBlock()
{
    int nr_images_in_database = static_cast<int>(m_database->m_tracked_frames.size());

    // Not enough images in the database yet -> extraction fails
    // Todo: why 2 times?
    if(nr_images_in_database < 2*(m_keyframe_spacing*m_min_keyframes_valid)+m_safe_zone_size)//2*(15*3)+20 = 110
    {
        return false;
    }
    
    // Create a new optimization block, delete the old block (if an old one exists)    
    if(m_optimization_block != NULL)
    {
        delete m_optimization_block;
    }
    m_optimization_block = new OptimizationBlock(m_patch_size);//3
    
    int nr_images_in_block = 0;
    m_optimization_block->deleteExposureTimes();
    
    // Iterate through all images in the database (except the most current ones used for exposure optimization)
    for(int i = 0;i < nr_images_in_database - m_safe_zone_size;i++)//20
    {
        // Only add keyframe images
        if(i%m_keyframe_spacing == 0)//15
        {
            nr_images_in_block++;
            
            // Store the image inside the optimization block for later correction
            m_optimization_block->addImage(m_database->m_tracked_frames.at(i).m_image);
            
            // Push exposure time estimate (use either the one found from rapid exposure time estimation or 1.0)
            // If 1.0 is used, then all the optimization parameters should also be initialized with the same constant values (for V,f,E,L)
            m_optimization_block->pushExposureTime(m_database->m_tracked_frames.at(i).m_exp_time,m_database->m_tracked_frames.at(i).m_gt_exp_time);
            //m_optimization_block->pushExposureTime(1.0);
        }
        
        // Get all features in the current keyframe and iterate them
        // Todo: change features to pointer?
        std::vector<Feature*> features = m_database->m_tracked_frames.at(i).m_features;
        
        for(int p = 0;p < features.size();p++)
        {
            // Skip features that are not new, don't attempt at creating a new optimization point
            if(features.at(p)->m_prev_feature != NULL)
                continue;
            
            // Find out in how many and which keyframes this point is visible
            Feature* feature_iterator = features.at(p);
            std::vector<int> keyframes_valid;
            int feature_iterator_image_index = i;
            
            // Track the feature forward until either we hit NULL or the feature is tracked out of the safe zone
            while(feature_iterator != NULL && feature_iterator_image_index < nr_images_in_database-m_safe_zone_size)
            {
                if(feature_iterator_image_index%m_keyframe_spacing == 0) //feature_iterator_image_index is a keyframe image
                    keyframes_valid.push_back(feature_iterator_image_index/m_keyframe_spacing);
                
                // Check the next feature, break if the min. number of keyframes necessary for this feature has been reached
                // -> then go on to extract image information for this feature
                feature_iterator = feature_iterator->m_next_feature;
                feature_iterator_image_index++;
                
                // Break early if the feature has been identified as tracked long enough
                if(keyframes_valid.size() >= m_min_keyframes_valid)
                    break;
            }
            
            // Feature not tracked long enough
            if(keyframes_valid.size() < m_min_keyframes_valid)
                continue;
            
            // Allocate new optimization point
            OptimizedPoint opt_p;
            opt_p.start_image_idx = keyframes_valid.at(0);
            
            // Initialize vector for radiance estimates
            std::vector<double> radiance_estimate;
            for(int r = 0;r < features.at(p)->m_radiance_estimates.size();r++)
            {
                radiance_estimate.push_back(0.0);
            }
            
            // Iterate the good feature again, now completely and extract its information
            feature_iterator = features.at(p);
            feature_iterator_image_index = i;
            int nr_keyframes_valid = 0;
            
            while(feature_iterator != NULL && feature_iterator_image_index < nr_images_in_database-m_safe_zone_size)
            {
                if(feature_iterator_image_index%m_keyframe_spacing != 0) //only extract data on keyframe images
                {
                    feature_iterator = feature_iterator->m_next_feature;
                    feature_iterator_image_index++;
                    continue;
                }
                
                nr_keyframes_valid++;
                
                // Accumulate radiance estimates (obtained using the previously corrected image data)
                for(int r = 0;r < feature_iterator->m_radiance_estimates.size();r++)
                {
                    radiance_estimate.at(r) += feature_iterator->m_radiance_estimates.at(r);
                }
                
                // Initialize the estimation problem with the average of the output intensities of the original images
                // This should be exchanged with the above initialization, if exposure values = 1.0 are used for initialization
                // Assuming unit response and no vignetting (for this, the avg. of outputs is the optimal radiance estimate)
                /*for(int r = 0;r < feature_iterator->m_output_values.size();r++)
                {
                    radiance_estimate.at(r) += feature_iterator->m_output_values.at(r);
                }*/
                
                // Store output intensities to the optimization point
                opt_p.output_intensities.push_back(feature_iterator->m_output_values);
                
                // Store xy locations
                opt_p.xy_image_locations.push_back(feature_iterator->m_xy_location);
                
                // Calculate point radius
                double radius = m_database->m_vignette_estimate.getNormalizedRadius(feature_iterator->m_xy_location);
                opt_p.radii.push_back(radius);
                
                // Set gradient weights
                std::vector<double> grad_weights;
                for(int r = 0;r < feature_iterator->m_gradient_values.size();r++)
                {
                    double grad_value = feature_iterator->m_gradient_values.at(r);
                    double weight = 1.0 - (grad_value/255.0);
                    grad_weights.push_back(weight);
                }
                opt_p.grad_weights.push_back(grad_weights);
                
                feature_iterator = feature_iterator->m_next_feature;
                feature_iterator_image_index++;
            }
            
            // Average radiance estimates
            for(int r = 0;r < features.at(p)->m_radiance_estimates.size();r++)
            {
                radiance_estimate.at(r) /= 255.0*nr_keyframes_valid;
            }
            
            // Store information about in how many keyframes this feature is valid
            opt_p.num_images_valid = nr_keyframes_valid;
            
            // Store the radiance estimates to the optimization point
            opt_p.radiances = radiance_estimate;
            
            // Add point to optimization block
            m_optimization_block->addOptimizationPoint(opt_p);
        }
    }
    
    return true;
}

double NonlinearOptimizer::evfOptimization(bool show_debug_prints)
{
    // Used for calculating first order derivatives, creating the Jacobian
    JacobianGenerator jacobian_generator;
    jacobian_generator.setResponseParameters(m_response_estimate);
    jacobian_generator.setVignettingParameters(m_vignette_estimate);
    
    // Find maximum number of residuals
    int points_per_patch = pow(2*m_patch_size+1,2);//49
    int num_residuals = m_optimization_block->getNrResiduals();
    
    // Number of parameters to optimize for (4 response, 3 vignette + exposure times)
    int num_parameters = C_NR_RESPONSE_PARAMS + C_NR_VIGNETTE_PARAMS + m_optimization_block->getNrImages();
    
    // Initialize empty matrices for the Jacobian and the residual vector
    cv::Mat Jacobian(num_residuals,num_parameters,CV_64F,0.0);
    cv::Mat Residuals(num_residuals,1,CV_64F,0.0);
    
    // Weight matrix for the Jacobian
    cv::Mat Weights_Jacobian(num_residuals,num_parameters,CV_64F,0.0);
    
    // Fill the Jacobian
    int residual_id = -1;
    double residual_sum = 0;
    int overall_image_index = 0;
    
    std::vector<OptimizedPoint>* points_to_optimize = m_optimization_block->getOptimizedPoints();
    
    // Iterate all tracked points
    for(int p = 0;p < points_to_optimize->size();p++)
    {
        int image_start_index = points_to_optimize->at(p).start_image_idx;
        int nr_img_valid = points_to_optimize->at(p).num_images_valid;
        
        // Iterate images the point is valid
        for(int i = 0;i < nr_img_valid;i++)
        {
            double radius = points_to_optimize->at(p).radii.at(i);
            double exposure = m_optimization_block->getExposureTime(image_start_index+i);
                
            //iterate all the points in the patch
            for(int r = 0;r < points_per_patch;r++)
            {
                double grad_weight = points_to_optimize->at(p).grad_weights.at(i).at(r);
                if(grad_weight < 0.001) // Dont include a point with close to 0 weight in optimization
                    continue;
                
                double radiance = points_to_optimize->at(p).radiances.at(r);
                double o_value = points_to_optimize->at(p).output_intensities.at(i).at(r);
                
                // Avoid I = 0 which leads to NaN errors in the Jacobian, also ignore implausible radiance estimates much larger than 1
                if(radiance < 0.001 || radiance > 1.1)
                    continue;
                if(radiance > 1)
                    radiance = 1;
                
                // Count the actual number of residuals up
                residual_id++;
                
                // Fill the Jacobian row
                jacobian_generator.getJacobianRow_eca(radiance,
                                                      radius,
                                                      exposure,
                                                      Jacobian,
                                                      overall_image_index + image_start_index + i,
                                                      residual_id);
                
                // For debugging
                /*if(show_debug_prints)
                {
                    std::cout << "JACOBIAN ROW ADDED " << std::endl;
                    for(int j = 0;j < num_parameters;j++)
                    {
                        std::cout << Jacobian.at<double>(residual_id,j) << " ";
                            
                        if(Jacobian.at<double>(residual_id,j) != Jacobian.at<double>(residual_id,j))
                        {
                            std::cout << "NAN" << std::endl;
                        }
                    }
                    std::cout << std::endl;
                }*/
                    
                // Write weight values to weight matrix
                for(int k = 0;k < num_parameters;k++)
                {
                    Weights_Jacobian.at<double>(residual_id,k) = grad_weight;
                }
                    
                // Fill the residual vector
                double residual = getResidualValue(o_value, radiance, radius, exposure);
                Residuals.at<double>(residual_id,0) = grad_weight * residual;
                
                residual_sum += std::abs(grad_weight * residual);
            }
        }
    }
        
    overall_image_index += m_optimization_block->getNrImages();
    
    int real_number_of_residuals = residual_id+1;
    
    // Get only the relevant part of the Jacobian (actual number of residuals)
    Jacobian = Jacobian(cv::Rect(0,0,num_parameters,real_number_of_residuals));
    Weights_Jacobian = Weights_Jacobian(cv::Rect(0,0,num_parameters,real_number_of_residuals));
    Residuals = Residuals(cv::Rect(0,0,1,real_number_of_residuals));
    
    // Transpose the Jacobian, calculate J^T * W *J * X = - J^T * W * r
    cv::Mat Jacobian_T;
    cv::transpose(Jacobian, Jacobian_T);
    
    cv::Mat A = Jacobian_T* (Weights_Jacobian.mul(Jacobian));
    //cv::Mat A = Jacobian.t()*Jacobian;
    cv::Mat b = - Jacobian.t() * Residuals; // Todo: reuse Jacobian_T to save time?
    
    // Get the current residual before optimization in order to compare progress
    double total_error, avg_error;
    getTotalResidualError(total_error,avg_error);
    
    if(show_debug_prints)
        std::cout << "Error before ECA adjustment: total: " << total_error << " avg: " << avg_error << std::endl;
    
    // Prepare identity matrix for Levenberg-Marquardt dampening
    cv::Mat Identity = cv::Mat::eye(num_parameters, num_parameters, CV_64F);
    Identity = Identity.mul(A);
    
    // Backup photometric parameters in order to revert if update is not good
    std::vector<double> response_param_backup    = m_response_estimate;
    std::vector<double> vignetting_param_backup = m_vignette_estimate;
    std::vector<double> exp_backups;
    for(int i = 0;i < m_optimization_block->getNrImages();i++)
    {
        exp_backups.push_back(m_optimization_block->getExposureTime(i));
    }
    
    // Perform update steps
    int max_rounds = 6; // Todo: change this to const class member
    cv::Mat BestStateUpdate(num_parameters,1,CV_64F,0.0);
    double current_best_error = total_error;
    
    double lambda = 1.0f;
    double lm_dampening = 1.0;

    // Todo: are these the right LM iterations???
    // Rui: may be because of the alternative optimization of evf and radiances, thus only one iteration in the evf GN.
    for(int round = 0;round < max_rounds;round++)
    {
        if(show_debug_prints)
            std::cout << "ECA Optimization round with dampening = " << lm_dampening << std::endl;
        
        //solve the linear equation system
        cv::Mat State_Update;

        lambda = 1.0;
  
        // Solve state update equation (+LM damping)
        // Todo: reuse Jacobian_T to save time?
        State_Update = - (Jacobian.t()* Weights_Jacobian.mul(Jacobian) + lm_dampening*Identity).inv(cv::DECOMP_SVD)*(Jacobian.t()*Residuals);
        
        // Update the estimated parameters
        for(int k = 0;k < m_response_estimate.size();k++)
        {
            m_response_estimate.at(k) = response_param_backup.at(k) + lambda * State_Update.at<double>(k,0);
        }
        
        for(int k = 0;k < m_vignette_estimate.size();k++)
        {
            m_vignette_estimate.at(k) = vignetting_param_backup.at(k) + lambda * State_Update.at<double>((int)m_response_estimate.size()+k,0);
        }
        
        int abs_image_index = 0;
        for(int i = 0;i < m_optimization_block->getNrImages();i++)
        {
            double new_exp_time = exp_backups.at(i) + lambda*State_Update.at<double>((int)m_response_estimate.size()+(int)m_vignette_estimate.size()+abs_image_index,0);
            m_optimization_block->setExposureTime(i,new_exp_time);
            abs_image_index++;
        }
        
        // Evaluate new residual error with new parameter estimate
        double current_error;
        getTotalResidualError(current_error,avg_error);
        
        if(show_debug_prints)
            std::cout << "error after ECA adjustment: total: " << current_error << " avg: " << avg_error << std::endl;
        
        // Improvement?
        if(current_error < current_best_error)
        {
            //increase damping factor, re-perform
            if(lm_dampening >= 0.0625)
                lm_dampening /= 2.0f;
            
            current_best_error = current_error;
            BestStateUpdate = State_Update;
        }
        else
        {
            if(lm_dampening <= 1000000)
            {
                lm_dampening *= 2;
            }
            else
            {
                if(show_debug_prints)
                    std::cout << "MAX DAMPING REACHED, BREAK EARLY " << std::endl;
                break;
            }
        }
    }
    
    // Apply the best of the found state updates
    
    for(int k = 0;k < m_response_estimate.size();k++)
    {
        m_response_estimate.at(k) = response_param_backup.at(k) + lambda * BestStateUpdate.at<double>(k,0);
    }
    
    for(int k = 0;k < m_vignette_estimate.size();k++)
    {
        m_vignette_estimate.at(k) = vignetting_param_backup.at(k) + lambda * BestStateUpdate.at<double>((int)m_response_estimate.size()+k,0);
    }
    
    int abs_image_index = 0;
    for(int i = 0;i < m_optimization_block->getNrImages();i++)
    {
        double new_exp_time = exp_backups.at(i) +
                              lambda*BestStateUpdate.at<double>((int)m_response_estimate.size() +
                                                                (int)m_vignette_estimate.size() +
                                                                abs_image_index,0);
        m_optimization_block->setExposureTime(i,new_exp_time);
        abs_image_index++;
    }
    
    if(show_debug_prints)
        std::cout << "Best update " << BestStateUpdate << std::endl;
    
    double error_after_optimization;
    getTotalResidualError(error_after_optimization,avg_error);
    
    if(show_debug_prints)
        std::cout << "error after ECA adjustment: total: " << error_after_optimization << " avg: " << avg_error << std::endl;
    
    return avg_error;
}


double NonlinearOptimizer::getResidualValue(double O, double I, double r, double e)
{
    double vignetting = applyVignetting(r);
    
    if(vignetting < 0)
    {
        vignetting = 0;
    }
    if(vignetting > 1)
    {
        vignetting = 1;
    }
    
    // Argument of response function
    double inside = e*I*vignetting;
    double response_value;
    
    // Apply response function
    if(inside < 0)
    {
        response_value = 0;
    }
    else if(inside > 1)
    {
        response_value = 255;
    }
    else
    {
        response_value = applyResponse(inside);
    }
    
    // Compute residual
    double residual = response_value - O;
    
    return residual;
}


void NonlinearOptimizer::getTotalResidualError(double& total_error,double& avg_error)
{
    int residual_id = -1;
    double residual_sum = 0;
    int points_per_patch = pow(2*m_patch_size+1,2);
    
    std::vector<OptimizedPoint>* points_to_optimize = m_optimization_block->getOptimizedPoints();
    
    // Iterate all tracked points
    for(int p = 0;p < points_to_optimize->size();p++)
    {
        int image_start_index = points_to_optimize->at(p).start_image_idx;
        int nr_images         = points_to_optimize->at(p).num_images_valid;
        
        // Iterate all images of the point
        for(int i = 0;i < nr_images;i++)
        {
            double radius = points_to_optimize->at(p).radii.at(i);
            double exposure = m_optimization_block->getExposureTime(image_start_index+i);
                
            // Iterate all the points in the patch
            for(int r = 0;r < points_per_patch;r++)
            {
                // Handle next residual
                residual_id++;
                    
                // Get the radiance value of this residual (image independent)
                double radiance = points_to_optimize->at(p).radiances.at(r);
                
                // Get image output value of this residual (original image)
                double o_value = points_to_optimize->at(p).output_intensities.at(i).at(r);
                    
                // Compute residual
                double residual = getResidualValue(o_value, radiance, radius, exposure);
                    
                // Get the weight of the residual
                double residual_weight = points_to_optimize->at(p).grad_weights.at(i).at(r);
                
                // Accumulate resdiual values
                residual_sum += residual_weight*std::abs(residual);
            }
        }
    }
    
    // Average residual error
    total_error = residual_sum;
    avg_error   = total_error/(residual_id+1);
}


double NonlinearOptimizer::getResidualErrorPoint(OptimizedPoint p,int r)
{
    int start_image = p.start_image_idx;
    int num_images  = p.num_images_valid;
    double radiance_guess = p.radiances.at(r);
    double totalError = 0;
    
    for(int i = 0;i < num_images;i++)
    {
        double radius = p.radii.at(i);
        double output_value = p.output_intensities.at(i).at(r);
        double exposure = m_optimization_block->getExposureTime(start_image+i);
        double residual = getResidualValue(output_value, radiance_guess, radius, exposure);
        totalError += std::abs(residual);
    }
    
    return totalError;
}

void NonlinearOptimizer::fetchResponseVignetteFromDatabase()
{
    // Fetch vignette estimate from database
    m_vignette_estimate = m_database->m_vignette_estimate.getVignetteEstimate();
    
    m_response_estimate = m_database->m_response_estimate.getResponseEstimate();
    // Set response estimate to unit response
    //m_response_estimate.clear();
    //m_response_estimate.push_back(6.3);
    //m_response_estimate.push_back(0.0);
    //m_response_estimate.push_back(0.0);
    //m_response_estimate.push_back(0.0);
}


double NonlinearOptimizer::applyVignetting(double r)
{
    double r_2 = r*r;
    double r_4 = r_2 * r_2;
    double r_6 = r_4 * r_2;
    
    double result_vignette = 1 + m_vignette_estimate.at(0) * r_2 + m_vignette_estimate.at(1)*r_4 + m_vignette_estimate.at(2) * r_6;
    
    if(result_vignette < 0)
        result_vignette = 0.0;
    if(result_vignette > 1)
        result_vignette = 1.0;
    
    return result_vignette;
}

double NonlinearOptimizer::applyResponse(double x)
{
    JacobianGenerator jacobian_generator;
    jacobian_generator.setResponseParameters(m_response_estimate);
    double result = jacobian_generator.applyGrossbergResponse(x);
    return 255*result;
}

double NonlinearOptimizer::visualizeOptimizationResult(double* inverse_response)
{
    // Define an exponential factor here to scale response + vignette
    // double exponent = 1.0;
    // To go through one point of the GT response of the TUM Mono camera
    //double exponent = determineGammaFixResponseAt(inverse_response, 206, 0.5);
    double exponent = determineGammaFixResponseAt(inverse_response, 148, 0.3);
    
    // Setup output image windows
    cv::namedWindow("Estimated Vignetting");
    cv::namedWindow("Estimated Response");
    
    double inverse_response_scaled[256];
    
    // Scale up inverse response from range [0,1] to range [0,255]
    for(int i = 0;i < 256;i++)
    {
        inverse_response_scaled[i] = 255*pow(inverse_response[i],exponent);
    }
    
    // Invert the inverse response to get the response
    double response_function[256];
    response_function[0] = 0;
    response_function[255] = 255;

    // For each response value i find s, such that inverse_response[s] = i
    for(int i=1;i<255;i++)
    {
        for(int s=0;s<255;s++)
        {
            if(inverse_response_scaled[s] <= i && inverse_response_scaled[s+1] >= i)
            {
                response_function[i] = s+(i - inverse_response_scaled[s]) / (inverse_response_scaled[s+1]-inverse_response_scaled[s]);
                break;
            }
        }
    }
    
    // Setup a 256x256 mat to display inverse response + response
    // Todo: change to class member
    cv::Mat response_vis_image(256,256,CV_8UC3,cv::Scalar(255,255,255));
    for(int i = 0;i < 256;i++)
    {
        int response_value = static_cast<int>(round(response_function[i]));
        int inv_response_value = static_cast<int>(round(inverse_response_scaled[i]));
        
        if(response_value < 0)
            response_value = 0;
        if(response_value > 255)
            response_value = 255;
        if(inv_response_value < 0)
            inv_response_value = 0;
        if(inv_response_value > 255)
            inv_response_value = 255;
        
        //plot the response
        response_vis_image.at<cv::Vec3b>(255-response_value,i)[0] = 255;
        response_vis_image.at<cv::Vec3b>(255-response_value,i)[1] = 255;
        response_vis_image.at<cv::Vec3b>(255-response_value,i)[2] = 0;
        
        //plot the inverse response
        //response_vis_image.at<cv::Vec3b>(255-inv_response_value,i)[0] = 0;
        //response_vis_image.at<cv::Vec3b>(255-inv_response_value,i)[1] = 0;
        //response_vis_image.at<cv::Vec3b>(255-inv_response_value,i)[2] = 255;
        
        //draw the diagonal
        //response_vis_image.at<cv::Vec3b>(255-i,i)[0] = 255;
        //response_vis_image.at<cv::Vec3b>(255-i,i)[1] = 255;
        //response_vis_image.at<cv::Vec3b>(255-i,i)[2] = 255;
        
        // Draw a variety of GT response functions
        double x = i/255.0;
        
        // [1] Draw the best gamma approximation for DSO response
        /*double dso_gamma_approx_y = pow(x,0.7-0.3*x);
         int dso_gamma_approx_y_int = static_cast<int>(dso_gamma_approx_y*255);
         if(dso_gamma_approx_y_int > 255)
             dso_gamma_approx_y_int = 255;
         response_vis_image.at<cv::Vec3b>(255-dso_gamma_approx_y_int,i)[0] = 255;
         response_vis_image.at<cv::Vec3b>(255-dso_gamma_approx_y_int,i)[1] = 255;
         response_vis_image.at<cv::Vec3b>(255-dso_gamma_approx_y_int,i)[2] = 0;*/
        
        /*double m;
        double t;
        //draw GT response for DSO camera
        if(x < 0.1)
        {
            m = (90/255.0)/0.1;
            t = 0.0f;
        }
        else if(x < 0.48)
        {
            m = (110.0/255)/0.38;
            t = (200.0/255.0) - m*0.48;
        }
        else
        {
            m = (55.0/255)/0.52;
            t = 1 - m*1;
        }
        double dso_value_f = m*x + t;
        int dso_value = static_cast<int>(dso_value_f*255);
        if(dso_value > 255)
            dso_value = 255;
        response_vis_image.at<cv::Vec3b>(255-dso_value,i)[0] = 255;
        response_vis_image.at<cv::Vec3b>(255-dso_value,i)[1] = 255;
        response_vis_image.at<cv::Vec3b>(255-dso_value,i)[2] = 0;*/
        
//        /*
//         * Draw GT response for artificial dataset
//         */
//        //draw the GT response for canon EOS 600 D
//        double artificial_value_f = pow(x,0.6-0.2*x);
//        int artificial_value = static_cast<int>(artificial_value_f*255);
//        if(artificial_value > 255)
//            artificial_value = 255;
//        response_vis_image.at<cv::Vec3b>(255-artificial_value,i)[0] = 0;
//        response_vis_image.at<cv::Vec3b>(255-artificial_value,i)[1] = 255;
//        response_vis_image.at<cv::Vec3b>(255-artificial_value,i)[2] = 255;
        
    }
    
    cv::imshow("Estimated Response", response_vis_image);
    // TODO: move only the first time the window is created,
    //       to allow the user to move it somewhere else.
    //       Same for other calls to "moveWindow".
    cv::moveWindow("Estimated Response", 20,20);

    // Show the vignetting
    
    //Setup a 256x256 mat to display vignetting
    cv::Mat vignette_vis_image(256,256,CV_8UC3,cv::Scalar(255,255,255));
    for(int i = 0;i < 256;i++)
    {
        double r = i/255.0f;
        
        double r_2 = r*r;
        double r_4 = r_2 * r_2;
        double r_6 = r_4 * r_2;
        
        double vignette = 1 + m_vignette_estimate.at(0) * r_2 + m_vignette_estimate.at(1)*r_4 + m_vignette_estimate.at(2) * r_6;
        
        vignette = pow(vignette,exponent);
        
        int y_pos = 245 - round(235*vignette);
        if(y_pos < 0)
            y_pos = 0;
        if(y_pos > 255)
            y_pos = 255;
        
        // Plot the vignetting
        vignette_vis_image.at<cv::Vec3b>(y_pos,i)[0] = 255;
        vignette_vis_image.at<cv::Vec3b>(y_pos,i)[1] = 255;
        vignette_vis_image.at<cv::Vec3b>(y_pos,i)[2] = 0;
        
        // Plot the reference line for V = 1
        //vignette_vis_image.at<cv::Vec3b>(10,i)[0] = 255;
        //vignette_vis_image.at<cv::Vec3b>(10,i)[1] = 255;
        //vignette_vis_image.at<cv::Vec3b>(10,i)[2] = 255;
        
        // Plot the reference line for V = 0
        //vignette_vis_image.at<cv::Vec3b>(235,i)[0] = 255;
        //vignette_vis_image.at<cv::Vec3b>(235,i)[1] = 255;
        //vignette_vis_image.at<cv::Vec3b>(235,i)[2] = 255;
        
        // Plot the vignetting for DSO sequence 47
         /*double dso_vignette_47 = 0.971 + 0.1891*r - 1.5958*r_2 + 1.4473*r_2*r - 0.5143* r_4;
         y_pos = 245 - round(235*dso_vignette_47  );
         vignette_vis_image.at<cv::Vec3b>(y_pos,i)[0] = 255;
         vignette_vis_image.at<cv::Vec3b>(y_pos,i)[1] = 255;
         vignette_vis_image.at<cv::Vec3b>(y_pos,i)[2] = 0;*/
        
//        // Plot the vignetting for artificial dataset
//        double art_vignette =  0.9983-0.0204*r -0.2341*r_2 - 0.0463*r_2*r;
//        y_pos = 245 - round(235*art_vignette  );
//        vignette_vis_image.at<cv::Vec3b>(y_pos,i)[0] = 0;
//        vignette_vis_image.at<cv::Vec3b>(y_pos,i)[1] = 255;
//        vignette_vis_image.at<cv::Vec3b>(y_pos,i)[2] = 255;
        
    }
    
    cv::imshow("Estimated Vignetting", vignette_vis_image);
    cv::moveWindow("Estimated Vignetting", 20,20+50+256);


    // Visualize exposure times 
    // If GT data is available, the estimated exposure times will be aligned 
    // to the GT by computing an optimal alignment factor alignment_alpha
    // If no GT data is available, the estimated exposure is simply scaled between [0,1]
    int exp_image_height = 150;
    int draw_spacing = 5;
    double alignment_alpha = 1.0;
    double max_exp = -10000.0;
    double min_exp = 10000.0;
    double top = 0;
    double bot = 0;
    int nr_block_images = m_optimization_block->getNrImages();
    std::vector<double> block_exp_estimates;
    std::vector<double> block_exp_gt;
    for(int i = 0;i < nr_block_images;i++)
    {
        double estimated_exp = m_optimization_block->getExposureTime(i);
        estimated_exp = pow(estimated_exp,exponent);
        double gt_exp = m_optimization_block->getGTExposureTime(i);

        // Store max and min exposure for normalization to [0,1] range 
        if(estimated_exp > max_exp)
            max_exp = estimated_exp;
        if(estimated_exp < min_exp)
            min_exp = estimated_exp;

        // Accumulate information for least square fit between GT and estimated exposure
        top += estimated_exp*gt_exp;
        bot += estimated_exp*estimated_exp;

        block_exp_estimates.push_back(estimated_exp);
        if(!(gt_exp < 0))
            block_exp_gt.push_back(gt_exp);
    }

    if(block_exp_estimates.size() == block_exp_gt.size())
    {
        alignment_alpha = top/bot;
    }
    else
    {
        // Normalize estimated exposures between [0,1] if no gt information is available for alignment
        for(int i = 0;i < block_exp_estimates.size();i++)
        {
            block_exp_estimates.at(i) = (block_exp_estimates.at(i)-min_exp)/(max_exp-min_exp);
        }
    }

    for(int i = 0;i < block_exp_estimates.size();i++)
    {
        block_exp_estimates.at(i) *= alignment_alpha;
    }

    // Create exposure time canvas
    draw_spacing = 20;
    //int block_exp_window_width = int(fmax(400,draw_spacing*block_exp_estimates.size()));
    int block_exp_window_width = int(draw_spacing*block_exp_estimates.size());
    cv::Mat block_exposure_vis_image(exp_image_height,block_exp_window_width,CV_8UC3,cv::Scalar(255,255,255));

    // Draw estimated exposure times as lines to graph
    for(int i = 0;i < block_exp_estimates.size()-1;i++)
    {
        int drawing_y_exp_1 = exp_image_height - exp_image_height*(block_exp_estimates.at(i));
        drawing_y_exp_1 = int(fmax(0,drawing_y_exp_1));
        drawing_y_exp_1 = int(fmin(exp_image_height-1,drawing_y_exp_1));

        int drawing_y_exp_2 = exp_image_height - exp_image_height*(block_exp_estimates.at(i+1));
        drawing_y_exp_2 = int(fmax(0,drawing_y_exp_2));
        drawing_y_exp_2 = int(fmin(exp_image_height-1,drawing_y_exp_2));

        //draw exposure lines
        cv::line(block_exposure_vis_image, cv::Point(draw_spacing*i,drawing_y_exp_1), cv::Point(draw_spacing*(i+1),drawing_y_exp_2), cv::Scalar(0,0,255));
     }

    // draw GT exposure line only if GT exposure data is available
    if(block_exp_estimates.size() == block_exp_gt.size())
    {
        for(int i = 0;i < block_exp_estimates.size()-1;i++)
        {
            int drawing_y_gt_exp_1 = exp_image_height - exp_image_height * block_exp_gt.at(i);
            drawing_y_gt_exp_1 = int(fmax(0,drawing_y_gt_exp_1));
            drawing_y_gt_exp_1 = int(fmin(exp_image_height-1,drawing_y_gt_exp_1));

            int drawing_y_gt_exp_2 = exp_image_height - exp_image_height * block_exp_gt.at(i+1);
            drawing_y_gt_exp_2 = int(fmax(0,drawing_y_gt_exp_2));
            drawing_y_gt_exp_2 = int(fmin(exp_image_height-1,drawing_y_gt_exp_2));

            cv::line(block_exposure_vis_image, cv::Point(draw_spacing*i,drawing_y_gt_exp_1), cv::Point(draw_spacing*(i+1),drawing_y_gt_exp_2), cv::Scalar(255,255,0));
        }
    }   

    cv::imshow("Estimated Keyframe Exposures (Backend)", block_exposure_vis_image);
    cv::moveWindow("Estimated Keyframe Exposures (Backend)", 20+20+256,20+40+exp_image_height);


    cv::waitKey(1);

    // Return gamma that was used for visualization
    return exponent;
}


double NonlinearOptimizer::getInverseResponseFixGamma(double* inverse_response_function)
{
    getInverseResponseRaw(inverse_response_function);
    double gamma = determineGammaFixResponseAt(inverse_response_function, 127, 0.5);
    
    // Scale the inverse response
    for(int i = 0;i < 256;i++)
    {
        inverse_response_function[i] = pow(inverse_response_function[i],gamma);
    }
    
    // Return the obtained gamma factor
    return gamma;
}

void NonlinearOptimizer::getInverseResponseRaw(double* inverse_response_function)
{
    //set boundaries of the inverse response
    inverse_response_function[0] = 0;
    inverse_response_function[255] = 1.0;
    
    // For each inverse response value i find s, such that response[s] = i
    for(int i=1;i<255;i++)
    {
        bool inversion_found = false;
        
        for(int s=0;s<255;s++)
        {
            double response_s1 = applyResponse(s/255.0f);
            double response_s2 = applyResponse((s+1)/255.0f);
            if(response_s1 <= i && response_s2 >= i)
            {
                inverse_response_function[i] = s+(i - response_s1) / (response_s2-response_s1);
                inverse_response_function[i] /= 255.0;
                inversion_found = true;
                break;
            }
        }
        
        if(!inversion_found)
        {
            std::cout << "Error, no inversion found in getInverseResponse(..)" << std::endl;
        }
    }
}

double NonlinearOptimizer::determineGammaFixResponseAt(double*inverse_response,int x,double y)
{
    double v_y = inverse_response[x];
    double gamma = log(y) / log(v_y);
    return gamma;
}

void NonlinearOptimizer::smoothResponse()
{
    // Get inverse response estimate, fixing the gamma value reasonably
    double inverse_response[256];
    double gamma = getInverseResponseFixGamma(inverse_response);
    
    // Scale up the inverse response to range [0,255]
    for(int i = 0;i < 256;i++)
    {
        inverse_response[i] = 255*inverse_response[i];
    }
    
    // Invert the inverse response to get the response again
    double response_function[256];
    response_function[0] = 0;
    response_function[255] = 255;
    
    // For each response value i find s, such that inverse_response[s] = i
    for(int i=1;i<255;i++)
    {
        for(int s=0;s<255;s++)
        {
            if(inverse_response[s] <= i && inverse_response[s+1] >= i)
            {
                response_function[i] = s+(i - inverse_response[s]) / (inverse_response[s+1]-inverse_response[s]);
                break;
            }
        }
    }
    
    // Fit the Grossberg parameters new to the acquired data
    JacobianGenerator generator;
    m_response_estimate = generator.fitGrossbergModelToResponseVector(response_function);
    
    // Scale vignette by gamma factor
    double vfactors[100];
    for(int r = 0;r < 100;r++)
    {
        double radius = r/100.0;
        double vfactor = applyVignetting(radius);
        vfactor = pow(vfactor,gamma);
        vfactors[r] = vfactor;
    }
    
    //[Note] Radiances of optimization should be scaled as well, but since these are not used anymore, its not done
    int nr_block_images = m_optimization_block->getNrImages();
    for(int k = 0;k < nr_block_images;k++)
    {
        double old_exposure = m_optimization_block->getExposureTime(k);
        double new_exposure = pow(old_exposure,gamma);
        m_optimization_block->setExposureTime(k,new_exposure);
    }

    // Fit new vignetting parameters in least square manner
    cv::Mat LeftSide(3,3,CV_64F,0.0);
    cv::Mat RightSide(3,1,CV_64F,0.0);
    
    int nr_bad_v = 0;
    
    for(int r = 0;r < 100;r++)
    {
        double w = 1.0;
        if(r > 0)
        {
            double diff = vfactors[r] - vfactors[r-1];
            if(diff > 0)
            {
                w = 0.5;
                vfactors[r] = vfactors[r-1];
                nr_bad_v++;
            }
        }

        double radius = r/100.0;
        double r2 = radius*radius;
        double r4 = r2*r2;
        double r6 = r4*r2;
        double r8 = r4*r4;
        double r10 = r6*r4;
        double r12 = r6*r6;
        
        LeftSide.at<double>(0,0) += w*r4;
        LeftSide.at<double>(0,1) += w*r6;
        LeftSide.at<double>(0,2) += w*r8;
        RightSide.at<double>(0,0) += (w*vfactors[r]*r2 - w*r2);
        
        LeftSide.at<double>(1,0) += w*r6;
        LeftSide.at<double>(1,1) += w*r8;
        LeftSide.at<double>(1,2) += w*r10;
        RightSide.at<double>(1,0) += (w*vfactors[r]*r4 - w*r4);
        
        LeftSide.at<double>(2,0) += w*r8;
        LeftSide.at<double>(2,1) += w*r10;
        LeftSide.at<double>(2,2) += w*r12;
        RightSide.at<double>(2,0) += (w*vfactors[r]*r6 - w*r6);
    }
    
    cv::Mat Solution;
    cv::solve(LeftSide, RightSide, Solution, cv::DECOMP_SVD);
    
    std::vector<double> solution_vig;
    solution_vig.push_back(Solution.at<double>(0,0));
    solution_vig.push_back(Solution.at<double>(1,0));
    solution_vig.push_back(Solution.at<double>(2,0));
    
    m_vignette_estimate = solution_vig;
}
//***************************************************************************//
OptimizationBlock::OptimizationBlock(int patch_size)
{
    m_patch_size = patch_size;//3
    m_nr_patch_points =(2*m_patch_size+1)*(2*m_patch_size+1);//7*7=49
}

void OptimizationBlock::addOptimizationPoint(OptimizedPoint p)
{
    m_optimized_points.push_back(p);
}

int OptimizationBlock::getNrResiduals()
{
    int nr_residuals = 0;
    
    for(int i = 0;i < m_optimized_points.size();i++)
    {
        int nr_images = m_optimized_points.at(i).num_images_valid;
        
        nr_residuals += (nr_images * m_nr_patch_points);
    }
    
    return nr_residuals;
}
//***************************************************************************//
JacobianGenerator::JacobianGenerator()
{
    // Nothing to initialize
}

void JacobianGenerator::getJacobianRow_eca(double I,
                                           double r,
                                           double e,
                                           cv::Mat jacobian,
                                           int image_index,
                                           int residual_index)
{
    // Get the vignetting value
    double a2 = m_vignetting_params.at(0);
    double a4 = m_vignetting_params.at(1);
    double a6 = m_vignetting_params.at(2);
    double r2 = r * r;
    double r4 = r2 * r2;
    double r6 = r4 * r2;
    double v  = 1 + a2*r2 + a4*r4 + a6*r6;
        
    // Evaluate the grossberg base functions' derivatives for the other derivatives
    double eIv = e * I * v;
    double h_0_d = evaluateGrossbergBaseFunction(0, true, eIv);
    double h_1_d = evaluateGrossbergBaseFunction(1, true, eIv);
    double h_2_d = evaluateGrossbergBaseFunction(2, true, eIv);
    double h_3_d = evaluateGrossbergBaseFunction(3, true, eIv);
    double h_4_d = evaluateGrossbergBaseFunction(4, true, eIv);

    double deriv_value = h_0_d +
                         m_response_params.at(0)*h_1_d +
                         m_response_params.at(1)*h_2_d +
                         m_response_params.at(2)*h_3_d +
                         m_response_params.at(3)*h_4_d;
        
    // Derive by the 4 Grossberg parameters
    jacobian.at<double>(residual_index,0) = 255*evaluateGrossbergBaseFunction(1, false, eIv);
    jacobian.at<double>(residual_index,1) = 255*evaluateGrossbergBaseFunction(2, false, eIv);
    jacobian.at<double>(residual_index,2) = 255*evaluateGrossbergBaseFunction(3, false, eIv);
    jacobian.at<double>(residual_index,3) = 255*evaluateGrossbergBaseFunction(4, false, eIv);
    
    // Derive by the 3 vignetting parameters
    jacobian.at<double>(residual_index,4) = 255 * deriv_value * e * I * r2;
    jacobian.at<double>(residual_index,5) = 255 * deriv_value * e * I * r4;
    jacobian.at<double>(residual_index,6) = 255 * deriv_value * e * I * r6;
        
    // Derive by exposure time
    jacobian.at<double>(residual_index,7+image_index) = 255 * deriv_value * (I*v);
}

void JacobianGenerator::getJacobianRadiance(double I,double r,double e,double& j_I)
{
    double a2 = m_vignetting_params.at(0);
    double a4 = m_vignetting_params.at(1);
    double a6 = m_vignetting_params.at(2);
    
    // Get the vignetting value
    double r2 = r * r;
    double r4 = r2 * r2;
    double r6 = r4 * r2;
    double v  = 1 + a2*r2 + a4*r4 + a6*r6;
        
    // Evaluate the Grossberg base functions' derivatives for the other derivatives
    double h_0_d = evaluateGrossbergBaseFunction(0, true, e*I*v);
    double h_1_d = evaluateGrossbergBaseFunction(1, true, e*I*v);
    double h_2_d = evaluateGrossbergBaseFunction(2, true, e*I*v);
    double h_3_d = evaluateGrossbergBaseFunction(3, true, e*I*v);
    double h_4_d = evaluateGrossbergBaseFunction(4, true, e*I*v);
        
    double deriv_value = h_0_d +
                         m_response_params.at(0)*h_1_d +
                         m_response_params.at(1)*h_2_d +
                         m_response_params.at(2)*h_3_d +
                         m_response_params.at(3)*h_4_d;
    
    j_I = 255 * deriv_value * (e*v);
}

double JacobianGenerator::evaluateGrossbergBaseFunction(int base_function_index,bool is_derivative,double x)
{
    if(x < 0)x = 0.0;
    else if(x > 1)x = 1.0;
    
    int x_int = round(x*1023);
    int x_der_int = round(x*1021);
    
    if(base_function_index == 0)
    {
        if(!is_derivative)
        {
            return m_f_0[x_int];
        }
        else
        {
            return m_f_0_der[x_der_int];
        }
    }
    
    if(base_function_index == 1)
    {
        if(!is_derivative)
        {
            return m_h_1[x_int];
        }
        else
        {
            return m_h_1_der[x_der_int];
        }
    }
    
    if(base_function_index == 2)
    {
        if(!is_derivative)
        {
            return m_h_2[x_int];
        }
        else
        {
            return m_h_2_der[x_der_int];
        }
    }
    
    if(base_function_index == 3)
    {
        if(!is_derivative)
        {
            return m_h_3[x_int];
        }
        else
        {
            return m_h_3_der[x_der_int];
        }
    }
    
    if(base_function_index == 4)
    {
        if(!is_derivative)
        {
            return m_h_4[x_int];
        }
        else
        {
            return m_h_4_der[x_der_int];
        }
    }
    
    // Error code
    return -1.0;
}

double JacobianGenerator::applyGrossbergResponse(double x)
{
    double v0 = evaluateGrossbergBaseFunction(0, false, x);
    double v1 = evaluateGrossbergBaseFunction(1, false, x);
    double v2 = evaluateGrossbergBaseFunction(2, false, x);
    double v3 = evaluateGrossbergBaseFunction(3, false, x);
    double v4 = evaluateGrossbergBaseFunction(4, false, x);
    
    double c1 = m_response_params.at(0);
    double c2 = m_response_params.at(1);
    double c3 = m_response_params.at(2);
    double c4 = m_response_params.at(3);
    
    return v0 + c1*v1 + c2*v2 + c3*v3 + c4*v4;
}

vector<double> JacobianGenerator::fitGrossbergModelToResponseVector(double* response)
{
    // Given a response vector, find Grossberg parameters that fit well
    cv::Mat LeftSide(4,4,CV_64F,0.0);
    cv::Mat RightSide(4,1,CV_64F,0.0);
    
    for(int i = 10;i < 240;i++)
    {
        response[i] /= 255.0;
        
        double input = i/256.0;
        
        double f0 = evaluateGrossbergBaseFunction(0, false, input);
        double f1 = evaluateGrossbergBaseFunction(1, false, input);
        double f2 = evaluateGrossbergBaseFunction(2, false, input);
        double f3 = evaluateGrossbergBaseFunction(3, false, input);
        double f4 = evaluateGrossbergBaseFunction(4, false, input);

        // For equation 1
        LeftSide.at<double>(0,0) += f1*f1;
        LeftSide.at<double>(0,1) += f1*f2;
        LeftSide.at<double>(0,2) += f1*f3;
        LeftSide.at<double>(0,3) += f1*f4;
        
        RightSide.at<double>(0,0) += (response[i]*f1 - f0*f1);
        
        // For equation 2
        LeftSide.at<double>(1,0) += f2*f1;
        LeftSide.at<double>(1,1) += f2*f2;
        LeftSide.at<double>(1,2) += f2*f3;
        LeftSide.at<double>(1,3) += f2*f4;
        
        RightSide.at<double>(1,0) += (response[i]*f2 - f0*f2);
        
        // For equation 3
        LeftSide.at<double>(2,0) += f3*f1;
        LeftSide.at<double>(2,1) += f3*f2;
        LeftSide.at<double>(2,2) += f3*f3;
        LeftSide.at<double>(2,3) += f3*f4;
        
        RightSide.at<double>(2,0) += (response[i]*f3 - f0*f3);
        
        // For equation 4
        LeftSide.at<double>(3,0) += f4*f1;
        LeftSide.at<double>(3,1) += f4*f2;
        LeftSide.at<double>(3,2) += f4*f3;
        LeftSide.at<double>(3,3) += f4*f4;
        
        RightSide.at<double>(3,0) += (response[i]*f4 - f0*f4);
    }
    
    cv::Mat Solution;
    cv::solve(LeftSide, RightSide, Solution,cv::DECOMP_SVD);
    
    std::vector<double> solution_response;
    solution_response.push_back(Solution.at<double>(0,0));
    solution_response.push_back(Solution.at<double>(1,0));
    solution_response.push_back(Solution.at<double>(2,0));
    solution_response.push_back(Solution.at<double>(3,0));
    
    return solution_response;
}
//***************************************************************************//
Tracker::Tracker(int patch_size,int nr_active_features,int nr_pyramid_levels,Database* database)
{
    // Simply store all passed arguments to object
    m_patch_size = patch_size;//3
    m_max_nr_active_features = nr_active_features;//200
    m_nr_pyramid_levels = nr_pyramid_levels;//2
    m_database = database;
    id_num = 0;
}

void Tracker::computeGradientImage(cv::Mat input_image,cv::Mat &gradient_image)
{
    // Blur the input image a little and apply discrete 3x3 sobel filter in x,y directions to obtain a gradient estimate
    // Todo: change to class member
    cv::Mat blurred_image;
    cv::GaussianBlur( input_image, blurred_image, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    // Todo: change to class member
    cv::Mat grad_x,grad_y;
    cv::Sobel( blurred_image, grad_x, CV_16S, 1, 0, 3, 1.0, 0, cv::BORDER_DEFAULT );
    cv::Sobel( blurred_image, grad_y, CV_16S, 0, 1, 3, 1.0, 0, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_x, grad_x );
    cv::convertScaleAbs( grad_y, grad_y );
    cv::addWeighted( grad_x, 0.5, grad_y, 0.5, 0, gradient_image );
}

void Tracker::trackNewFrame(cv::Mat input_image,double gt_exp_time)
{
    // Compute gradient (necessary for weighting factors)
    // Todo: move to class member
    //cv::Mat gradient_image;
    //computeGradientImage(input_image, gradient_image);
    
    // Correct the input image based on the current response and vignette estimate (exposure time not known yet)
    // Todo: move to class member
    cv::Mat corrected_frame = input_image.clone();
    photometricallyCorrectImage(corrected_frame);


    cv::Mat gradient_corrected_image;
    computeGradientImage(corrected_frame, gradient_corrected_image);
 
    // Empty database -> First frame - extract features and push them back
    /*if(m_database->m_tracked_frames.size() == 0)
    {
        initialFeatureExtraction(input_image,gradient_image,gt_exp_time);
        return;
    }*/
    
    if(m_database->m_tracked_frames.size() == 0)
    {
        initialFeatureExtraction(corrected_frame,gradient_corrected_image,gt_exp_time);
        return;
    }
    // Database not empty
    
    // Fetch the old active feature locations together with their image
    vector<cv::Point2f> feature_locations = m_database->fetchActiveFeatureLocations();
    vector<int> feature_id = m_database->fetchActiveFeatureID();
    cv::Mat last_frame = m_database->fetchActiveImage();
    
    // Track the feature locations forward using gain robust KLT
    vector<cv::Point2f> tracked_points_new_frame;
    vector<unsigned char> tracked_point_status;
    vector<float> tracking_error_values;

    GainRobustTracker gain_robust_klt_tracker(C_KLT_PATCH_SIZE,C_NR_PYRAMID_LEVELS);
    vector<int> tracked_point_status_int;
    gain_robust_klt_tracker.trackImagePyramids(last_frame,
                                               corrected_frame,
                                               feature_locations,
                                               tracked_points_new_frame,
                                               tracked_point_status_int);
    for(int i = 0;i < tracked_point_status_int.size();i++)
    {
        if(tracked_point_status_int.at(i) == 0)
        {
            tracked_point_status.push_back(0);
        }
        else
        {
            tracked_point_status.push_back(1);
        }
    }
     
    // Bidirectional tracking filter: Track points backwards and make sure its consistent
    vector<cv::Point2f> tracked_points_backtracking;
    vector<unsigned char> tracked_point_status_backtracking;
    vector<float> tracking_error_values_backtracking;
    GainRobustTracker gain_robust_klt_tracker_2(C_KLT_PATCH_SIZE,C_NR_PYRAMID_LEVELS);
    vector<int> tracked_point_status_int2;
    gain_robust_klt_tracker_2.trackImagePyramids(corrected_frame,
                                                 last_frame,
                                                 tracked_points_new_frame,
                                                 tracked_points_backtracking,
                                                 tracked_point_status_int2);
    for(int i = 0;i < tracked_point_status_int2.size();i++)
    {
        if(tracked_point_status_int2.at(i) == 0)
        {
            tracked_point_status_backtracking.push_back(0);
        }
        else
        {
            tracked_point_status_backtracking.push_back(1);
        }
    }
    
    // Tracked points from backtracking and old frame should be the same -> check and filter by distance
    for(int p = 0;p < feature_locations.size();p++)
    {
        if(tracked_point_status.at(p) == 0) // Point already set invalid by forward tracking -> ignore
            continue;
        
        if(tracked_point_status_backtracking.at(p) == 0) // Invalid in backtracking -> set generally invalid
            tracked_point_status.at(p) = 0;
        
        // Valid in front + backtracked images -> calculate displacement error
        cv::Point2d d_p = feature_locations.at(p) - tracked_points_backtracking.at(p);
        double distance = sqrt(d_p.x*d_p.x + d_p.y*d_p.y);
        
        if(distance > C_FWD_BWD_TRACKING_THRESH)
        {
            tracked_point_status.at(p) = 0;
        }
    }
    
    Frame frame;
    frame.m_image = input_image;
    frame.m_image_corrected = corrected_frame;
    frame.m_gradient_image = gradient_corrected_image;
    frame.m_exp_time = 1.0;
    frame.m_gt_exp_time = gt_exp_time;

    // Reject features that have been tracked to the side of the image
    // Todo: validity_vector can be combined with tracked_point_status to one vector?
    vector<int> validity_vector = checkLocationValidity(tracked_points_new_frame);
    
    int nr_pushed_features = 0;
    for(int i = 0;i < feature_locations.size();i++)
    {
        // If the feature became invalid don't do anything, otherwise if its still valid, push it to the database and set the feature pointers
        if(tracked_point_status.at(i) == 0 || validity_vector.at(i) == 0)
            continue;
        
        // Feature is valid, set its data and push it back
        Feature* f = new Feature();
        // Todo: remove os, is, gs
        std::vector<double> os = bilinearInterpolateImagePatch(input_image,tracked_points_new_frame.at(i).x,tracked_points_new_frame.at(i).y);
        f->m_output_values = os;
        std::vector<double> is = bilinearInterpolateImagePatch(corrected_frame,tracked_points_new_frame.at(i).x,tracked_points_new_frame.at(i).y);
        f->m_radiance_estimates = is;
        std::vector<double> gs = bilinearInterpolateImagePatch(gradient_corrected_image, tracked_points_new_frame.at(i).x, tracked_points_new_frame.at(i).y);
        f->m_gradient_values = gs;

        f->m_xy_location = tracked_points_new_frame.at(i);

	cv::Point2f m_xy_location_undistorted_;
	liftProjective(tracked_points_new_frame.at(i),m_xy_location_undistorted_);
	f->m_xy_location_undistorted = m_xy_location_undistorted_;

	f->id_feature = feature_id.at(i);
        f->m_next_feature = NULL;
        f->m_prev_feature =  m_database->m_tracked_frames.at(m_database->m_tracked_frames.size()-1).m_features.at(i);
        
        m_database->m_tracked_frames.at(m_database->m_tracked_frames.size()-1).m_features.at(i)->m_next_feature = f;
        
        frame.m_features.push_back(f);
        nr_pushed_features++;
    }
    
    m_database->m_tracked_frames.push_back(frame);
    
    // Extract new features
    vector<cv::Point2f> new_feature_locations = extractFeatures(corrected_frame,m_database->fetchActiveFeatureLocations());
    vector<int> new_validity_vector = checkLocationValidity(new_feature_locations);
    for(int p = 0;p < new_feature_locations.size();p++)
    {
        // Skip invalid points (too close to the side of image)
        if(new_validity_vector.at(p) == 0)
            continue;
        
        // Push back new feature information
        Feature* f = new Feature();
        f->m_xy_location = new_feature_locations.at(p);
	cv::Point2f m_xy_location_undistorted_;
	liftProjective(new_feature_locations.at(p),m_xy_location_undistorted_);
	f->m_xy_location_undistorted = m_xy_location_undistorted_;
    	f->id_feature = id_num;
        f->m_next_feature = NULL;
        f->m_prev_feature = NULL;
        // Todo: remove os, is, gs
        std::vector<double> os = bilinearInterpolateImagePatch(input_image,new_feature_locations.at(p).x,new_feature_locations.at(p).y);
        f->m_output_values = os;
        std::vector<double> is = bilinearInterpolateImagePatch(corrected_frame,new_feature_locations.at(p).x,new_feature_locations.at(p).y);
        f->m_radiance_estimates = is;
        std::vector<double> gs = bilinearInterpolateImagePatch(gradient_corrected_image, new_feature_locations.at(p).x, new_feature_locations.at(p).y);
        f->m_gradient_values = gs;
        m_database->m_tracked_frames.at(m_database->m_tracked_frames.size()-1).m_features.push_back(f);
	id_num++;
    }
}

void Tracker::liftProjective(const cv::Point2f& p, cv::Point2f& P)
{
    double mx_d, my_d,mx2_d, mxy_d, my2_d, mx_u, my_u;
    double rho2_d, rho4_d, radDist_d, Dx_d, Dy_d, inv_denom_d;
    double m_inv_K11, m_inv_K13, m_inv_K22, m_inv_K23;
    m_inv_K11 = 1.0 / 4.616e+02;
    m_inv_K13 = -3.630e+02 / 4.616e+02;
    m_inv_K22 = 1.0 / 4.603e+02;
    m_inv_K23 = -2.481e+02 / 4.603e+02;

    mx_d = m_inv_K11 * p.x + m_inv_K13;
    my_d = m_inv_K22 * p.y + m_inv_K23;

    int n = 8;
    Eigen::Vector2d d_u;
    distortion(Eigen::Vector2d(mx_d, my_d), d_u);
    // Approximate value
    mx_u = mx_d - d_u(0);
    my_u = my_d - d_u(1);

    for (int i = 1; i < n; ++i)
     {
      	distortion(Eigen::Vector2d(mx_u, my_u), d_u);
      	mx_u = mx_d - d_u(0);
      	my_u = my_d - d_u(1);
     }
    P.x = mx_u;
    P.y = my_u;
}

void Tracker::distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u) 
{
    double k1 = -2.917e-01;
    double k2 = 8.228e-02;
    double p1 = 5.333e-05;
    double p2 = -1.578e-04;

    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = p_u(0) * p_u(0);
    my2_u = p_u(1) * p_u(1);
    mxy_u = p_u(0) * p_u(1);
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
           p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
}


// Todo: change both types to reference
vector<cv::Point2f> Tracker::extractFeatures(cv::Mat frame,vector<cv::Point2f> old_features)
{
    std::vector<cv::Point2f> new_features;

    // No new features have to be extracted
    if(old_features.size() >= m_max_nr_active_features)
    {
        return new_features;
    }
    
    int nr_features_to_extract = static_cast<int>(m_max_nr_active_features-old_features.size());
    
    // Build spatial distribution map to check where to extract features
    int cells_r = 10;
    int cells_c = 10;
    
    double im_width  = m_database->m_image_width;
    double im_height = m_database->m_image_height;
        
    int cell_height = floor(im_height / cells_r);
    int cell_width  = floor(im_width  / cells_c);

    // Todo: change to class member
    int pointDistributionMap[cells_r][cells_c];
    for(int r = 0;r < cells_r;r++)
    {
        for(int c = 0;c < cells_c;c++)
        {
            pointDistributionMap[r][c] = 0;
        }
    }
    
    // Build the point distribution map to check where features need to be extracted mostly
    for(int p = 0;p < old_features.size();p++)
    {
        double x_value = old_features.at(p).x;
        double y_value = old_features.at(p).y;
        
        int c_bin = x_value / cell_width;
        if(c_bin >= cells_c)
            c_bin = cells_c - 1;
        
        int r_bin = y_value / cell_height;
        if(r_bin >= cells_r)
            r_bin = cells_r - 1;
        
        pointDistributionMap[r_bin][c_bin]++;
    }
    
    // Identify empty cells
    vector<int> empty_row_indices;
    vector<int> empty_col_indices;
    
    for(int r = 0;r < cells_r;r++)
    {
        for(int c = 0;c < cells_c;c++)
        {
            if(pointDistributionMap[r][c] == 0)
            {
                empty_row_indices.push_back(r);
                empty_col_indices.push_back(c);
            }
        }
    }

    // Todo: empty_col_indices might be 0!!!
    // Todo: Another bad case is: only one cell is empty and all other cells have only 1 feature inside,
    // Todo: then all the features to extract will be extracted from the single empty cell.
    int points_per_cell = ceil(nr_features_to_extract / (empty_col_indices.size()*1.0));
    
    // Extract "points per cell" features from each empty cell
    for(int i = 0;i < empty_col_indices.size();i++)
    {
        // Select random cell from where to extract features
        //int random_index = rand() % empty_row_indices.size();
        
        // Select row and col
        //int selected_row = empty_row_indices.at(random_index);
        //int selected_col = empty_col_indices.at(random_index);
        
        int selected_row = empty_row_indices.at(i);
        int selected_col = empty_col_indices.at(i);

        // Define the region of interest where to detect a feature
        cv::Rect ROI(selected_col * cell_width,selected_row * cell_height,cell_width,cell_height);
        
        // Extract features from this frame
        cv::Mat frame_roi = frame(ROI);
        
        // Extract features
        std::vector<cv::Point2f> good_corners;
        cv::goodFeaturesToTrack(frame_roi,
                                good_corners,
                                points_per_cell,
                                0.01,
                                7,
                                cv::Mat(),
                                7,
                                false,
                                0.04);

        //goodPixelsToTrack(frame_roi, good_corners, points_per_cell);

        // Add the strongest "points per cell" features from this extraction
        for(int k = 0;k < good_corners.size();k++)
        {
            if(k == points_per_cell)
                break;
            
            // Add the offset to the point location
            cv::Point2f point_location = good_corners.at(k);
            point_location.x += selected_col*cell_width;
            point_location.y += selected_row*cell_height;
            
            new_features.push_back(point_location);
        }
    }
    
    return new_features;
}

void Tracker::goodPixelsToTrack(cv::Mat frame,std::vector<cv::Point2f> &good_corners,int points_per_cell)
{ 
    int rows = frame.rows - 1;
    int cols = frame.cols - 1;

    double total_gradient;
    double avg_gradient;
    int num_pixel;

    num_pixel = rows * cols;

    std::vector<int> Pixels_row_indices;
    std::vector<int> Pixels_col_indices;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int dx = frame.at<uchar>(i, j + 1) - frame.at<uchar>(i, j);
            int dy = frame.at<uchar>(i + 1, j) - frame.at<uchar>(i, j);
            double ds = std::sqrt((dx*dx + dy*dy) / 2);
            total_gradient = total_gradient + ds;
        }
    }
   avg_gradient = total_gradient/num_pixel;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int dx = frame.at<uchar>(i, j + 1) - frame.at<uchar>(i, j);
            int dy = frame.at<uchar>(i + 1, j) - frame.at<uchar>(i, j);
            double ds = std::sqrt((dx*dx + dy*dy) / 2);
            if(ds > avg_gradient)
	    {
		Pixels_row_indices.push_back(i);
		Pixels_col_indices.push_back(j);
	    }
        }
    }
	
    for(int i = 0;i < points_per_cell;i++)
    {
	int random_index = rand() % Pixels_col_indices.size();
		
        int selected_row = Pixels_row_indices.at(random_index);
        int selected_col = Pixels_col_indices.at(random_index);

	good_corners.push_back(cv::Point2f(selected_row, selected_col));
    }
}

double Tracker::bilinearInterpolateImage(cv::Mat image,double x,double y)
{
    double floor_x = std::floor(x);
    double ceil_x  = std::ceil(x);
    
    double floor_y = std::floor(y);
    double ceil_y  = std::ceil(y);
    
    // Normalize x,y to be in [0,1)
    double x_normalized = x - floor_x;
    double y_normalized = y - floor_y;
    
    // Get bilinear interpolation weights
    double w1 = (1-x_normalized)*(1-y_normalized);
    double w2 = x_normalized*(1-y_normalized);
    double w3 = (1-x_normalized)*y_normalized;
    double w4 = x_normalized*y_normalized;
    
    // Evaluate image locations
    double i1 = static_cast<double>(image.at<uchar>(floor_y,floor_x));
    double i2 = static_cast<double>(image.at<uchar>(floor_y,ceil_x));
    double i3 = static_cast<double>(image.at<uchar>(ceil_y,floor_x));
    double i4 = static_cast<double>(image.at<uchar>(ceil_y,ceil_x));
    
    // Interpolate the result
    return w1*i1 + w2*i2 + w3*i3 + w4*i4;
}

vector<double> Tracker::bilinearInterpolateImagePatch(cv::Mat image,double x,double y)
{
    vector<double> result;
    
    for(int x_offset = -m_patch_size;x_offset <= m_patch_size;x_offset++)
    {
        for(int y_offset = -m_patch_size;y_offset <= m_patch_size;y_offset++)
        {
            double o_value = bilinearInterpolateImage(image,x+x_offset,y+y_offset);
            result.push_back(o_value);
        }
    }
    
    return result;
}

// Todo: change return to parameter passed by ref
vector<int> Tracker::checkLocationValidity(vector<cv::Point2f> points)
{
    // Check for each passed point location if the patch centered around it falls completely within the input images
    // Return 0 for a point if not, 1 if yes
    
    int min_x = m_patch_size+1;  //Todo: should be m_patch_size?
    int min_y = m_patch_size+1;
    
    int max_x = m_database->m_image_width-m_patch_size-1;
    int max_y = m_database->m_image_height-m_patch_size-1;
    
    vector<int> is_valid;
    
    for(int i = 0;i < points.size();i++)
    {
        if(points.at(i).x < min_x || points.at(i).x > max_x || points.at(i).y < min_y || points.at(i).y > max_y)
        {
            is_valid.push_back(0);
        }
        else
        {
            is_valid.push_back(1);
        }
    }
    
    return is_valid;
}

// Todo: change parameter type to reference (or const reference)
void Tracker::initialFeatureExtraction(cv::Mat input_image,cv::Mat gradient_image,double gt_exp_time)
{
    std::vector<cv::Point2f> old_f;
    std::vector<cv::Point2f> feature_locations = extractFeatures(input_image,old_f);
    std::vector<int> validity_vector = checkLocationValidity(feature_locations);
    
    // Initialize new tracking Frame
    Frame frame;
    frame.m_image = input_image;
    frame.m_image_corrected = input_image.clone();
    frame.m_exp_time = 1.0;
    frame.m_gt_exp_time = gt_exp_time;
    
    // Push back tracked feature points to the tracking Frame
    for(int p = 0;p < feature_locations.size();p++)
    {
        // Skip invalid points (too close to the side of image)
        if(validity_vector.at(p) == 0)
            continue;
        
        // Create new feature object and associate with it output intensities, gradient values, etc.
        Feature* f = new Feature();
        f->m_xy_location = feature_locations.at(p);
	cv::Point2f m_xy_location_undistorted_;
	liftProjective(feature_locations.at(p),m_xy_location_undistorted_);
	f->m_xy_location_undistorted = m_xy_location_undistorted_;
        f->id_feature = id_num;
        f->m_next_feature = NULL;
        f->m_prev_feature = NULL;
        std::vector<double> os = bilinearInterpolateImagePatch(input_image,feature_locations.at(p).x,feature_locations.at(p).y);
        std::vector<double> gs = bilinearInterpolateImagePatch(gradient_image,feature_locations.at(p).x,feature_locations.at(p).y);
        f->m_gradient_values = gs;
        f->m_output_values = os;
        f->m_radiance_estimates = os;
        frame.m_features.push_back(f);
	id_num++;
    }
    
    m_database->m_tracked_frames.push_back(frame);
}

void Tracker::photometricallyCorrectImage(cv::Mat &corrected_frame)
{
    for(int r = 0;r < corrected_frame.rows;r++)
    {
        for(int c = 0;c < corrected_frame.cols;c++)
        {
            int o_value = corrected_frame.at<uchar>(r,c);
            double radiance = m_database->m_response_estimate.removeResponse(o_value);
            double vig = m_database->m_vignette_estimate.getVignetteFactor(cv::Point2f(c,r));
            radiance /= vig;
            if(radiance > 255)radiance = 255;
            if(radiance < 0)radiance = 0;
            corrected_frame.at<uchar>(r,c) = (uchar)radiance;
        }
    }
}
//***************************************************************************//
GainRobustTracker::GainRobustTracker(int patch_size,int pyramid_levels)
{
    // Initialize patch size and pyramid levels
    m_patch_size = patch_size;
    m_pyramid_levels = pyramid_levels;
}


// Todo: change frame_1 frame 2 to ref (or const ref), pts_1 to ref
double GainRobustTracker::trackImagePyramids(cv::Mat frame_1,
                                             cv::Mat frame_2,
                                             vector<cv::Point2f> pts_1,
                                             vector<cv::Point2f>& pts_2,
                                             vector<int>& point_status)
{
    // All points valid in the beginning of tracking
    std::vector<int> point_validity;
    for(int i = 0;i < pts_1.size();i++)
    {
        point_validity.push_back(1);
    }
    
    // Calculate image pyramid of frame 1 and frame 2
    vector<cv::Mat> new_pyramid;
    cv::buildPyramid(frame_2, new_pyramid, m_pyramid_levels);
    
    vector<cv::Mat> old_pyramid;
    cv::buildPyramid(frame_1, old_pyramid, m_pyramid_levels);
    
    // Temporary vector to update tracking estiamtes over time
    vector<cv::Point2f> tracking_estimates = pts_1;
    
    double all_exp_estimates = 0.0;
    int nr_estimates = 0;
    
    // Iterate all pyramid levels and perform gain robust KLT on each level (coarse to fine)
    for(int level = (int)new_pyramid.size()-1;level >= 0;level--)
    {
        // Scale the input points and tracking estimates to the current pyramid level
        std::vector<cv::Point2f> scaled_tracked_points;
        std::vector<cv::Point2f> scaled_tracking_estimates;
        for(int i = 0;i < pts_1.size();i++)
        {
            cv::Point2f scaled_point;
            scaled_point.x = (float)(pts_1.at(i).x/pow(2,level));
            scaled_point.y = (float)(pts_1.at(i).y/pow(2,level));
            scaled_tracked_points.push_back(scaled_point);
            
            cv::Point2f scaled_estimate;
            scaled_estimate.x = (float)(tracking_estimates.at(i).x/pow(2,level));
            scaled_estimate.y = (float)(tracking_estimates.at(i).y/pow(2,level));
            scaled_tracking_estimates.push_back(scaled_estimate);
        }
        
        // Perform tracking on current level
        double exp_estimate = trackImageExposurePyr(old_pyramid.at(level),
                                                    new_pyramid.at(level),
                                                    scaled_tracked_points,
                                                    scaled_tracking_estimates,
                                                    point_validity);
        
        // Optional: Do something with the estimated exposure ratio
        // std::cout << "Estimated exposure ratio of current level: " << exp_estimate << std::endl;
        
        // Average estimates of each level later
        all_exp_estimates += exp_estimate;
        nr_estimates++;
        
        // Update the current tracking result by scaling down to pyramid level 0
        for(int i = 0;i < scaled_tracking_estimates.size();i++)
        {
            if(point_validity.at(i) == 0)
                continue;
            
            cv::Point2f scaled_point;
            scaled_point.x = (float)(scaled_tracking_estimates.at(i).x*pow(2,level));
            scaled_point.y = (float)(scaled_tracking_estimates.at(i).y*pow(2,level));
            
            tracking_estimates.at(i) = scaled_point;
        }
    }
    
    // Write result to output vectors passed by reference
    pts_2 = tracking_estimates;
    point_status = point_validity;
    
    // Average exposure ratio estimate
    double overall_exp_estimate = all_exp_estimates / nr_estimates;
    return overall_exp_estimate;
}


/**
 * For a reference on the meaning of the optimization variables and the overall concept of this function
 * refer to the photometric calibration paper 
 * introducing gain robust KLT tracking by Kim et al.
 */
 // Todo: change Mat and vector to ref
double GainRobustTracker::trackImageExposurePyr(cv::Mat old_image,
                                                cv::Mat new_image,
                                                vector<cv::Point2f> input_points,
                                                vector<cv::Point2f>& output_points,
                                                vector<int>& point_validity)
{
    // Number of points to track
    int nr_points = static_cast<int>(input_points.size());
    
    // Updated point locations which are updated throughout the iterations
    if(output_points.size() == 0)
    {
        output_points = input_points;
    }
    else if(output_points.size() != input_points.size())
    {
        std::cout << "ERROR - OUTPUT POINT SIZE != INPUT POINT SIZE!" << std::endl;
        return -1;
    }
    
    // Input image dimensions
    int image_rows = new_image.rows;
    int image_cols = new_image.cols;
    
    // Final exposure time estimate
    double K_total = 0.0;
    
    for(int round = 0;round < 1;round++)
    {
        // Get the currently valid points
        int nr_valid_points = getNrValidPoints(point_validity);
        
        // Allocate space for W,V matrices
        cv::Mat W(2*nr_valid_points,1,CV_64F,0.0);
        cv::Mat V(2*nr_valid_points,1,CV_64F,0.0);
        
        // Allocate space for U_INV and the original Us
        cv::Mat U_INV(2*nr_valid_points,2*nr_valid_points,CV_64F,0.0);
        vector<cv::Mat> Us;
        
        double lambda = 0;
        double m = 0;

        int absolute_point_index = -1;
        
        for(int p = 0;p < input_points.size();p++)
        {
            if(point_validity.at(p) == 0)
            {
                continue;
            }
            
            absolute_point_index++;
            
            // Build U matrix
            cv::Mat U(2,2, CV_64F, 0.0);
            
            // Bilinear image interpolation
            cv::Mat patch_intensities_1;
            cv::Mat patch_intensities_2;
            int absolute_patch_size = ((m_patch_size+1)*2+1);  // Todo: why m_patch_size+1?  9
            cv::getRectSubPix(new_image, cv::Size(absolute_patch_size,absolute_patch_size), output_points.at(p), patch_intensities_2,CV_32F);
            cv::getRectSubPix(old_image, cv::Size(absolute_patch_size,absolute_patch_size), input_points.at(p), patch_intensities_1,CV_32F);
            
            // Go through image patch around this point
            for(int r = 0; r < 2*m_patch_size+1;r++)
            {
                for(int c = 0; c < 2*m_patch_size+1;c++)
                {
                    // Fetch patch intensity values
                    double i_frame_1 = patch_intensities_1.at<float>(1+r,1+c);
                    double i_frame_2 = patch_intensities_2.at<float>(1+r,1+c);
                    
                    if(i_frame_1 < 1)
                        i_frame_1 = 1;
                    if(i_frame_2 < 1)
                        i_frame_2 = 1;
                    
                    // Estimate patch gradient values
                    double grad_1_x = (patch_intensities_1.at<float>(1+r,1+c+1) - patch_intensities_1.at<float>(1+r,1+c-1))/2;
                    double grad_1_y = (patch_intensities_1.at<float>(1+r+1,1+c) - patch_intensities_1.at<float>(1+r-1,1+c))/2;
                    
                    double grad_2_x = (patch_intensities_2.at<float>(1+r,1+c+1) - patch_intensities_2.at<float>(1+r,1+c-1))/2;
                    double grad_2_y = (patch_intensities_2.at<float>(1+r+1,1+c) - patch_intensities_2.at<float>(1+r-1,1+c))/2;
                    
                    double a = (1.0/i_frame_2)*grad_2_x + (1.0/i_frame_1)*grad_1_x;
                    double b = (1.0/i_frame_2)*grad_2_y + (1.0/i_frame_1)*grad_1_y;
                    double beta = log(i_frame_2/255.0) - log(i_frame_1/255.0);
                    
                    U.at<double>(0,0) += 0.5*a*a;
                    U.at<double>(1,0) += 0.5*a*b;
                    U.at<double>(0,1) += 0.5*a*b;
                    U.at<double>(1,1) += 0.5*b*b;
                    
                    W.at<double>(2*absolute_point_index,0)   -= a;
                    W.at<double>(2*absolute_point_index+1,0) -= b;
                    
                    V.at<double>(2*absolute_point_index,0)   -= beta*a;
                    V.at<double>(2*absolute_point_index+1,0) -= beta*b;
                    
                    lambda += 2;
                    m += 2*beta;
                }
            }
            
            //Back up U for re-substitution
            Us.push_back(U);
            
            //Invert matrix U for this point and write it to diagonal of overall U_INV matrix
            cv::Mat U_INV_p = U.inv();
            //std::cout << cv::determinant(U_INV_p) << std::endl;
            //std::cout << U_INV_p << std::endl;
            //std::cout << U << std::endl;
            
            U_INV.at<double>(2*absolute_point_index,2*absolute_point_index) = U_INV_p.at<double>(0,0);
            U_INV.at<double>(2*absolute_point_index+1,2*absolute_point_index) = U_INV_p.at<double>(1,0);
            U_INV.at<double>(2*absolute_point_index,2*absolute_point_index+1) = U_INV_p.at<double>(0,1);
            U_INV.at<double>(2*absolute_point_index+1,2*absolute_point_index+1) = U_INV_p.at<double>(1,1);
        }

        // Todo: check if opencv utilizes the sparsity of U
        //solve for the exposure
        cv::Mat K_MAT;
        cv::solve(-W.t()*U_INV*W+lambda, -W.t()*U_INV*V+m, K_MAT);
        double K = K_MAT.at<double>(0,0);
        
        //std::cout << -W.t()*U_INV*W+lambda << std::endl;
        //std::cout << -W.t()*U_INV*V+m << std::endl;
        //std::cout << K_MAT << std::endl;
        
        // Solve for the displacements
        absolute_point_index = -1;
        for(int p = 0;p < nr_points;p++)
        {
            if(point_validity.at(p) == 0)
                continue;
            
            absolute_point_index++;
            
            cv::Mat U_p = Us.at(absolute_point_index);
            cv::Mat V_p = V(cv::Rect(0,2*absolute_point_index,1,2));
            cv::Mat W_p = W(cv::Rect(0,2*absolute_point_index,1,2));
            
            cv::Mat displacement;
            cv::solve(U_p, V_p - K*W_p, displacement);
            
            //std::cout << displacement << std::endl;
            
            output_points.at(p).x += displacement.at<double>(0,0);
            output_points.at(p).y += displacement.at<double>(1,0);
            
            // Filter out this point if too close at the boundaries
            int filter_margin = 2;
            double x = output_points.at(p).x;
            double y = output_points.at(p).y;
            // Todo: the latter two should be ">=" ?
            if(x < filter_margin || y < filter_margin || x > image_cols-filter_margin || y > image_rows-filter_margin)
            {
                point_validity.at(p) = 0;
            }
        }
        
        K_total += K;
    }
    
    return exp(K_total);
}

int GainRobustTracker::getNrValidPoints(vector<int> validity_vector)
{
    // Simply sum up the validity vector
    int result = 0;
    for(int i = 0;i < validity_vector.size();i++)
    {
        result += validity_vector.at(i);
    }
    return result;
}
//***************************************************************************//



























































































































































































