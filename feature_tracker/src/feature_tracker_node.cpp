#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

int start_image_index = 0;
int end_image_index   = -1;
int image_width       = 752;
int image_height      = 480;  
int visualize_cnt       = 1; 
int tracker_patch_size  = 3; 
int nr_pyramid_levels   = 2; 
int nr_active_features  = 200;  
int nr_images_rapid_exp = 15;
int nr_active_frames    = 250;  
int keyframe_spacing    = 15; 
int min_keyframes_valid = 3;

int safe_zone_size = 20;
int num_frame = 0;
int optimize_cnt = 0;
double vis_exponent = 1.0;

FeatureTracker trackerData[NUM_OF_CAM];
Database database(image_width,image_height);
RapidExposureTimeEstimator exposure_estimator(nr_images_rapid_exp, &database);
NonlinearOptimizer backend_optimizer(keyframe_spacing,//15
                                         &database,
                                         safe_zone_size,//20
                                         min_keyframes_valid,//3
                                         tracker_patch_size);//3
Tracker tracker(tracker_patch_size,nr_active_features,nr_pyramid_levels,&database);

// Optimization thread handle
pthread_t opt_thread = 0;

// This variable indicates if currently an optimization task is running in a second thread
pthread_mutex_t g_is_optimizing_mutex;
bool g_is_optimizing = false;

double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

void *run_optimization_task(void* thread_arg)
{
    //std::cout << "START OPTIMIZATION" << std::endl;
    ROS_INFO("START OPTIMIZATION");
    pthread_mutex_lock(&g_is_optimizing_mutex);
    g_is_optimizing = true;
    pthread_mutex_unlock(&g_is_optimizing_mutex);
    
    // The nonlinear optimizer contains all the optimization information
    NonlinearOptimizer* optimizer = (NonlinearOptimizer*)thread_arg;
    
    optimizer->fetchResponseVignetteFromDatabase();

    // Perform optimization
    optimizer->evfOptimization(false);
    optimizer->evfOptimization(false);
    optimizer->evfOptimization(false);
    
    // Smooth optimization data
    optimizer->smoothResponse();
    
    // Initialize the inverse response vector with the current inverse response estimate
    // (in order to write it to the database later + visualization)
    // better to do this here since currently the inversion is done rather inefficiently and not to slow down tracking
    optimizer->getInverseResponseRaw(optimizer->m_raw_inverse_response);
    
    pthread_mutex_lock(&g_is_optimizing_mutex);
    g_is_optimizing = false;
    pthread_mutex_unlock(&g_is_optimizing_mutex);
    //std::cout << "END OPTIMIZATION" << std::endl;
    ROS_INFO("END OPTIMIZATION");
    pthread_exit(NULL);
}

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    num_frame++;

    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }
    // detect unstable camera stream
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = img_msg->header.stamp.toSec();
    // frequency control
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    if(num_frame == nr_images_rapid_exp*2 + safe_zone_size)//50
    {
        for(int ii = 0;ii < nr_images_rapid_exp;ii++)//15
        {
            database.removeLastFrame();
        }
    }
        
     // If the database is large enough, start removing old frames
    if(num_frame > nr_active_frames)//200
        database.removeLastFrame();

    cv::Mat show_img = ptr->image;
    TicToc t_r;
    double gt_exp_time = -1.0;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)
            //trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
		tracker.trackNewFrame(ptr->image.rowRange(ROW * i, ROW * (i + 1)), gt_exp_time);
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    /*for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }*/
    double exposure_time = exposure_estimator.estimateExposureTime();
    database.m_tracked_frames.at(database.m_tracked_frames.size()-1).m_exp_time = exposure_time;
    database.visualizeRapidExposureTimeEstimates(vis_exponent);
    //cout << "exposure_time"<< exposure_time << endl;
    ROS_INFO("exposure_time  %f", exposure_time);

    std::vector<Feature*>* features = &database.m_tracked_frames.at(database.m_tracked_frames.size()-1).m_features;
    for(int k = 0;k < features->size();k++)
    {
        for(int r = 0;r < features->at(k)->m_radiance_estimates.size();r++)
        {
            features->at(k)->m_radiance_estimates.at(r) /= exposure_time;
        }
    }

   if (PUB_THIS_FRAME)
   {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            vector<cv::Point2f> cur_pts = database.fetchActiveFeatureLocations();
	    vector<cv::Point2f> un_pts = database.fetchActiveFeatureLocationsUndistorted();
	    vector<int> ids = database.fetchActiveFeatureID();
	    
            //auto &un_pts = trackerData[i].cur_un_pts;
            //auto &cur_pts = trackerData[i].cur_pts;
            //auto &ids = trackerData[i].ids;
            //auto &pts_velocity = trackerData[i].pts_velocity;
            
                for (unsigned int j = 0; j < ids.size(); j++)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(0);
                    velocity_y_of_point.values.push_back(0);
                }
            
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);
    }
    //ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
    
    pthread_mutex_lock(&g_is_optimizing_mutex);
    bool is_optimizing = g_is_optimizing;
    pthread_mutex_unlock(&g_is_optimizing_mutex);
        
    // Optimization is still running, don't do anything and keep tracking
    if(is_optimizing)
    {
        return;
    }

    if(optimize_cnt > 0)
    {
    	database.m_vignette_estimate.setVignetteParameters(backend_optimizer.m_vignette_estimate);
    	database.m_response_estimate.setGrossbergParameterVector(backend_optimizer.m_response_estimate);
    	database.m_response_estimate.setInverseResponseVector(backend_optimizer.m_raw_inverse_response);

    	vis_exponent = backend_optimizer.visualizeOptimizationResult(backend_optimizer.m_raw_inverse_response);
    }

    bool succeeded = backend_optimizer.extractOptimizationBlock();

    if(succeeded)
    {
	//ROS_DEBUG("succeeded");
        // TODO: reuse thread
        //start a new optimization task here
        pthread_create(&opt_thread, NULL, run_optimization_task, (void*)&backend_optimizer);
	optimize_cnt++;
    }
    //pthread_join(opt_thread,NULL);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);

    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin();
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?
