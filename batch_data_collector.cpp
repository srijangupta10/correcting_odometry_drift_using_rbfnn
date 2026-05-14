#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <linux/input.h>
#include <pigpio.h>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <vector>
#include <cmath>
#include <iomanip>
#include <unordered_map>
#include <Eigen/Dense>

#include "sl_lidar.h"
#include "sl_lidar_driver.h"
#include "nanoflann.hpp"

using namespace std;
using namespace Eigen;
using namespace sl;

#define IN1 24
#define IN2 23
#define IN3 17
#define IN4 22
#define ENA 13
#define ENB 12

#define MOUSE_LEFT "/dev/input/by-id/usb-Logitech_G102_LIGHTSYNC_Gaming_Mouse_2067327C4D47-event-mouse"
#define MOUSE_RIGHT "/dev/input/by-id/usb-Logitech_G102_LIGHTSYNC_Gaming_Mouse_206337685947-event-mouse"

#define OUTPUT_DIR "results_mouse_lidar"
const std::string SURFACE_NAME = "weighted_12_tiles";

const double L_SENSOR_SEPARATION = 140.0;
double CPI_LEFT_X = 314.409;  double CPI_LEFT_Y = 307.341;
double CPI_RIGHT_X = 324.244; double CPI_RIGHT_Y = 321.594;

const int NUM_ITERATIONS = 10; 

struct PositionConfig {
    std::string id;
    double x, y, heading;
};

struct RunConfig {
    std::string run_id;
    std::string category;
    int motion_type;
    int speed_left;
    int speed_right;
    double drive_time;
    double coast_time;
    std::string description;
    std::string start_position_id;
    double start_heading;
};

std::unordered_map<std::string, PositionConfig> positions;
std::vector<RunConfig> all_runs;

void init_embedded_configs() {
    int idx = 1;
    for (int y : {1500, 900, 300}) {
        for (int x : {300, 900, 1500, 2100}) {
            std::string id = "T" + std::to_string(idx) + "_CENTER";
            positions[id] = {id, (double)x, (double)y, 0};
            idx++;
        }
    }
    positions["T9_LEFT"] = {"T9_LEFT", 150, 300, 90};
    positions["T5_LEFT"] = {"T5_LEFT", 150, 900, 90};
    positions["T1_LEFT"] = {"T1_LEFT", 150, 1500, 90};
    positions["T9_BOTTOM"] = {"T9_BOTTOM", 300, 150, 0};
    positions["T10_BOTTOM"] = {"T10_BOTTOM", 900, 150, 0};
    positions["T11_BOTTOM"] = {"T11_BOTTOM", 1500, 150, 0};
    positions["T12_BOTTOM"] = {"T12_BOTTOM", 2100, 150, 0};
    
    positions["T1T2_CIRCLE"] = {"T1T2_CIRCLE", 600, 1200, 0};
    positions["T2T3_CIRCLE"] = {"T2T3_CIRCLE", 1200, 1200, 0};
    positions["T3T4_CIRCLE"] = {"T3T4_CIRCLE", 1800, 1200, 0};
    positions["T5T6_CIRCLE"] = {"T5T6_CIRCLE", 600, 600, 0};
    positions["T6T7_CIRCLE"] = {"T6T7_CIRCLE", 1200, 600, 0};
    positions["T7T8_CIRCLE"] = {"T7T8_CIRCLE", 1800, 600, 0};
    positions["GRID_CENTER"] = {"GRID_CENTER", 1200, 900, 0};

    double coast = 5.0;
    
    all_runs.push_back({"SR1", "straight", 1, 60, 60, 141, coast, "Row1_Bottom_PWM60", "T9_LEFT", 90});
    all_runs.push_back({"SR2", "straight", 1, 157, 157, 54, coast, "Row1_Bottom_PWM157", "T9_LEFT", 90});
    all_runs.push_back({"SR3", "straight", 1, 255, 255, 33, coast, "Row1_Bottom_PWM255", "T9_LEFT", 90});
    all_runs.push_back({"SR4", "straight", 1, 60, 60, 141, coast, "Row2_Mid_PWM60", "T5_LEFT", 90});
    all_runs.push_back({"SR5", "straight", 1, 157, 157, 54, coast, "Row2_Mid_PWM157", "T5_LEFT", 90});
    all_runs.push_back({"SR6", "straight", 1, 255, 255, 33, coast, "Row2_Mid_PWM255", "T5_LEFT", 90});
    all_runs.push_back({"SR7", "straight", 1, 60, 60, 141, coast, "Row3_Top_PWM60", "T1_LEFT", 90});
    all_runs.push_back({"SR8", "straight", 1, 157, 157, 54, coast, "Row3_Top_PWM157", "T1_LEFT", 90});
    all_runs.push_back({"SR9", "straight", 1, 255, 255, 33, coast, "Row3_Top_PWM255", "T1_LEFT", 90});

    all_runs.push_back({"SC1", "straight", 1, 60, 60, 100, coast, "Col1_Left_PWM60", "T9_BOTTOM", 0});
    all_runs.push_back({"SC2", "straight", 1, 157, 157, 37, coast, "Col1_Left_PWM157", "T9_BOTTOM", 0});
    all_runs.push_back({"SC3", "straight", 1, 255, 255, 23, coast, "Col1_Left_PWM255", "T9_BOTTOM", 0});
    all_runs.push_back({"SC4", "straight", 1, 60, 60, 100, coast, "Col2_MidL_PWM60", "T10_BOTTOM", 0});
    all_runs.push_back({"SC5", "straight", 1, 157, 157, 37, coast, "Col2_MidL_PWM157", "T10_BOTTOM", 0});
    all_runs.push_back({"SC6", "straight", 1, 255, 255, 23, coast, "Col2_MidL_PWM255", "T10_BOTTOM", 0});
    all_runs.push_back({"SC7", "straight", 1, 60, 60, 100, coast, "Col3_MidR_PWM60", "T11_BOTTOM", 0});
    all_runs.push_back({"SC8", "straight", 1, 157, 157, 37, coast, "Col3_MidR_PWM157", "T11_BOTTOM", 0});
    all_runs.push_back({"SC9", "straight", 1, 255, 255, 23, coast, "Col3_MidR_PWM255", "T11_BOTTOM", 0});
    all_runs.push_back({"SC10", "straight", 1, 60, 60, 100, coast, "Col4_Right_PWM60", "T12_BOTTOM", 0});
    all_runs.push_back({"SC11", "straight", 1, 157, 157, 37, coast, "Col4_Right_PWM157", "T12_BOTTOM", 0});
    all_runs.push_back({"SC12", "straight", 1, 255, 255, 23, coast, "Col4_Right_PWM255", "T12_BOTTOM", 0});

    for (int t = 1; t <= 12; t++) {
        std::string loc = "T" + std::to_string(t);
        std::string pos = loc + "_CENTER";
        all_runs.push_back({loc+"_SP1", "spin", 6, 0, 60, 15, coast, loc+"_PivotLeft_PWM60", pos, 0});
        all_runs.push_back({loc+"_SP2", "spin", 6, 0, 157, 15, coast, loc+"_PivotLeft_PWM157", pos, 0});
        all_runs.push_back({loc+"_SP3", "spin", 6, 0, 255, 15, coast, loc+"_PivotLeft_PWM255", pos, 0});
        all_runs.push_back({loc+"_SP4", "spin", 7, 60, 0, 15, coast, loc+"_PivotRight_PWM60", pos, 0});
        all_runs.push_back({loc+"_SP5", "spin", 7, 157, 0, 15, coast, loc+"_PivotRight_PWM157", pos, 0});
        all_runs.push_back({loc+"_SP6", "spin", 7, 255, 0, 15, coast, loc+"_PivotRight_PWM255", pos, 0});
    }

    std::vector<std::string> quads = {"T1T2", "T2T3", "T3T4", "T5T6", "T6T7", "T7T8"};
    for (const auto& q : quads) {
        std::string pos = q + "_CIRCLE";
        all_runs.push_back({q+"_T1", "turn", 4, 24, 60, 71, coast, q+"_TightLeft_PWM60", pos, 0});
        all_runs.push_back({q+"_T2", "turn", 4, 62, 157, 27, coast, q+"_TightLeft_PWM157", pos, 0});
        all_runs.push_back({q+"_T3", "turn", 4, 102, 255, 17, coast, q+"_TightLeft_PWM255", pos, 0});
        all_runs.push_back({q+"_T4", "turn", 2, 42, 60, 71, coast, q+"_GentleLeft_PWM60", pos, 0});
        all_runs.push_back({q+"_T5", "turn", 2, 110, 157, 27, coast, q+"_GentleLeft_PWM157", pos, 0});
        all_runs.push_back({q+"_T6", "turn", 2, 178, 255, 17, coast, q+"_GentleLeft_PWM255", pos, 0});
        all_runs.push_back({q+"_T7", "turn", 5, 60, 24, 71, coast, q+"_TightRight_PWM60", pos, 0});
        all_runs.push_back({q+"_T8", "turn", 5, 157, 62, 27, coast, q+"_TightRight_PWM157", pos, 0});
        all_runs.push_back({q+"_T9", "turn", 5, 255, 102, 17, coast, q+"_TightRight_PWM255", pos, 0});
        all_runs.push_back({q+"_T10", "turn", 3, 60, 42, 71, coast, q+"_GentleRight_PWM60", pos, 0});
        all_runs.push_back({q+"_T11", "turn", 3, 157, 110, 27, coast, q+"_GentleRight_PWM157", pos, 0});
        all_runs.push_back({q+"_T12", "turn", 3, 255, 178, 17, coast, q+"_GentleRight_PWM255", pos, 0});
    }
}

struct Point { float x, y; };

struct LidarPose {
    double timestamp;
    double x, y, theta, dx, dy, dtheta;
    bool valid;
    LidarPose() : timestamp(0), x(0), y(0), theta(0), dx(0), dy(0), dtheta(0), valid(false) {}
};

struct PointCloud {
    std::vector<Point> pts;
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const { return (dim == 0) ? pts[idx].x : pts[idx].y; }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};
typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 2> my_kd_tree_t;

struct MouseOdometry {
    double timestamp;
    int dx_L_raw, dy_L_raw, dx_R_raw, dy_R_raw;
    double lat_mm, fwd_mm, rot_rad, x, y, theta;
    MouseOdometry() : timestamp(0), dx_L_raw(0), dy_L_raw(0), dx_R_raw(0), dy_R_raw(0), lat_mm(0), fwd_mm(0), rot_rad(0), x(0), y(0), theta(0) {}
};

std::vector<MouseOdometry> mouse_trajectory;
std::vector<LidarPose> lidar_trajectory;

std::atomic<int> dx_l(0), dy_l(0), dx_r(0), dy_r(0);
std::atomic<bool> running(true);
std::atomic<bool> iteration_running(false); 
std::atomic<bool> motors_started(false);
std::atomic<bool> map_is_built(false);      
std::atomic<bool> collision_imminent(false);

std::chrono::steady_clock::time_point global_start_time;
std::mutex lidar_mutex; 

int get_choice() {
    int choice;
    while (!(std::cin >> choice)) {
        std::cin.clear();
        std::cin.ignore(10000, '\n');
        std::cout << "Invalid input. Please enter a number: ";
    }
    std::cin.ignore(10000, '\n');
    return choice;
}

void wait_for_enter() {
    std::string dummy;
    std::getline(std::cin, dummy);
}

void setupMotor() {
    if (gpioInitialise() < 0) exit(1);
    for (int pin : {IN1, IN2, IN3, IN4, ENA, ENB}) gpioSetMode(pin, PI_OUTPUT);
}

void run_motors(int speed_left, int speed_right) {
    gpioWrite(IN1, 1); gpioWrite(IN2, 0); 
    gpioWrite(IN3, 1); gpioWrite(IN4, 0); 
    
    gpioPWM(ENA, speed_left);
    gpioPWM(ENB, speed_right);
}

void stopMotor() {
    gpioPWM(ENA, 0); gpioPWM(ENB, 0);
    gpioWrite(IN1, 0); gpioWrite(IN2, 0); gpioWrite(IN3, 0); gpioWrite(IN4, 0);
}

void read_mouse(const char* device, std::atomic<int>& dx, std::atomic<int>& dy, const char* name) {
    int fd = open(device, O_RDONLY | O_NONBLOCK);
    if (fd == -1) return;
    struct pollfd pfd = {fd, POLLIN, 0}; struct input_event ev;
    std::cout << name << " mouse connected.\n";
    while (running.load()) {
        if (poll(&pfd, 1, 100) > 0 && (pfd.revents & POLLIN)) {
            if (read(fd, &ev, sizeof(ev)) == sizeof(ev) && ev.type == EV_REL) {
                if (ev.code == REL_X) dx.fetch_add(ev.value);
                else if (ev.code == REL_Y) dy.fetch_add(ev.value);
            }
        }
    }
    close(fd);
}

vector<Point> process_scan(sl_lidar_response_measurement_node_hq_t *nodes, size_t count) {
    vector<Point> pts;
    
    const float ROBOT_HALF_WIDTH = 0.145f; 
    const float CRITICAL_ZONE = 0.200f;    
    
    bool local_collision = false;

    for (size_t i = 0; i < count; i++) {
        float angle = nodes[i].angle_z_q14 * 90.f / (1 << 14);
        float dist  = nodes[i].dist_mm_q2 / 4.0f / 1000.0f;
        if (dist < 0.10 || dist > 2.5) continue;
        
        float rad = angle * M_PI / 180.0;
        
        float x_local = -dist * sin(rad); 
        float y_local = -dist * cos(rad); 
        
        if (y_local > 0 && y_local < CRITICAL_ZONE && abs(x_local) < ROBOT_HALF_WIDTH) {
            local_collision = true;
        }
        pts.push_back({x_local, y_local});
    }
    
    if (local_collision && motors_started.load()) collision_imminent.store(true);
    return pts;
}

vector<Point> downsample(const vector<Point>& cloud, float leaf_size) {
    vector<Point> filtered; unordered_map<string, bool> grid;
    for (const auto& p : cloud) {
        string key = to_string((int)round(p.x/leaf_size)) + "," + to_string((int)round(p.y/leaf_size));
        if (grid.find(key) == grid.end()) { grid[key] = true; filtered.push_back(p); }
    }
    return filtered;
}

bool icp_fast(const vector<Point>& source_pts, const vector<Point>& target_pts, Matrix2f &R_out, Vector2f &t_out, int max_iters = 50, float tolerance = 1e-5) {
    PointCloud target_cloud = {target_pts};
    my_kd_tree_t index(2, target_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10)); index.buildIndex();
    vector<Vector2f> src_eigen; for (auto& p : source_pts) src_eigen.push_back({p.x, p.y});
    Matrix2f R = Matrix2f::Identity(); Vector2f t = Vector2f::Zero(); float prev_error = 1e9;
    
    for (int iter = 0; iter < max_iters; iter++) {
        vector<Vector2f> src_matched, dst_matched; float total_error = 0;
        for (auto& p : src_eigen) {
            float query_pt[2] = {p.x(), p.y()}; size_t ret_index; float out_dist_sqr;
            nanoflann::KNNResultSet<float> res(1); res.init(&ret_index, &out_dist_sqr);
            index.findNeighbors(res, query_pt, nanoflann::SearchParameters(10));
            if (out_dist_sqr < 0.0225f) { src_matched.push_back(p); dst_matched.push_back({target_pts[ret_index].x, target_pts[ret_index].y}); total_error += out_dist_sqr; }
        }
        if (src_matched.size() < 10) return false;
        
        Vector2f cA(0,0), cB(0,0);
        for (size_t i = 0; i < src_matched.size(); i++) { cA += src_matched[i]; cB += dst_matched[i]; }
        cA /= src_matched.size(); cB /= dst_matched.size();
        
        Matrix2f H = Matrix2f::Zero();
        for (size_t i = 0; i < src_matched.size(); i++) H += (src_matched[i] - cA) * (dst_matched[i] - cB).transpose();
        JacobiSVD<Matrix2f> svd(H, ComputeFullU | ComputeFullV); Matrix2f dR = svd.matrixV() * svd.matrixU().transpose();
        if (dR.determinant() < 0) { Matrix2f V = svd.matrixV(); V.col(1) *= -1; dR = V * svd.matrixU().transpose(); }
        Vector2f dt = cB - dR * cA; R = dR * R; t = dR * t + dt;
        for (auto& p : src_eigen) p = dR * p + dt;
        if (abs(prev_error - total_error) < tolerance) break;
        prev_error = total_error;
    }
    R_out = R; t_out = t; return true;
}

MouseOdometry compute_mouse_odometry(int dx_L, int dy_L, int dx_R, int dy_R, double p_x, double p_y, double p_th) {
    MouseOdometry o; o.dx_L_raw = dx_L; o.dy_L_raw = dy_L; o.dx_R_raw = dx_R; o.dy_R_raw = dy_R;
    o.lat_mm = ((dx_L / CPI_LEFT_X) + (dx_R / CPI_RIGHT_X)) / 2.0;
    o.fwd_mm = ((-dy_L / CPI_LEFT_Y) + (-dy_R / CPI_RIGHT_Y)) / 2.0;
    o.rot_rad = ((-dy_R / CPI_RIGHT_Y) - (-dy_L / CPI_LEFT_Y)) / L_SENSOR_SEPARATION;
    double th_mid = p_th + o.rot_rad / 2.0;
    o.x = p_x + o.lat_mm * cos(th_mid) - o.fwd_mm * sin(th_mid);
    o.y = p_y + o.lat_mm * sin(th_mid) + o.fwd_mm * cos(th_mid);
    o.theta = p_th + o.rot_rad; return o;
}

void lidar_tracking_thread(ILidarDriver* drv) {
    sl_lidar_response_measurement_node_hq_t nodes[8192];
    size_t count; vector<Point> global_map;
    bool map_locked = false; int warmup_count = 0;
    double current_x = 0.0, current_y = 0.0, current_theta = 0.0;
    double prev_x = 0.0, prev_y = 0.0, prev_theta = 0.0;
    
    global_map.clear();
    map_is_built.store(false);
    
    for(int i = 0; i < 5; i++) {
        count = sizeof(nodes) / sizeof(nodes[0]);
        drv->grabScanDataHq(nodes, count);
        drv->ascendScanData(nodes, count);
    }
    
    while (iteration_running.load()) {
        count = sizeof(nodes) / sizeof(nodes[0]);
        if (SL_IS_FAIL(drv->grabScanDataHq(nodes, count))) continue;
        drv->ascendScanData(nodes, count);
        auto raw_pts = process_scan(nodes, count); auto pts = downsample(raw_pts, 0.015f);
        if (pts.size() < 50) continue;
        
        if (!map_locked) {
            for (auto p : pts) global_map.push_back(p);
            warmup_count++;
            if (warmup_count >= 15) { 
                global_map = downsample(global_map, 0.01f);
                map_locked = true; map_is_built.store(true); 
            }
            continue;
        }

        if (!motors_started.load()) {
            current_x = 0; current_y = 0; current_theta = 0;
            prev_x = 0; prev_y = 0; prev_theta = 0;
            continue; 
        }
        
        LidarPose pose;
        vector<Point> guessed_pts; double cos_t = cos(current_theta); double sin_t = sin(current_theta);
        for (auto p : pts) guessed_pts.push_back({(float)(p.x * cos_t - p.y * sin_t + current_x), (float)(p.x * sin_t + p.y * cos_t + current_y)});
        
        Matrix2f R_step; Vector2f t_step;
        if (!icp_fast(guessed_pts, global_map, R_step, t_step)) {
            pose.valid = false; pose.timestamp = chrono::duration<double>(chrono::steady_clock::now() - global_start_time).count();
            std::lock_guard<std::mutex> lock(lidar_mutex); lidar_trajectory.push_back(pose); continue;
        }
        
        double new_x = R_step(0,0) * current_x + R_step(0,1) * current_y + t_step(0);
        double new_y = R_step(1,0) * current_x + R_step(1,1) * current_y + t_step(1);
        
        current_x = new_x;
        current_y = new_y;
        current_theta += atan2(R_step(1,0), R_step(0,0));
        
        double dX_gl = current_x - prev_x; double dY_gl = current_y - prev_y;
        double c_pr = cos(prev_theta); double s_pr = sin(prev_theta);
        
        pose.dx = dX_gl * c_pr + dY_gl * s_pr;   
        pose.dy = -dX_gl * s_pr + dY_gl * c_pr;  
        pose.dtheta = current_theta - prev_theta;
        
        while (pose.dtheta > M_PI) pose.dtheta -= 2.0 * M_PI; while (pose.dtheta < -M_PI) pose.dtheta += 2.0 * M_PI;
        pose.x = current_x; pose.y = current_y; pose.theta = current_theta; pose.valid = true;
        pose.timestamp = chrono::duration<double>(chrono::steady_clock::now() - global_start_time).count();
        
        std::lock_guard<std::mutex> lock(lidar_mutex); lidar_trajectory.push_back(pose);
        prev_x = current_x; prev_y = current_y; prev_theta = current_theta;
    }
}

void save_integrated_dataset(int iteration, const RunConfig& run) {
    system(("mkdir -p " + std::string(OUTPUT_DIR)).c_str());
    auto time_t = chrono::system_clock::to_time_t(chrono::system_clock::now());
    std::stringstream ts; ts << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    
    std::string fn = string(OUTPUT_DIR) + "/" + SURFACE_NAME + "_" + run.run_id + "_iter" + to_string(iteration) + "_" + ts.str() + ".csv";
    
    std::ofstream csv(fn);
    csv << "timestamp_sec,mouse_dx_L,mouse_dy_L,mouse_dx_R,mouse_dy_R,mouse_lat_mm,mouse_fwd_mm,mouse_rot_rad,mouse_x,mouse_y,mouse_theta,lidar_x_mm,lidar_y_mm,lidar_theta_rad,lidar_dx_mm,lidar_dy_mm,lidar_dtheta_rad,lidar_valid\n";
    
    size_t m_idx = 0, l_idx = 0;
    std::lock_guard<std::mutex> lock(lidar_mutex);
    
    while (m_idx < mouse_trajectory.size() || l_idx < lidar_trajectory.size()) {
        bool h_m = m_idx < mouse_trajectory.size(), h_l = l_idx < lidar_trajectory.size();
        double t_m = h_m ? mouse_trajectory[m_idx].timestamp : 1e9, t_l = h_l ? lidar_trajectory[l_idx].timestamp : 1e9;
        
        if (h_m && (!h_l || t_m <= t_l)) {
            auto& m = mouse_trajectory[m_idx++];
            csv << std::fixed << std::setprecision(4) << m.timestamp << "," << m.dx_L_raw << "," << m.dy_L_raw << "," << m.dx_R_raw << "," << m.dy_R_raw << "," << m.lat_mm << "," << m.fwd_mm << "," << m.rot_rad << "," << m.x << "," << m.y << "," << m.theta << ",,,,,,,\n";
        } else {
            auto& l = lidar_trajectory[l_idx++];
            csv << std::fixed << std::setprecision(4) << l.timestamp << ",,,,,,,,,,," << l.x * 1000.0 << "," << l.y * 1000.0 << "," << l.theta << "," << l.dx * 1000.0 << "," << l.dy * 1000.0 << "," << l.dtheta << "," << (int)l.valid << "\n";
        }
    }
    csv.close();
    std::cout << "Saved: " << fn << "\n";
}

int main() {
    init_embedded_configs();

    std::cout << "\nINTERACTIVE RBFNN DATA COLLECTOR (10x TRAINING LOOP)\n";
    
    std::cout << "\nSelect Run Category:\n";
    std::cout << "1. Straight Line\n";
    std::cout << "2. Spin / Pivot\n";
    std::cout << "3. Turn (Quadrants)\n";
    std::cout << "> ";
    int cat_choice = get_choice();

    std::vector<RunConfig> category_runs;
    
    if (cat_choice == 1) { 
        std::cout << "\nSelect Straight Line Direction:\n";
        std::cout << "1. Horizontal (Rows)\n";
        std::cout << "2. Vertical (Columns)\n";
        std::cout << "> ";
        int dir_choice = get_choice();
        
        for(auto& r : all_runs) {
            if(r.category == "straight") {
                if (dir_choice == 1 && r.run_id.find("SR") != std::string::npos) category_runs.push_back(r);
                if (dir_choice == 2 && r.run_id.find("SC") != std::string::npos) category_runs.push_back(r);
            }
        }
    } else if (cat_choice == 2) {
        for(auto& r : all_runs) if(r.category == "spin") category_runs.push_back(r);
    } else if (cat_choice == 3) {
        for(auto& r : all_runs) if(r.category == "turn") category_runs.push_back(r);
    }

    if (category_runs.empty()) {
        std::cout << "No matching runs found.\n";
        return 0;
    }

    std::map<std::string, std::vector<RunConfig>> pos_groups;
    std::vector<std::string> pos_order;
    for (auto& r : category_runs) {
        if (pos_groups.find(r.start_position_id) == pos_groups.end()) pos_order.push_back(r.start_position_id);
        pos_groups[r.start_position_id].push_back(r);
    }

    std::cout << "\nSelect Start Position (Tile ID):\n";
    for (size_t i = 0; i < pos_order.size(); i++) {
        std::cout << i+1 << ". " << pos_order[i] << "\n";
    }
    std::cout << "> ";
    int pos_choice = get_choice();
    if (pos_choice < 1 || pos_choice > pos_order.size()) return 0;
    std::string selected_pos = pos_order[pos_choice - 1];

    auto& final_runs = pos_groups[selected_pos];
    std::cout << "\nSelect Specific Run Profile:\n";
    for (size_t i = 0; i < final_runs.size(); i++) {
        std::cout << i+1 << ". " << final_runs[i].run_id << " (" << final_runs[i].description 
                  << ") | PWM: " << final_runs[i].speed_left << "L/" << final_runs[i].speed_right << "R\n";
    }
    std::cout << final_runs.size() + 1 << ". Run ALL above sequentially (" << NUM_ITERATIONS << "x iterations each)\n";
    std::cout << "> ";
    int run_choice = get_choice();

    std::vector<RunConfig> runs_to_execute;
    if (run_choice >= 1 && run_choice <= final_runs.size()) {
        runs_to_execute.push_back(final_runs[run_choice - 1]);
    } else {
        runs_to_execute = final_runs;
    }

    setupMotor();
    auto channelRes = createSerialPortChannel("/dev/ttyUSB0", 115200);
    if (!channelRes) return -1;
    ILidarDriver* drv = *createLidarDriver();
    if (SL_IS_FAIL(drv->connect(*channelRes))) return -1;
    
    drv->startScan(false, true);
    
    std::thread t_left(read_mouse, MOUSE_LEFT, std::ref(dx_l), std::ref(dy_l), "LEFT");
    std::thread t_right(read_mouse, MOUSE_RIGHT, std::ref(dx_r), std::ref(dy_r), "RIGHT");
    
    for (size_t i = 0; i < runs_to_execute.size(); ++i) {
        const auto& run = runs_to_execute[i];
        PositionConfig pos = positions[run.start_position_id];

        std::cout << "\nSTARTING TRAINING BLOCK: " << run.run_id << " (" << run.description << ")\n";

        for (int iter = 1; iter <= NUM_ITERATIONS; ++iter) {
            collision_imminent.store(false);
            
            std::cout << "\nITERATION " << iter << " / " << NUM_ITERATIONS << " | " << run.run_id << "\n";
            std::cout << "   PWM: " << run.speed_left << "L / " << run.speed_right << "R | Time: " << run.drive_time << "s\n";
            std::cout << "ACTION REQUIRED: PHYSICALLY REPOSITION ROBOT\n";
            std::cout << "   Place Robot at -> X: " << pos.x << "mm, Y: " << pos.y << "mm\n";
            std::cout << "   Heading        -> " << pos.heading << "° (Tile ID: " << pos.id << ")\n";
            
            std::cout << "   Press ENTER when perfectly aligned to build map...";
            wait_for_enter();
            
            drv->stop();
            
            dx_l = dy_l = dx_r = dy_r = 0;
            mouse_trajectory.clear();
            lidar_mutex.lock(); lidar_trajectory.clear(); lidar_mutex.unlock();
            
            drv->startScan(false, true);
            iteration_running.store(true);
            motors_started.store(false);
            std::thread t_lidar(lidar_tracking_thread, drv);
            
            std::cout << "\nBuilding Global Map...";
            while (!map_is_built.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << " Map Locked.\n";
            
            std::cout << "CLEAR PATH. Press ENTER to start motors...";
            wait_for_enter();
            
            global_start_time = std::chrono::steady_clock::now();
            
            LidarPose initial_pose; initial_pose.timestamp = 0.0; initial_pose.valid = true;
            lidar_mutex.lock(); lidar_trajectory.push_back(initial_pose); lidar_mutex.unlock();
            
            motors_started.store(true);
            
            run_motors(run.speed_left, run.speed_right);
            
            auto start_time = global_start_time;
            int last_dx_l = 0, last_dy_l = 0, last_dx_r = 0, last_dy_r = 0;
            double prev_x = 0.0, prev_y = 0.0, prev_theta = 0.0;
            bool motor_running = true;
            
            double actual_drive_time = run.drive_time; 
            
            while (true) {
                double elapsed = chrono::duration<double>(chrono::steady_clock::now() - start_time).count();
                
                if (motor_running && (elapsed >= run.drive_time || collision_imminent.load())) { 
                    actual_drive_time = elapsed; 
                    stopMotor(); 
                    motor_running = false; 
                    if (collision_imminent.load()) std::cout << "\n\n🚨 [E-STOP SHIELD] WALL DETECTED! Coasting for " << run.coast_time << "s...\n";
                    else std::cout << "\n[COASTING]\n"; 
                }
                
                if (!motor_running && elapsed >= (actual_drive_time + run.coast_time)) break;
                
                int curr_dx_l = dx_l.load(), curr_dy_l = dy_l.load(), curr_dx_r = dx_r.load(), curr_dy_r = dy_r.load();
                
                MouseOdometry o = compute_mouse_odometry(curr_dx_l - last_dx_l, curr_dy_l - last_dy_l, curr_dx_r - last_dx_r, curr_dy_r - last_dy_r, prev_x, prev_y, prev_theta);
                o.timestamp = elapsed; 
                mouse_trajectory.push_back(o);
                
                prev_x = o.x; prev_y = o.y; prev_theta = o.theta;
                last_dx_l = curr_dx_l; last_dy_l = curr_dy_l; last_dx_r = curr_dx_r; last_dy_r = curr_dy_r;
                
                static int last_sec = -1;
                if ((int)elapsed != last_sec) {
                    std::cout << "\r[t=" << (int)elapsed << "s] M: (" << (int)prev_x << "," << (int)prev_y << ") | ";
                    lidar_mutex.lock();
                    if (!lidar_trajectory.empty() && lidar_trajectory.back().valid) std::cout << "L: (" << (int)(lidar_trajectory.back().x * 1000) << "," << (int)(lidar_trajectory.back().y * 1000) << ")";
                    lidar_mutex.unlock();
                    std::cout << "    " << std::flush;
                    last_sec = (int)elapsed;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            if (motor_running) stopMotor();
            motors_started.store(false);
            
            iteration_running.store(false);
            t_lidar.join();
            
            save_integrated_dataset(iter, run);
            
            std::cout << "\nRun " << iter << " Complete. Please prepare to pick up the robot.\n";
        }
    }
    
    std::cout << "\nALL SELECTED RUNS COMPLETE! Shutting down...\n";
    running = false;
    t_left.join(); t_right.join(); 
    
    drv->stop(); delete drv; gpioTerminate();
    return 0;
}
