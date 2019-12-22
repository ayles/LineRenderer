//
// Created by selya on 19.12.2019.
//

#ifndef LINERENDERER_UTILS_HPP
#define LINERENDERER_UTILS_HPP

#include <mutex>
#include <condition_variable>

#include <opencv2/core/core.hpp>

#ifdef __GNUC__
#define INLINE inline
#else
#define INLINE inline
#endif

namespace lr {

class Locker {
public:
    void Wait() {
        std::unique_lock lock(m);
        if (locked) {
            cv.wait(lock, [&]() { return !locked; });
        }
        locked = true;
    }

    void Wake() {
        std::unique_lock lock(m);
        locked = false;
        lock.unlock();
        cv.notify_all();
    }

    std::mutex &GetMutex() {
        return m;
    }

private:
    std::mutex m;
    std::condition_variable cv;
    bool locked = true;
};

struct Color {
    Color() = delete;

    static INLINE cv::Vec3d FromWeb(const char* c) {
        std::string s(c);
        auto color = s.find('#') != std::string::npos ? std::string(s.c_str() + 1) : s;
        int clr = std::stoi(color, nullptr, 16);
        return cv::Vec3d((clr / 256 / 256) % 256, (clr / 256) % 256, clr % 256) / 255.0;
    }


    static INLINE double DistanceSquared(const cv::Vec3d& c1, const cv::Vec3d& c2) {
        auto diff = c2 - c1;
        return diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
    }


    static INLINE double Distance(const cv::Vec3b& c1, const cv::Vec3b& c2) {
        return std::sqrt(DistanceSquared(c1, c2));
    }
};

struct Line {
    double k;
    double A, B, C;
    double L;
    double invL;
    cv::Point2d p1, p2;

    Line(const cv::Point2d &p1_, const cv::Point2d &p2_) : p1(p1_), p2(p2_) {
        if (std::abs(p2.x - p1.x) > std::abs(p2.y - p2.y)) {
            if (p1.x > p2.x) {
                std::swap(p1, p2);
            }
        } else {
            if (p1.y > p2.y) {
                std::swap(p1, p2);
            }
        }

        A = p1.y - p2.y;
        B = p2.x - p1.x;
        C = p1.x * p2.y - p2.x * p1.y;
        k = - A / B;
        L = std::sqrt(A * A + B * B);
        invL = 1.0 / L;
    }

    bool operator<(const Line &other) const {
        if (p1.x != other.p1.x) return p1.x < other.p1.x;
        if (p2.x != other.p2.x) return p2.x < other.p2.x;
        if (p1.y != other.p1.y) return p1.y < other.p1.y;
        return p2.y < other.p2.y;
    }
};

}

#endif //LINERENDERER_UTILS_HPP
