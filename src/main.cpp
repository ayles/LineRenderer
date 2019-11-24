#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <random>
#include <list>
#include <limits>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include "wu_line.hpp"

class Lock {
public:
    void Wait() {
        if (locked) {
            std::unique_lock lock(m);
            cv.wait(lock, [&]() { return !locked; });
            lock.unlock();
        }
        locked = true;
    }

    void Wake() {
        {
            std::lock_guard lg(m);
            locked = false;
        }
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

struct Point {
    double x = 0;
    double y = 0;

    Point() {}
    Point(double x, double y) : x(x), y(y) {}

    Point(const Point &) = default;
    Point &operator=(const Point &) = default;
};

struct Matrix {
    int width;
    int height;
    std::vector<double> m;

    Matrix(int width, int height, double dflt = 0.0) : width(width), height(height), m(width * height, dflt) {}

    Matrix(const Matrix &) = default;
    Matrix &operator=(const Matrix &) = default;

    double *operator[](int index) {
        return m.data() + height * index;
    }

    const double *operator[](int index) const {
        return m.data() + height * index;
    }
};

struct Line {
    double k;
    double A, B, C;
    double L;
    double invL;
    Point p1, p2;

    Line(const Point &p1_, const Point &p2_) : p1(p1_), p2(p2_) {
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

class A {
public:
    int scale;
    double inv_scale;

    Matrix result;
    Matrix result_dwn;
    Matrix image_dwn;
    Matrix deviations_dwn;
    double deviations_sum;

    A(const Matrix &image_dwn, int width, int height)
        : result(width, height, 0.0)
        , result_dwn(image_dwn.width, image_dwn.height, 0.0)
        , image_dwn(image_dwn)
        , deviations_dwn(image_dwn.width, image_dwn.height)
        , deviations_sum(0.0) {
        if ((width & (width - 1)) || (height & (height - 1))) {
            throw std::runtime_error("width or height is not power of 2");
        }

        if ((image_dwn.width & (image_dwn.width - 1)) || (image_dwn.height & (image_dwn.height - 1))) {
            throw std::runtime_error("image width or height is not power of 2");
        }

        scale = width / image_dwn.width;

        if (scale != height / image_dwn.height) {
            throw std::runtime_error("width and height multipliers differ");
        }

        inv_scale = 1.0 / scale;

        for (int x = 0; x < result.width; ++x) {
            for (int y = 0; y < result.height; ++y) {
                DrawPixel(x, y, 1, 1);
            }
        }
    }

    static inline double CalculateDeviation(double goal, double current) {
        double r = goal - current;
        return r * r;
    }

    static inline double PreCalculatePixelValue(double old_color, double color, double intensity) {
        return old_color * (1 - intensity) + color * intensity;
    }

    inline double PreCalculateDeviation(int x, int y, double color, double intensity) {
        int x_dwn = x / scale;
        int y_dwn = y / scale;
        double average =
            (result_dwn[x_dwn][y_dwn] - result[x][y] + PreCalculatePixelValue(result[x][y], color, intensity)) * inv_scale * inv_scale;
        return deviations_sum - deviations_dwn[x_dwn][y_dwn] + CalculateDeviation(image_dwn[x_dwn][y_dwn], average);
    }

    inline void DrawPixel(int x, int y, double color, double intensity) {
        int x_dwn = x / scale;
        int y_dwn = y / scale;
        auto new_sum = PreCalculateDeviation(x, y, color, intensity);
        result_dwn[x_dwn][y_dwn] -= result[x][y];
        result[x][y] = PreCalculatePixelValue(result[x][y], color, intensity);
        result_dwn[x_dwn][y_dwn] += result[x][y];
        deviations_dwn[x_dwn][y_dwn] += new_sum - deviations_sum;
        deviations_sum = new_sum;
    }

    inline double PreCalculateDeviation(const Line &line, double color) {
        double temp_deviations_sum = deviations_sum;
        WuDrawLine(line.p1.x, line.p1.y, line.p2.x, line.p2.y, [&](int x, int y, double intensity) {
             temp_deviations_sum += PreCalculateDeviation(x, y, color, intensity) - deviations_sum;
        });
        return temp_deviations_sum;
    }

    inline void DrawLine(const Line &line, double color) {
        WuDrawLine(line.p1.x, line.p1.y, line.p2.x, line.p2.y, [&](int x, int y, double intensity) {
            DrawPixel(x, y, color, intensity);
        });
    }
};

constexpr const double pi = 3.141592653589793238462643383279502884;

inline Matrix CVToM(const cv::Mat &m) {
    Matrix mm(m.cols, m.rows);
    for (int x = 0; x < mm.width; ++x) {
        for (int y = 0; y < mm.height; ++y) {
            mm[x][y] = m.at<uchar>(y, x) / 255.0;
        }
    }
    return mm;
}

inline cv::Mat MToCV(const Matrix &m) {
    cv::Mat mm(m.width, m.height, CV_8UC1);
    for (int x = 0; x < m.width; ++x) {
        for (int y = 0; y < m.height; ++y) {
            mm.at<uchar>(y, x) = m[x][y] * 255.0;
        }
    }
    return mm;
}

inline std::list<Line> GenerateLinesSimple(const Point &center, double radius, int count) {
    std::list<Line> lines;
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            lines.emplace_back(
                Point(std::cos(i * 2 * pi / count) * radius + center.x, std::sin(i * 2 * pi / count) * radius + center.y),
                Point(std::cos(j * 2 * pi / count) * radius + center.x, std::sin(j * 2 * pi / count) * radius + center.y));
        }
    }
    return lines;
}

inline std::list<std::pair<double, Line>> MakeWeightedLinesList(const std::list<Line> &lines) {
    std::list<std::pair<double, Line>> lines_weighted;
    for (auto &l : lines) {
        lines_weighted.emplace_back(std::numeric_limits<double>::infinity(), l);
    }
    return lines_weighted;
}

int main() {
    cv::Mat src = cv::imread("monro.png", cv::ImreadModes::IMREAD_GRAYSCALE);
    cv::resize(src, src, cv::Size(256, 256));

    int width = 2048;
    int height = 2048;

    A a(CVToM(src), width, height);

    auto lines = MakeWeightedLinesList(GenerateLinesSimple(Point(width / 2, height / 2), width * 0.45, 200));

    const size_t threads_count = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4;

    std::cout << "Will compute in " << threads_count << " threads" << std::endl;

    std::vector<Lock> worker_threads_locks(threads_count);
    std::vector<std::pair<decltype(lines)::iterator, decltype(lines)::iterator>> worker_threads_ranges(threads_count);

    Lock main_thread_lock;
    std::mutex push_pair_mutex;

    size_t busy_threads = 0;
    decltype(lines)::iterator best_line;

    auto push_line = [&](decltype(lines)::iterator &p) {
        std::lock_guard lg(push_pair_mutex);
        if (p->first < best_line->first) {
            best_line = p;
        }
        {
            const std::lock_guard<std::mutex> lock_main(main_thread_lock.GetMutex());
            --busy_threads;
        }
        if (busy_threads == 0) {
            main_thread_lock.Wake();
        }
    };

    auto calc = [&](size_t index) {
        while (index < worker_threads_ranges.size()) {
            worker_threads_locks[index].Wait();
            if (index >= worker_threads_ranges.size()) {
                break;
            }
            auto min_iter = worker_threads_ranges[index].first;
            for (auto it = worker_threads_ranges[index].first; it != worker_threads_ranges[index].second; ++it) {
                it->first = a.PreCalculateDeviation(it->second, 0.1);
                if (it->first < min_iter->first) {
                    min_iter = it;
                }
            }
            push_line(min_iter);
        }
    };

    std::vector<std::thread> worker_threads;
    for (size_t i = 0; i < threads_count; ++i) {
        worker_threads.emplace_back(calc, i);
    }

    int lines_elapsed = 0;
    while (!lines.empty()) {
        best_line = lines.begin();
        busy_threads = threads_count;

        bool full_step = lines_elapsed % 10 == 0;

        if (full_step) {
            // Do full step
            size_t per_thread_count = lines.size() / threads_count;
            auto iter = lines.begin();
            for (size_t i = 0; i < threads_count - 1; ++i) {
                auto next_iter = iter;
                std::advance(next_iter, per_thread_count);
                worker_threads_ranges[i] = std::pair(iter, next_iter);
                iter = next_iter;
            }
            worker_threads_ranges[worker_threads_ranges.size() - 1] = std::pair(iter, lines.end());
        } else {
            // Do partial step
            size_t per_thread_count = std::min(lines.size() / threads_count, 200 / threads_count);

            auto iter = lines.begin();
            for (size_t i = 0; i < threads_count; ++i) {
                auto next_iter = iter;
                std::advance(next_iter, per_thread_count);
                worker_threads_ranges[i] = std::pair(iter, next_iter);
                iter = next_iter;
            }
        }

        lines_elapsed++;

        for (auto &lock : worker_threads_locks) {
            lock.Wake();
        }

        main_thread_lock.Wait();

        a.DrawLine(best_line->second, 0.1);
        lines.erase(best_line);
        lines.sort();

        if (lines_elapsed % 50 == 0) {
            cv::imwrite("img/" + std::to_string(lines_elapsed) + "_" + std::to_string(a.deviations_sum) + ".png", MToCV(a.result));
            std::cout << lines_elapsed << std::endl;
        }
    }

    worker_threads_ranges.clear();
    for (auto &lock : worker_threads_locks) {
        lock.Wake();
    }
    for (auto &th : worker_threads) {
        th.join();
    }

    return 0;
}