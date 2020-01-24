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
#include <functional>
#include <cfenv>

#include "utils.hpp"
#include "wu_line.hpp"

constexpr const double pi = 3.141592653589793238462643383279502884;

struct TextureNode {
    cv::Point origin; // Top left of the rectangle this node represents
    cv::Size size;   // Size of the rectangle this node represents
    bool empty;        // true if this node is a leaf and is filled

    std::unique_ptr<TextureNode> left;  // Left (or top) subdivision
    std::unique_ptr<TextureNode> right; // Right (or bottom) subdivision
};
/*
TextureNode* pack(TextureNode* node, const cv::Size& size, cv::Size& textureSize) {
    if (!node->empty) {
        // The node is filled, not gonna fit anything else here
        assert(!node->left && !node->right);
        return NULL;
    } else if (node->left && node->right) {
        // Non-leaf, try inserting to the left and then to the right
        TextureNode* retval = pack(node->left.get(), size);
        if (retval != NULL) {
            return retval;
        }
        return pack(node->right.get(), size);
    } else {
        // This is an unfilled leaf - let's see if we can fill it
        cv::Point realSize(node->size.width, node->size.height);

        // If we're along a boundary, calculate the actual size
        if (node->origin.x + node->size.width == INT_MAX) {
            realSize.x = textureSize.width - node->origin.x;
        }
        if (node->origin.y + node->size.height == INT_MAX) {
            realSize.y = textureSize.height - node->origin.y;
        }

        if (node->size.width == size.width && node->size.height == size.height) {
            // Perfect size - just pack into this node
            node->empty = false;
            return node;
        } else if (realSize.x < size.width || realSize.y < size.height) {
            // Not big enough
            return NULL;
        } else {
            // Large enough - split until we get a perfect fit
            TextureNode* left;
            TextureNode* right;

            // Determine how much space we'll have left if we split each way
            int remainX = realSize.x - size.width;
            int remainY = realSize.y - size.height;

            // Split the way that will leave the most room
            bool verticalSplit = remainX < remainY;
            if (remainX == 0 && remainY == 0) {
                // Edge case - we are are going to hit the border of
                // the texture atlas perfectly, split at the border instead
                if (node->size.width > node->size.height) {
                    verticalSplit = false;
                } else {
                    verticalSplit = true;
                }
            }

            if (verticalSplit) {
                // Split vertically (left is top)
                left = new TextureNode(node->origin, cv::Size(node->size.width, size.height);
                right = new TextureNode(cv::Point(node->origin.x, node->origin.y + size.width),
                                        cv:Size(node->size.width, node->size.height - size.height));
            } else {
                // Split horizontally
                left = new TextureNode(node->origin, cv::Size(size.width, node->size.height);
                right = new TextureNode(cv::Point(node->origin.x + size.width, node->origin.y), cv::Size_(node->size.width - size.width, node->size.height));
            }

            node->left = std::unique_ptr<TextureNode>(left);
            node->right = std::unique_ptr<TextureNode>(right);
            return pack(node->left.get(), size);
        }
    }
}

cv::Point packTexture(cv::Mat* textureBuffer, const cv::Size &  bufferSize, cv::Size& textureSize) {
    TextureNode* node = pack(textureBuffer, bufferSize);
    if (node == NULL)
    {
        this->resizeBuffer(cv::Size(textureSize.width * 2, textureSize.height * 2), textureSize);
        node = pack(textureBuffer, bufferSize);

        // Note: this assert will be hit if we try to pack a texture larger
        // than the current size of the texture
        assert(node != NULL);
    }

    assert(bufferSize.x == node->size.x);
    assert(bufferSize.y == node->size.y);

    // Copy the texture to the texture atlas' buffer
    for (int ly = 0; ly < bufferSize.height; ly++) {
        for (int lx = 0; lx < bufferSize.width; lx++) {
            int y = node->origin.y + ly;
            int x = node->origin.x + lx;
            this->buffer[y * textureSize.width + x] = textureBuffer[ly * bufferSize.width + lx];
        }
    }

    return node->origin;
}
*/
/*
    size_t width = 0;
    for (auto& m : mats) {
        m = cvt(m);
        width += m.cols;
    }

    cv::Mat mmm(mats[0].rows, width, mats[0].type());
    size_t curr_width = 0;
    for (auto& m : mats) {
        m.copyTo(mmm(cv::Rect(curr_width, 0, m.cols, m.rows)));
        curr_width += m.cols;
    }
*/

/*void resizeBuffer(const cv::Point& newSize) {
    std::vector<cv::Mat> newBuffer;
    newBuffer.resize(newSize.y*newSize.x);
    for (int y = 0; y < textureSize.y; y++) {
        for (int x = 0; x < textureSize.x; x++) {
            newBuffer[y * newSize.x + x] = buffer[y * textureSize.x + x];
        }
    }

    textureSize = newSize;
    buffer = std::move(newBuffer);
}*/
/*
void resizeBuffer(const cv::Size& newSize, cv::Size& textureSize) {
    std::vector<cv::Mat> newBuffer;
    newBuffer.resize(newSize.height * newSize.width);
    size_t curr_width = 0;
    size_t curr_height = 0;
    for (auto& m : mats) {
        mmm(cv::Rect(curr_width, curr_height, m.cols, m.rows));
        curr_width += m.cols;
        curr_height += m.rows;
    }

    textureSize = newSize;
    buffer = std::move(newBuffer);
*/
class A {
public:
    cv::Mat image;
    cv::Mat image_dwn;
    cv::Mat image_grayscale_dwn;
    cv::Mat details_dwn;
    cv::Mat result;
    cv::Mat result_dwn;
    cv::Mat deviations_dwn;

    int dwn_factor;
    double dwn_factor_inv;
    double deviations_sum;

public:
    // =============== This section can be adjusted to change behaviour ============= //
    INLINE void CalculatePotential(double& dst, double src, double distance_squared) const {
        dst += src * 1 / (std::sqrt(distance_squared) * 30 + 15);
    }

    INLINE double CalculateDetail(double potential) const {
        return potential * potential * potential * potential * potential * potential;
    }

    INLINE double CalculateDeviation(const cv::Vec3d &goal, const cv::Vec3d &current, double potential = 1.0) const {
        return lr::Color::DistanceSquared(goal, current) * CalculateDetail(potential);
    }

    INLINE double GetLineIntensity() const {
        return 0.3;
    }

    INLINE bool ShouldDeleteLineAfterDraw() const {
        return false;
    }
    ///////////////////////////////////////////////////////////////////////////////////

    void CalculateDetailsMatrix() {
        cv::Mat m;
        image_grayscale_dwn.convertTo(m, CV_64FC1);
        cv::Sobel(m, m, CV_64F, 1, 0);
        cv::Sobel(m, m, CV_64F, 0, 1);
        m = cv::abs(m);
        details_dwn = cv::Mat(m.rows, m.cols, CV_64FC1);

        double mx = 0.0;
        for (size_t r = 0; r < details_dwn.rows; ++r) {
            for (size_t c = 0; c < details_dwn.cols; ++c) {
                for (size_t rr = 0; rr < m.rows; ++rr) {
                    for (size_t cc = 0; cc < m.cols; ++cc) {
                        double ccc = double(cc) - c;
                        double rrr = double(rr) - r;
                        double distance_squared = ccc * ccc + rrr * rrr;
                        CalculatePotential(details_dwn.at<double>(r, c), m.at<double>(rr, cc), distance_squared);
                        mx = std::max(details_dwn.at<double>(r, c), mx);
                    }
                }
            }
        }

        double mx_inv = 1.0 / mx;
        details_dwn *= mx_inv;
    }

    A(const cv::Mat& image_, int dwn_width, int dwn_height, int width, int height)
        : result(cv::Size(width, height), CV_64FC3, cv::Scalar(0, 0, 0))
        , result_dwn(cv::Size(dwn_width, dwn_height), CV_64FC3, cv::Scalar(0, 0, 0))
        , deviations_dwn(cv::Size(dwn_width, dwn_height), CV_64FC1, 0.0)
        , deviations_sum(0.0)
    {
        if ((width & (width - 1)) || (height & (height - 1))) {
            throw std::runtime_error("width or height is not power of 2");
        }

        if ((dwn_width & (dwn_width - 1)) || (dwn_height & (dwn_height - 1))) {
            throw std::runtime_error("image width or height is not power of 2");
        }

        dwn_factor = width / dwn_width;

        if (dwn_factor != height / dwn_height) {
            throw std::runtime_error("width and height multipliers differ");
        }

        dwn_factor_inv = 1.0 / dwn_factor;

        cv::cvtColor(image_, image, cv::ColorConversionCodes::COLOR_BGR2RGB);
        image.convertTo(image, CV_64FC3, 1 / 255.0);
        cv::resize(image, image, cv::Size(width, height));
        cv::resize(image, image_dwn, cv::Size(dwn_width, dwn_height));
        image_dwn.convertTo(image_grayscale_dwn, CV_32FC3);
        cv::cvtColor(image_grayscale_dwn, image_grayscale_dwn, cv::ColorConversionCodes::COLOR_RGB2GRAY);
        image_grayscale_dwn.convertTo(image_grayscale_dwn, CV_64FC3);

        CalculateDetailsMatrix();

        for (int x = 0; x < result.cols; ++x) {
            for (int y = 0; y < result.rows; ++y) {
                DrawPixel(cv::Point(x, y), cv::Vec3d(1, 1, 1), 1);
            }
        }
    }

    static INLINE cv::Vec3d PreCalculatePixelValue(const cv::Vec3d &old_color, const cv::Vec3d &color, double intensity) {
        return old_color * (1 - intensity) + color * intensity;
    }

    INLINE double PreCalculateDeviation(const cv::Point& p, const cv::Vec3d &color, double intensity) {
        cv::Point p_dwn = p / dwn_factor;
        cv::Vec3d average =
            (result_dwn.at<cv::Vec3d>(p_dwn) - result.at<cv::Vec3d>(p) + PreCalculatePixelValue(result.at<cv::Vec3d>(p), color, intensity)) * dwn_factor_inv * dwn_factor_inv;
        return deviations_sum - deviations_dwn.at<double>(p_dwn) + CalculateDeviation(image_dwn.at<cv::Vec3d>(p_dwn), average, details_dwn.at<double>(p_dwn));
    }

    INLINE void DrawPixel(const cv::Point& p, const cv::Vec3d &color, double intensity) {
        cv::Point p_dwn = p / dwn_factor;
        auto new_sum = PreCalculateDeviation(p, color, intensity);
        result_dwn.at<cv::Vec3d>(p_dwn) -= result.at<cv::Vec3d>(p);
        result.at<cv::Vec3d>(p) = PreCalculatePixelValue(result.at<cv::Vec3d>(p), color, intensity);
        result_dwn.at<cv::Vec3d>(p_dwn) += result.at<cv::Vec3d>(p);
        deviations_dwn.at<double>(p_dwn) += new_sum - deviations_sum;
        deviations_sum = new_sum;
    }

    INLINE double PreCalculateDeviation(const lr::Line &line, const cv::Vec3d& color) {
        double temp_deviations_sum = deviations_sum;
        WuDrawLine(line.p1.x, line.p1.y, line.p2.x, line.p2.y, [&](int x, int y, double intensity) {
             temp_deviations_sum += PreCalculateDeviation(cv::Point(x, y), color, intensity * GetLineIntensity()) - deviations_sum;
        });
        return temp_deviations_sum;
    }

    INLINE void DrawLine(const lr::Line &line, const cv::Vec3d& color) {
        WuDrawLine(line.p1.x, line.p1.y, line.p2.x, line.p2.y, [&](int x, int y, double intensity) {
            DrawPixel(cv::Point(x, y), color, intensity * GetLineIntensity());
        });
    }

    void Generate(const std::list<lr::Line>& lines, const std::vector<cv::Vec3d>& colors, const std::function<void(const A&, size_t)>& callback) {
        struct LineExt {
            lr::Line line;
            double deviation;
            cv::Vec3d color;

            bool operator<(const LineExt& other) const {
                if (deviation != other.deviation) return deviation < other.deviation;
                return line < other.line;
            }
        };

        std::list<LineExt> lines_ext;
        for (auto& line : lines) {
            for (auto& color : colors) {
                lines_ext.emplace_back(LineExt { line, std::numeric_limits<double>::infinity(), color });
            }
        }

        const size_t threads_count = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4;

        std::cout << "Will compute in " << threads_count << " threads" << std::endl;

        std::vector<lr::Locker> worker_threads_locks(threads_count);
        std::vector<std::pair<decltype(lines_ext)::iterator, decltype(lines_ext)::iterator>> worker_threads_ranges(threads_count);

        lr::Locker main_thread_lock;
        std::mutex push_pair_mutex;

        size_t busy_threads = 0;
        decltype(lines_ext)::iterator best_line;

        auto push_line = [&](decltype(lines_ext)::iterator &p) {
            std::lock_guard lg(push_pair_mutex);
            if (p->deviation < best_line->deviation) {
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
                    it->deviation = PreCalculateDeviation(it->line, it->color);
                    if (it->deviation < min_iter->deviation) {
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
        while (!lines_ext.empty()) {
            best_line = lines_ext.begin();
            busy_threads = threads_count;

            bool full_step = lines_elapsed % 10 == 0;

            if (full_step) {
                // Do full step
                size_t per_thread_count = lines_ext.size() / threads_count;
                auto iter = lines_ext.begin();
                for (size_t i = 0; i < threads_count - 1; ++i) {
                    auto next_iter = iter;
                    std::advance(next_iter, per_thread_count);
                    worker_threads_ranges[i] = std::pair(iter, next_iter);
                    iter = next_iter;
                }
                worker_threads_ranges[worker_threads_ranges.size() - 1] = std::pair(iter, lines_ext.end());
            } else {
                // Do partial step
                size_t per_thread_count = std::min(lines_ext.size() / threads_count, 200 / threads_count);

                auto iter = lines_ext.begin();
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

            DrawLine(best_line->line, best_line->color);
            if (ShouldDeleteLineAfterDraw()) {
                lines_ext.erase(best_line);
            }
            lines_ext.sort();

            callback(*this, lines_elapsed);
        }

        worker_threads_ranges.clear();
        for (auto &lock : worker_threads_locks) {
            lock.Wake();
        }
        for (auto &th : worker_threads) {
            th.join();
        }
    }
};

inline std::list<lr::Line> GenerateLinesRandom(const cv::Point &center, double radius, int count) {
    std::list<lr::Line> lines;
    std::vector<cv::Point> points;
    double epsilon = 0;
    for (int i = 0; i < count; ++i) {
        epsilon = 0.00008 * (rand() % 101);
        points.emplace_back(std::cos(i * 2 * pi / count + epsilon) * radius + center.x, std::sin(i * 2 * pi / count + epsilon) * radius + center.y);
    }
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            lines.emplace_back(points[i], points[j]);
        }
    }
    return lines;
}

std::list<lr::Line> GenerateLinesSimple(const cv::Point &center, double radius, int count) {
    std::list<lr::Line> lines;
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            lines.emplace_back(
                cv::Point(std::cos(i * 2 * pi / count) * radius + center.x,
                      std::sin(i * 2 * pi / count) * radius + center.y),
                cv::Point(std::cos(j * 2 * pi / count) * radius + center.x,
                      std::sin(j * 2 * pi / count) * radius + center.y));
        }
    }
    return lines;
}
/*
cv::Mat combine(std::vector<cv::Mat> mats) {
    auto cvt = [](const cv::Mat& m) {
        cv::Mat r;
        // Now works only with grayscale images, but we have CV_64FC3 images - RGB. Should convert all images to RGB
        // Use cv::cvtColor to convert RGB to grayscale and m.depth(), m.channels() to check is it should be converted or its already ok
        m.convertTo(r, CV_64FC1);
        // Do not resize and just pack it as is
        //cv::resize(r, r, cv::Size(256, 256), 0, 0, cv::InterpolationFlags::INTER_NEAREST);
        return r;
    };

    size_t width = 0;
    for (auto& m : mats) {
        m = cvt(m);
        width += m.cols;
    }

    cv::Mat mmm(mats[0].rows, width, mats[0].type());
    size_t curr_width = 0;
    for (auto& m : mats) {
        m.copyTo(mmm(cv::Rect(curr_width, 0, m.cols, m.rows)));
        curr_width += m.cols;
    }

    size_t width = 0;
    for (auto& m : mats) {
        m = cvt(m);
        m = packTexture(m);
    }


    return mmm;
}
*/
int main() {
    cv::Mat src = cv::imread("mona2.png", cv::ImreadModes::IMREAD_COLOR);

    int dwn_width = 128;
    int dwn_height = 128;

    int width = 1024;
    int height = 1024;

    A a(src, dwn_width, dwn_height, width, height);

    auto lines = GenerateLinesRandom(cv::Point(width / 2, height / 2), width * 0.45, 200);
    auto colors = std::vector<cv::Vec3d> { lr::Color::FromWeb("ffffff"), lr::Color::FromWeb("000000"), lr::Color::FromWeb("EFA636"), lr::Color::FromWeb("6A2C15"), lr::Color::FromWeb("798032") //"f6beb6", "d9ac75", "cd5459", "3d5e99", "7c902d",
                                 //"ff2c36", "ffc5bd", "ef972f", "3066ef", "3bb70b"
    };

    cv::namedWindow("w");

    a.Generate(lines, colors, [](const A& a, size_t n) {
        if (n % 5 == 0) {
            cv::Mat m;
            a.result.convertTo(m, CV_8UC3, 255.0);
            cv::cvtColor(m, m, cv::ColorConversionCodes::COLOR_RGB2BGR);
            if (n % 50 == 0) {
                cv::imwrite("img/" + std::to_string(n) + "_" + std::to_string(a.deviations_sum) + ".png", m);
                std::cout << n << std::endl;
            }

            // Here 'a' contains many matrices of different types (RGB or grayscale double floating point)
            // a.details_dwn - details float matrix downscaled
            // a.deviations_dwn - deviations float matrix downscaled
            // a.result and a.result_downscaled rgb matrices and so on
            // combine should combine passed matrices and return one big matrix containing all them packed
            // before combining they should all be converted from grayscale to rgb (if needed)
            //cv::imshow("w", combine({ a.details_dwn, a.deviations_dwn }));
            // this line is for realtime drawing process debug
            // cv::imshow("w", a.result);
            //cv::waitKey(1);
        }
    });

    return 0;
}