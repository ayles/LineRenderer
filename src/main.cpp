#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <random>
#include <functional>
#include <list>

#include "wu_line.hpp"

struct Line {
    double k;
    double A, B, C;
    double L;
    double invL;
    cv::Point2i p1, p2;

    Line(const cv::Point2i &p1_, const cv::Point2i &p2_) : p1(p1_), p2(p2_) {
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

    inline double DistanceToPoint(double x, double y) const {
        return std::abs(A * x + B * y + C) * invL;
    }

    inline double DistanceToPoint(const cv::Point2d &p) const {
        return DistanceToPoint(p.x, p.y);
    }
};

class A {
public:
    cv::Mat image;
    cv::Mat result;

    int multiplier;

    cv::Mat deviations;
    double deviations_sum;

    A(cv::Mat &image_, int width, int height)
    : image(image_.rows, image_.cols, CV_64F), result(height, width, CV_64F), deviations(image_.rows, image_.cols, CV_64F), deviations_sum(0.0) {
        if ((width & (width - 1)) || (height & (height - 1))) {
            throw std::runtime_error("width or height is not power of 2");
        }

        if ((image.cols & (image.cols - 1)) || (image.rows & (image.rows - 1))) {
            throw std::runtime_error("image width or height is not power of 2");
        }

        multiplier = width / image.cols;

        if (multiplier != height / image.rows) {
            throw std::runtime_error("width and height multipliers differ");
        }

        for (int x = 0; x < image_.cols; ++x) {
            for (int y = 0; y < image_.rows; ++y) {
                image.at<double>(y, x) = image_.at<uchar>(y, x) / 255.0;
            }
        }

        for (int x = 0; x < result.cols; ++x) {
            for (int y = 0; y < result.rows; ++y) {
                DrawPixel(x, y, 1, 1);
            }
            int a = 0;
        }
    }

    static inline double CalculateDeviation(double a, double b) {
        double r = a - b;
        return r * r;
    }

    inline double PreCalculatePixelValue(int x, int y, double color, double intensity) {
        return result.at<double>(y, x) * (1 - intensity) + color * intensity;
    }

    inline double PreCalculateDeviation(int x, int y, double color, double intensity) {
        double average = -result.at<double>(y, x) + PreCalculatePixelValue(x, y, color, intensity);
        int xl = x / multiplier * multiplier;
        int xu = xl + multiplier;
        int yl = y / multiplier * multiplier;
        int yu = yl + multiplier;
        for (int xx = xl; xx < xu; ++xx) {
            for (int yy = yl; yy < yu; ++yy) {
                average += result.at<double>(yy, xx);
            }
        }
        average /= multiplier * multiplier;
        return deviations_sum - deviations.at<double>(y / multiplier, x / multiplier) +
            CalculateDeviation(average, image.at<double>(y / multiplier, x / multiplier));
    }

    inline void DrawPixel(int x, int y, double color, double intensity) {
        result.at<double>(y, x) = PreCalculatePixelValue(x, y, color, intensity);
        deviations_sum = PreCalculateDeviation(x, y, color, intensity);
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

int main() {
    cv::Mat src = cv::imread("monro.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::resize(src, src, cv::Size(128, 128));

    int width = 1024;
    int height = 1024;

    A a(src, width, height);


    double lines_count = 200;
    std::list<Line> lines;

    for (int i = 0; i < lines_count; ++i) {
        for (int j = i + 1; j < lines_count; ++j) {
            int x = width / 2;
            int y = height / 2;
            lines.emplace_back(
                cv::Point2i(std::cos(i / lines_count * 2 * pi) * x * 0.9 + x, std::sin(i / lines_count * 2 * pi) * y * 0.9 + y),
                cv::Point2i(std::cos(j / lines_count * 2 * pi) * x * 0.9 + x, std::sin(j / lines_count * 2 * pi) * y * 0.9 + y));
        }
    }

    int lines_elapsed = 0;
    while (!lines.empty()) {
        lines_elapsed++;
        double min_deviations_sum = 1.0 / 0.0;
        std::list<Line>::iterator min_iter;
        for (auto it = lines.begin(); it != lines.end(); ++it) {
            auto tmp_deviations_sum = a.PreCalculateDeviation(*it, 0.3);
            if (tmp_deviations_sum < min_deviations_sum) {
                min_deviations_sum = tmp_deviations_sum;
                min_iter = it;
            }
        }
        a.DrawLine(*min_iter, 0.0);
        lines.erase(min_iter);
        if (lines_elapsed % 50 == 0) {
            cv::Mat m(a.result.rows, a.result.cols, CV_8UC1);
            for (int x = 0; x < m.cols; ++x) {
                for (int y = 0; y < m.rows; ++y) {
                    m.at<uchar>(y, x) = a.result.at<double>(y, x) * 255.0;
                }
            }
            cv::imwrite("img/" + std::to_string(lines_elapsed) + "_" + std::to_string(a.deviations_sum) + ".png", m);
            std::cout << lines_elapsed << std::endl;
        }
    }

    return 0;
}