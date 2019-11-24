//
// Created by selya on 28.10.2019.
//

#ifndef LINERENDERER_WU_LINE_HPP
#define LINERENDERER_WU_LINE_HPP

#include <cmath>
#include <algorithm>
#include <functional>

inline void WuDrawLine(double x0, double y0, double x1, double y1,
                const std::function<void(int x, int y, double brightess)> &plot) {
    auto ipart = [](double x) -> int {return int(std::floor(x));};
    auto round = [](double x) -> double {return std::round(x);};
    auto fpart = [](double x) -> double {return x - std::floor(x);};
    auto rfpart = [=](double x) -> double {return 1 - fpart(x);};

    const bool steep = abs(y1 - y0) > abs(x1 - x0);
    if (steep) {
        std::swap(x0,y0);
        std::swap(x1,y1);
    }
    if (x0 > x1) {
        std::swap(x0,x1);
        std::swap(y0,y1);
    }

    const double dx = x1 - x0;
    const double dy = y1 - y0;
    const double gradient = (dx == 0) ? 1 : dy/dx;

    int xpx11;
    double intery;
    {
        const double xend = round(x0);
        const double yend = y0 + gradient * (xend - x0);
        const double xgap = rfpart(x0 + 0.5);
        xpx11 = int(xend);
        const int ypx11 = ipart(yend);
        if (steep) {
            plot(ypx11,     xpx11, rfpart(yend) * xgap);
            plot(ypx11 + 1, xpx11,  fpart(yend) * xgap);
        } else {
            plot(xpx11, ypx11,    rfpart(yend) * xgap);
            plot(xpx11, ypx11 + 1, fpart(yend) * xgap);
        }
        intery = yend + gradient;
    }

    int xpx12;
    {
        const double xend = round(x1);
        const double yend = y1 + gradient * (xend - x1);
        const double xgap = rfpart(x1 + 0.5);
        xpx12 = int(xend);
        const int ypx12 = ipart(yend);
        if (steep) {
            plot(ypx12,     xpx12, rfpart(yend) * xgap);
            plot(ypx12 + 1, xpx12,  fpart(yend) * xgap);
        } else {
            plot(xpx12, ypx12,    rfpart(yend) * xgap);
            plot(xpx12, ypx12 + 1, fpart(yend) * xgap);
        }
    }

    if (steep) {
        for (int x = xpx11 + 1; x < xpx12; x++) {
            plot(ipart(intery),     x, rfpart(intery));
            plot(ipart(intery) + 1, x,  fpart(intery));
            intery += gradient;
        }
    } else {
        for (int x = xpx11 + 1; x < xpx12; x++) {
            plot(x, ipart(intery),     rfpart(intery));
            plot(x, ipart(intery) + 1,  fpart(intery));
            intery += gradient;
        }
    }
}

#endif //LINERENDERER_WU_LINE_HPP
