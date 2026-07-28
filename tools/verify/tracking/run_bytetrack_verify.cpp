// ByteTracker(tracker.cpp) 격리 검증. 합성 검출 시퀀스(seq.json) → track ID 덤프.
#include "visp/tracker.h"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <fstream>
#include <vector>
using namespace visp;
using json = nlohmann::json;
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <seq.json> <out.bin>\n", argv[0]); return 1; }
    json seq = json::parse(std::ifstream(argv[1]));   // [[ [x1,y1,x2,y2,score,label],... ], ...]
    ByteTracker tracker;
    std::vector<float> out;   // [K,6] frame,x1,y1,x2,y2,id
    int f = 0;
    for (auto const& frame : seq) {
        std::vector<detection> dets;
        for (auto const& d : frame)
            dets.push_back({d[0], d[1], d[2], d[3], d[4], (int)d[5]});
        auto res = tracker.track(dets, f);
        for (auto const& r : res) {
            out.push_back((float)f); out.push_back(r.det.x1); out.push_back(r.det.y1);
            out.push_back(r.det.x2); out.push_back(r.det.y2); out.push_back((float)r.id);
        }
        ++f;
    }
    FILE* fo = fopen(argv[2], "wb"); fwrite(out.data(), sizeof(float), out.size(), fo); fclose(fo);
    printf("frames=%d track rows=%zu\n", f, out.size() / 6);
    return 0;
}
