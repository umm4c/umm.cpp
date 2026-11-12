#pragma once
extern "C" {
#include "sha256.h"
}
#include <chrono>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
namespace umm::profiling::detail {
using clock = std::chrono::steady_clock;
inline double elapsed(clock::time_point start) { return std::chrono::duration<double,std::milli>(clock::now()-start).count(); }
inline std::string hash(const std::string & value) {
    sha256_t state;sha256_init(&state);sha256_update(&state,reinterpret_cast<const unsigned char*>(value.data()),value.size());
    unsigned char bytes[32];sha256_final(&state,bytes);std::ostringstream out;
    for(auto byte:bytes)out<<std::hex<<std::setw(2)<<std::setfill('0')<<int(byte);
    return out.str();
}
inline std::string base64(const std::vector<uint8_t> & bytes) {
    constexpr char alphabet[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";std::string out;
    for(size_t i=0;i<bytes.size();i+=3){uint32_t word=uint32_t(bytes[i])<<16;if(i+1<bytes.size())word|=uint32_t(bytes[i+1])<<8;if(i+2<bytes.size())word|=bytes[i+2];
        out+=alphabet[(word>>18)&63];out+=alphabet[(word>>12)&63];out+=i+1<bytes.size()?alphabet[(word>>6)&63]:'=';out+=i+2<bytes.size()?alphabet[word&63]:'=';}
    return out;
}
}
