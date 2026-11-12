#pragma once
#include "ggml-backend.h"
#include <array>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace umm::profiling {
// An address-independent execution contract. Construction belongs to the structure
// pass. Checking uses preallocated storage: no JSON, SHA or pointer-ID map rebuild.
class graph_contract {
    struct tensor_record {
        ggml_type type; ggml_op op; int flags;
        std::array<int64_t,GGML_MAX_DIMS> ne;
        std::array<size_t,GGML_MAX_DIMS> nb;
        std::array<int32_t,GGML_MAX_OP_PARAMS/sizeof(int32_t)> params;
        std::array<int,GGML_MAX_SRC> src;
        int view=-1,parent=-1,slot=-1;
        size_t offset=0;
        std::string name,buffer;
        int usage=-1;
    };
    int n_nodes;
    bool scheduler;
    std::vector<tensor_record> records;
    std::vector<ggml_backend_dev_t> placement;
    std::vector<ggml_tensor*> current,unique;
    static const char * buffer_name(const ggml_tensor * t){return t->buffer?ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer)):"";}
    static int usage(const ggml_tensor * t){return t->buffer?int(ggml_backend_buffer_get_usage(t->buffer)):-1;}
    static ggml_backend_dev_t device(ggml_backend_t backend,ggml_backend_sched_t sched,ggml_tensor * t){auto target=backend?backend:ggml_backend_sched_get_tensor_backend(sched,t);return target?ggml_backend_get_device(target):nullptr;}
    [[noreturn]] static void mismatch(){throw std::runtime_error("Graph differs from frozen structure contract; collect structure again");}
public:
    graph_contract(ggml_backend_t backend,ggml_backend_sched_t sched,ggml_cgraph * graph):n_nodes(ggml_graph_n_nodes(graph)),scheduler(sched!=nullptr){
        std::unordered_map<const ggml_tensor*,int> ids;
        struct reference{int parent,slot;};std::vector<reference> references;
        auto add=[&](ggml_tensor * t,int parent,int slot){if(t&&!ids.count(t)){ids[t]=int(current.size());current.push_back(t);references.push_back({parent,slot});}};
        for(int i=0;i<n_nodes;++i){auto t=ggml_graph_node(graph,i);if(ids.count(t))throw std::runtime_error("Duplicate graph node");add(t,-1,-1);placement.push_back(device(backend,sched,t));}
        for(size_t i=0;i<current.size();++i){auto t=current[i];add(t->view_src,int(i),-1);for(int k=0;k<GGML_MAX_SRC;++k)add(t->src[k],int(i),k);}
        for(size_t i=0;i<current.size();++i){auto t=current[i];tensor_record r{};
            r.type=t->type;r.op=t->op;r.flags=t->flags;r.offset=t->view_offs;r.name=t->name;r.buffer=buffer_name(t);r.usage=usage(t);
            std::memcpy(r.ne.data(),t->ne,sizeof(t->ne));std::memcpy(r.nb.data(),t->nb,sizeof(t->nb));std::memcpy(r.params.data(),t->op_params,GGML_MAX_OP_PARAMS);
            for(int k=0;k<GGML_MAX_SRC;++k)r.src[k]=t->src[k]?ids.at(t->src[k]):-1;
            r.view=t->view_src?ids.at(t->view_src):-1;r.parent=references[i].parent;r.slot=references[i].slot;records.push_back(std::move(r));
        }
        // No model/graph pointers survive from the structure pass.
        std::fill(current.begin(),current.end(),nullptr);
        size_t capacity=1;while(capacity<current.size()*2)capacity*=2;unique.resize(capacity,nullptr);
    }
    void check(ggml_backend_t backend,ggml_backend_sched_t sched,ggml_cgraph * graph){
        if((sched!=nullptr)!=scheduler || ggml_graph_n_nodes(graph)!=n_nodes)mismatch();
        for(int i=0;i<n_nodes;++i)current[i]=ggml_graph_node(graph,i);
        for(size_t i=n_nodes;i<records.size();++i){const auto & r=records[i];auto parent=current[r.parent];if(!parent)mismatch();current[i]=r.slot<0?parent->view_src:parent->src[r.slot];}
        std::fill(unique.begin(),unique.end(),nullptr);
        for(auto t:current){if(!t)mismatch();size_t h=reinterpret_cast<uintptr_t>(t)>>4;h^=h>>17;h*=size_t(0x9e3779b97f4a7c15ULL);size_t slot=h&(unique.size()-1);
            while(unique[slot] && unique[slot]!=t)slot=(slot+1)&(unique.size()-1);
            if(unique[slot])mismatch();unique[slot]=t;
        }
        for(size_t i=0;i<records.size();++i){auto t=current[i];const auto & r=records[i];
            if(!t || t->type!=r.type || t->op!=r.op || t->flags!=r.flags || t->view_offs!=r.offset || r.name!=t->name ||
                std::memcmp(r.ne.data(),t->ne,sizeof(t->ne)) || std::memcmp(r.nb.data(),t->nb,sizeof(t->nb)) || std::memcmp(r.params.data(),t->op_params,GGML_MAX_OP_PARAMS) ||
                r.buffer!=buffer_name(t) || r.usage!=usage(t))mismatch();
            if(t->view_src!=(r.view<0?nullptr:current[r.view]))mismatch();
            for(int k=0;k<GGML_MAX_SRC;++k)if(t->src[k]!=(r.src[k]<0?nullptr:current[r.src[k]]))mismatch();
            if(i<size_t(n_nodes) && device(backend,sched,t)!=placement[i])mismatch();
        }
    }
};
}
