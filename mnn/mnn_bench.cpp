// Exact pp128/tg32: teacher-forced forwards, independent of generated EOS.
#include "llm/llm.hpp"
#include "jsonhpp/json.hpp"
#include <chrono>
#include <array>
#include <memory>
using json=nlohmann::json;
using Clock=std::chrono::steady_clock;
using Llm=MNN::Transformer::Llm;
int main(int argc,char**argv){
    if(argc!=3 || (std::string(argv[2])!="cpu" && std::string(argv[2])!="opencl")){
        std::cerr<<"usage: mnn_bench CONFIG cpu|opencl\n";return 2;
    }
    const bool opencl=std::string(argv[2])=="opencl";
    MNN::BackendConfig backend_config;
    auto executor=MNN::Express::Executor::newExecutor(opencl?MNN_FORWARD_OPENCL:MNN_FORWARD_CPU,backend_config,1);
    MNN::Express::ExecutorScope scope(executor);
    std::unique_ptr<Llm> llm(Llm::createLLM(argv[1]));
    if(!llm || !llm->load())return 3;
    llm->set_config("{\"async\":false}");
    const std::vector<int> prompt(128,16),token(1,16);
    json::array_t samples;samples.reserve(3);
    std::array<int,32> warmup_position_ids;
    for(int repetition=-1;repetition<3;++repetition){
        llm->reset();llm->generate_init(nullptr,"");
        if(opencl)llm->switchMode(Llm::Prefill);
        const auto pp_start=Clock::now();
        auto prompt_logits=llm->forward(prompt,true);
        if(!prompt_logits.get() || !prompt_logits->readMap<float>())return 4;
        const double pp_seconds=std::chrono::duration<double>(Clock::now()-pp_start).count();
        const int actual_prompt_tokens=llm->getContext()->all_seq_len;
        llm->reset();llm->generate_init(nullptr,"");
        if(opencl)llm->switchMode(Llm::Decode);
        const auto tg_start=Clock::now();
        int actual_decode_calls=0;
        std::array<int,32> positions_before,positions_after;
        std::array<size_t,32> kv_before,kv_after;
        for(int i=0;i<32;++i){
            positions_before[i]=llm->getContext()->all_seq_len;
            kv_before[i]=llm->getCurrentHistory();
            if(positions_before[i]!=i)return 7;
            if(repetition<0){
                auto position_ids=llm->gen_position_ids(1);
                if(!position_ids.get())return 8;
                const int* position_data=position_ids->readMap<int>();
                if(!position_data)return 8;
                warmup_position_ids[i]=position_data[0];
                if(warmup_position_ids[i]!=i)return 9;
            }
            auto logits=llm->forward(token,false);
            if(!logits.get() || !logits->readMap<float>())return 5;
            ++actual_decode_calls;
            positions_after[i]=llm->getContext()->all_seq_len;
            kv_after[i]=llm->getCurrentHistory();
            if(positions_after[i]!=i+1 || kv_after[i]!=static_cast<size_t>(i+1)){
                std::cerr<<"KV progression failure at step "<<i<<": context="<<positions_after[i]<<" history="<<kv_after[i]<<"\n";
                return 10;
            }
        }
        const double tg_seconds=std::chrono::duration<double>(Clock::now()-tg_start).count();
        const auto context=llm->getContext();
        if(actual_prompt_tokens!=128 || actual_decode_calls!=32 || context->gen_seq_len!=32 || context->all_seq_len!=32)return 6;
        if(repetition>=0)samples.push_back({{"repetition",repetition},{"actual_prompt_tokens",actual_prompt_tokens},{"actual_prompt_forward_calls",1},{"actual_decode_forward_calls",actual_decode_calls},{"actual_decode_context_generated",context->gen_seq_len},{"actual_decode_context_tokens",context->all_seq_len},{"decode_context_positions_before",positions_before},{"decode_context_positions_after",positions_after},{"kv_history_before",kv_before},{"kv_history_after",kv_after},{"pp_seconds",pp_seconds},{"decode_seconds",tg_seconds},{"pp_tok_s",128/pp_seconds},{"decode_tok_s",32/tg_seconds}});
    }
    json result={{"engine","MNN"},{"version","3.6.1"},{"backend_requested",argv[2]},{"prompt_token_id",16},{"decode_token_id",16},{"prompt_tokens",128},{"decode_forward_calls",32},{"warmup_repetitions",1},{"measured_repetitions",3},{"warmup_actual_position_ids",warmup_position_ids},{"sampling","none: teacher-forced fixed-token forward calls"},{"timing","native steady_clock around synchronous MNN forward plus output readMap"},{"samples",std::move(samples)}};
    std::cout<<"EDGE_MNN_BENCH "<<result.dump()<<std::endl;
}
