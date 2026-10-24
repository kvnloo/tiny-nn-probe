// Native Android MNN transport. Raw inputs bypass all model chat templates.
#include "llm/llm.hpp"
#include "httplib.h"
#include "jsonhpp/json.hpp"
#include <cstdlib>
#include <ctime>
#include <memory>
#include <mutex>
using json=nlohmann::json;
using Llm=MNN::Transformer::Llm;
static json infer(Llm* llm,const std::vector<int>& tokens,int limit){
    llm->reset();std::srand(1729);std::ostringstream answer;
    llm->response(tokens,&answer,"",limit);auto c=llm->getContext();
    return {{"content",answer.str()},{"prompt_tokens",c->prompt_len},{"generated_tokens",c->gen_seq_len},{"prefill_us",c->prefill_us},{"decode_us",c->decode_us},{"sample_us",c->sample_us},{"status",static_cast<int>(c->status)},{"output_tokens",c->output_tokens}};
}
int main(int argc,char**argv){
    if(argc<3){std::cerr<<"usage: edge_mnn CONFIG raw|serve [PORT] [MODEL_ID]\n";return 2;}
    std::unique_ptr<Llm> llm(Llm::createLLM(argv[1]));if(!llm || !llm->load())return 3;
    if(std::string(argv[2])=="serve"){
        const int port=argc>3?std::atoi(argv[3]):9090;const std::string model=argc>4?argv[4]:"mnn";
        auto configured=json::parse(llm->dump_config(),nullptr,false);
        configured["jinja"]["context"]["enable_thinking"]=false;
        llm->set_config(configured.dump());
        httplib::Server server;std::mutex mutex;
        server.Get("/health",[&](const httplib::Request&,httplib::Response& res){res.set_content(json({{"status","ok"},{"engine","MNN"},{"engine_version","3.6.1"},{"model",model},{"model_id",model},{"backend_type",configured.value("backend_type",std::string("unknown"))},{"thread_num",configured.value("thread_num",0)},{"backend_evidence","loaded configuration; not operator-placement proof"}}).dump(),"application/json");});
        server.Get("/v1/models",[&](const httplib::Request&,httplib::Response& res){res.set_content(json({{"object","list"},{"data",json::array({{{"id",model},{"object","model"},{"owned_by","local"}}})}}).dump(),"application/json");});
        server.Post("/v1/chat/completions",[&](const httplib::Request& req,httplib::Response& res){
            auto j=json::parse(req.body,nullptr,false);
            if(j.is_discarded() || !j.is_object() || !j.contains("messages") || !j["messages"].is_array() || (j.contains("max_tokens") && !j["max_tokens"].is_number_integer())){res.status=400;res.set_content("{\"error\":\"invalid messages or max_tokens\"}","application/json");return;}
            int limit=j.value("max_tokens",8);if(limit<1){res.status=400;res.set_content("{\"error\":\"max_tokens must be positive\"}","application/json");return;}
            MNN::Transformer::ChatMessages messages;
            for(const auto& message:j["messages"]){
                if(!message.is_object() || !message.contains("role") || !message["role"].is_string() || !message.contains("content") || !message["content"].is_string()){res.status=400;res.set_content("{\"error\":\"text role/content required\"}","application/json");return;}
                messages.emplace_back(message["role"].get<std::string>(),message["content"].get<std::string>());
            }
            if(j.contains("temperature") && (!j["temperature"].is_number() || j["temperature"].get<double>()!=0)){res.status=400;res.set_content("{\"error\":\"this deterministic endpoint supports temperature 0\"}","application/json");return;}
            std::lock_guard<std::mutex> guard(mutex);auto prompt=llm->apply_chat_template(messages);auto result=infer(llm.get(),llm->tokenizer_encode(prompt),limit);
            if(result["status"].get<int>()>2 || result["status"].get<int>()<0){res.status=500;res.set_content(json({{"error","MNN generation failed"},{"engine_result",result}}).dump(),"application/json");return;}
            const auto now=std::time(nullptr);const std::string id="chatcmpl-"+std::to_string(now);
            const std::string finish=result["generated_tokens"].get<int>()>=limit?"length":"stop";
            if(j.value("stream",false)){
                json chunk={{"id",id},{"object","chat.completion.chunk"},{"created",now},{"model",model},{"choices",json::array({{{"index",0},{"delta",{{"role","assistant"},{"content",result["content"]}}},{"finish_reason",nullptr}}})}};
                json end={{"id",id},{"object","chat.completion.chunk"},{"created",now},{"model",model},{"choices",json::array({{{"index",0},{"delta",json::object()},{"finish_reason",finish}}})}};
                res.set_content("data: "+chunk.dump()+"\n\ndata: "+end.dump()+"\n\ndata: [DONE]\n\n","text/event-stream");
            }else{
                json response={{"id",id},{"object","chat.completion"},{"created",now},{"model",model},{"choices",json::array({{{"index",0},{"message",{{"role","assistant"},{"content",result["content"]}}},{"finish_reason",finish}}})},{"mnn_timings",result}};
                // Optional OpenAI usage is omitted: engine-native counters stay explicitly named.
                res.set_content(response.dump(),"application/json");
            }
        });
        return server.listen("127.0.0.1",port)?0:5;
    }
    if(std::string(argv[2])!="raw")return 2;
    std::string line;while(std::getline(std::cin,line)){
        auto j=json::parse(line,nullptr,false);if(j.is_discarded() || !j.contains("prompt"))return 4;
        auto out=infer(llm.get(),llm->tokenizer_encode(j["prompt"].get<std::string>()),8);
        std::cout<<"EDGE_MNN_RESULT "<<out.dump()<<std::endl;
    }
}
