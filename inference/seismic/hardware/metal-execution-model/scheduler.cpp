// Discrete-event SIMD scheduler. No Metal API, source compiler, measurements,
// candidate identity, or fitted candidate correction is available here.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>

extern "C" {
struct Instruction { int32_t kind,dst,a,b,c; uint32_t bytes,lanes,flags; };
struct Phase { int32_t offset,count,iterations,live; uint64_t footprint; };
struct Launch { int32_t threads,group,shared,registers; int32_t cores,partitions,max_waves,register_file; };
struct Result { double ns; uint64_t issued,spill_loads,spill_stores,barriers; int32_t resident_groups; };
}

// Parameter indices are shared with machine.py. All times are nanoseconds.
enum Parameter { DISPATCH, WARP_ISSUE, FRONT_ISSUE, FP_ISSUE, FP_LATENCY,
 INT_ISSUE, INT_LATENCY, MUL_ISSUE, MUL_LATENCY, WIDE_ISSUE,WIDE_LATENCY,
 SFU_ISSUE,SFU_LATENCY, MEMORY_ISSUE,L1_LATENCY,L2_LATENCY,DRAM_LATENCY,
 L1_BW,L2_BW,DRAM_BW,L1_SIZE,L2_SIZE,SHARED_ISSUE,SHARED_LATENCY,
 BARRIER_LATENCY,SHUFFLE_ISSUE,SHUFFLE_LATENCY,MMA_ISSUE,MMA_LATENCY,
 ATOMIC_ISSUE,ATOMIC_LATENCY,USABLE_REGS,LOOP_LATENCY,CONVERT_ISSUE,CONVERT_LATENCY,
 BRANCH_LATENCY,CALL_LATENCY,CMUL_ISSUE,CMUL_LATENCY,CMAD_ISSUE,CMAD_LATENCY,RANDOM_SECTOR_BYTES,FENCE_LATENCY };
enum Kind { FMA, ALU, MUL, WIDE, SFU, LOAD, STORE, SHLOAD, SHSTORE, CONTROL,
 BARRIER, SHUFFLE, REDUCE, MMA, ATOMIC, CAS, NOP, CONVERT, BRANCH,CALL,CMUL,CMAD };

struct Wave { int group,partition,phase=0,pc=0,iteration=0; bool blocked=false,done=false,spill_prepared=false;
 double ready=0; uint64_t generation=0; std::vector<double> values; };
struct Group { int alive=0,arrived=0; double arrival=0; };
struct Event { double time; uint64_t sequence; int wave; uint64_t generation;
 bool operator<(const Event& b)const{return time>b.time || (time==b.time && sequence>b.sequence);} };

extern "C" Result simulate(const Instruction* code,const Phase* phases,int phase_count,
 const Launch* launch,const double* p) {
 Result out{};
 if(launch->threads==0 || phase_count==0){return out;}
 const int groups=(launch->threads+launch->group-1)/launch->group;
 const int active_cores=std::min(groups,launch->cores);
 const int core_groups=(groups+active_cores-1)/active_cores;
 const int waves_per_group=(std::min(launch->group,launch->threads)+31)/32;
 const int register_reservation=std::max(1,std::min(launch->registers,int(p[USABLE_REGS])));
 int residency=std::max(1,launch->max_waves/waves_per_group);
 if(launch->shared)residency=std::min(residency,std::max(1,32768/launch->shared));
 residency=std::min(residency,std::max(1,launch->register_file/(register_reservation*launch->group)));
 residency=std::min(residency,core_groups);out.resident_groups=residency;
 std::vector<Wave> waves;waves.reserve(core_groups*waves_per_group);std::vector<Group> cohorts(core_groups);
 std::vector<double> partitions(launch->partitions,0);
 std::array<double,10> ports{};
 double front=0,finish=0;int admitted=0,completed=0;
 std::priority_queue<Event> events;uint64_t sequence=0;
 auto latency=[&](const Instruction& i,uint64_t footprint)->double{
   if(i.flags==2)footprint=64; // fixed helper constant table, distinct from input
   switch(i.kind){
   case FMA:return p[FP_LATENCY];case ALU:return p[INT_LATENCY];case MUL:return p[MUL_LATENCY];
   case WIDE:return p[WIDE_LATENCY];case SFU:return p[SFU_LATENCY];
   case LOAD:case STORE:return footprint<=p[L1_SIZE]?p[L1_LATENCY]:(footprint<=p[L2_SIZE]?p[L2_LATENCY]:p[DRAM_LATENCY]);
   case SHLOAD:case SHSTORE:return p[SHARED_LATENCY];
   case SHUFFLE:case REDUCE:return p[SHUFFLE_LATENCY];case MMA:return p[MMA_LATENCY];
   case ATOMIC:case CAS:return p[ATOMIC_LATENCY];case CONTROL:return p[LOOP_LATENCY];
   case CONVERT:return p[CONVERT_LATENCY];default:return 0;
   case BRANCH:return p[BRANCH_LATENCY];case CALL:return p[CALL_LATENCY];
   case CMUL:return p[CMUL_LATENCY];case CMAD:return p[CMAD_LATENCY];
   }
 };
 auto port=[&](const Instruction& i)->int{
   switch(i.kind){case FMA:return 0;case ALU:case CONTROL:return 1;case MUL:return 2;case WIDE:return 3;
   case SFU:case CONVERT:return 4;case LOAD:case STORE:return 5;case SHLOAD:case SHSTORE:return 6;
   case SHUFFLE:case REDUCE:return 7;case MMA:return 8;case ATOMIC:case CAS:return 9;case CMUL:case CMAD:return 2;default:return 1;}
 };
 auto issue=[&](const Instruction& i,uint64_t footprint)->double{
   if(i.flags==2)footprint=64;
   switch(i.kind){case FMA:return p[FP_ISSUE];case ALU:case CONTROL:return p[INT_ISSUE];
   case MUL:return p[MUL_ISSUE];case WIDE:return p[WIDE_ISSUE];case SFU:return p[SFU_ISSUE];
   case LOAD:case STORE:{double bw=footprint<=p[L1_SIZE]?p[L1_BW]:(footprint<=p[L2_SIZE]?p[L2_BW]:p[DRAM_BW]/active_cores);
     double bytes=i.flags==1 && footprint>p[L1_SIZE]?std::max(double(i.bytes),p[RANDOM_SECTOR_BYTES]):i.bytes;
     return std::max(p[MEMORY_ISSUE],bytes*std::max(1u,i.lanes)/bw);}
   case SHLOAD:case SHSTORE:return p[SHARED_ISSUE]*std::max(1.0,i.bytes/4.0);case SHUFFLE:case REDUCE:return p[SHUFFLE_ISSUE];
   case MMA:return p[MMA_ISSUE];case ATOMIC:case CAS:return p[ATOMIC_ISSUE];
   case CONVERT:return p[CONVERT_ISSUE];case CMUL:return p[CMUL_ISSUE];case CMAD:return p[CMAD_ISSUE];
   case BRANCH:case CALL:return p[INT_ISSUE];default:return 0;}
 };
 auto enqueue=[&](int w,double time){auto& v=waves[w];v.generation++;events.push({time,sequence++,w,v.generation});};
 auto admit=[&](double time){
   int g=admitted++;cohorts[g].alive=waves_per_group;
   for(int j=0;j<waves_per_group;j++){
     Wave w;w.group=g;w.partition=(g*waves_per_group+j)%launch->partitions;w.ready=time;
     w.values.resize(std::max(launch->registers+8,512),time);waves.push_back(std::move(w));enqueue(int(waves.size())-1,time);
   }
 };
 for(int i=0;i<residency;i++)admit(0);
 while(!events.empty()){
   Event e=events.top();events.pop();auto& w=waves[e.wave];
   if(w.done||w.blocked||w.generation!=e.generation)continue;
   const auto& ph=phases[w.phase];const auto& ins=code[ph.offset+w.pc];
   double ready=w.ready;
   for(int r:{ins.a,ins.b,ins.c})if(r>=0){ready=std::max(ready,w.values[r]);}
   ready=std::max(ready,partitions[w.partition]);ready=std::max(ready,front);
   ready=std::max(ready,ports[port(ins)]);
   if(ready>e.time && !events.empty() && events.top().time<ready){enqueue(e.wave,ready);continue;}
   // Spill accesses are real resource operations introduced by the explicit
   // register realization. Their count follows allocation, never kernel name.
   int spills=0;for(int r:{ins.a,ins.b,ins.c})if(r>=int(p[USABLE_REGS]))spills++;
   if(ins.a>=int(p[USABLE_REGS]) && ins.a==ins.b)spills--;
   if(ins.c>=int(p[USABLE_REGS]) && (ins.c==ins.a||ins.c==ins.b))spills--;
   if(spills && !w.spill_prepared){ready=std::max(ready,ports[5]);ports[5]=ready+spills*std::max(p[MEMORY_ISSUE],128.0/p[L1_BW]);w.ready=ready+p[L1_LATENCY];out.spill_loads+=spills;w.spill_prepared=true;enqueue(e.wave,w.ready);continue;}
   w.spill_prepared=false;
   const double end=ready+latency(ins,ph.footprint);
   ports[port(ins)]=ready+issue(ins,ph.footprint);
   front=ready+p[FRONT_ISSUE];partitions[w.partition]=ready+p[FRONT_ISSUE]*launch->partitions;
   w.ready=ready+p[WARP_ISSUE];
   if(ins.kind==CONTROL || ins.kind==BRANCH || ins.kind==CALL)w.ready=std::max(w.ready,end);
   if(ins.dst>=0){w.values[ins.dst]=end;if(ins.dst>=int(p[USABLE_REGS])){ports[5]=std::max(ports[5],end)+std::max(p[MEMORY_ISSUE],128.0/p[L1_BW]);w.values[ins.dst]=ports[5];out.spill_stores++;}}
   finish=std::max(finish,end);out.issued++;
   const bool barrier=ins.kind==BARRIER;
   if(barrier){w.blocked=true;auto& g=cohorts[w.group];g.arrived++;g.arrival=std::max(g.arrival,end);out.barriers++;}
   w.pc++;
   if(w.pc==ph.count){w.pc=0;w.iteration++;if(w.iteration==ph.iterations){w.iteration=0;w.phase++;}}
   if(w.phase==phase_count){
     double done=end;for(double v:w.values)done=std::max(done,v);finish=std::max(finish,done);
     w.done=true;auto& g=cohorts[w.group];g.alive--;
     if(g.alive==0){completed++;if(admitted<core_groups)admit(done);}
   }else if(!barrier)enqueue(e.wave,w.ready);
   if(barrier){auto& g=cohorts[w.group];if(g.arrived==g.alive){
     const double release=g.arrival+p[waves_per_group==1?FENCE_LATENCY:BARRIER_LATENCY];g.arrived=0;g.arrival=0;
     for(int j=0;j<int(waves.size());j++)if(waves[j].group==w.group&&waves[j].blocked){waves[j].blocked=false;waves[j].ready=release;enqueue(j,release);}
   }}
 }
 out.ns=finish+p[DISPATCH];return out;
}

// Explicit SIMT compare/exchange retry execution. A successful lane waits at
// reconvergence while failed lanes retry using the value returned by CAS.
// Atomic service is ordered per address and cache line. No CAS timings enter.
extern "C" struct AtomicResult { double ns; uint64_t attempts,successes,rounds; };
struct AtomicWave { int core,iteration=0; std::array<uint32_t,32> expected{};
 uint32_t active=0xffffffffu; double ready=0; };
extern "C" AtomicResult simulate_cas(int threads,int destinations,int iterations,int cores,
 double helper_latency,double helper_service,double atomic_latency,double address_service,
 double line_service,double global_service,double fail_address,double fail_line,double fail_global,double load_latency) {
 AtomicResult result{};int count=(threads+31)/32;
 std::vector<AtomicWave> waves(count);std::vector<uint32_t> values(destinations,0);
 std::vector<double> addresses(destinations,0),lines((destinations+31)/32,0),core_ports(cores,0);
 std::priority_queue<Event> queue;double global=0;uint64_t seq=0;
 for(int w=0;w<count;w++){waves[w].core=w%cores;queue.push({load_latency,seq++,w,0});}
 while(!queue.empty()){
   auto e=queue.top();queue.pop();auto& w=waves[e.wave];
   double start=std::max(e.time,core_ports[w.core]);
   if(!queue.empty() && queue.top().time<start){queue.push({start,seq++,e.wave,0});continue;}
   core_ports[w.core]=start+helper_service;
   double arrival=start+helper_latency,done=arrival;
   // Each wave's memory requests are injected as a batch, with per-address
   // ordered service. Inter-wave execution is ordered by helper completion.
   for(int lane=0;lane<32;lane++)if(w.active&(1u<<lane)){
     int address=(e.wave*32+lane)%destinations,line=address/32;
     double at=std::max({arrival,addresses[address],lines[line],global});
     bool success=w.expected[lane]==values[address];
     addresses[address]=at+(success?address_service:fail_address);lines[line]=at+(success?line_service:fail_line);global=at+(success?global_service:fail_global);
     result.attempts++;
     if(success){values[address]++;w.active&=~(1u<<lane);result.successes++;}
     else w.expected[lane]=values[address];
     done=std::max(done,at+atomic_latency);
   }
   result.rounds++;result.ns=std::max(result.ns,done);
   if(!w.active){
     w.iteration++;if(w.iteration==iterations)continue;
     w.active=0xffffffffu;
     for(int lane=0;lane<32;lane++)w.expected[lane]=values[(e.wave*32+lane)%destinations];
     done+=load_latency;
   }
   queue.push({done,seq++,e.wave,0});
 }
 return result;
}
