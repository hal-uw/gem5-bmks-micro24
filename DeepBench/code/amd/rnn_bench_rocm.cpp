#include <iomanip>
#include <memory>
#include <chrono>
#include <vector>
#include <thread>
#include <tuple>
#include <sched.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "tensor.h"
#include "miopen_helper.h"
#include "rnn_problems.h"

#include <gem5/m5ops.h>

//#define NOSIM

template <typename T>
class miopenRNN
{
public:
    MIOpenHandle miopen_handle_;
private:
    RNNDescriptor rnnDesc_;

    int sequenceLen_;

    TensorDescriptorArray<T> xDescArray_;
    TensorDescriptorArray<T> yDescArray_;
    TensorDescriptor<T> hxDesc_;
    TensorDescriptor<T> hyDesc_;
    TensorDescriptor<T> cxDesc_;
    TensorDescriptor<T> cyDesc_;
    TensorDescriptor<T> wDesc_;

    TensorDescriptorArray<T> dxDescArray_;
    TensorDescriptorArray<T> dyDescArray_;
    TensorDescriptor<T> dhxDesc_;
    TensorDescriptor<T> dhyDesc_;
    TensorDescriptor<T> dcxDesc_;
    TensorDescriptor<T> dcyDesc_;

    size_t weight_size_byte_;
    size_t workspace_size_byte_;
    size_t trainspace_size_byte_;

    Tensor<T> weights_;
    Tensor<float> workspace_;
    Tensor<float> trainspace_;

public:
    miopenRNN(int hidden_size, int batch_size, int time_steps, const std::string& rnn_type,
              uint64_t deadline) :
        miopen_handle_(),
        sequenceLen_(time_steps),
        xDescArray_ ({batch_size, hidden_size}, {hidden_size, 1}, time_steps),
        yDescArray_ ({batch_size, hidden_size}, {hidden_size, 1}, time_steps),
        dxDescArray_({batch_size, hidden_size}, {hidden_size, 1}, time_steps),
        dyDescArray_({batch_size, hidden_size}, {hidden_size, 1}, time_steps),
        hxDesc_ ({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        hyDesc_ ({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        dhxDesc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        dhyDesc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        cxDesc_ ({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        cyDesc_ ({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        dcxDesc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        dcyDesc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1})
    {
        miopenRNNMode_t rnn_mode;
        if( rnn_type == "vanilla")
            rnn_mode = miopenRNNRELU;
        else if( rnn_type == "gru")
            rnn_mode = miopenGRU;
        else if( rnn_type == "lstm")
            rnn_mode = miopenLSTM;
        else
            throw std::runtime_error("Unknow rnn mode in miopenRNN");

        miopenDataType_t data_type;
        if(std::is_same<T,float>::value)
            data_type = miopenFloat;
        else
            throw std::runtime_error("Unknow data type in miopenRNN");

        //miopen_handle_ = MIOpenHandle(deadline);

        rnnDesc_ = RNNDescriptor(hidden_size,
                                 1,
                                 miopenRNNskip,
                                 miopenRNNunidirection,
                                 rnn_mode,
                                 miopenRNNNoBias,
                                 miopenRNNdefault,
                                 data_type);

        CHECK_MIOPEN_ERROR(miopenGetRNNParamsDescriptor(  miopen_handle_.handle(),
                                                          rnnDesc_.desc(),
                                                          xDescArray_.ptr()[0],
                                                          wDesc_.desc(),
                                                          data_type));

        CHECK_MIOPEN_ERROR( miopenGetRNNParamsSize( miopen_handle_.handle(),
                                                    rnnDesc_.desc(),
                                                    xDescArray_.ptr()[0],
                                                    &weight_size_byte_,
                                                    data_type) );


        CHECK_MIOPEN_ERROR( miopenGetRNNWorkspaceSize(  miopen_handle_.handle(),
                                                        rnnDesc_.desc(),
                                                        sequenceLen_,
                                                        xDescArray_.ptr(),
                                                        &workspace_size_byte_) );

        CHECK_MIOPEN_ERROR( miopenGetRNNTrainingReserveSize(miopen_handle_.handle(),
                                                            rnnDesc_.desc(),
                                                            sequenceLen_,
                                                            xDescArray_.ptr(),
                                                            &trainspace_size_byte_) );

        weights_    = rand<T>(std::vector<int>{static_cast<int>(weight_size_byte_/sizeof(T))});
        workspace_  = zeros<float>(std::vector<int>{static_cast<int>(workspace_size_byte_/sizeof(float))});
        trainspace_ = zeros<float>(std::vector<int>{static_cast<int>(trainspace_size_byte_/sizeof(float))});

    }

    void forward(Tensor<T> x, Tensor<T> hx, Tensor<T> cx,
                 Tensor<T> y, Tensor<T> hy, Tensor<T> cy)
    {
        CHECK_MIOPEN_ERROR(miopenRNNForwardTraining(miopen_handle_.handle(),
                                                    rnnDesc_.desc(),
                                                    sequenceLen_,
                                                    xDescArray_.ptr(),
                                                    (void *)x.begin(),
                                                    hxDesc_.desc(),
                                                    (void *)hx.begin(),
                                                    cxDesc_.desc(),
                                                    (void *)cx.begin(),
                                                    wDesc_.desc(),
                                                    (void *)weights_.begin(),
                                                    yDescArray_.ptr(),
                                                    (void *)y.begin(),
                                                    hyDesc_.desc(),
                                                    (void *)hy.begin(),
                                                    cyDesc_.desc(),
                                                    (void *)cy.begin(),
                                                    (void *)workspace_.begin(),
                                                    workspace_size_byte_,
                                                    (void *)trainspace_.begin(),
                                                    trainspace_size_byte_) );
    }

    void backward_data( Tensor<T> y, Tensor<T> dy, Tensor<T> dhy,
                        Tensor<T> dcy, Tensor<T> hx, Tensor<T> cx,
                        Tensor<T> dx, Tensor<T> dhx, Tensor<T> dcx) {
        CHECK_MIOPEN_ERROR(miopenRNNBackwardData(miopen_handle_.handle(),
                                                rnnDesc_.desc(),
                                                sequenceLen_,
                                                yDescArray_.ptr(),
                                                (void *)y.begin(),
                                                dyDescArray_.ptr(),
                                                (void *)dy.begin(),
                                                dhyDesc_.desc(),
                                                (void *)dhy.begin(),
                                                dcyDesc_.desc(),
                                                (void *)dcy.begin(),
                                                wDesc_.desc(),
                                                (void *)weights_.begin(),
                                                hxDesc_.desc(),
                                                (void *)hx.begin(),
                                                cxDesc_.desc(),
                                                (void *)cx.begin(),
                                                dxDescArray_.ptr(),
                                                (void *)dx.begin(),
                                                dhxDesc_.desc(),
                                                (void *)dhx.begin(),
                                                dcxDesc_.desc(),
                                                (void *)dcx.begin(),
                                                (void *)workspace_.begin(),
                                                workspace_size_byte_,
                                                (void *)trainspace_.begin(),
                                                trainspace_size_byte_) );
    }

};

//template<typename T>
//void *call_rnn(void *id){
//    miopenRNN<T> *rnn;
//    int time_steps, batch_size, hidden_size;
//
//    std::tie(rnn, time_steps, batch_size, hidden_size) = *(std::tuple<miopenRNN<T>*, int, int, int> *)id;
//
//    auto x  = rand<T>({time_steps, batch_size, hidden_size});
//    auto y  = rand<T>({time_steps, batch_size, hidden_size});
//    auto dx = rand<T>({time_steps, batch_size, hidden_size});
//    auto dy = rand<T>({time_steps, batch_size, hidden_size});
//
//    auto hx  = rand<T>({1, batch_size, hidden_size});
//    auto hy  = rand<T>({1, batch_size, hidden_size});
//    auto cx  = rand<T>({1, batch_size, hidden_size});
//    auto cy  = rand<T>({1, batch_size, hidden_size});
//    auto dhx = rand<T>({1, batch_size, hidden_size});
//    auto dhy = rand<T>({1, batch_size, hidden_size});
//    auto dcx = rand<T>({1, batch_size, hidden_size});
//    auto dcy = rand<T>({1, batch_size, hidden_size});
//
//#ifdef NOSIM
//    //Warm up
//    rnn->forward(x, hx, cx, y, hy, cy);
//
//    hipHccModuleRingDoorbell(rnn->miopen_handle_.stream_);
//    hipStreamSynchronize(rnn->miopen_handle_.stream_);
//    //hipDeviceSynchronize();
//#endif
//
//    auto start = std::chrono::steady_clock::now();
//
//    for (int i = 0; i < 1; ++i) {
//        rnn->forward(x, hx, cx, y, hy, cy);
//    }
//
//    hipHccModuleRingDoorbell(rnn->miopen_handle_.stream_);
//    hipStreamSynchronize(rnn->miopen_handle_.stream_);
//    //hipDeviceSynchronize();
//
//    auto end = std::chrono::steady_clock::now();
//    std::cout << "Out of rnn call\n";
//    return 0;
//}

template<typename T>
int time_rnn(int hidden_size, int batch_size, int time_steps, const std::string & type, int inference, int num_streams)
{
    int num_repeats = 1;
    std::cout << "In time_rnn\n";

    auto x  = rand<T>({time_steps, batch_size, hidden_size});
    auto y  = rand<T>({time_steps, batch_size, hidden_size});
    auto dx = rand<T>({time_steps, batch_size, hidden_size});
    auto dy = rand<T>({time_steps, batch_size, hidden_size});

    auto hx  = rand<T>({1, batch_size, hidden_size});
    auto hy  = rand<T>({1, batch_size, hidden_size});
    auto cx  = rand<T>({1, batch_size, hidden_size});
    auto cy  = rand<T>({1, batch_size, hidden_size});
    auto dhx = rand<T>({1, batch_size, hidden_size});
    auto dhy = rand<T>({1, batch_size, hidden_size});
    auto dcx = rand<T>({1, batch_size, hidden_size});
    auto dcy = rand<T>({1, batch_size, hidden_size});

    miopenRNN<T> rnn[1] = {miopenRNN<T>(hidden_size, batch_size, time_steps, type, -1)};
    //miopenRNN<T> rnn[16] = {miopenRNN<T>(hidden_size, batch_size, time_steps, type, -1), miopenRNN<T>(hidden_size, batch_size, time_steps, type, -1), miopenRNN<T>(hidden_size, batch_size, time_steps, type, -1), miopenRNN<T>(hidden_size, batch_size, time_steps, type, -1),
    //                        miopenRNN<T>(hidden_size, batch_size, time_steps, type, -1), miopenRNN<T>(hidden_size, batch_size, time_steps, type, -1), miopenRNN<T>(hidden_size, batch_size, time_steps, type, -1), miopenRNN<T>(hidden_size, batch_size, time_steps, type, -1),
    //                        miopenRNN<T>(hidden_size, batch_size, time_steps, type, -1), miopenRNN<T>(hidden_size, batch_size, time_steps, type, -1), miopenRNN<T>(hidden_size, batch_size, time_steps, type, -1), miopenRNN<T>(hidden_size, batch_size, time_steps, type, -1),
    //                        miopenRNN<T>(hidden_size, batch_size, time_steps, type, -1), miopenRNN<T>(hidden_size, batch_size, time_steps, type, -1), miopenRNN<T>(hidden_size, batch_size, time_steps, type, -1), miopenRNN<T>(hidden_size, batch_size, time_steps, type, 10000000)};

    //m5_dump_reset_stats(0, 0);
#ifdef NOSIM
    Warm up
    for (int i = 0; i < num_streams; i++) {
        rnn[i].forward(x, hx, cx, y, hy, cy);

        // hipHccModuleRingDoorbell(rnn[i].miopen_handle_.stream_);
    }
    hipStreamSynchronize(rnn[0].miopen_handle_.stream_);
    //hipDeviceSynchronize();
#endif

    //auto start = std::chrono::steady_clock::now();

    for (int j = 0; j < num_streams; j++) {
        for (int i = 0; i < num_repeats; ++i) {
            rnn[j].forward(x, hx, cx, y, hy, cy);
        }

        // hipHccModuleRingDoorbell(rnn[j].miopen_handle_.stream_);
    }
    hipStreamSynchronize(rnn[0].miopen_handle_.stream_);
    //hipDeviceSynchronize();

    //auto end = std::chrono::steady_clock::now();
    //int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

    //int bwd_inputs_time = 0;
    //int bwd_params_time = 0;

    if (!inference)
    {
#ifdef NOSIM
        //Warm up
        for (int i = 0; i < num_streams; i++) {
            rnn[i].backward_data(y, dy, dhy, dcy,
                              hx, cx, dx, dhx, dcx);

            // hipHccModuleRingDoorbell(rnn[i].miopen_handle_.stream_);
        }
        hipStreamSynchronize(rnn[0].miopen_handle_.stream_);
        //hipDeviceSynchronize();
#endif

        //start = std::chrono::steady_clock::now();

        for (int j = 0; j < num_streams; j++) {
            for (int i = 0; i < num_repeats; ++i)
            {
                rnn[j].backward_data(y, dy, dhy, dcy,
                                  hx, cx, dx, dhx, dcx);
            }
            // hipHccModuleRingDoorbell(rnn[j].miopen_handle_.stream_);
        }
        hipStreamSynchronize(rnn[0].miopen_handle_.stream_);
        //hipDeviceSynchronize();

        //end = std::chrono::steady_clock::now();
        //bwd_inputs_time = std::chrono::duration<double, std::micro>(end - start).count() / num_repeats;
    }
    //m5_dump_reset_stats(0, 0);

    std::cout << "Out time_rnn\n";
    return 0;
}


int main(int argc, char **argv) {

    hipFree(0);

    int inference = 0;
    int num_streams = 1;

    if (argc > 5) {
        std::string inf = "inference";
        inference = argv[5] == inf ? 1 : 0;
    }
    if (argc > 6) {
        num_streams = atoi(argv[6]);
    }

    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(190) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "hidden_size     batch_size     time_steps     rnn_type     fwd_time (usec)  bwd_inputs_time (usec)  bwd_params_time (usec)  total_time (usec)  " << std::endl;
    std::cout << std::setfill('-') << std::setw(190) << "-" << std::endl;
    std::cout << std::setfill(' ');

    int total_fwd_time=0, total_bwd_inputs_time=0, total_bwd_params_time=0;
    int hidden_size, batch_size, time_steps;
    std::string rnn_type;
    std::tie(hidden_size, batch_size, time_steps, rnn_type) = std::make_tuple(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), argv[4]);

    std::tuple<int , int, int, std::string, int, int> input = std::make_tuple(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), argv[4], inference, 1);

    int fwd_time, bwd_inputs_time, bwd_params_time;

    //pthread_t *threads = new pthread_t[num_streams];

    //for (int i = 0; i < num_streams; i++) {
    //    std::cout << "Create thread " << i << "\n";
    //    pthread_create(&threads[i], NULL, time_rnn<float>, (void *)&input);
    //}

    //for (int i = num_streams-1; i >= 0; i++) {
    //    pthread_join(threads[i], NULL);
    //    std::cout << "After join " << i << "\n";
    //}

    std::vector<std::thread> threads;

    int n_worker_threads = 0;
    std::vector<int> outputs( num_streams, 0 );
    for ( int tid = 0; tid < num_streams; ++tid ) {
        try {
            threads.push_back( std::thread( [&] (size_t thread_id ) {
                        std::cout << "Hello from thread " <<  thread_id
                                  << std::endl;
                        outputs[thread_id] = thread_id;
                    }, tid ) );
        } catch ( const std::system_error& err ) {
            break;
        }
        n_worker_threads++;
    }
    std::cout << "Hello from master thread" << std::endl;
    // sync up all threads
    for (int i = 0; i < n_worker_threads; ++i) {
        threads[i].join();
    }

//    for (int i = 0; i < num_streams; i++) {
//        threads.emplace_back(time_rnn<float>, hidden_size, batch_size, time_steps, rnn_type, inference, 1);
//    }
//
//    std::cout << "Before join\n";
//
//    for (int i = 0; i < num_streams; i++) {
//        threads[i].join();
//        std::cout << "After join " << i << "\n";
//    }

    //hipDeviceSynchronize();
    std::cout << "DONE@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n";

    //time_rnn<float>(hidden_size, batch_size, time_steps, rnn_type, inference, num_streams);

    //std::cout << std::setw(5)  << hidden_size;
    //std::cout << std::setw(15) << batch_size;
    //std::cout << std::setw(15) << time_steps;
    //std::cout << std::setw(19) << rnn_type;
    //std::cout << std::setw(11) << std::setprecision(7) << fwd_time;
    //std::cout << std::setw(24) << std::setprecision(7) << bwd_inputs_time;
    //std::cout << std::setw(24) << std::setprecision(7) << bwd_params_time;
    //std::cout << std::setw(19) << std::setprecision(8) << fwd_time + bwd_inputs_time + bwd_params_time;

    //std::cout << std::endl;

    //total_fwd_time += fwd_time;
    //total_bwd_inputs_time += bwd_inputs_time;
    //total_bwd_params_time += bwd_params_time;

    //std::cout << std::setw(82) << "Totals" ;
    //std::cout << std::setw(14) << std::setprecision(7) << total_fwd_time;
    //std::cout << std::setw(24) << std::setprecision(7) << total_bwd_inputs_time;
    //std::cout << std::setw(24) << std::setprecision(7) << total_bwd_params_time;
    //std::cout << std::setw(19) << std::setprecision(8) << total_fwd_time + total_bwd_inputs_time + total_bwd_params_time;
    //std::cout << std::endl;

    return 0;

}


