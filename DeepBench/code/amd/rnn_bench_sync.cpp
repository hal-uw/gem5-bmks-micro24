#include <iomanip>
#include <memory>
#include <chrono>
#include <vector>
#include <tuple>

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
    Tensor<T> x,y,dx,dy;

public:
    miopenRNN(int hidden_size, int batch_size, int time_steps, const std::string& rnn_type, uint64_t deadline, bool ind_gpu, uint32_t gpu_id) :
        miopen_handle_(deadline, ind_gpu, gpu_id),
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

        x = rand<T>({time_steps, batch_size, hidden_size});
        y = rand<T>({time_steps, batch_size, hidden_size});
        dx = rand<T>({time_steps, batch_size, hidden_size});
        dy = rand<T>({time_steps, batch_size, hidden_size});

    }

    void forward(Tensor<T> hx, Tensor<T> cx,
                 Tensor<T> hy, Tensor<T> cy)
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

    void backward_data( Tensor<T> dhy,
                        Tensor<T> dcy, Tensor<T> hx, Tensor<T> cx,
                        Tensor<T> dhx, Tensor<T> dcx) {
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

template<typename T>
std::tuple<int, int, int> time_rnn(
    int hidden_size, int batch_size,
    const std::string & type, int inference, int num_streams,
    bool ind_gpu)
{
    int num_repeats = 1;

    auto hx  = rand<T>({1, batch_size, hidden_size});
    auto hy  = rand<T>({1, batch_size, hidden_size});
    auto cx  = rand<T>({1, batch_size, hidden_size});
    auto cy  = rand<T>({1, batch_size, hidden_size});
    auto dhx = rand<T>({1, batch_size, hidden_size});
    auto dhy = rand<T>({1, batch_size, hidden_size});
    auto dcx = rand<T>({1, batch_size, hidden_size});
    auto dcy = rand<T>({1, batch_size, hidden_size});


    int sizes[128] = {4,38,31,19,40,8,9,7,23,25,49,18,12,7,10,18,39,13,28,34,27,
                      41,6,21,69,13,9,14,21,7,28,17,57,54,21,40,10,47,17,20,10,
                      22,16,12,18,6,10,6,15,17,32,25,22,12,28,8,12,16,52,14,15,
                      13,10,13,7,13,19,1,37,10,3,32,15,8,15,56,36,24,22,27,26,
                      31,17,12,35,15,105,62,97,26,5,30,14,18,7,10,11,7,16,7,17,
                      39,9,4,10,35,54,5,12,27,5,28,29,24,18,82,19,14,5,32,7,64,
                      26,53,62,28,5,19};

    std::vector<std::shared_ptr<miopenRNN<T>>> rnn(num_streams);

    std::cout << "Initializing RNNs" << std::endl;

    for (int i = 0; i < num_streams; i++) {
        std::cout << "rnn " << i << std::endl;
        rnn[i] = std::make_shared<miopenRNN<T>>(hidden_size, batch_size, sizes[i], type, 7000000000, ind_gpu, i%4);
    }

    std::cout << "Starting to run" << std::endl;

    m5_dump_reset_stats(0, 0);
    auto start = std::chrono::steady_clock::now();

    for (int j = 0; j < num_streams; j++) {
        for (int i = 0; i < num_repeats; ++i) {
            rnn[j].get()->forward(hx, cx, hy, cy);
        }

    }
    // for (int j = 0; j < num_streams; j++) {
    //     hipHccModuleRingDoorbell(rnn[j].get()->miopen_handle_.stream_);
    // }

    for (int j = 0; j < num_streams; j++) {
        hipStreamSynchronize(rnn[j].get()->miopen_handle_.stream_);
    }

    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

    int bwd_inputs_time = 0;
    int bwd_params_time = 0;

    if (!inference)
    {
        start = std::chrono::steady_clock::now();

        for (int j = 0; j < num_streams; j++) {
            for (int i = 0; i < num_repeats; ++i)
            {
                rnn[j].get()->backward_data(dhy, dcy,
                                  hx, cx, dhx, dcx);
            }
        }
        // for (int j = 0; j < num_streams; j++) {
        //     hipHccModuleRingDoorbell(rnn[j].get()->miopen_handle_.stream_);
        // }

        for (int j = 0; j < num_streams; j++) {
            hipStreamSynchronize(rnn[j].get()->miopen_handle_.stream_);
        }

        end = std::chrono::steady_clock::now();
        bwd_inputs_time = std::chrono::duration<double, std::micro>(end - start).count() / num_repeats;
    }
    m5_dump_reset_stats(0, 0);
    return std::tuple<int, int, int>(fwd_time, bwd_inputs_time, bwd_params_time);
}


int main(int argc, char **argv) {

    hipFree(0);

    int inference = 1;
    int num_streams = 1;
    bool ind_gpus = false;

    // rnn_type, num_streams, ind_gpus, mmap file

    if (argc < 4 || argc > 5) {
        std::cout << "Invalid # of args" << std::endl;
    }

    if (argc == 5) {
        // Global vars, located in tensor.h
        g_tensor_use_mmap = true;
        g_tensor_mmap_file = argv[4];
    }

    num_streams = atoi(argv[2]);
    ind_gpus = atoi(argv[3]);

    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(190) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "hidden_size     batch_size     rnn_type     fwd_time (usec)  bwd_inputs_time (usec)  bwd_params_time (usec)  total_time (usec)  " << std::endl;
    std::cout << std::setfill('-') << std::setw(190) << "-" << std::endl;
    std::cout << std::setfill(' ');

    int total_fwd_time=0, total_bwd_inputs_time=0, total_bwd_params_time=0;
    int hidden_size, batch_size, time_steps;
    std::string rnn_type;

    rnn_type = argv[1];
    if (rnn_type == "vanilla") {
        hidden_size = 256;
        batch_size = 8;
    } else {
        hidden_size = 128;
        batch_size = 4;
    }

    int fwd_time, bwd_inputs_time, bwd_params_time;

    std::tie(fwd_time, bwd_inputs_time, bwd_params_time) =
        time_rnn<float>(hidden_size, batch_size, rnn_type, inference, num_streams, ind_gpus);

    std::cout << std::setw(5)  << hidden_size;
    std::cout << std::setw(15) << batch_size;
    std::cout << std::setw(19) << rnn_type;
    std::cout << std::setw(11) << std::setprecision(7) << fwd_time;
    std::cout << std::setw(24) << std::setprecision(7) << bwd_inputs_time;
    std::cout << std::setw(24) << std::setprecision(7) << bwd_params_time;
    std::cout << std::setw(19) << std::setprecision(8) << fwd_time + bwd_inputs_time + bwd_params_time;

    std::cout << std::endl;

    total_fwd_time += fwd_time;
    total_bwd_inputs_time += bwd_inputs_time;
    total_bwd_params_time += bwd_params_time;

    std::cout << std::setw(82) << "Totals" ;
    std::cout << std::setw(14) << std::setprecision(7) << total_fwd_time;
    std::cout << std::setw(24) << std::setprecision(7) << total_bwd_inputs_time;
    std::cout << std::setw(24) << std::setprecision(7) << total_bwd_params_time;
    std::cout << std::setw(19) << std::setprecision(8) << total_fwd_time + total_bwd_inputs_time + total_bwd_params_time;
    std::cout << std::endl;

    return 0;

}


