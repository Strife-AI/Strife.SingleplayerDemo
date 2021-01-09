#pragma once

#include "../../Strife.ML/NewStuff.hpp"
#include "ML/ML.hpp"
#include "../../Strife.ML/TensorPacking.hpp"
#include <torch/torch.h>

#include "Tools/MetricsManager.hpp"

struct Observation : StrifeML::ISerializable
{
    void Serialize(StrifeML::ObjectSerializer& serializer) override
    {
        serializer
            .Add(grid);
    }

    GridSensorOutput<40, 40> grid;
};

struct TrainingLabel : StrifeML::ISerializable
{
    void Serialize(StrifeML::ObjectSerializer& serializer) override
    {
        serializer
            .Add(actionIndex);
    }

    int actionIndex;
};

struct BehaviorCloningNetwork : StrifeML::NeuralNetwork<Observation, TrainingLabel, 1>
{
    torch::nn::Embedding embedding{ nullptr };
    torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr }, conv3{ nullptr }, conv4{ nullptr };
    torch::nn::Linear dense{ nullptr };
    std::shared_ptr<torch::optim::Adam> optimizer;

    BehaviorCloningNetwork()
    {
        embedding = register_module("embedding", torch::nn::Embedding(3, 4));
        conv1 = register_module("conv1", torch::nn::Conv2d(4, 8, 5));
        conv2 = register_module("conv2", torch::nn::Conv2d(8, 16, 3));
        conv3 = register_module("conv3", torch::nn::Conv2d(16, 32, 3));
        conv4 = register_module("conv4", torch::nn::Conv2d(32, 64, 3));
        dense = register_module("dense", torch::nn::Linear(64, 9));
        optimizer = std::make_shared<torch::optim::Adam>(parameters(), 1e-3);
    }

    void TrainBatch(Grid<const SampleType> input, StrifeML::TrainingBatchResult& outResult) override
    {
        //Log("Train batch start\n");
        optimizer->zero_grad();

        //Log("Pack spatial\n");
        torch::Tensor spatialInput = PackIntoTensor(input, [=](auto& sample) { return sample.input.grid; });

        //Log("Pack labels\n");
        torch::Tensor labels = PackIntoTensor(input, [=](auto& sample) { return static_cast<int64_t>(sample.output.actionIndex); }).squeeze();

        //Log("Predicting...\n");
        torch::Tensor prediction = Forward(spatialInput).squeeze();

        //std::cout << prediction.sizes() << std::endl;
        //std::cout << labels << std::endl;

        //Log("Calculate loss\n");
        torch::Tensor loss = torch::nn::functional::nll_loss(prediction, labels);

        //Log("Call backward\n");
        loss.backward();

        //Log("Call optimizer step\n");
        optimizer->step();

        outResult.loss = loss.item<float>();

        //Log("Train batch end\n");
    }

    void MakeDecision(Grid<const InputType> input, OutputType& output) override
    {
        auto spatialInput = PackIntoTensor(input, [=](auto& sample) { return sample.grid; });
        torch::Tensor action = Forward(spatialInput).squeeze();
        torch::Tensor index = std::get<1>(torch::max(action, 0));
        int maxIndex = *index.data_ptr<int64_t>();
        output.actionIndex = maxIndex;
    }

    torch::Tensor Forward(const torch::Tensor& spatialInput)
    {
        // spatialInput's shape: B x S x Rows x Cols
        auto batchSize = spatialInput.size(0);
        auto sequenceLength = spatialInput.size(1);
        auto height = spatialInput.size(2);
        auto width = spatialInput.size(3);

        torch::Tensor x;

        if (sequenceLength > 1)
        {
            x = spatialInput.view({ -1, height, width });
        }
        else
        {
            x = squeeze(spatialInput, 1);
        }

        x = embedding->forward(x);     // N x 80 x 80 x 4
        x = x.permute({ 0, 3, 1, 2 }); // N x 4 x 80 x 80


        x = leaky_relu(conv1->forward(x)); // N x 8 x 76 x 76
        x = dropout(x, 0.5, is_training());
        x = max_pool2d(x, { 2, 2 }); // N x 8 x 38 x 38

        x = leaky_relu(conv2->forward(x)); // N x 16 x 36 x 36
        x = dropout(x, 0.5, is_training());
        x = max_pool2d(x, { 2, 2 }); // N x 16 x 18 x 18

        x = leaky_relu(conv3->forward(x)); // N x 32 x 16 x 16
        x = dropout(x, 0.5, is_training());
        x = max_pool2d(x, { 2, 2 }); // N x 32 x 8 x 8

        x = leaky_relu(conv4->forward(x)); // N x 64 x 6 x 6
        x = dropout(x, 0.5, is_training());

        if (sequenceLength > 1)
        {
            x = x.view({sequenceLength, batchSize, 128});
        }
        else
        {
            x = x.view({batchSize, 64});
        }

        x = dense->forward(x);
        x = log_softmax(x, 1);

        return x;
    }
};

struct BCDecider : StrifeML::Decider<BehaviorCloningNetwork>
{

};

struct BCTrainer : StrifeML::Trainer<BehaviorCloningNetwork>
{
    BCTrainer(Metric* lossMetric)
        : Trainer<BehaviorCloningNetwork>(32, 10000),
          lossMetric(lossMetric)
    {
        LogStartup();
        samples = sampleRepository.CreateSampleSet("player-samples");
        samplesByActionType = samples
            ->CreateGroupedView<int>()
            ->GroupBy([=](const SampleType& sample) { return sample.output.actionIndex; });
    }

    void LogStartup() const
    {
        std::cout << "Trainer starting" << std::endl;

        std::cout << std::boolalpha;
        std::cout << "CUDA is available: " << torch::cuda::is_available() << std::endl;

        std::cout << torch::show_config() << std::endl;
        std::cout << "Torch Inter-Op Threads: " << torch::get_num_interop_threads() << std::endl;
        std::cout << "Torch Intra-Op Threads: " << torch::get_num_threads() << std::endl;
    }

    void ReceiveSample(const SampleType& sample) override
    {
        samples->AddSample(sample);
    }

    bool TrySelectSequenceSamples(gsl::span<SampleType> outSequence) override
    {
        return samplesByActionType->TryPickRandomSequence(outSequence);
    }

    void OnTrainingComplete(const StrifeML::TrainingBatchResult& result) override
    {
        lossMetric->Add(result.loss);
    }

    StrifeML::SampleSet<SampleType>* samples;
    StrifeML::GroupedSampleView<SampleType, int>* samplesByActionType;
    Metric* lossMetric;
};