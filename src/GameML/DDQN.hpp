#pragma once

#include "../../Strife.ML/NewStuff.hpp"
#include "ML/ML.hpp"
#include "../../Strife.ML/TensorPacking.hpp"
#include <torch/torch.h>
#include "Tools/MetricsManager.hpp"

const int GridSize = 40;

struct InitialState : StrifeML::ISerializable
{
	void Serialize(StrifeML::ObjectSerializer& serializer) override
	{
		serializer
			.Add(grid);
	}

	GridSensorOutput<GridSize, GridSize> grid;
};

struct Transition : StrifeML::ISerializable
{
	void Serialize(StrifeML::ObjectSerializer& serializer) override
	{
		serializer
			.Add(actionIndex)
			.Add(reward)
			.Add(grid);
	}

	int actionIndex;
	float reward;
	GridSensorOutput<GridSize, GridSize> grid;
	bool isFinalState = false;
};

struct DeepQNetwork : StrifeML::NeuralNetwork<InitialState, Transition, 1>
{
	float discount = 0.999f; // gamma
	
    torch::nn::Embedding embedding{ nullptr };
    torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr }, conv3{ nullptr }, conv4{ nullptr };
    torch::nn::Linear dense{ nullptr };
    std::shared_ptr<torch::optim::Adam> optimizer;
	
    std::shared_ptr<DeepQNetwork> targetNetwork;
	torch::Device device = torch::Device(torch::kCPU);
	
    DeepQNetwork()
    {
        auto totalObservables = 5;
    	auto embeddingSize = 4;
    	auto totalActions = 9;
    	
        embedding = register_module("embedding", torch::nn::Embedding(totalObservables, embeddingSize));

    	// todo brendan we can write a loop for these
    	// todo brendan add back in batch norm
        conv1 = register_module("conv1", torch::nn::Conv2d(4, 8, 3));
        conv2 = register_module("conv2", torch::nn::Conv2d(8, 16, 3));
        conv3 = register_module("conv3", torch::nn::Conv2d(16, 32, 3));
        conv4 = register_module("conv4", torch::nn::Conv2d(32, 64, 3));
        dense = register_module("dense", torch::nn::Linear(64, totalActions));
        optimizer = std::make_shared<torch::optim::Adam>(parameters(), 1e-3);
    }

	// todo brendan can we move this function back to the trainer?
    void TrainBatch(Grid<const SampleType> input, StrifeML::TrainingBatchResult& outResult) override
    {
        torch::Tensor initialStates = PackIntoTensor(input, [=](auto& sample) { return sample.input.grid; }).to(device);
        torch::Tensor actions = PackIntoTensor(input, [=](auto& sample) { return static_cast<int64_t>(sample.output.actionIndex); }).to(device);
        torch::Tensor rewards = PackIntoTensor(input, [=](auto& sample) { return static_cast<float_t>(sample.output.reward); }).squeeze().to(device);
    	torch::Tensor nextStates = PackIntoTensor(input, [=](auto& sample) { return sample.output.grid; }).to(device);

    	//std::cout << initialStates.sizes() << std::endl;
        //std::cout << actions.sizes() << std::endl;
        
        auto forward = Forward(initialStates);
        //std::cout << forward.sizes() << std::endl;

    	//std::cout << actions << std::endl;
        torch::Tensor currentValues = forward.gather(1, actions);
    	torch::Tensor nextValues = std::get<0>(targetNetwork->Forward(nextStates).max(1));
        torch::Tensor expectedValues = (nextValues * discount) + rewards;

        std::cout << rewards << std::endl;
        //std::cout << expectedValues.sizes() << std::endl;

        torch::Tensor loss = torch::nn::functional::smooth_l1_loss(currentValues.squeeze(), expectedValues);

    	optimizer->zero_grad();
        loss.backward();
        optimizer->step();
 	
        outResult.loss = loss.item<float>();
    }

	// todo brendan can we move this function back to the decider?
    void MakeDecision(Grid<const InputType> input, OutputType& output) override
    {
        SetDevice(torch::kCPU);

        auto spatialInput = PackIntoTensor(input, [=](auto& sample) { return sample.grid; });
        torch::Tensor action = Forward(spatialInput);
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

    	// todo brendan add in dense layers and then LSTM

        x = dense->forward(x);

        return x.squeeze();
    }

	void SetDevice(torch::DeviceType deviceType)
    {
	    device = torch::Device(deviceType);
    	to(device);
    }
};

struct DQNDecider : StrifeML::Decider<DeepQNetwork>
{

};

struct DQNTrainer : StrifeML::Trainer<DeepQNetwork>
{
    DQNTrainer(Metric* lossMetric, int targetUpdatePeriod)
        : Trainer<DeepQNetwork>(32, 10000),
          lossMetric(lossMetric),
		  targetUpdatePeriod(targetUpdatePeriod)
    {
    	network->targetNetwork = std::make_shared<DeepQNetwork>();
        network->SetDevice(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    	network->targetNetwork->SetDevice(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    	
        LogStartup();
        samples = sampleRepository.CreateSampleSet("player-samples");
        samplesByActionType = samples
            ->CreateGroupedView<int>()
            //->GroupBy([=](const SampleType& sample) { return sample.output.actionIndex; });
            ->GroupBy([=](const SampleType& sample) { return 0; });// todo brendan hacky 1 group
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
    	++trainingCount;
    	if (trainingCount % targetUpdatePeriod == 0)
    	{
    		std::stringstream stream;
    		torch::save(network, stream);
    		torch::load(network->targetNetwork, stream);
    	}
        lossMetric->Add(result.loss);
    }

    StrifeML::SampleSet<SampleType>* samples;
    StrifeML::GroupedSampleView<SampleType, int>* samplesByActionType;
    Metric* lossMetric;
	int trainingCount = 0;
	int targetUpdatePeriod;
};