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
	const static int LayerCount = 4;
	
    torch::nn::Embedding embedding{ nullptr };
    torch::nn::Conv2d conv[LayerCount]{ {nullptr}, {nullptr}, {nullptr}, {nullptr} };
    torch::nn::BatchNorm2d batchNorm[LayerCount]{ {nullptr}, {nullptr}, {nullptr}, {nullptr} };
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

    	int convChannels = embeddingSize;
        for (int i = 0; i < LayerCount; ++i)
        {
	        conv[i] = register_module("conv" + std::to_string(i), torch::nn::Conv2d(convChannels, convChannels * 2, 3));
	        batchNorm[i] = register_module("batchNorm" + std::to_string(i), torch::nn::BatchNorm2d(convChannels * 2));
        	convChannels *= 2;
        }
    	
        dense = register_module("dense", torch::nn::Linear(convChannels, totalActions));
        optimizer = std::make_shared<torch::optim::Adam>(parameters(), 1e-5);
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

        //std::cout << rewards << std::endl;
        //std::cout << expectedValues.sizes() << std::endl;

        torch::Tensor loss = torch::nn::functional::smooth_l1_loss(currentValues.squeeze(), expectedValues);

    	optimizer->zero_grad();
        loss.backward();

   // 	for(auto param : parameters()) {
			//param.mutable_grad() = param.grad().data().clamp_(-1, 1);
   //     }
    	
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

        x = embedding->forward(x);
        x = x.permute({ 0, 3, 1, 2 });

        try
        {
	        for (int i = 0; i < LayerCount; ++i)
	        {
	        	bool lastLayer = i == LayerCount - 1;
		        x = conv[i]->forward(x);
	        	
	            if (!lastLayer)
	            {
		            x = batchNorm[i]->forward(x);
	            }
		        
		        x = leaky_relu(x);
		        x = dropout(x, 0.5, is_training());

	            if (!lastLayer)
	            {
			        x = max_pool2d(x, {2, 2});
		        }
	        }
        }
        catch (std::exception e)
        {
	        std::cout << e.what() << std::endl;
        	throw;
        }

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