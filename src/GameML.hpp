#pragma once

#include "../../Strife.ML/NewStuff.hpp"
#include "Math/Vector2.hpp"
#include "ML/ML.hpp"
#include "../../Strife.ML/TensorPacking.hpp"

struct PlayerModelInput : StrifeML::ISerializable
{
    void Serialize(StrifeML::ObjectSerializer& serializer) override
    {
        serializer
            .Add(velocity)
            .Add(grid);
    }

    Vector2 velocity;
    GridSensorOutput<40, 40> grid;
};

enum class PlayerAction
{
    Nothing,
    Up,
    Down,
    Left,
    Right
};

struct PlayerDecision : StrifeML::ISerializable
{
    void Serialize(StrifeML::ObjectSerializer& serializer) override
    {
        serializer
            .Add(velocity)
            .Add(action);
    }

    Vector2 velocity;
    PlayerAction action;
};

struct PlayerNetwork : StrifeML::NeuralNetwork<PlayerModelInput, PlayerDecision, 1>
{
    void MakeDecision(Grid<const InputType> input, OutputType& output) override
    {
        auto spacialInput = StrifeML::PackIntoTensor(input, [=](auto& input) { return input.grid; });
    }

    void TrainBatch(Grid<const SampleType> input, StrifeML::TrainingBatchResult& outResult) override
    {
        auto spacialInput = StrifeML::PackIntoTensor(input, [=](auto& sample) { return sample.input.grid; });
    }
};

struct PlayerDecider : StrifeML::Decider<PlayerNetwork>
{
    
};

struct PlayerTrainer : StrifeML::Trainer<PlayerNetwork>
{
    PlayerTrainer()
        : Trainer<PlayerNetwork>(32, 1)
    {
        samples = sampleRepository.CreateSampleSet("player-samples");
        samplesByActionType = samples
            ->CreateGroupedView<PlayerAction>()
            ->GroupBy([=](const SampleType& sample) { return sample.output.action; });
    }

    void ReceiveSample(const SampleType& sample) override
    {
        samples->AddSample(sample);
    }

    bool TrySelectSequenceSamples(gsl::span<SampleType> outSequence) override
    {
        return samplesByActionType->TryPickRandomSequence(outSequence);
    }

    StrifeML::SampleSet<SampleType>* samples;
    StrifeML::GroupedSampleView<SampleType, PlayerAction>* samplesByActionType;
};