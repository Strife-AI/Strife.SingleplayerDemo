#pragma once

#include "PlayerNeuralNetworkService.hpp"
#include "InputService.hpp"

PlayerNeuralNetworkService::PlayerNeuralNetworkService(StrifeML::NetworkContext<PlayerNetwork>* context, InputService* inputService)
	: NeuralNetworkService<PlayerEntity, PlayerNetwork>(context, 128),
	inputService(inputService)
{
}

void PlayerNeuralNetworkService::CollectInput(PlayerEntity* entity, InputType& input)
{
	entity->gridSensor->Read(input.grid);
}

void PlayerNeuralNetworkService::ReceiveDecision(PlayerEntity* entity, OutputType& output)
{
	entity->SetMoveDirection(MoveDirectionToVector2(static_cast<MoveDirection>(output.actionIndex)) * 200);
}

void PlayerNeuralNetworkService::CollectTrainingSamples(TrainerType* trainer)
{
	PlayerEntity* player;
	if (inputService->activePlayer.TryGetValue(player))
	{
		SampleType sample;
		CollectInput(player, sample.input);
		sample.output.actionIndex = static_cast<int>(player->lastDirection);
		trainer->AddSample(sample);
	}
}
