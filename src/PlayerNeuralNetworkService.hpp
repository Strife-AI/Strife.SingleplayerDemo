#pragma once

#include "PlayerEntity.hpp"
#include "ML/NeuralNetworkService.hpp"

class InputService;

struct PlayerNeuralNetworkService : NeuralNetworkService<PlayerEntity, PlayerNetwork>
{
	PlayerNeuralNetworkService(StrifeML::NetworkContext<PlayerNetwork>* context, InputService* inputService);
	
	void CollectInput(PlayerEntity* entity, InputType& input) override;

	void ReceiveDecision(PlayerEntity* entity, OutputType& output) override;

	void CollectTrainingSamples(TrainerType* trainer) override;

	InputService* inputService;
};