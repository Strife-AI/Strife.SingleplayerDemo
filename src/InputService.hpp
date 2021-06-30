#pragma once

#include "PlayerEntity.hpp"
#include "Scene/IEntityEvent.hpp"
#include "Scene/Scene.hpp"

struct CastleEntity;
class PlayerNeuralNetworkService;

struct InputService : ISceneService
{
    void HandleInput();
    void Render(Renderer* renderer);
    void ReceiveEvent(const IEntityEvent& ev) override;
    void SpawnPlayer(CastleEntity* spawn, int playerId);

    static MoveDirection GetInputDirection();

    EntityReference<PlayerEntity> activePlayer;
    std::vector<PlayerEntity*> players;
    std::vector<CastleEntity*> spawns;
	PlayerNeuralNetworkService* nnService;

    bool gameOver = false;
};
