#pragma once

#include "PlayerEntity.hpp"
#include "Scene/IEntityEvent.hpp"
#include "Scene/Scene.hpp"

struct CastleEntity;

struct InputService : ISceneService
{
    void HandleInput();
    void Render(Renderer* renderer);
    void ReceiveEvent(const IEntityEvent& ev) override;
    void SpawnPlayer(Vector2 position, int playerId);

    static MoveDirection GetInputDirection();

    EntityReference<PlayerEntity> activePlayer;
    std::vector<PlayerEntity*> players;
    std::vector<CastleEntity*> spawns;

    bool gameOver = false;
};
