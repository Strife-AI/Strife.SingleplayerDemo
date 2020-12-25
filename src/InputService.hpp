#pragma once

#include "PlayerEntity.hpp"
#include "Scene/IEntityEvent.hpp"
#include "Scene/Scene.hpp"

struct CastleEntity;

struct InputService : ISceneService
{
    void OnAdded() override;
    void HandleInput();
    void Render(Renderer* renderer);
    void ReceiveEvent(const IEntityEvent& ev) override;

    EntityReference<PlayerEntity> activePlayer;
    std::vector<PlayerEntity*> players;
    std::vector<CastleEntity*> spawns;
    std::vector<Vector2> spawnPositions;

    float sendUpdateTimer = 0;

    bool gameOver = false;
};
