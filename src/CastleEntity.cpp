#include "CastleEntity.hpp"

#include "Engine.hpp"
#include "PlayerEntity.hpp"
#include "Components/RigidBodyComponent.hpp"
#include "Components/SpriteComponent.hpp"
#include "Physics/PathFinding.hpp"
#include "Net/ReplicationManager.hpp"

void CastleEntity::OnAdded()
{
    spriteComponent = AddComponent<SpriteComponent>("castle");
    spriteComponent->scale = Vector2(5.0f);

    Vector2 size{ 67 * 5, 55 * 5 };
    SetDimensions(size);
    scene->GetService<PathFinderService>()->AddObstacle(Bounds());

    auto rigidBody = AddComponent<RigidBodyComponent>(b2_staticBody);
    rigidBody->CreateBoxCollider(size);

    auto health = AddComponent<HealthBarComponent>();
    health->offsetFromCenter = -size.YVector() / 2 - Vector2(0, 5);
    health->maxHealth = 1000;
    health->health = 1000;

    auto offset = size / 2 + Vector2(40, 40);

    _spawnSlots[0] = Center() + offset.XVector();
    _spawnSlots[1] = Center() - offset.XVector();
    _spawnSlots[2] = Center() + offset.YVector();
    _spawnSlots[3] = Center() - offset.YVector();

    _light = AddComponent<LightComponent<PointLight>>();
    _light->position = Center();
    _light->intensity = 0.5;
    _light->maxDistance = 500;
}

void CastleEntity::Update(float deltaTime)
{
    _light->color = playerId == 0
        ? Color::Green()
        : Color::White();
}

void CastleEntity::SpawnPlayer()
{
    auto position = _spawnSlots[_nextSpawnSlotId];
    _nextSpawnSlotId = (_nextSpawnSlotId + 1) % 4;

    auto player = scene->CreateEntity<PlayerEntity>({ position });
    player->playerId = playerId;
}

void CastleEntity::OnDestroyed()
{
    for (auto player : scene->GetEntitiesOfType<PlayerEntity>())
    {
        if (player->playerId == playerId)
        {
            player->Destroy();
        }
    }

    scene->GetService<PathFinderService>()->RemoveObstacle(Bounds());
}

void CastleEntity::ReceiveEvent(const IEntityEvent& ev)
{
    if (ev.Is<OutOfHealthEvent>())
    {
        Destroy();
    }
}
