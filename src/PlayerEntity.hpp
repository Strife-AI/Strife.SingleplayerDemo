#pragma once

#include <Components/PathFollowerComponent.hpp>
#include "GameML.hpp"
#include "Components/NetComponent.hpp"
#include "ML/ML.hpp"
#include "Scene/BaseEntity.hpp"
#include "Scene/IEntityEvent.hpp"
#include "HealthBarComponent.hpp"

struct PlayerEntity;

DEFINE_EVENT(PlayerAddedToGame)
{
    PlayerAddedToGame(PlayerEntity* player_)
        : player(player_)
    {

    }

    PlayerEntity* player;
};

DEFINE_EVENT(PlayerRemovedFromGame)
{
    PlayerRemovedFromGame(PlayerEntity* player_)
        : player(player_)
    {

    }

    PlayerEntity* player;
};

enum class PlayerState
{
    None,
    Moving,
    Attacking
};

DEFINE_ENTITY(PlayerEntity, "player")
{
    using NeuralNetwork = NeuralNetworkComponent<PlayerNetwork>;

    void MoveTo(Vector2 position);
    void Attack(Entity* entity);

    void OnAdded() override;
    void ReceiveServerEvent(const IEntityEvent& ev) override;
    void OnDestroyed() override;

    void Render(Renderer* renderer) override;
    void ServerFixedUpdate(float deltaTime) override;

    RigidBodyComponent* rigidBody;
    PathFollowerComponent* pathFollower;
    NetComponent* net;
    HealthBarComponent* health;

    EntityReference<Entity> attackTarget;
    PlayerState state = PlayerState::None;
    float attackCoolDown = 0;

    void Die(const OutOfHealthEvent* outOfHealth);
};