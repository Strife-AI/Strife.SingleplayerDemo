#pragma once

#include <Components/PathFollowerComponent.hpp>
#include "GameML.hpp"
#include "Components/NetComponent.hpp"
#include "ML/ML.hpp"
#include "Scene/BaseEntity.hpp"
#include "Scene/IEntityEvent.hpp"
#include "HealthBarComponent.hpp"

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
    HealthBarComponent* health;

    EntityReference<Entity> attackTarget;
    PlayerState state = PlayerState::None;
    float attackCoolDown = 0;
    int playerId;

    void Die(const OutOfHealthEvent* outOfHealth);
};