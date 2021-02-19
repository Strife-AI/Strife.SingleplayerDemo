#pragma once

#include <Components/PathFollowerComponent.hpp>
#include "GameML/DDQN.hpp"
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

enum class MoveDirection
{
    None,
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest,
    TotalDirections
};

enum class RewardType
{
	None,
    ScoreGoal
};

DEFINE_EVENT(RewardEvent)
{
    RewardEvent(RewardType rewardType)
        : rewardType(rewardType)
    {
    }

    RewardType rewardType;
};

DEFINE_EVENT(TimerEvent)
{
    TimerEvent()
    {
    }
};

DEFINE_ENTITY(PlayerEntity, "player")
{
    using NeuralNetwork = NeuralNetworkComponent<DeepQNetwork>;

    void Attack(Entity* entity);
    void SetMoveDirection(Vector2 direction);

    void OnAdded() override;
    void ReceiveEvent(const IEntityEvent& ev) override;
    void OnDestroyed() override;

    void Render(Renderer* renderer) override;
    void FixedUpdate(float deltaTime) override;

    RigidBodyComponent* rigidBody;
    PathFollowerComponent* pathFollower;
    HealthBarComponent* health;

    EntityReference<Entity> attackTarget;
    PlayerState state = PlayerState::None;
    float attackCoolDown = 0;
    int playerId;
    MoveDirection lastDirection = MoveDirection::None;
	int lastDirectionIndex = 0; // todo brendan redundant hack with lastDirection
	float currentReward = 0.0f;
	EntityReference<Entity> goal;

    void Die(const OutOfHealthEvent* outOfHealth);
};

Vector2 MoveDirectionToVector2(MoveDirection direction);
MoveDirection GetClosestMoveDirection(Vector2 v);