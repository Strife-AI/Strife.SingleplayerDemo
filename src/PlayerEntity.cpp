#include <Memory/Util.hpp>
#include <Components/LightComponent.hpp>
#include "PlayerEntity.hpp"
#include "InputService.hpp"
#include "Components/RigidBodyComponent.hpp"
#include "Net/ReplicationManager.hpp"
#include "Physics/PathFinding.hpp"
#include "Renderer/Renderer.hpp"

#include "CastleEntity.hpp"
#include "FireballEntity.hpp"
#include "GoalEntity.hpp"
#include "Math/Random.hpp"

Vector2 MoveDirectionToVector2(MoveDirection direction)
{
    const float halfSqrt2 = sqrt(2) / 2;

    switch(direction)
    {
    case MoveDirection::None: return Vector2(0, 0);
    case MoveDirection::West: return Vector2(-1, 0);;
    case MoveDirection::East: return Vector2(1, 0);;
    case MoveDirection::North: return Vector2(0, -1);;
    case MoveDirection::South: return Vector2(0, 1);;
    case MoveDirection::NorthEast: return Vector2(halfSqrt2, -halfSqrt2);;
    case MoveDirection::SouthEast: return Vector2(halfSqrt2, halfSqrt2);;
    case MoveDirection::NorthWest: return Vector2(-halfSqrt2, -halfSqrt2);;
    case MoveDirection::SouthWest: return Vector2(-halfSqrt2, halfSqrt2);;
    default: return Vector2(0, 0);
    }
}

MoveDirection GetClosestMoveDirection(Vector2 v)
{
    if (v == Vector2(0, 0))
    {
        return MoveDirection::None;
    }

    float maxDot = -INFINITY;
    MoveDirection closestDirection;

    for (int i = 1; i < (int)MoveDirection::TotalDirections; ++i)
    {
        auto direction = static_cast<MoveDirection>(i);
        Vector2 dir = MoveDirectionToVector2(direction);
        float dot = dir.Dot(v);

        if (dot > maxDot)
        {
            maxDot = dot;
            closestDirection = direction;
        }
    }

    return closestDirection;
}

void PlayerEntity::OnAdded()
{
    auto light = AddComponent<LightComponent<PointLight>>();
    light->position = Center();
    light->color = Color(255, 255, 255, 255);
    light->maxDistance = 400;
    light->intensity = 0.6f;

    health = AddComponent<HealthBarComponent>();
    health->offsetFromCenter = Vector2(0, -20);

    rigidBody = AddComponent<RigidBodyComponent>(b2_dynamicBody);
    pathFollower = AddComponent<PathFollowerComponent>(rigidBody);

    SetDimensions({ 30, 30 });
    auto box = rigidBody->CreateBoxCollider(Dimensions());

    box->SetDensity(1);
    box->SetFriction(0);

    scene->GetService<InputService>()->players.push_back(this);

	// Setup network and sensors
    {
        auto nn = AddComponent<NeuralNetworkComponent<DeepQNetwork>>(10);
        nn->SetNetwork("nn");
    	nn->mode = NeuralNetworkMode::ReinforcementLearning;

        auto gridSensor = AddComponent<GridSensorComponent<GridSize, GridSize>>(Vector2(16, 16));
    	gridSensor->render = true;
    	    	            	
        // Called when:
        //  * Collecting input to make a decision
        //  * Adding a training sample
        nn->collectInput = [=](InitialState& input)
        {
            gridSensor->Read(input.grid);
        };

        // Called when the decider makes a decision
        nn->receiveDecision = [=](Transition& decision)
        {
        	if (scene->GetService<InputService>()->activePlayer.GetValueOrNull() == this)
        	{
        		return;
        	}
        	lastDirectionIndex = decision.actionIndex;
            SetMoveDirection(MoveDirectionToVector2(static_cast<MoveDirection>(decision.actionIndex)) * 200);
        };

        // Collects what decision/state the player made
        nn->collectDecision = [=](Transition& outDecision)
        {
        	gridSensor->Read(outDecision.grid); // todo brendan I think we are reading the sensor input twice in a row essentially

        	outDecision.reward = currentReward;
        	outDecision.isFinalState = IsApproximately(currentReward, 0);
        	currentReward = -0.1f;
        	outDecision.actionIndex = lastDirectionIndex;

            if (outDecision.isFinalState)
            {
	            SetCenter(Rand(Vector2(150), Vector2(600)));
            }
        };
    }
}

void PlayerEntity::ReceiveEvent(const IEntityEvent& ev)
{
    if (auto outOfHealth = ev.Is<OutOfHealthEvent>())
    {
        Die(outOfHealth);
    }

	if (auto contact = ev.Is<ContactBeginEvent>())
	{
		if (contact->OtherIs<GoalEntity>())
		{
			currentReward = 0;
			//StartTimer(0.05f, [=]
			//{
			//	std::cout << "Jumping" << std::endl;
			//	SetCenter(Rand(Vector2(150), Vector2(600)));
			//});
		}
	}

    if (auto reward = ev.Is<RewardEvent>())
    {
	    switch (reward->rewardType)
	    {
	    case RewardType::ScoreGoal:
		    currentReward = 0;
		    break;
	    case RewardType::None:
		    currentReward -= 0.1f;
	    }
    }
}

void PlayerEntity::Die(const OutOfHealthEvent* outOfHealth)
{
    Destroy();

    for (auto spawn : scene->GetService<InputService>()->spawns)
    {
        if (spawn->playerId == playerId)
        {
            spawn->StartTimer(10, [=]
            {
                spawn->SpawnPlayer();
            });

            break;
        }
    }
}

void PlayerEntity::OnDestroyed()
{
    RemoveFromVector(scene->GetService<InputService>()->players, this);
}

void PlayerEntity::Render(Renderer* renderer)
{
    auto position = Center();
	
    // Render player
    {
        Color c[5] = {
            Color::CornflowerBlue(),
            Color::Green(),
            Color::Orange(),
            Color::HotPink(),
            Color::Yellow()
        };

        auto color = c[playerId];
        renderer->RenderRectangle(Rectangle(position - Dimensions() / 2, Dimensions()), color, -0.99);
    }
}

void PlayerEntity::FixedUpdate(float deltaTime)
{
    attackCoolDown -= deltaTime;

    if (state == PlayerState::Attacking)
    {
        Entity* target;
        RaycastResult hitResult;
        if (attackTarget.TryGetValue(target))
        {
            bool withinAttackingRange = (target->Center() - Center()).Length() < 200;
            bool canSeeTarget = scene->Raycast(Center(), target->Center(), hitResult)
                                && hitResult.handle.OwningEntity() == target;

            if (withinAttackingRange && canSeeTarget)
            {
                rigidBody->SetVelocity({ 0, 0 });
                auto dir = (target->Center() - Center()).Normalize();

                if (attackCoolDown <= 0)
                {
                    auto fireball = scene->CreateEntity<FireballEntity>(Center(), dir * 400);
                    fireball->playerId = playerId;
                    fireball->ownerId = id;

                    attackCoolDown = 1;
                }

                return;
            }
        }
        else
        {
            pathFollower->Stop(true);
            state = PlayerState::None;
        }
    }
}

void PlayerEntity::Attack(Entity* entity)
{
    attackTarget = entity;
    state = PlayerState::Attacking;
}

void PlayerEntity::SetMoveDirection(Vector2 direction)
{
    rigidBody->SetVelocity(direction);

    if (direction != Vector2(0, 0))
    {
        state = PlayerState::Moving;
    }
}
