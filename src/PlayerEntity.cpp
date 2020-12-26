#include <Memory/Util.hpp>
#include <Components/LightComponent.hpp>
#include "PlayerEntity.hpp"
#include "InputService.hpp"
#include "Components/RigidBodyComponent.hpp"
#include "Net/ReplicationManager.hpp"
#include "Physics/PathFinding.hpp"
#include "Renderer/Renderer.hpp"
#include "torch/torch.h"
#include "MessageHud.hpp"

#include "CastleEntity.hpp"
#include "FireballEntity.hpp"

void PlayerEntity::OnAdded()
{
    auto light = AddComponent<LightComponent<PointLight>>();
    light->position = Center();
    light->color = Color(255, 255, 255, 255);
    light->maxDistance = 400;
    light->intensity = 0.6;

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
#if false
    {
        auto nn = AddComponent<NeuralNetworkComponent<PlayerNetwork>>();
        //nn->SetNetwork("nn");

        // Network only runs on server
        if (scene->isServer) nn->mode = NeuralNetworkMode::Deciding;

        auto gridSensor = AddComponent<GridSensorComponent<40, 40>>(Vector2(16, 16));

        // Called when:
        //  * Collecting input to make a decision
        //  * Adding a training sample
        nn->collectInput = [=](PlayerModelInput& input)
        {
            input.velocity = rigidBody->GetVelocity();
            gridSensor->Read(input.grid);
        };

        // Called when the decider makes a decision
        nn->receiveDecision = [=](PlayerDecision& decision)
        {

        };

        // Collects what decision the player made
        nn->collectDecision = [=](PlayerDecision& outDecision)
        {
            outDecision.action = PlayerAction::Down;
        };
    }
#endif
}

void PlayerEntity::ReceiveServerEvent(const IEntityEvent& ev)
{
    if (auto outOfHealth = ev.Is<OutOfHealthEvent>())
    {
        Die(outOfHealth);
    }
}

void PlayerEntity::Die(const OutOfHealthEvent* outOfHealth)
{
    Destroy();

    auto& selfName = scene->replicationManager->GetClient(
        outOfHealth->killer->GetComponent<NetComponent>()->ownerClientId).clientName;
    auto& otherName = scene->replicationManager->GetClient(playerId).clientName;

    scene->SendEvent(BroadcastToClientMessage(selfName + " killed " + otherName + "'s bot!"));

    for (auto spawn : scene->GetService<InputService>()->spawns)
    {
        if (spawn->playerId == playerId)
        {
            auto server = GetEngine()->GetServerGame();

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
