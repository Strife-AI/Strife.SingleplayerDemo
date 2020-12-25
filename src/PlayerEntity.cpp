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

    net = AddComponent<NetComponent>();

    scene->SendEvent(PlayerAddedToGame(this));

    scene->GetService<InputService>()->players.push_back(this);

    // Setup network and sensors
    {
        auto nn = AddComponent<NeuralNetworkComponent<PlayerNetwork>>();
        nn->SetNetwork("nn");

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
    auto& otherName = scene->replicationManager->GetClient(net->ownerClientId).clientName;

    scene->SendEvent(BroadcastToClientMessage(selfName + " killed " + otherName + "'s bot!"));

    for (auto spawn : scene->GetService<InputService>()->spawns)
    {
        if (spawn->net->ownerClientId == net->ownerClientId)
        {
            auto server = GetEngine()->GetServerGame();

            spawn->StartTimer(10, [=]
            {
                spawn->SpawnPlayer();
                server->ExecuteRpc(spawn->net->ownerClientId, MessageHudRpc("Your bot has respawned at your base"));
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

        auto color = c[net->ownerClientId];
        renderer->RenderRectangle(Rectangle(position - Dimensions() / 2, Dimensions()), color, -0.99);
    }

    // Render name
    {
        FontSettings font;
        font.spriteFont = ResourceManager::GetResource<SpriteFont>("console-font"_sid);
        font.scale = 0.75;

        auto& name = scene->replicationManager->GetClient(net->ownerClientId).clientName;
        Vector2 size = font.spriteFont->MeasureStringWithNewlines(name.c_str(), 0.75).AsVectorOfType<float>();
        renderer->RenderString(font, name.c_str(), Center() - Vector2(0, 32) - size / 2, -1);
    }
}

void PlayerEntity::ServerFixedUpdate(float deltaTime)
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
                    fireball->GetComponent<NetComponent>()->ownerClientId = net->ownerClientId;
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

void PlayerEntity::MoveTo(Vector2 position)
{
    pathFollower->SetTarget(position);
    state = PlayerState::Moving;
}

void PlayerEntity::Attack(Entity* entity)
{
    attackTarget = entity;
    state = PlayerState::Attacking;
    pathFollower->FollowEntity(entity, 200);
}
