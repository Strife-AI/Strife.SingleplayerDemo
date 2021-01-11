
#include "InputService.hpp"

#include <slikenet/BitStream.h>
#include <slikenet/peerinterface.h>

#include "Engine.hpp"
#include "PuckEntity.hpp"
#include "Components/RigidBodyComponent.hpp"
#include "Memory/Util.hpp"
#include "Net/NetworkPhysics.hpp"
#include "Net/ReplicationManager.hpp"
#include "Physics/PathFinding.hpp"
#include "Renderer/Renderer.hpp"
#include "Tools/Console.hpp"

#include "CastleEntity.hpp"

InputButton g_quit = InputButton(SDL_SCANCODE_ESCAPE);
InputButton g_upButton(SDL_SCANCODE_W);
InputButton g_downButton(SDL_SCANCODE_S);
InputButton g_leftButton(SDL_SCANCODE_A);
InputButton g_rightButton(SDL_SCANCODE_D);

void InputService::ReceiveEvent(const IEntityEvent& ev)
{
    if (ev.Is<SceneLoadedEvent>())
    {
        auto spawnPoints = scene->GetEntitiesOfType<CastleEntity>();

        for (int i = 0; i < spawnPoints.size(); ++i)
        {
            SpawnPlayer(spawnPoints[i], i);
        }
    }
    if (ev.Is<UpdateEvent>())
    {
        HandleInput();
    }
    else if (auto renderEvent = ev.Is<RenderEvent>())
    {
        Render(renderEvent->renderer);
    }
}

void InputService::HandleInput()
{
    if (g_quit.IsPressed())
    {
        scene->GetEngine()->QuitGame();
    }

    if (!scene->isServer)
    {
        if (scene->deltaTime == 0)
        {
            return;
        }

        auto mouse = scene->GetEngine()->GetInput()->GetMouse();

        if (mouse->LeftPressed())
        {
            for (auto player : players)
            {
                if (player->Bounds().ContainsPoint(scene->GetCamera()->ScreenToWorld(mouse->MousePosition()))
                    && player->playerId == 0)
                {
                    PlayerEntity* oldPlayer;
                    if (activePlayer.TryGetValue(oldPlayer))
                    {
                        oldPlayer->GetComponent<PlayerEntity::NeuralNetwork>()->mode = NeuralNetworkMode::Deciding;
                    }

                    activePlayer = player;
                    player->GetComponent<PlayerEntity::NeuralNetwork>()->mode = NeuralNetworkMode::CollectingSamples;

                    scene->GetCameraFollower()->FollowEntity(player);

                    break;
                }
            }
        }

        PlayerEntity* self;
        if (activePlayer.TryGetValue(self))
        {
            auto direction = GetInputDirection();
            self->SetMoveDirection(MoveDirectionToVector2(direction) * 200);
            self->lastDirection = direction;

            if (mouse->RightPressed())
            {
                for (auto entity : scene->GetEntities())
                {
                    if (entity->GetComponent<HealthBarComponent>(false) == nullptr)
                    {
                        continue;
                    }

                    if (entity->Bounds().ContainsPoint(scene->GetCamera()->ScreenToWorld(mouse->MousePosition())))
                    {
                        self->Attack(entity);
                        break;
                    }
                }
            }
        }
    }
}

void InputService::Render(Renderer* renderer)
{
    PlayerEntity* currentPlayer;
    if (activePlayer.TryGetValue(currentPlayer))
    {
        renderer->RenderRectangleOutline(currentPlayer->Bounds(), Color::White(), -1);
    }
}

MoveDirection InputService::GetInputDirection()
{
    Vector2 inputDir;
    if (g_leftButton.IsDown()) --inputDir.x;
    if (g_rightButton.IsDown()) ++inputDir.x;
    if (g_upButton.IsDown()) --inputDir.y;
    if (g_downButton.IsDown()) ++inputDir.y;

    return GetClosestMoveDirection(inputDir);
}

void InputService::SpawnPlayer(CastleEntity* spawn, int playerId)
{
    spawn->playerId = playerId;

    for (int i = 0; i < 4; ++i)
    {
        spawn->SpawnPlayer();
    }

    spawns.push_back(spawn);

    if (playerId == 0)
    {
        scene->GetCameraFollower()->CenterOn(spawn->Center());
    }
}
