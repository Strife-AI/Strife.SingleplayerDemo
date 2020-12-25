
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
#include "MessageHud.hpp"

#include "CastleEntity.hpp"

InputButton g_quit = InputButton(SDL_SCANCODE_ESCAPE);
InputButton g_upButton(SDL_SCANCODE_W);
InputButton g_downButton(SDL_SCANCODE_S);
InputButton g_leftButton(SDL_SCANCODE_A);
InputButton g_rightButton(SDL_SCANCODE_D);

DEFINE_COMMAND(MoveToCommand)
{
    void DoSerialize(ReadWriteBitStream& stream) override
    {
        stream.Add(position).Add(netId);
    }

    Vector2 position;
    uint32_t netId;
};

DEFINE_COMMAND(AttackCommand)
{
    void DoSerialize(ReadWriteBitStream& stream) override
    {
        stream.Add(netId).Add(attackNetId);
    }

    uint32_t netId;
    uint32_t attackNetId;
};

void InputService::OnAdded()
{
    auto& handler = scene->replicationManager->playerCommandHandler;
    ReplicationManager* replicationManager = scene->replicationManager;

    handler.RegisterCommandType<MoveToCommand>(1, [=](const MoveToCommand& command)
    {
        auto entity = replicationManager->GetEntityByNetId(command.netId);
        PlayerEntity* player;
        if (entity != nullptr && entity->Is(player))
        {
            player->MoveTo(command.position);
        }
    });

    handler.RegisterCommandType<AttackCommand>(2, [=](const AttackCommand& command)
    {
        auto entity = replicationManager->GetEntityByNetId(command.netId);
        PlayerEntity* player;
        if (entity != nullptr && entity->Is(player))
        {
            auto attack = replicationManager->GetEntityByNetId(command.attackNetId);

            if (attack != nullptr)
            {
                player->Attack(attack);
            }
        }
    });
}


void InputService::ReceiveEvent(const IEntityEvent& ev)
{
    if (ev.Is<SceneLoadedEvent>())
    {
        if (scene->isServer)
        {
            for (auto spawn : scene->GetEntitiesOfType<CastleEntity>())
            {
                spawnPositions.push_back(spawn->Center());
                spawn->Destroy();
            }

            spawnPositions.push_back({ 1000, 1000 });
        }
    }
    if (ev.Is<UpdateEvent>())
    {
        if (scene->isServer && !gameOver)
        {
            bool multipleClientsConnected = scene->GetEngine()->GetServerGame()->TotalConnectedClients() >= 2;

            if (multipleClientsConnected)
            {
                auto spawnsLeft = scene->GetEntitiesOfType<CastleEntity>();

                if (spawnsLeft.size() == 0)
                {
                    scene->SendEvent(BroadcastToClientMessage("Draw!"));
                    gameOver = true;
                }
                else
                {
                    if (spawnsLeft.size() == 1)
                    {
                        gameOver = true;
                        auto& name = scene->replicationManager->GetClient(players[0]->net->ownerClientId).clientName;
                        scene->SendEvent(BroadcastToClientMessage(name + " wins!"));
                    }
                }
            }
        }

        HandleInput();
    }
    else if (auto renderEvent = ev.Is<RenderEvent>())
    {
        Render(renderEvent->renderer);
    }
    else if (auto joinedServerEvent = ev.Is<JoinedServerEvent>())
    {
        scene->replicationManager->localClientId = joinedServerEvent->selfId;
    }
    else if (auto connectedEvent = ev.Is<PlayerConnectedEvent>())
    {
        if (scene->isServer)
        {
            auto spawnPoint = spawnPositions[spawnPositions.size() - 1];
            spawnPositions.pop_back();

            auto spawn = scene->CreateEntity<CastleEntity>(spawnPoint);

            spawn->net->ownerClientId = connectedEvent->id;

            for (int i = 0; i < 4; ++i)
            {
                spawn->SpawnPlayer();
            }

            spawns.push_back(spawn);
        }
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
                    && player->net->ownerClientId == scene->replicationManager->localClientId)
                {
                    PlayerEntity* oldPlayer;
                    if (activePlayer.TryGetValue(oldPlayer))
                    {
                        oldPlayer->GetComponent<PlayerEntity::NeuralNetwork>()->mode = NeuralNetworkMode::Deciding;
                    }

                    activePlayer = player;
                    player->GetComponent<PlayerEntity::NeuralNetwork>()->mode = NeuralNetworkMode::CollectingSamples;

                    break;
                }
            }
        }

        PlayerEntity* self;
        if (activePlayer.TryGetValue(self))
        {
            if (mouse->RightPressed())
            {
                bool attack = false;
                for (auto entity : scene->GetEntities())
                {
                    auto netComponent = entity->GetComponent<NetComponent>(false);

                    if (netComponent == nullptr)
                    {
                        continue;
                    }

                    if (entity->GetComponent<HealthBarComponent>(false) == nullptr)
                    {
                        continue;
                    }

                    if (entity->Bounds().ContainsPoint(scene->GetCamera()->ScreenToWorld(mouse->MousePosition())))
                    {
                        AttackCommand command;
                        command.netId = self->net->netId;
                        command.attackNetId = netComponent->netId;
                        scene->replicationManager->playerCommandHandler.AddCommand(command);

                        attack = true;
                        break;
                    }
                }

                if (!attack)
                {
                    MoveToCommand command;
                    command.position = scene->GetCamera()->ScreenToWorld(mouse->MousePosition());
                    command.netId = self->net->netId;
                    scene->replicationManager->playerCommandHandler.AddCommand(command);
                }
            }
        }

        scene->replicationManager->Client_SendUpdateRequest(scene->deltaTime, scene->GetEngine()->GetClientGame());
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
