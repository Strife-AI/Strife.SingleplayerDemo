
#include "InputService.hpp"
#include "Engine.hpp"
#include "PuckEntity.hpp"
#include "Memory/Util.hpp"
#include "Net/ReplicationManager.hpp"
#include "Renderer/Renderer.hpp"
#include "Tools/Console.hpp"
#include "GoalEntity.hpp"

InputButton g_quit = InputButton(SDL_SCANCODE_ESCAPE);
InputButton g_upButton(SDL_SCANCODE_W);
InputButton g_downButton(SDL_SCANCODE_S);
InputButton g_leftButton(SDL_SCANCODE_A);
InputButton g_rightButton(SDL_SCANCODE_D);

void InputService::ReceiveEvent(const IEntityEvent& ev)
{
	if (ev.Is<SceneLoadedEvent>())
	{
		scene->CreateEntity<GoalEntity>(Vector2(320, 200));
		
		auto player = scene->CreateEntity<PlayerEntity>(Vector2(320, 320));
		player->playerId = 0;
		scene->GetCameraFollower()->FollowEntity(player);

		auto position = Vector2(400, 400);
        auto puck = scene->CreateEntity<PuckEntity>(position);
		puck->spawn = position;
		puck->player = player;
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

// todo brendan clicking the bot should change its mode to receive data only, not try to control the player too
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
        	bool playerAssigned = false;
            for (auto player : players)
            {
                if (player->Bounds().ContainsPoint(scene->GetCamera()->ScreenToWorld(mouse->MousePosition()))
                    && player->playerId == 0)
                {
                    ReleaseActivePlayer();

                    activePlayer = player;
                    player->GetComponent<PlayerEntity::NeuralNetwork>()->mode = NeuralNetworkMode::CollectingSamples;

                    scene->GetCameraFollower()->FollowEntity(player);
                	playerAssigned = true;

                    break;
                }
            }

        	if (!playerAssigned)
        	{
                ReleaseActivePlayer();
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

void InputService::ReleaseActivePlayer()
{
	PlayerEntity* player;
	if (activePlayer.TryGetValue(player))
	{
		player->GetComponent<PlayerEntity::NeuralNetwork>()->mode = NeuralNetworkMode::ReinforcementLearning;
		activePlayer.Invalidate();
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
