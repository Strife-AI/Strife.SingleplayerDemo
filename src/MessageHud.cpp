#include <Net/ReplicationManager.hpp>
#include "Renderer/Renderer.hpp"
#include "MessageHud.hpp"

void MessageHud::ReceiveEvent(const IEntityEvent& ev)
{
    if (ev.Is<UpdateEvent>())
    {
        _expireTimer -= scene->deltaTime;

        if (_expireTimer <= 0)
        {
            if (!_messages.IsEmpty())
            {
                _messages.Dequeue();
                _expireTimer = ExpireTime;
            }
        }
    }
    else if (auto renderEvent = ev.Is<RenderHudEvent>())
    {
        FontSettings font;
        font.spriteFont = ResourceManager::GetResource<SpriteFont>("console-font"_sid);
        font.scale = 1;

        auto height = font.spriteFont->CharacterDimension(font.scale).y;
        int i = 0;

        for (auto& message : _messages)
        {
            Vector2 messagePos(0, i * height);
            renderEvent->renderer->RenderRectangle(
                Rectangle(0, i * height, renderEvent->renderer->GetCamera()->ScreenSize().x, height),
                Color(64, 0, 128, 128),
                -0.99);
            renderEvent->renderer->RenderString(font, message.c_str(), messagePos, -1);
            ++i;
        }
    }
    else if (auto broadcastToClientMessage = ev.Is<BroadcastToClientMessage>())
    {
        auto server = scene->GetEngine()->GetServerGame();
        if (server != nullptr)
        {
            server->BroadcastRpc(MessageHudRpc(broadcastToClientMessage->message));
        }
    }
    else if (ev.Is<SceneLoadedEvent>())
    {
        if (!scene->isServer)
        {
            scene->GetEngine()->GetClientGame()->rpcManager.Register<MessageHudRpc>();
        }
    }
    else if (auto updatedInfo = ev.Is<PlayerInfoUpdatedEvent>())
    {
        AddStatus(scene->replicationManager->GetClient(updatedInfo->clientId).clientName + " connected");
    }
}

void MessageHud::AddStatus(const std::string& message)
{
    if (_messages.IsFull())
    {
        _messages.Dequeue();
        _expireTimer = ExpireTime;
    }
    else if (_messages.IsEmpty())
    {
        _expireTimer = ExpireTime;
    }

    _messages.Enqueue(message);
}

void MessageHudRpc::Execute()
{
    engine->GetClientGame()->GetScene()->GetService<MessageHud>()->AddStatus(message);
}
