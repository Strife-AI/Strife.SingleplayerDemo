#pragma once

#include "InputService.hpp"

DEFINE_RPC(MessageHudRpc)
{
    MessageHudRpc() = default;

    explicit MessageHudRpc(const std::string& message)
        : message(message)
    {

    }

    void Serialize(ReadWriteBitStream& stream)
    {
        stream.Add(message);
    }

    void Execute() override;

    std::string message;
};

DEFINE_EVENT(BroadcastToClientMessage)
{
    BroadcastToClientMessage(const std::string& message)
        : message(message)
    {

    }

    const std::string& message;
};

class MessageHud : public ISceneService
{
public:
    void AddStatus(const std::string& message);

private:
    static constexpr float ExpireTime = 10.0;

    void ReceiveEvent(const IEntityEvent& ev) override;

    CircularQueue<std::string, 5> _messages;
    float _expireTimer = 0;
};