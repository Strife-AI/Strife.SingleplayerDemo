#pragma once

#include "Scene/EntityComponent.hpp"
#include "Net/SyncVar.hpp"
#include "Scene/IEntityEvent.hpp"

DEFINE_EVENT(OutOfHealthEvent)
{
    OutOfHealthEvent(Entity* killer)
        : killer(killer)
    {

    }

    Entity* killer;
};

DEFINE_COMPONENT(HealthBarComponent)
{
    void Render(Renderer* renderer) override;
    void TakeDamage(int amount, Entity* fromEntity = nullptr);

    SyncVar<uint16_t> health {100 };
    Vector2 offsetFromCenter;
    int maxHealth = 100;
};