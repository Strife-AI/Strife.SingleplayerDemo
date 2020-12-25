#pragma once

#include "Scene/BaseEntity.hpp"

DEFINE_ENTITY(FireballEntity, "fireball")
{
    static constexpr float Radius = 5;

    FireballEntity() = default;

    FireballEntity(Vector2 velocity)
        : velocity(velocity)
    {

    }

    void OnAdded() override;
    void OnDestroyed() override;
    void ReceiveServerEvent(const IEntityEvent& ev) override;
    void Update(float deltaTime) override;

    PointLight light;
    int ownerId;
    Vector2 velocity;
};