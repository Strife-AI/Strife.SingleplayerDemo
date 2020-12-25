#pragma once

#include "Scene/BaseEntity.hpp"

DEFINE_ENTITY(PuckEntity, "puck")
{
    void OnAdded() override;
    void Render(Renderer* renderer) override;

    void ChangeDirection();
};