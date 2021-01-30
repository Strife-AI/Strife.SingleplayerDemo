#pragma once

#include "Scene/BaseEntity.hpp"

DEFINE_ENTITY(GoalEntity, "goal")
{
    void OnAdded() override;
    void Render(Renderer* renderer) override;
	void ReceiveEvent(const IEntityEvent& ev) override;
};