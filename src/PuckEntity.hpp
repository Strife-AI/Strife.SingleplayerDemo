#pragma once

#include "PlayerEntity.hpp"
#include "Scene/BaseEntity.hpp"
#include "Components/RigidBodyComponent.hpp"

DEFINE_ENTITY(PuckEntity, "puck")
{
    void OnAdded() override;
    void Render(Renderer* renderer) override;
	void ReceiveEvent(const IEntityEvent& ev) override;

	RigidBodyComponent* rigidBody;
	PlayerEntity* player;
	Vector2 spawn;
};