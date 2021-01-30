#include "PuckEntity.hpp"

#include "GoalEntity.hpp"
#include "Renderer/Renderer.hpp"
#include "Scene/IEntityEvent.hpp"

void PuckEntity::OnAdded()
{
	SetDimensions({ 30, 30 });
    rigidBody = AddComponent<RigidBodyComponent>(b2_dynamicBody);
    auto box = rigidBody->CreateBoxCollider(Dimensions());

    box->SetFriction(1);
    box->SetRestitution(1);
    box->SetDensity(10);
    rigidBody->body->ResetMassData();
}

void PuckEntity::Render(Renderer* renderer)
{
    renderer->RenderRectangle(Bounds(), Color::Red(), -1);
}

void PuckEntity::ReceiveEvent(const IEntityEvent& ev)
{
	if (auto contact = ev.Is<ContactBeginEvent>())
	{
		if (contact->OtherIs<GoalEntity>())
		{
			//player->SendEvent(RewardEvent(RewardType::ScoreGoal));
			rigidBody->SetVelocity({ 0, 0 });
			SetCenter(spawn);
		}
	}
}
