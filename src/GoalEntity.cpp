#include "GoalEntity.hpp"

#include "PuckEntity.hpp"
#include "Components/RigidBodyComponent.hpp"
#include "Renderer/Renderer.hpp"
#include "Scene/IEntityEvent.hpp"

void GoalEntity::OnAdded()
{
	SetDimensions({ 100, 100 });
    auto rb = AddComponent<RigidBodyComponent>(b2_dynamicBody);
    rb->CreateBoxCollider(Dimensions(), true);
}

void GoalEntity::Render(Renderer* renderer)
{
    renderer->RenderRectangle(Bounds(), Color::Blue(), -1);
}

void GoalEntity::ReceiveEvent(const IEntityEvent& ev)
{
	if (auto contact = ev.Is<ContactBeginEvent>())
	{
		if (contact->OtherIs<PuckEntity>())
		{
			// do nothing for now
		}
	}
}
