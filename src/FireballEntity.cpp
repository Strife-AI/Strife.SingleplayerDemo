#include "FireballEntity.hpp"
#include "Components/RigidBodyComponent.hpp"
#include "Components/NetComponent.hpp"
#include "HealthBarComponent.hpp"

void FireballEntity::OnAdded()
{
    auto rb = AddComponent<RigidBodyComponent>(b2_dynamicBody);
    rb->CreateCircleCollider(Radius, true);

    AddComponent<NetComponent>();

    light.color = Color::Orange();
    light.maxDistance = Radius;
    light.intensity = 2;

    if (!scene->isServer) scene->GetLightManager()->AddLight(&light);

    rb->SetVelocity(velocity);
}

void FireballEntity::OnDestroyed()
{
    if (!scene->isServer) scene->GetLightManager()->RemoveLight(&light);
}

void FireballEntity::ReceiveServerEvent(const IEntityEvent& ev)
{
    if (auto contactBegin = ev.Is<ContactBeginEvent>())
    {
        auto other = contactBegin->other.OwningEntity();
        if (other->id == ownerId) return;
        if (contactBegin->other.IsTrigger()) return;

        auto healthBar = other->GetComponent<HealthBarComponent>(false);

        if (healthBar != nullptr)
        {
            healthBar->TakeDamage(5, this);
        }

        Destroy();
    }
}

void FireballEntity::Update(float deltaTime)
{
    light.position = Center();
}
