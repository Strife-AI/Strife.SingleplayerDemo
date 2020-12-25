#include "HealthBarComponent.hpp"
#include "Renderer/Renderer.hpp"

void HealthBarComponent::Render(Renderer* renderer)
{
    Vector2 healthBarSize(32, 4);
    renderer->RenderRectangle(Rectangle(
        owner->Center() + offsetFromCenter - healthBarSize / 2,
        Vector2(healthBarSize.x * health.currentValue / maxHealth, healthBarSize.y)),Color::White(), -1);
}

void HealthBarComponent::TakeDamage(int amount, Entity* fromEntity)
{
    health.SetValue(Max(0, (int)health.currentValue - amount));

    if(health.Value() == 0)
    {
        owner->SendEvent(OutOfHealthEvent(fromEntity));
    }
}
