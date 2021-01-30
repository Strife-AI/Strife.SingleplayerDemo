#include <SDL2/SDL.h>

#include "Engine.hpp"
#include "GoalEntity.hpp"
#include "InputService.hpp"
#include "PlayerEntity.hpp"
#include "PuckEntity.hpp"
#include "GameML/DDQN.hpp"
#include "Net/NetworkPhysics.hpp"
#include "Scene/IGame.hpp"
#include "Scene/Scene.hpp"
#include "Scene/TilemapEntity.hpp"
#include "Tools/Console.hpp"

ConsoleVar<int> g_serverPort("port", 60001);
extern ConsoleVar<bool> g_isServer;

struct Game : IGame
{
    void ConfigureGame(GameConfig& config) override
    {
        config
            .SetDefaultScene("erebor"_sid)
            .SetWindowCaption("Soccer")
            .SetGameName("soccer")
            .ExecuteUserConfig("user.cfg")
            .EnableDevConsole("console-font");

        auto resourceManager = ResourceManager::GetInstance();
        resourceManager->SetBaseAssetPath("../assets");
        resourceManager->LoadResourceFromFile("Sprites/castle.png", "castle");
        resourceManager->LoadResourceFromFile("Tilemaps/Soccer.tmx", "soccer");
        resourceManager->LoadResourceFromFile("Sprites/Spritesheets/font.png", "console-font", ".sfnt");
    }

    void ConfigureEngine(EngineConfig& config) override
    {
        config.initialConsoleCmd = initialConsoleCmd;
    }

    void BuildScene(Scene* scene) override
    {
        if (scene->MapSegmentName() != "empty-map"_sid)
        {
            scene->AddService<InputService>();
            scene->AddService<NetworkPhysics>(scene->isServer);
        }
    }

    void OnGameStart() override
    {
        auto map = "soccer";
        auto engine = GetEngine();

        auto neuralNetworkManager = engine->GetNeuralNetworkManager();

        // Create networks
        {
            auto decider = neuralNetworkManager->CreateDecider<DQNDecider>();
            auto trainer = neuralNetworkManager->CreateTrainer<DQNTrainer>(engine->GetMetricsManager()->GetOrCreateMetric("loss"), 10);

            neuralNetworkManager->CreateNetwork("nn", decider, trainer);
        }

        // Add types of objects the sensors can pick up
        {
            SensorObjectDefinition sensorDefinition;
            sensorDefinition.Add<PuckEntity>(1).SetColor(Color::Red()).SetPriority(3);
            sensorDefinition.Add<GoalEntity>(2).SetColor(Color::Blue()).SetPriority(2);
            sensorDefinition.Add<PlayerEntity>(3).SetColor(Color::Green()).SetPriority(1);
            sensorDefinition.Add<TilemapEntity>(4).SetColor(Color::Gray()).SetPriority(0);

            neuralNetworkManager->SetSensorObjectDefinition(sensorDefinition);
        }

        engine->StartSinglePlayerGame(map);
    }

    std::string initialConsoleCmd;
};

int main(int argc, char* argv[])
{
    Game game;

    if (argc >= 2)
    {
        game.initialConsoleCmd = argv[1];
    }

    game.Run();

    return 0;
}
