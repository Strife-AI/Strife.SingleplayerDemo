#include <SDL2/SDL.h>

#include "Engine.hpp"
#include "InputService.hpp"
#include "PlayerEntity.hpp"
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
            .SetWindowCaption("Strife Singleplayer Demo")
            .SetGameName("Strife Singleplayer Demo")
            .ExecuteUserConfig("user.cfg")
            .SetProjectFile("../assets/SingleplayerDemo.sfProj")
            .EnableDevConsole("console-font");

        auto resourceManager = ResourceManager::GetInstance();
        resourceManager->SetBaseAssetPath("../assets");
        resourceManager->LoadResourceFromFile("Sprites/Spritesheets/font.sfnt", "console-font", ".sfnt");
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
        auto map = "erebor";
        auto engine = GetEngine();

        auto neuralNetworkManager = engine->GetNeuralNetworkManager();

        // Create networks
        {
            auto playerDecider = neuralNetworkManager->CreateDecider<PlayerDecider>();
            auto playerTrainer = neuralNetworkManager->CreateTrainer<PlayerTrainer>(engine->GetMetricsManager()->GetOrCreateMetric("loss"));

            neuralNetworkManager->CreateNetwork("nn", playerDecider, playerTrainer);
        }

        // Add types of objects the sensors can pick up
        {
            SensorObjectDefinition sensorDefinition;
            sensorDefinition.Add<PlayerEntity>(1).SetColor(Color::Red()).SetPriority(1);
            sensorDefinition.Add<TilemapEntity>(2).SetColor(Color::Gray()).SetPriority(0);

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
