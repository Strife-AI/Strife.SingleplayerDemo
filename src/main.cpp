#include <SDL2/SDL.h>

#include "Engine.hpp"
#include "InputService.hpp"
#include "PlayerEntity.hpp"
#include "Net/NetworkPhysics.hpp"
#include "Scene/IGame.hpp"
#include "Scene/Scene.hpp"
#include "Scene/TilemapEntity.hpp"
#include "Tools/Console.hpp"
#include "MessageHud.hpp"

ConsoleVar<int> g_serverPort("port", 60001);
extern ConsoleVar<bool> g_isServer;

struct Game : IGame
{
    void ConfigureGame(GameConfig& config) override
    {
        config
            .SetDefaultScene("erebor"_sid)
            .SetWindowCaption("Breakout")
            .SetGameName("breakout")
            .ExecuteUserConfig("user.cfg")
            .EnableDevConsole("console-font")
            .AddResourcePack("MultiplayerDemo.x2rp");
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
            scene->AddService<MessageHud>();

            //scene->GetEngine()->GetConsole()->Execute("light 0");
        }
    }

    void OnGameStart() override
    {
        auto map = "erebor"_sid;
        auto engine = GetEngine();

        if(!g_isServer.Value())
        {
            engine->StartLocalServer(g_serverPort.Value(), map);
        }
        else
        {
            engine->StartServer(g_serverPort.Value(), map);
        }

        auto neuralNetworkManager = engine->GetNeuralNetworkManager();

        // Create networks
        {
            auto playerDecider = neuralNetworkManager->CreateDecider<PlayerDecider>();
            auto playerTrainer = neuralNetworkManager->CreateTrainer<PlayerTrainer>();

            neuralNetworkManager->CreateNetwork("nn", playerDecider, playerTrainer);
        }

        // Add types of objects the sensors can pick up
        {
            SensorObjectDefinition sensorDefinition;
            sensorDefinition.Add<PlayerEntity>(1).SetColor(Color::Red()).SetPriority(1);
            sensorDefinition.Add<TilemapEntity>(2).SetColor(Color::Gray()).SetPriority(0);

            neuralNetworkManager->SetSensorObjectDefinition(sensorDefinition);
        }
    }

    std::string initialConsoleCmd;
};

void UploadToServer(ConsoleCommandBinder& binder)
{
    std::string fileName;
    binder
        .Bind(fileName, "fileName")
        .Help("Uploads a file to the server");

    auto client = binder.GetEngine()->GetClientGame();

    if (client != nullptr)
    {
        bool successfullyStarted = client->fileTransferService.TryUploadFile(fileName, client->serverAddress);
        if (!successfullyStarted)
        {
            Log("Failed to initiate file transfer for file %s\n", fileName.c_str());
        }
    }
}

ConsoleCmd g_upload("upload", UploadToServer);

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
