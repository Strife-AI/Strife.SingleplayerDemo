# Strife.SingleplayerDemo

![MadeWithStrifeEngine_BG](https://user-images.githubusercontent.com/7697514/112562676-35b7dc80-8dae-11eb-839f-78d6cf3826d9.png)

Sample single-player game built using the [Strife Engine](https://github.com/Strife-AI/Strife.Engine)
feel free to use this as a template if you're interested in using the Strife.Engine for your game /
machine-learning project. This starter project is provided under a [modified UIUC/NCSA license](https://github.com/Strife-AI/Strife.Engine/blob/master/LICENSE.txt).

[Join us on Discord!](https://discord.gg/rNrKTKY)

### Getting Started
Clone the repo (replace the repo url below with your own if you're mirroring the repo):
```shell
git clone --recurse-submodules -j8 git@github.com:Strife-AI/Strife.SingleplayerDemo.git
```

Run git lfs and fetch assets
```shell
cd Strife.SingleplayerDemo
git lfs install && git lfs fetch --all
``` 

### Visual Studio Instructions (Windows)
Prerequisite: In the Visual Studio installer, check "Desktop Development with C++".
1. Open Visual Studio
1. Open the project with File → Open Folder → `Strife.SinglePlayerDemo`
1. CMake will automatically start running but needs to be cancelled because the default will use the ninja cmake generator.
    1. Project → `Cancel CMake Cache Generation`
1. Configure cmake
    1. Click the dropdown by x64-Debug and open `Manage Configurations...`
    1. Under x64-Debug, click `Show advanced settings`
    1. Then under Cmake generator dropdown, select `Visual Studio 16 2019 Win64`
    1. Repeat the above steps for x64-RelWithDebInfo
1.  Run cmake
    1. Project → `Generate Cache`.
    1. The cmake build will download all the dependencies including PyTorch, so this could take upwards of 20 minutes.  Vcpkg will cache most of its work so future builds will be much faster.

### Non-Visual Studio - Generating Makefiles
Our team uses CLion, we can guarantee that the engine and the demo game build successfully with it.
That being said, CMake is used as the buildsystem, so any IDE with CMake support should suffice.

#### Using CMake in the command line
Create a build directory:
```shell
mkdir build && cd build
```

Generate makefiles:
```shell
cmake ..
```

* Add a `-G` flag followed by a supported project generator, for example. If generating an Xcode project, do:
```shell
cmake -G Xcode .. <CMAKE_VARIABLES>
```

Build project:
```shell
make
```

#### Using CLion
Open the root `CMakeLists.txt` as a project in CLion, then navigate to CLion
settings (File → Settings or CLion → Preferences on Mac).

Once updated, CLion will automatically attempt to generate the CMakeCache files.

---
After that, the game should successfully compile!

Feel free to join our [engine discussion on Discord](https://discord.gg/544ctNNHzD) 
if you have any questions or feedback.
