# Strife.SingleplayerDemo

![MadeWithStrifeEngine_BG](https://user-images.githubusercontent.com/7697514/112562676-35b7dc80-8dae-11eb-839f-78d6cf3826d9.png)

Sample single-player game built using the [Strife Engine](https://github.com/Strife-AI/Strife.Engine)
feel free to use this as a template if you're interested in using the Strife.Engine for your game /
machine-learning project. This starter project is provided under a [modified UIUC/NCSA license](https://github.com/Strife-AI/Strife.Engine/blob/master/LICENSE.txt).

[Join us on Discord!](https://discord.gg/rNrKTKY)

## Setup
These instructions are written in mind to be used with the CLion IDE. The underlying build system is
CMake so any build environment will suffice.

### Prerequisites
* [Make sure you have all the prerequisites for the engine](https://github.com/Strife-AI/Strife.Engine#getting-started)
* Some assets are included with this repository, [git LFS](https://git-lfs.github.com) is required to get
  the sample environment working.
* [Install dependencies using vpckg](https://github.com/Strife-AI/Strife.Engine#setting-up-vcpkg)

### Getting Started
Clone the repo (replace the repo url below with your own if you're mirroring the repo):
```shell
git clone --recurse-submodules -j8 git@github.com:Strife-AI/Strife.SingleplayerDemo.git
```

Run git lfs and fetch assets
```shell
git lfs install && git lfs fetch --all
``` 

If you haven't done so already, unzip libtorch and store the unarchived files someplace useful.

* If you're on Windows, set up environment variables ([How to access them in Windows 10](https://www.wikihow.com/Create-an-Environment-Variable-in-Windows-10))
named `TORCH_DEBUG_DIR` and `TORCH_RELEASE_DIR` which point to the root 
directory of the debug and release versions of libtorch.
  * Once these are set, restart Windows to ensure the changes take place.

### Generating Makefiles
Our team uses CLion, we can guarantee that the engine and the demo game build successfully with it.
That being said, CMake is used as the buildsystem, so any IDE with CMake support should suffice.

#### Using CMake in the command line
Create a build directory:
```shell
mkdir build && cd build
```

Generate makefiles:
```shell
cmake .. -DCMAKE_TOOLCHAIN_FILE={Path/To/Vcpkg}/scripts/buildsystems/vcpkg.cmake
 -DCMAKE_PREFIX_PATH={Path/To/Libtorch}/share/cmake/Torch/
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

Under **Build Execution & Deployment**, select **CMake**. Set the **CMake 
Options** field to the CMake variables:
```shell
-DCMAKE_TOOLCHAIN_FILE={Path/To/Vcpkg}/scripts/buildsystems/vcpkg.cmake
-DCMAKE_PREFIX_PATH={Path/To/Libtorch}/share/cmake/Torch/
```

Once updated, CLion will automatically attempt to generate the CMakeCache files.

---
After that, the game should successfully compile!

Feel free to join our [engine discussion on Discord](https://discord.gg/544ctNNHzD) 
if you have any questions or feedback.
