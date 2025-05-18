Steel Hunters/Tech & Tools/How to - Tech /How to - Connect to Local Dedicated Server Without Backend
https://confluence.wargaming.net/pages/viewpage.action?pageId=2021328376
a_zvyagintsev

# Connecting to Local Dedicated Server Without Backend
1. Step: 1; Copy the `LocalDeploy.json` file to the root directory of the dedicated server.
2. Step: 2; Copy the `LocalDeploy.json` file to any folder on your computer.
3. Step: 3; Copy the path to `LocalDeploy.json` and paste it into the `startserver.bat` file after `localdeploy=`.
4. Step: 4; Copy the `LocalDeploy.json` file to the root directory of the client. (Only one client connection is supported using this method. For more info on connecting multiple clients, see [Connecting as Client](#connecting-as-client).)

## Launching Dedicated Server

To start a dedicated server, you need to pass two parameters to it. First is the name of the map you want to load, second is the special parameter `localdeploy=PathToConfig` where `PathToConfig` is the absolute path to the configuration file that the server needs to properly start a match. If you are using `startserver.bat` linked to this page, you only need to specify the path to your config file inside `startserver.bat`. By default, `startserver.bat` loads `TerraMagna_WC`. To change the map, simply replace this name with the map name you want to load.

## Awaiting Connections

After launching the server, it will eventually reach a stage where it waits for client connections. The server assigns a `PlayerId` for each player that it waits for. This is just a simple number counting from 0 and incrementing for each player. So if the server waits for two players, their ids will be 0 and 1. To make the server aware of the number of players to wait for, you need to create and set up a configuration file. It can have any name but is required to have a `.json` extension. This `.json` file can be anywhere on your computer.

## Config

The `LocalDeploy.json` file looks like this:

```json
{
  "ConnectInfo": {
    "NumPlayersToWait": "2",
    "PlayerInfos": [
      {
        "TeamId": "0",
        "MechPreset": "Mech_Razor"
      },
      {
        "TeamId": "0",
        "MechPreset": "Mech_Fenris"
      }
    ]
  }
}
```

- `NumPlayersToWait`: The number of players expected to connect before the server starts the game. A total of 12 is supported.
- `PlayerInfos`: An array of JSON objects that contains information about each player. The first object of this array will belong to the player with id 0, the second to the player with id 1, etc.
- `TeamId`: The team id of this player. If you want players on the same team, their ids must be identical.
- `MechPreset`: The mech that will be given to this player. `MechPreset` should have the same value as the name of the `.uparam` file for this specific mech. Uparams for mechs can be found in the `/GameParams/uparams/Hunters` folder of your editor project directory.

The only mandatory field that you must specify is `NumPlayersToWait`. The rest are optional and if you don't specify them, the default values will be taken. Default values are the same as in the editor and can be found in `BP_R3GameInstance`. You can also partially specify the `PlayerInfos` array. E.g.,

```json
{
  "ConnectInfo": {
    "NumPlayersToWait": "2",
    "PlayerInfos": [
      {
        "TeamId": "0",
        "MechPreset": "Mech_Razor"
      }
    ]
  }
}
```

## Connecting as Client

For more information on connecting multiple clients, see the [Connecting as Client](#connecting-as-client) section.# Connecting to Server

There are three ways of connecting to a server:

## Using `startclient.bat`

1. Execute the provided `startclient.bat` script, and the client will start connecting. Note that only one connection is supported when using this method.

## Using Console

1. Run the `open` command in the console. You must specify the server's address (e.g., `127.0.0.1` for local).
2. To add options, paste a '?' (question mark) after the address without any whitespace. For example:
   - To connect as the first client: `open 127.0.0.1?PlayerId=0`
   - To connect as the second client: `open 127.0.0.1?PlayerId=1`
3. The order in which clients are connected to the server doesn't matter.
4. There is also an alias for joining the server with `PlayerId=0`, named `r3joinlocal`. Simply type it in the console, and the client will start connecting.

## Using Command Line

1. It is the same as using the console except that you don't need to type `open`. For example: `projectr3 127.0.0.1?PlayerId=0`

## Troubleshooting

If any issues arise while trying to connect, search for the `LogLocalDeploy` log category in the server log file. If everything is ok, a log will be printed giving some info about the config.

## Connecting from Editor to Server (Editor/Build)

1. Launch the map Lobby in net mode standalone and use command `r3joinlocal`.
2. You can connect to the server after that part in logs, but you have 100 sec to connect or the game will start with bots only.

## Starting Server from Editor

To start the server from the editor, you need to launch `UE3editor.exe` with parameters `-server`, maps, and the path to projects and localDeploy:

1. Start `UE4Editor.exe` with the following command (place it in `R3\unreal_engine\R3-4.27.2-d2a35e2a [engine version]\Engine\Binaries\Win64`):
   ```
   start "" UE4Editor.exe (path to ProjectR3.uproject) TerraMagnaLite -server -log localdeploy=(path to LocalDeploy.json)
   ```
   Example:
   ```
   start "" UE4Editor.exe C:\Source\R3\game\ProjectR3.uproject TerraMagnaLite -server -log localdeploy=C:\LocalConnect\LocalDeploy.json
   ``