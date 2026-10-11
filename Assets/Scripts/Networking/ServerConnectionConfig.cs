using System;
using System.IO;
using UnityEngine;

/// <summary>
/// Host/port settings for the two PC services. Edited on the device in
/// persistentDataPath/server_connections.json, which is created with these defaults on first run.
/// </summary>
[Serializable]
public class ServerConnectionSettings
{
    public string detectionServerHost = "localhost";
    public int detectionServerUdpPort = 8001;

    public string bundleCompilerHost = "localhost";
    public int bundleCompilerPort = 8765;

    public string BundleCompilerUri => $"ws://{bundleCompilerHost}:{bundleCompilerPort}/";
}

/// <summary>
/// Loads <see cref="ServerConnectionSettings"/> from the device and exposes them as a singleton.
/// </summary>
public class ServerConnectionConfig : MonoBehaviour
{
    private const string ConfigFileName = "server_connections.json";

    [Tooltip("Written to the device config on first run; edit the file on the device afterwards.")]
    [SerializeField] private ServerConnectionSettings defaults = new();

    [SerializeField] private Logger logger;

    public static ServerConnectionConfig Instance { get; private set; }

    public ServerConnectionSettings Settings { get; private set; }

    public string BundleCompilerUri => Settings.BundleCompilerUri;

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(this);
            return;
        }

        Instance = this;
        Settings = DeviceConfigStore.LoadOrCreate(
            DeviceConfigStore.ResolvePath(ConfigFileName), logger, nameof(ServerConnectionConfig), defaults);

        logger?.LogInfo(
            $"Servers: detection udp://{Settings.detectionServerHost}:{Settings.detectionServerUdpPort}, " +
            $"bundle compiler {Settings.BundleCompilerUri}", nameof(ServerConnectionConfig));
    }

    private void OnDestroy()
    {
        if (Instance == this) Instance = null;
    }
}
