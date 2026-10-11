using System;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Networking
{
    /// <summary>
    /// Sends a generated shader to the bundle compiler (QuestPackager) over WebSocket and stores
    /// the compiled Android AssetBundle in the shader's history session.
    /// </summary>
    public class ShaderBundleWebSocketClient : MonoBehaviour
    {
        private const int BufferSize = 8192;

        [Tooltip("Seconds to wait for the compiled bundle. A cold Unity batchmode build takes a while, " +
                 "but an unreachable or stuck compiler must still end in an error instead of silence.")]
        [SerializeField] private int bundleTimeoutSeconds = 300;

        [SerializeField] private Logger logger;

        [Header("References")]
        [SerializeField] private LLMShaderGenerator shaderGenerator;

        private string ServerUri => ServerConnectionConfig.Instance.BundleCompilerUri;

        /// <summary>
        /// Event fired when an asset bundle has been received and saved.
        /// </summary>
        public event Action OnBundleReceived;

        private void OnEnable()
        {
            if (shaderGenerator != null)
            {
                shaderGenerator.OnShaderGenerated += SendShaderAndReceiveBundle;
            }
            else
            {
                logger?.LogWarning("LLMShaderGenerator reference not set in ShaderBundleWebSocketClient.", nameof(ShaderBundleWebSocketClient));
            }
        }

        private void OnDisable()
        {
            if (shaderGenerator != null)
            {
                shaderGenerator.OnShaderGenerated -= SendShaderAndReceiveBundle;
            }
        }

        /// <summary>
        /// Sends the session's shader file to the compiler and saves the returned bundle into
        /// the same session.
        /// </summary>
        public async void SendShaderAndReceiveBundle(ShaderHistorySession session)
        {
            string shaderPath = session.ShaderFilePath;

            if (!File.Exists(shaderPath))
            {
                logger?.LogError($"Shader file not found: {shaderPath}", nameof(ShaderBundleWebSocketClient));
                return;
            }

            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(bundleTimeoutSeconds));
            try
            {
                using var ws = new ClientWebSocket();
                logger?.LogInfo($"Connecting to {ServerUri}...", nameof(ShaderBundleWebSocketClient));
                logger?.SetStatus("Compiling shader...", busy: true);
                await ws.ConnectAsync(new Uri(ServerUri), timeout.Token);

                PipelineBenchmarkLogger.Instance?.RecordTimestamp("script_sent_to_quest_packer");
                await SendShaderFile(ws, shaderPath, timeout.Token);
                await ReceiveBundle(ws, session, timeout.Token);

                await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Done", timeout.Token);
            }
            catch (OperationCanceledException)
            {
                logger?.LogError($"Bundle compiler at {ServerUri} did not deliver within {bundleTimeoutSeconds} s.",
                    nameof(ShaderBundleWebSocketClient));
            }
            catch (Exception ex)
            {
                logger?.LogError(ex.ToString(), nameof(ShaderBundleWebSocketClient));
            }
        }

        private async Task SendShaderFile(ClientWebSocket ws, string shaderPath, CancellationToken token)
        {
            var nameBytes = Encoding.UTF8.GetBytes(Path.GetFileName(shaderPath));
            await ws.SendAsync(new ArraySegment<byte>(nameBytes), WebSocketMessageType.Text, true, token);

            byte[] shaderBytes = File.ReadAllBytes(shaderPath);
            await ws.SendAsync(new ArraySegment<byte>(shaderBytes), WebSocketMessageType.Binary, true, token);

            logger?.LogInfo($"Shader file sent: {shaderPath}", nameof(ShaderBundleWebSocketClient));
        }

        private async Task ReceiveBundle(ClientWebSocket ws, ShaderHistorySession session, CancellationToken token)
        {
            using var ms = new MemoryStream();
            var buffer = new byte[BufferSize];
            while (true)
            {
                var result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), token);

                // The compiler signals every server-side failure (Unity build failed, bundle not at
                // the expected path) by closing the socket without sending a byte. Treating that
                // close frame as a zero-length message would save an empty, unloadable bundle.
                if (result.MessageType == WebSocketMessageType.Close)
                {
                    logger?.LogError(
                        $"Bundle server closed without sending a bundle ({result.CloseStatus}: " +
                        $"{result.CloseStatusDescription}) — the compile failed on the server. " +
                        "Check the QuestPackager console.", nameof(ShaderBundleWebSocketClient));
                    return;
                }

                ms.Write(buffer, 0, result.Count);

                if (result.EndOfMessage)
                    break;
            }

            byte[] bundleData = ms.ToArray();
            if (bundleData.Length == 0)
            {
                logger?.LogError("Bundle server sent an EMPTY bundle — not saving it. " +
                                 "Check the QuestPackager console.", nameof(ShaderBundleWebSocketClient));
                return;
            }

            await session.SaveBundleAsync(bundleData);
            logger?.LogInfo($"Asset bundle received and saved to: {session.BundlePath}", nameof(ShaderBundleWebSocketClient));
            PipelineBenchmarkLogger.Instance?.RecordTimestamp("asset_bundle_received");

            OnBundleReceived?.Invoke();
        }
    }
}
