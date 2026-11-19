using System;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Threading;

namespace Networking
{
    /// <summary>
    /// Sends image frames to the detection server as chunked UDP datagrams and receives its
    /// single-datagram responses on a background thread.
    /// </summary>
    public class UdpImageTransmitter : IDisposable
    {
        // Reserved frame-id sentinels on the wire (vocabulary control = 0xFFFFFFFF, ack = 0xFFFFFFFE).
        private const uint MaxDataFrameId = 0xFFFFFFFDu;

        // Flow control (~30 FPS ceiling), measured with Stopwatch so it is thread-safe and frame-independent.
        private const double MinSendIntervalSeconds = 0.033;
        private static readonly long MinSendIntervalTicks =
            (long)(MinSendIntervalSeconds * Stopwatch.Frequency);

        // Chunking constants
        private const int MaxChunkSize = 60000;
        private const int TransportHeaderSize = 8; // FrameID(4) + ChunkIdx(2) + TotalChunks(2)

        private UdpClient udpClient;
        private IPEndPoint serverEndpoint;
        private Thread receiveThread;
        private volatile bool isRunning;
        private uint nextFrameId;
        private long lastSendTicks;

        // Reused send buffer (single-threaded send path) to avoid 30-60 KB GC allocations per frame.
        private readonly byte[] sendPacketBuffer = new byte[MaxChunkSize];

        // Callbacks
        public event Action<byte[]> OnDataReceived;
        public event Action<string> OnError;

        public bool IsConnected => udpClient != null && isRunning;

        /// <summary>
        /// True when the socket is connected and the flow-control interval since the last sent
        /// frame has elapsed, so callers can avoid capturing/encoding a frame that would be dropped.
        /// </summary>
        public bool CanSendFrame =>
            IsConnected && (lastSendTicks == 0 || Stopwatch.GetTimestamp() - lastSendTicks >= MinSendIntervalTicks);

        /// <summary>
        /// Connects to the specified UDP server.
        /// </summary>
        public void Connect(string host, int port)
        {
            try
            {
                IPAddress ipAddress = null;
                string trimmedHost = host?.Trim() ?? "";

                if (string.Equals(trimmedHost, "localhost", StringComparison.OrdinalIgnoreCase))
                {
                    ipAddress = IPAddress.Loopback;
                }
                else if (!IPAddress.TryParse(trimmedHost, out ipAddress))
                {
                    IPHostEntry entry = Dns.GetHostEntry(trimmedHost);
                    foreach (var ip in entry.AddressList)
                    {
                        if (ip.AddressFamily == AddressFamily.InterNetwork)
                        {
                            ipAddress = ip;
                            break;
                        }
                    }
                    if (ipAddress == null && entry.AddressList.Length > 0)
                    {
                        ipAddress = entry.AddressList[0];
                    }
                }

                if (ipAddress == null)
                    throw new Exception($"Could not resolve host: {host}");

                serverEndpoint = new IPEndPoint(ipAddress, port);
                udpClient = new UdpClient(ipAddress.AddressFamily);
                udpClient.Client.ReceiveTimeout = 1000;

                isRunning = true;
                receiveThread = new Thread(ReceiveLoop) { IsBackground = true, Name = "UdpDetectionReceive" };
                receiveThread.Start();

                UnityEngine.Debug.Log($"[UDP] Connected to {host}:{port}");
            }
            catch (Exception e)
            {
                OnError?.Invoke($"Connection error: {e.Message}");
            }
        }

        /// <summary>
        /// Sends a frame with optional metadata (fire-and-forget wrapper).
        /// </summary>
        public void SendFrame(byte[] imageBytes, byte[] metadata = null)
        {
            TrySendFrame(imageBytes, metadata, out _);
        }

        /// <summary>
        /// Sends a frame with optional metadata. Returns false when rate-limited, disconnected,
        /// or a socket error occurred, so callers never wait for a response to an unsent frame.
        /// </summary>
        public bool TrySendFrame(byte[] imageBytes, byte[] metadata, out uint frameId)
        {
            frameId = 0;
            var client = udpClient;
            if (client == null || !isRunning || imageBytes == null || imageBytes.Length == 0)
                return false;

            long nowTicks = Stopwatch.GetTimestamp();
            if (lastSendTicks != 0 && nowTicks - lastSendTicks < MinSendIntervalTicks)
                return false;

            try
            {
                nextFrameId = nextFrameId >= MaxDataFrameId ? 1u : nextFrameId + 1u;
                frameId = nextFrameId;

                int metaLength = metadata?.Length ?? 0;
                int headerSize = TransportHeaderSize + metaLength;
                int dataPerChunk = MaxChunkSize - headerSize;
                if (dataPerChunk <= 0)
                    throw new ArgumentException("Metadata exceeds MaxChunkSize.");

                int totalChunks = (imageBytes.Length + dataPerChunk - 1) / dataPerChunk;

                for (int i = 0; i < totalChunks; i++)
                {
                    int chunkStart = i * dataPerChunk;
                    int chunkLength = Math.Min(dataPerChunk, imageBytes.Length - chunkStart);
                    int packetLength = headerSize + chunkLength;

                    int offset = 0;

                    // 1. Transport Header (8 bytes): FrameID(4) + ChunkIdx(2) + TotalChunks(2)
                    WriteUInt32LE(sendPacketBuffer, offset, frameId); offset += 4;
                    WriteUInt16LE(sendPacketBuffer, offset, (ushort)i); offset += 2;
                    WriteUInt16LE(sendPacketBuffer, offset, (ushort)totalChunks); offset += 2;

                    // 2. Metadata (application header), repeated in every chunk so each is self-describing
                    if (metadata != null)
                    {
                        Buffer.BlockCopy(metadata, 0, sendPacketBuffer, offset, metaLength);
                        offset += metaLength;
                    }

                    // 3. Payload (Image Chunk)
                    Buffer.BlockCopy(imageBytes, chunkStart, sendPacketBuffer, offset, chunkLength);

                    client.Send(sendPacketBuffer, packetLength, serverEndpoint);
                }

                lastSendTicks = Stopwatch.GetTimestamp();
                return true;
            }
            catch (Exception e)
            {
                OnError?.Invoke($"Send error: {e.Message}");
                return false;
            }
        }

        /// <summary>
        /// Sends a single unchunked control datagram, prefixed with the reserved
        /// frame-id sentinel 0xFFFFFFFF so the server can distinguish it from image
        /// frames.
        /// </summary>
        public void SendControl(byte[] payload)
        {
            var client = udpClient;
            if (client == null || !isRunning || payload == null) return;

            try
            {
                byte[] packet = new byte[4 + payload.Length];
                WriteUInt32LE(packet, 0, 0xFFFFFFFFu);
                Buffer.BlockCopy(payload, 0, packet, 4, payload.Length);
                client.Send(packet, packet.Length, serverEndpoint);
            }
            catch (Exception e)
            {
                OnError?.Invoke($"Control send error: {e.Message}");
            }
        }

        private void ReceiveLoop()
        {
            // Block directly on Receive() with ReceiveTimeout = 1000 ms so packets wake the thread
            // immediately instead of paying 0-5 ms of Thread.Sleep polling jitter per RTT.
            var anyEndpoint = new IPEndPoint(
                serverEndpoint?.AddressFamily == AddressFamily.InterNetworkV6 ? IPAddress.IPv6Any : IPAddress.Any, 0);

            while (isRunning)
            {
                var client = udpClient;
                if (client == null) break;

                try
                {
                    IPEndPoint remote = anyEndpoint;
                    byte[] data = client.Receive(ref remote);
                    if (data != null && data.Length > 0)
                    {
                        OnDataReceived?.Invoke(data);
                    }
                }
                catch (SocketException se) when (
                    !isRunning ||
                    se.SocketErrorCode == SocketError.TimedOut ||
                    se.SocketErrorCode == SocketError.Interrupted ||
                    se.SocketErrorCode == SocketError.OperationAborted)
                {
                    // Expected periodic wakeup from ReceiveTimeout or socket close.
                }
                catch (ObjectDisposedException)
                {
                    break;
                }
                catch (Exception e)
                {
                    if (isRunning) OnError?.Invoke($"Receive error: {e.Message}");
                }
            }
        }

        private static void WriteUInt32LE(byte[] buffer, int offset, uint value)
        {
            buffer[offset] = (byte)value;
            buffer[offset + 1] = (byte)(value >> 8);
            buffer[offset + 2] = (byte)(value >> 16);
            buffer[offset + 3] = (byte)(value >> 24);
        }

        private static void WriteUInt16LE(byte[] buffer, int offset, ushort value)
        {
            buffer[offset] = (byte)value;
            buffer[offset + 1] = (byte)(value >> 8);
        }

        public void Close()
        {
            isRunning = false;
            var client = udpClient;
            udpClient = null;
            try { client?.Close(); } catch { }
            if (receiveThread != null && receiveThread.IsAlive) receiveThread.Join(200);
            receiveThread = null;
        }

        public void Dispose()
        {
            Close();
        }
    }
}
