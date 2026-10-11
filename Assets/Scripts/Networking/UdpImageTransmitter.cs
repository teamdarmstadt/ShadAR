using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

namespace Networking
{
    /// <summary>
    /// Sends image frames to the detection server as chunked UDP datagrams and receives its
    /// single-datagram responses on a background thread.
    /// </summary>
    public class UdpImageTransmitter : IDisposable
    {
        private UdpClient udpClient;
        private IPEndPoint serverEndpoint;
        private Thread receiveThread;
        private volatile bool isRunning;
        private uint nextFrameId = 0;

        // Flow control
        private const float MinSendInterval = 0.033f; // ~30 FPS
        private float lastSendTime;
        
        // Callbacks
        public event Action<byte[]> OnDataReceived;
        public event Action<string> OnError;
        
        // Chunking constants
        private const int MaxChunkSize = 60000;
        private const int TransportHeaderSize = 8; // FrameID(4) + ChunkIdx(2) + TotalChunks(2)
        
        public bool IsConnected => udpClient != null && isRunning;

        /// <summary>
        /// Connects to the specified UDP server.
        /// </summary>
        /// <param name="host">Server Hostname</param>
        /// <param name="port">Server Port</param>
        public void Connect(string host, int port)
        {
            try
            {
                IPAddress ipAddress;
                if (host == "localhost")
                {
                    ipAddress = IPAddress.Loopback;
                }
                else if (!IPAddress.TryParse(host, out ipAddress))
                {
                    IPHostEntry entry = Dns.GetHostEntry(host);
                    foreach (var ip in entry.AddressList)
                    {
                        if (ip.AddressFamily == AddressFamily.InterNetwork)
                        {
                            ipAddress = ip;
                            break;
                        }
                    }
                }
                
                if (ipAddress == null)
                    throw new Exception($"Could not resolve host: {host}");

                serverEndpoint = new IPEndPoint(ipAddress, port);
                udpClient = new UdpClient();
                udpClient.Client.ReceiveTimeout = 1000;
                
                // Start receive thread
                isRunning = true;
                receiveThread = new Thread(ReceiveLoop) { IsBackground = true };
                receiveThread.Start();
                
                Debug.Log($"[UDP] Connected to {host}:{port}");
            }
            catch (Exception e)
            {
                OnError?.Invoke($"Connection error: {e.Message}");
            }
        }

        /// <summary>
        /// Sends a frame with optional metadata.
        /// </summary>
        public void SendFrame(byte[] imageBytes, byte[] metadata = null)
        {
            if (udpClient == null || !isRunning) return;
            
            // Simple flow control
            if (Time.time - lastSendTime < MinSendInterval) return;
            
            try
            {
                nextFrameId++;
                lastSendTime = Time.time;
                
                int metaLength = metadata?.Length ?? 0;
                int headerSize = TransportHeaderSize + metaLength;
                int dataPerChunk = MaxChunkSize - headerSize;
                
                int totalChunks = (imageBytes.Length + dataPerChunk - 1) / dataPerChunk;
                
                for (int i = 0; i < totalChunks; i++)
                {
                    int chunkStart = i * dataPerChunk;
                    int chunkLength = Math.Min(dataPerChunk, imageBytes.Length - chunkStart);
                    
                    byte[] packet = new byte[headerSize + chunkLength];
                    int offset = 0;
                    
                    // 1. Transport Header (8 bytes)
                    Buffer.BlockCopy(BitConverter.GetBytes(nextFrameId), 0, packet, offset, 4); offset += 4;
                    Buffer.BlockCopy(BitConverter.GetBytes((ushort)i), 0, packet, offset, 2); offset += 2;
                    Buffer.BlockCopy(BitConverter.GetBytes((ushort)totalChunks), 0, packet, offset, 2); offset += 2;
                    
                    // 2. Metadata (application header), repeated in every chunk so each is self-describing
                    if (metadata != null)
                    {
                        Buffer.BlockCopy(metadata, 0, packet, offset, metaLength);
                        offset += metaLength;
                    }
                    
                    // 3. Payload (Image Chunk)
                    Buffer.BlockCopy(imageBytes, chunkStart, packet, offset, chunkLength);
                    
                    udpClient.Send(packet, packet.Length, serverEndpoint);
                }
            }
            catch (Exception e)
            {
                OnError?.Invoke($"Send error: {e.Message}");
            }
        }

        /// <summary>
        /// Sends a single unchunked control datagram, prefixed with the reserved
        /// frame-id sentinel 0xFFFFFFFF so the server can distinguish it from image
        /// frames. Used to deliver the open-vocabulary class list over the same
        /// host:port that carries frames (guaranteed-reachable, unlike a separate
        /// HTTP port). Best-effort: UDP may drop it, so callers should send a few times.
        /// </summary>
        public void SendControl(byte[] payload)
        {
            if (udpClient == null || !isRunning || payload == null) return;

            try
            {
                byte[] packet = new byte[4 + payload.Length];
                Buffer.BlockCopy(BitConverter.GetBytes(0xFFFFFFFFu), 0, packet, 0, 4);
                Buffer.BlockCopy(payload, 0, packet, 4, payload.Length);
                udpClient.Send(packet, packet.Length, serverEndpoint);
            }
            catch (Exception e)
            {
                OnError?.Invoke($"Control send error: {e.Message}");
            }
        }

        private void ReceiveLoop()
        {
            while (isRunning)
            {
                try
                {
                    if (udpClient.Available > 0)
                    {
                        // Accept responses from any sender address (the server may answer from another interface).
                        IPEndPoint anyEndpoint = new IPEndPoint(IPAddress.Any, 0);
                        byte[] data = udpClient.Receive(ref anyEndpoint);
                        OnDataReceived?.Invoke(data);
                    }
                    else
                    {
                        Thread.Sleep(5); // Sleep slightly to reduce CPU usage
                    }
                }
                catch (Exception e)
                {
                    if (isRunning) Debug.LogWarning($"[UDP] Receive error: {e.Message}");
                }
            }
        }

        public void Close()
        {
            isRunning = false;
            if (receiveThread != null && receiveThread.IsAlive) receiveThread.Join(200);
            udpClient?.Close();
            udpClient = null;
        }

        public void Dispose()
        {
            Close();
        }
    }
}
