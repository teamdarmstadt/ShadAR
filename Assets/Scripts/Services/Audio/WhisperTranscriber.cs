using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using OpenAI.Audio;

/// <summary>
/// Push-to-talk voice prompts: records the microphone while the trigger button is held and
/// transcribes the recording with the OpenAI speech-to-text API.
/// </summary>
public class WhisperTranscriber : MonoBehaviour
{
    private const int DefaultSampleRate = 44100;
    private const int DefaultMaxRecordingLength = 60;
    private const float MinRecordingSeconds = 0.15f;
    private const float NormalizationNoiseFloor = 0.01f;

    [Header("OpenAI Configuration")]
    [Tooltip("Reference to the OpenAI Config Manager (optional - will use singleton if not set)")]
    [SerializeField] private OpenAIConfigManager openAIConfigManager;

    [Tooltip("Seconds to wait for the transcription before giving up with an error.")]
    [SerializeField] private int requestTimeoutSeconds = 30;

    [Header("Logger")]
    [SerializeField] private Logger logger;

    [Header("Input Settings")]
    [SerializeField] private OVRInput.Button triggerButton;

    [Header("Audio Settings")]
    [Tooltip("Recording sample rate (Hz)")]
    [SerializeField] private int sampleRate = DefaultSampleRate;

    [Tooltip("Maximum recording length (seconds)")]
    [SerializeField] private int maxLengthSec = DefaultMaxRecordingLength;

    [Tooltip("Normalize audio before uploading")]
    [SerializeField] private bool normalizeAudio = true;

    private AudioClip audioClip;
    private string micDevice;
    private bool recording;

    /// <summary>
    /// Event fired when a transcription is completed and ready for processing.
    /// </summary>
    public event Action<string> OnTranscriptionReady;

    private ControllerTooltips.Hint tooltip;

    private void Awake() => tooltip = ControllerTooltips.Register(triggerButton, "Hold to speak");

    private void OnDestroy()
    {
        ControllerTooltips.Unregister(tooltip);
        if (recording && !string.IsNullOrEmpty(micDevice))
        {
            Microphone.End(micDevice);
            recording = false;
        }
        DestroyAudioClip();
    }

    private void Update()
    {
        if (OVRInput.GetDown(triggerButton))
        {
            BeginRecording();
        }
        else if (OVRInput.GetUp(triggerButton))
        {
            EndRecordingAndTranscribe();
        }
    }

    public void BeginRecording()
    {
        if (recording) return;

#if UNITY_ANDROID && !UNITY_EDITOR
        if (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(UnityEngine.Android.Permission.Microphone))
        {
            UnityEngine.Android.Permission.RequestUserPermission(UnityEngine.Android.Permission.Microphone);
            logger?.LogWarning("Microphone permission requested — grant it and hold B again.", nameof(WhisperTranscriber));
            logger?.SetStatus("Grant microphone permission, then hold B to speak.");
            return;
        }
#endif

        micDevice = Microphone.devices.Length > 0 ? Microphone.devices[0] : null;
        if (micDevice == null)
        {
            logger?.LogError("No microphone device found.", nameof(WhisperTranscriber));
            return;
        }

        DestroyAudioClip();
        audioClip = Microphone.Start(micDevice, false, maxLengthSec, sampleRate);
        if (audioClip == null)
        {
            logger?.LogError("Failed to start microphone recording.", nameof(WhisperTranscriber));
            return;
        }

        recording = true;
        logger?.LogInfo("Recording...", nameof(WhisperTranscriber));
        logger?.SetStatus("Listening...", busy: true);
    }

    public void EndRecordingAndTranscribe()
    {
        if (!recording || audioClip == null)
        {
            return;
        }

        // Check whether the non-looping clip was still recording before Microphone.End():
        // if it was still recording and micPos == 0 (quick tap), 0 samples were captured — NOT
        // the full 60s buffer! Only when recording stopped on its own did it hit maxLengthSec.
        bool wasStillRecording = Microphone.IsRecording(micDevice);
        int micPos = Microphone.GetPosition(micDevice);
        Microphone.End(micDevice);
        recording = false;

        int channels = audioClip.channels;
        int frequency = audioClip.frequency;
        int totalSamples = audioClip.samples;
        int usedSamples = wasStillRecording
            ? Mathf.Clamp(micPos, 0, totalSamples)
            : (micPos > 0 && micPos <= totalSamples ? micPos : totalSamples);

        int minSamples = Mathf.Max(1, Mathf.RoundToInt(frequency * MinRecordingSeconds));
        if (usedSamples < minSamples)
        {
            DestroyAudioClip();
            logger?.LogInfo("Recording too short — hold B while speaking.", nameof(WhisperTranscriber));
            logger?.SetStatus("Hold B while speaking.");
            return;
        }

        float[] samples = new float[usedSamples * channels];
        audioClip.GetData(samples, 0);
        // Free the 60s native PCM AudioClip buffer (~10.6 MB) immediately after reading samples.
        DestroyAudioClip();

        var configManager = openAIConfigManager != null ? openAIConfigManager : OpenAIConfigManager.Instance;
        if (configManager == null || !configManager.IsInitialized || configManager.Client == null)
        {
            logger?.LogError(
                $"Speech-to-text needs an OpenAI API key. Add one to {configManager?.GetDeviceConfigPath() ?? "openai_config.json"}.",
                nameof(WhisperTranscriber));
            return;
        }

        PipelineBenchmarkLogger.Instance?.BeginSession();
        PipelineBenchmarkLogger.Instance?.RecordTimestamp("speech_complete");
        logger?.SetStatus("Transcribing...", busy: true);

        TranscribeAsync(samples, usedSamples, channels, frequency, configManager);
    }

    private void DestroyAudioClip()
    {
        if (audioClip != null)
        {
            Destroy(audioClip);
            audioClip = null;
        }
    }

    private async void TranscribeAsync(
        float[] samples,
        int usedSamples,
        int channels,
        int frequency,
        OpenAIConfigManager configManager)
    {
        string wavPath = Path.Combine(Application.temporaryCachePath, $"voice_{DateTime.UtcNow.Ticks}.wav");
        bool doNormalize = normalizeAudio;

        try
        {
            // Normalize and encode WAV off the main thread to avoid VR frame hitches on long prompts.
            await Task.Run(() =>
            {
                if (doNormalize)
                {
                    Normalize(samples);
                }
                WriteWavFile(wavPath, samples, usedSamples, channels, frequency);
            });

            var request = new AudioTranscriptionRequest(Path.GetFullPath(wavPath), language: "en");
            PipelineBenchmarkLogger.Instance?.RecordTimestamp("audio_sent_to_whisper");

            var transcription = configManager.Client.AudioEndpoint.CreateTranscriptionTextAsync(request);
            if (await Task.WhenAny(transcription, Task.Delay(TimeSpan.FromSeconds(requestTimeoutSeconds))) != transcription)
            {
                logger?.LogError($"Speech-to-text did not answer within {requestTimeoutSeconds} s. Please try again.", nameof(WhisperTranscriber));
                return;
            }

            string text = await transcription;
            if (text != null)
            {
                OnTranscriptionReady?.Invoke(text);
            }
            else
            {
                logger?.LogError("OpenAI transcription failed.", nameof(WhisperTranscriber));
            }
        }
        catch (Exception ex)
        {
            logger?.LogError(ex.ToString(), nameof(WhisperTranscriber));
        }
        finally
        {
            try
            {
                if (File.Exists(wavPath)) File.Delete(wavPath);
            }
            catch (IOException)
            {
                // Cache cleanup is best-effort
            }
        }
    }

    private static void Normalize(float[] samples)
    {
        float max = 0f;
        for (int i = 0; i < samples.Length; i++)
        {
            float abs = Math.Abs(samples[i]);
            if (abs > max) max = abs;
        }

        // Ignore near-zero background hiss so silent recordings are not boosted into full-scale noise.
        if (max > NormalizationNoiseFloor && max < 0.99f)
        {
            float gain = 1f / max;
            for (int i = 0; i < samples.Length; i++)
            {
                samples[i] *= gain;
            }
        }
    }

    private static void WriteWavFile(string filePath, float[] samples, int sampleCount, int channels, int frequency)
    {
        int byteRate = frequency * channels * 2;
        int dataSize = sampleCount * channels * 2;
        byte[] wavBytes = new byte[44 + dataSize];

        // RIFF header, 16-bit little-endian PCM
        Encoding.ASCII.GetBytes("RIFF", 0, 4, wavBytes, 0);
        WriteInt32LE(wavBytes, 4, 36 + dataSize);
        Encoding.ASCII.GetBytes("WAVE", 0, 4, wavBytes, 8);
        Encoding.ASCII.GetBytes("fmt ", 0, 4, wavBytes, 12);
        WriteInt32LE(wavBytes, 16, 16);
        WriteInt16LE(wavBytes, 20, 1);
        WriteInt16LE(wavBytes, 22, (short)channels);
        WriteInt32LE(wavBytes, 24, frequency);
        WriteInt32LE(wavBytes, 28, byteRate);
        WriteInt16LE(wavBytes, 32, (short)(channels * 2));
        WriteInt16LE(wavBytes, 34, 16);
        Encoding.ASCII.GetBytes("data", 0, 4, wavBytes, 36);
        WriteInt32LE(wavBytes, 40, dataSize);

        int byteOffset = 44;
        int totalValues = Math.Min(samples.Length, sampleCount * channels);
        for (int i = 0; i < totalValues; i++)
        {
            float scaled = samples[i] * 32767f;
            short pcm = scaled >= 32767f ? (short)32767 : scaled <= -32768f ? (short)-32768 : (short)scaled;
            wavBytes[byteOffset++] = (byte)pcm;
            wavBytes[byteOffset++] = (byte)(pcm >> 8);
        }

        File.WriteAllBytes(filePath, wavBytes);
    }

    private static void WriteInt32LE(byte[] buf, int offset, int value)
    {
        buf[offset] = (byte)value;
        buf[offset + 1] = (byte)(value >> 8);
        buf[offset + 2] = (byte)(value >> 16);
        buf[offset + 3] = (byte)(value >> 24);
    }

    private static void WriteInt16LE(byte[] buf, int offset, short value)
    {
        buf[offset] = (byte)value;
        buf[offset + 1] = (byte)(value >> 8);
    }
}
