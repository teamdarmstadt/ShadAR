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

    private void OnDestroy() => ControllerTooltips.Unregister(tooltip);

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

        micDevice = Microphone.devices.Length > 0 ? Microphone.devices[0] : null;
        if (micDevice == null)
        {
            logger?.LogError("No microphone device found.", nameof(WhisperTranscriber));
            return;
        }

        audioClip = Microphone.Start(micDevice, false, maxLengthSec, sampleRate);
        recording = true;
        logger?.LogInfo("Recording...", nameof(WhisperTranscriber));
        logger?.SetStatus("Listening...", busy: true);
    }

    public void EndRecordingAndTranscribe()
    {
        if (!recording || audioClip == null)
        {
            logger?.LogWarning("No recording in progress.", nameof(WhisperTranscriber));
            return;
        }

        PipelineBenchmarkLogger.Instance?.BeginSession();
        PipelineBenchmarkLogger.Instance?.RecordTimestamp("speech_complete");

        int micPos = Microphone.GetPosition(micDevice);
        Microphone.End(micDevice);
        recording = false;
        logger?.SetStatus("Transcribing...", busy: true);
        Transcribe(micPos);
    }

    private async void Transcribe(int micPos)
    {
        int channels = audioClip.channels;
        int frequency = audioClip.frequency;
        int totalSamples = audioClip.samples;
        int usedSamples = (micPos > 0 && micPos <= totalSamples) ? micPos : totalSamples;

        float[] samples = new float[usedSamples * channels];
        audioClip.GetData(samples, 0);

        if (normalizeAudio)
        {
            Normalize(samples);
        }

        // Unique name per recording: with a fixed name, a new recording truncated the file
        // while the previous upload was still reading it, which corrupted that transcript.
        string wavPath = Path.Combine(Application.temporaryCachePath, $"voice_{DateTime.UtcNow.Ticks}.wav");
        WriteWavFile(wavPath, samples, usedSamples, channels, frequency);

        var configManager = openAIConfigManager != null ? openAIConfigManager : OpenAIConfigManager.Instance;
        if (configManager == null || !configManager.IsInitialized || configManager.Client == null)
        {
            logger?.LogError(
                $"Speech-to-text needs an OpenAI API key. Add one to {configManager?.GetDeviceConfigPath() ?? "openai_config.json"}.",
                nameof(WhisperTranscriber));
            return;
        }

        try
        {
            var request = new AudioTranscriptionRequest(Path.GetFullPath(wavPath), language: "en");
            PipelineBenchmarkLogger.Instance?.RecordTimestamp("audio_sent_to_whisper");

            // Same reason as in LLMShaderGenerator: without a deadline a request on a dead
            // connection never returns, and the prompt silently disappears.
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
            try { File.Delete(wavPath); } catch (IOException) { /* cache cleanup is best-effort */ }
        }
    }

    private static void Normalize(float[] samples)
    {
        float max = 0f;
        for (int i = 0; i < samples.Length; i++)
        {
            max = Mathf.Max(max, Mathf.Abs(samples[i]));
        }

        if (max > 0f && max < 0.99f)
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
        using var fileStream = new FileStream(filePath, FileMode.Create);
        using var writer = new BinaryWriter(fileStream);

        int byteRate = frequency * channels * 2;
        int dataSize = sampleCount * channels * 2;

        // RIFF header, 16-bit PCM
        writer.Write(Encoding.UTF8.GetBytes("RIFF"));
        writer.Write(36 + dataSize);
        writer.Write(Encoding.UTF8.GetBytes("WAVE"));
        writer.Write(Encoding.UTF8.GetBytes("fmt "));
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)channels);
        writer.Write(frequency);
        writer.Write(byteRate);
        writer.Write((short)(channels * 2));
        writer.Write((short)16);
        writer.Write(Encoding.UTF8.GetBytes("data"));
        writer.Write(dataSize);

        for (int i = 0; i < samples.Length; i++)
        {
            writer.Write((short)Mathf.Clamp(samples[i] * 32767f, -32768f, 32767f));
        }
    }
}
