using System;
using System.Collections.Generic;

public interface IDetectionProvider
{
    /// <summary>
    /// Gets the list of currently detected objects.
    /// </summary>
    List<DetectedObject> CurrentDetections { get; }
    
    /// <summary>
    /// Event fired when detections are updated.
    /// </summary>
    event Action<List<DetectedObject>> OnDetectionsUpdated;

    /// <summary>
    /// Starts the detection process (e.g., loading model, connecting to server).
    /// </summary>
    void StartDetection();

    /// <summary>
    /// Stops the detection process.
    /// </summary>
    void StopDetection();

    /// <summary>
    /// Sets the open-vocabulary target classes for the active shader. The order is
    /// significant: a detection's returned <c>LabelID</c> is the slot index into this
    /// list (slot 0 = first class), which the generated shader filters against.
    /// Passing null or an empty list reverts the detector to its default vocabulary.
    /// Implementations should treat this as a rare, set-once-per-shader operation and
    /// avoid per-frame work (open-vocab text encoding happens here, not in the hot path).
    /// </summary>
    /// <param name="classes">Natural-language target classes in slot order.</param>
    /// <param name="confidence">Optional confidence threshold for this vocabulary (&lt;0 = leave unchanged).</param>
    /// <param name="cocoClasses">
    /// Optional COCO class per slot (same length/order as <paramref name="classes"/>, empty
    /// string = no equivalent), written by the LLM when the shader was generated. Only the
    /// closed-vocabulary detector uses it; without it that detector has to guess the
    /// translation from the phrase. Null/empty keeps the server's own heuristic.
    /// </param>
    void SetVocabulary(IReadOnlyList<string> classes, float confidence = -1f,
                       IReadOnlyList<string> cocoClasses = null);

    /// <summary>
    /// Switches the server-side detector. <c>"yoloe"</c> is the open-vocabulary segmentation
    /// model driven by the shader's <c>@detect-classes</c> list; <c>"yolo"</c> is the closed
    /// COCO yolo26-seg model, which is faster and needs no text encoder but can only find the
    /// 80 COCO classes. The server maps COCO hits back onto the shader's slot indices, so
    /// shaders keep working either way — they just see fewer objects under <c>"yolo"</c>.
    /// Rare, user-driven operation: not for the per-frame path.
    /// </summary>
    void SetDetector(string detector);

    /// <summary>Last detector requested via <see cref="SetDetector"/>; empty = server default.</summary>
    string ActiveDetector { get; }
}
