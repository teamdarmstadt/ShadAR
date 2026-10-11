using System;
using UnityEngine;

/// <summary>
/// One detection from the detection server, in LEFT-camera viewport space (origin bottom-left).
/// </summary>
[Serializable]
public struct DetectedObject
{
    public string Label;
    public int LabelID;          // slot index into the active shader's @detect-classes list
    public Rect BoundingBox;     // x,y = CENTER (normalized [0,1]), width,height = size (normalized)
    public float Confidence;
    public Texture MaskTexture;  // rasterized segmentation mask, null if none
    public bool HasMask;
    public Vector2[] MaskPolygon; // mask outline, normalized to the bounding box (0-1)
    public bool HasPolygon;
}
