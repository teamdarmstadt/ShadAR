using System.Threading.Tasks;

/// <summary>
/// Input and feedback for the hold-to-reset of an LLM conversation, shared by the OpenAI and
/// Claude controls.
/// </summary>
public static class ConversationResetInput
{
    /// <summary>
    /// True only while EVERY button in <paramref name="buttons"/> is held. OVRInput.Get on a
    /// combined mask is true when ANY of them is down, which let a single trigger held for two
    /// seconds (e.g. while using the UI) silently wipe the conversation.
    /// </summary>
    public static bool AllHeld(OVRInput.Button buttons)
    {
        for (uint bit = 1; bit != 0 && bit <= (uint)buttons; bit <<= 1)
        {
            if (((uint)buttons & bit) != 0 && !OVRInput.Get((OVRInput.Button)bit)) return false;
        }
        return buttons != OVRInput.Button.None;
    }

    /// <summary>
    /// A short buzz on both controllers: confirmation you feel even when the status line is out
    /// of view.
    /// </summary>
    public static async void Buzz()
    {
        OVRInput.SetControllerVibration(1f, 0.6f, OVRInput.Controller.Touch);
        await Task.Delay(150);
        OVRInput.SetControllerVibration(0f, 0f, OVRInput.Controller.Touch);
    }
}
