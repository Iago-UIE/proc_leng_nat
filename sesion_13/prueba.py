from elevenlabs.client import ElevenLabs
from elevenlabs import play

client = ElevenLabs(
    api_key="sk_34b4e057d386dbba8ca0ecd8c103ff6d0fab50ab937348a8",
)

audio = client.text_to_speech.convert(
    text="Diego es un mono y hace UU AA UU AA UU AA UU.",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
)

play(audio)
