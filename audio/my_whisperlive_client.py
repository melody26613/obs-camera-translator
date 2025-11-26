import pyaudio

from whisper_live.client import TranscriptionClient, TranscriptionTeeClient, Client

TRANSCRIPTION_TEXT_FILENAME = "clean_transcription.txt"

my_pyaudio_input_device = 0


def my_transcription_callback(texts: str, transcriptions: list):
    is_first_line = True

    with open(TRANSCRIPTION_TEXT_FILENAME, "w", encoding="utf-8") as f:
        for trans in transcriptions:
            if not is_first_line:
                f.write("\n\n")

            text = trans["text"].rstrip()
            f.write(text)
            print(text)

            is_first_line = False

    print("========================================\n")


class MyTranscriptionTeeClient(TranscriptionTeeClient):
    """
    Override from whisper_live.client TranscriptionTeeClient

    Client for handling audio recording, streaming, and transcription tasks via one or more
    WebSocket connections.

    Acts as a high-level client for audio transcription tasks using a WebSocket connection. It can be used
    to send audio data for transcription to one or more servers, and receive transcribed text segments.
    Args:
        clients (list): one or more previously initialized Client instances

    Attributes:
        clients (list): the underlying Client instances responsible for handling WebSocket connections.
    """

    def __init__(
        self,
        clients,
        save_output_recording=False,
        output_recording_filename="./output_recording.wav",
        mute_audio_playback=False,
    ):
        self.clients = clients
        if not self.clients:
            raise Exception("At least one client is required.")
        self.chunk = 4096
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 60000
        self.save_output_recording = save_output_recording
        self.output_recording_filename = output_recording_filename
        self.mute_audio_playback = mute_audio_playback
        self.frames = b""
        self.p = pyaudio.PyAudio()
        try:
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                input_device_index=my_pyaudio_input_device,
            )
        except OSError as error:
            print(f"[WARN]: Unable to access microphone. {error}")
            self.stream = None


class MyTranscriptionClient(TranscriptionClient):
    """
    Override from whisper_live.client TranscriptionClient

    Client for handling audio transcription tasks via a single WebSocket connection.

    Acts as a high-level client for audio transcription tasksoutput_transcription_path using a WebSocket connection. It can be used
    to send audio data for transcription to a server and receive transcribed text segments.

    Args:
        host (str): The hostname or IP address of the server.
        port (int): The port number to connect to on the server.
        lang (str, optional): The primary language for transcription. Default is None, which defaults to English ('en').
        translate (bool, optional): If True, the task will be translation instead of transcription. Default is False.
        model (str, optional): The whisper model to use (e.g., "small", "base"). Default is "small".
        use_vad (bool, optional): Whether to enable voice activity detection. Default is True.
        save_output_recording (bool, optional): Whether to save the microphone recording. Default is False.
        output_recording_filename (str, optional): Path to save the output recording WAV file. Default is "./output_recording.wav".
        output_transcription_path (str, optional): File path to save the output transcription (SRT file). Default is "./output.srt".
        log_transcription (bool, optional): Whether to log transcription output to the console. Default is True.
        mute_audio_playback (bool, optional): If True, mutes audio playback during file playback. Default is False.
        send_last_n_segments (int, optional): Number of most recent segments to send to the client. Defaults to 10.
        no_speech_thresh (float, optional): Segments with no speech probability above this threshold will be discarded. Defaults to 0.45.
        clip_audio (bool, optional): Whether to clip audio with no valid segments. Defaults to False.
        same_output_threshold (int, optional): Number of repeated outputs before considering it as a valid segment. Defaults to 10.
        transcription_callback (callable, optional): A callback function to handle transcription results. Default is None.
        enable_translation (float, optional): Whether to enable translation from any to any language. Defaults to False.
        target_language (str, optional): Target language for translation. Defaults to 'fr'.
        translation_callback (callable, optional): A callback function to handle translation results. Default is None.
        translation_srt_file_path (str, optional): The file path to save the translated output SRT file. Default is "output_translated.srt".

    Attributes:
        client (Client): An instance of the underlying Client class responsible for handling the WebSocket connection.

    Example:
        To create a TranscriptionClient and start transcription on microphone audio:
        ```python
        transcription_client = TranscriptionClient(host="localhost", port=9090)
        transcription_client()
        ```
    """

    def __init__(
        self,
        host,
        port,
        lang=None,
        translate=False,
        model="small",
        use_vad=True,
        use_wss=False,
        save_output_recording=False,
        output_recording_filename="./output_recording.wav",
        output_transcription_path="./output.srt",
        log_transcription=False,
        mute_audio_playback=False,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=10,
        transcription_callback=my_transcription_callback,
        enable_translation=False,
        target_language="fr",
        translation_callback=None,
        translation_srt_file_path="./output_translated.srt",
    ):
        self.client = Client(
            host,
            port,
            lang,
            translate,
            model,
            srt_file_path=output_transcription_path,
            use_vad=use_vad,
            use_wss=use_wss,
            log_transcription=log_transcription,
            send_last_n_segments=send_last_n_segments,
            no_speech_thresh=no_speech_thresh,
            clip_audio=clip_audio,
            same_output_threshold=same_output_threshold,
            transcription_callback=transcription_callback,
            enable_translation=enable_translation,
            target_language=target_language,
            translation_callback=translation_callback,
            translation_srt_file_path=translation_srt_file_path,
        )

        if save_output_recording and not output_recording_filename.endswith(".wav"):
            raise ValueError(
                f"Please provide a valid `output_recording_filename`: {output_recording_filename}"
            )
        if not output_transcription_path.endswith(".srt"):
            raise ValueError(
                f"Please provide a valid `output_transcription_path`: {output_transcription_path}. The file extension should be `.srt`."
            )
        if not translation_srt_file_path.endswith(".srt"):
            raise ValueError(
                f"Please provide a valid `translation_srt_file_path`: {translation_srt_file_path}. The file extension should be `.srt`."
            )
        MyTranscriptionTeeClient.__init__(
            self,
            [self.client],
            save_output_recording=save_output_recording,
            output_recording_filename=output_recording_filename,
            mute_audio_playback=mute_audio_playback,
        )
