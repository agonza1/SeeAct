"""
This script leverages aiortc and RealtimeSTT (based on fast-whisper) to connect to a Janus WebRTC Media Server and be able to capture audio and relay commands that trigger actions during a video call
"""

import argparse
import asyncio
import random
import string
import json
import time
import logging
import os
from datetime import datetime, timedelta
import numpy as np
from scipy.signal import resample
from dataclasses import dataclass
from RealtimeSTT import AudioToTextRecorder
from seeactRunningProcess import SeeactRunningProcess
from store import Store

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceServer, RTCConfiguration
from aiortc.contrib.media import MediaPlayer, MediaRecorder, MediaBlackhole
import toml

channel = None
time_start = None
first_chunk = True
displayed_text = ""
audio_track_process = None
seeact_process = SeeactRunningProcess("python seeact.py -c config/remote_mode.toml")
pcs = set()
store = Store("../data/online_tasks/sample_task.json")
stdout = []

@dataclass
class SessionControl:
    pages = []
    cdp_sessions = []
    active_page = None
    active_cdp_session = None
    context = None
    browser = None

session_control = SessionControl()

def run_seeact_process():
    # Seeact process line receive and forward via data channels
    def line_received(line):
        global stdout
        print("SeeAct LAM:", line)
        stdout.append(line)

    seeact_process.set_callback(line_received)
    seeact_process.set_error_callback(line_received)
    try:
        seeact_process.run()
    except Exception as e:
        logging.error(f"Error running SeeAct process: {e}")

def transaction_id():
    return "".join(random.choice(string.ascii_letters) for x in range(12))

def current_stamp():
    global time_start

    if time_start is None:
        time_start = time.time()
        return 0
    else:
        return int((time.time() - time_start) * 1000000)
    
class JanusPlugin:
    def __init__(self, session, url):
        self._queue = asyncio.Queue()
        self._session = session
        self._url = url

    async def send(self, payload):
        message = {"janus": "message", "transaction": transaction_id()}
        message.update(payload)
        async with self._session._http.post(self._url, json=message) as response:
            data = await response.json()
            assert data["janus"] == "ack"

        response = await self._queue.get()
        assert response["transaction"] == message["transaction"]
        return response


class JanusSession:
    def __init__(self, url):
        self._http = None
        self._poll_task = None
        self._plugins = {}
        self._root_url = url
        self._session_url = None

    async def attach(self, plugin_name: str) -> JanusPlugin:
        message = {
            "janus": "attach",
            "plugin": plugin_name,
            "transaction": transaction_id(),
        }
        async with self._http.post(self._session_url, json=message) as response:
            data = await response.json()
            assert data["janus"] == "success"
            plugin_id = data["data"]["id"]
            plugin = JanusPlugin(self, self._session_url + "/" + str(plugin_id))
            self._plugins[plugin_id] = plugin
            return plugin

    async def create(self):
        self._http = aiohttp.ClientSession()
        message = {"janus": "create", "transaction": transaction_id()}
        async with self._http.post(self._root_url, json=message) as response:
            data = await response.json()
            assert data["janus"] == "success"
            session_id = data["data"]["id"]
            self._session_url = self._root_url + "/" + str(session_id)

        self._poll_task = asyncio.ensure_future(self._poll())

    async def destroy(self):
        if self._poll_task:
            self._poll_task.cancel()
            self._poll_task = None

        if self._session_url:
            message = {"janus": "destroy", "transaction": transaction_id()}
            async with self._http.post(self._session_url, json=message) as response:
                data = await response.json()
                assert data["janus"] == "success"
            self._session_url = None

        if self._http:
            await self._http.close()
            self._http = None

    async def _poll(self):
        while True:
            params = {"maxev": 1, "rid": int(time.time() * 1000)}
            async with self._http.get(self._session_url, params=params) as response:
                data = await response.json()
                if data["janus"] == "event":
                    plugin = self._plugins.get(data["sender"], None)
                    if plugin:
                        await plugin._queue.put(data)
                    else:
                        print(data)

class AudioTrackProcessor:
    def __init__(self):
        self.start_requested = False
        self.track = None
        self.speech_recorder = None
        self.last_text_detected_time = None

    def add_message_to_queue(self, type: str, content):
        global stdout
        message = {
            "type": type,
            "content": content
        }
        message_json = json.dumps(message)
        stdout.append(message_json)

    async def speech_to_text(self):
        def decode_and_resample(
                audio_frame,
                original_sample_rate,
                target_sample_rate):

            # Convert AudioFrame to NumPy array
            audio_data = np.array(audio_frame.to_ndarray())
            # Calculate the number of samples after resampling
            num_original_samples = len(audio_data)
            num_target_samples = int(num_original_samples * target_sample_rate /
                                    original_sample_rate)
            
            # Resample the audio
            resampled_audio = resample(audio_data, num_target_samples)
            return resampled_audio.astype(np.int16).tobytes()

        def text_detected(text):
            global displayed_text, first_chunk
            print("Text detected!")

            if text != displayed_text:
                first_chunk = False
                displayed_text = text
                self.add_message_to_queue("realtime", text)
                print(f"\r{text}", end='', flush=True)
                self.last_text_detected_time = datetime.now()

        def recording_started():
            self.add_message_to_queue("record_start", "") 

        # Initialize RealtimeSTT recorder
        recorder_config = {
            'spinner': False,
            'use_microphone': False,
            'model': 'base',
            'language': 'en',
            'silero_sensitivity': 0.4,
            'webrtc_sensitivity': 3,
            'post_speech_silence_duration': 0.7,
            'min_length_of_recording': 0.3,
            'min_gap_between_recordings': 0.1,
            'enable_realtime_transcription': True,
            'realtime_processing_pause': 0.3,
            'realtime_model_type': 'base.en',
            'on_recording_start': recording_started,
            'on_realtime_transcription_stabilized': text_detected,
            # 'level': logging.DEBUG
        }

        self.speech_recorder = AudioToTextRecorder(**recorder_config)
        print("AudioToTextRecorder() Initialized")
        self.speech_recorder.start()

        # Simulated function to get audio frames from the MediaStreamTrack
        async def get_audio_frames(track):
            while self.start_requested:  # Continue loop only if start_requested is True
                # Read audio frames from the MediaStreamTrack (replace this with actual implementation)
                audio_frame = await track.recv()  # Assuming track is a MediaStreamTrack instance
                # Convert the frame to raw audio data
                resampled_frame = decode_and_resample(audio_frame, audio_frame.sample_rate, 16000)
                # print(f"Resampled audio data size: {len(resampled_frame)} bytes")
                self.speech_recorder.feed_audio(resampled_frame)
            
        # Start the audio processing loop
        await get_audio_frames(self.track)

    async def start_speech_processing(self, audio_track):
        self.start_requested = True
        self.track = audio_track
        await self.speech_to_text()

    async def stop_speech_processing(self):

        def extract_task(full_sentence):
            # Find the index of the last occurrence of 'bot' or 'but' in the string
            last_bot_index = full_sentence.lower().rfind('bot')
            if last_bot_index == -1:
                last_bot_index = full_sentence.lower().rfind('but')
                if last_bot_index == -1:
                    return full_sentence
            return full_sentence[last_bot_index + 3:].strip()

        self.start_requested = False
        if self.speech_recorder:
            self.speech_recorder.stop()
        full_sentence = self.speech_recorder.text()
        self.add_message_to_queue('fullSentence', full_sentence)
        store.add_task(extract_task(full_sentence), "https://www.aa.com/homePage.do?locale=en_US")
        print(f"\rSentence: {full_sentence}")
        self.add_message_to_queue('Info', 'Started action process...')

def channel_send(channel, message):
    print(channel, ">", message)
    try:
        channel.send(message)
    except Exception as e:
        print("Error sending data through data channels")
        print(e)

async def publish(plugin, player):
    """
    Send video to the room.
    """
    pc = RTCPeerConnection(configuration=RTCConfiguration(
        iceServers=[RTCIceServer("stun:stun.l.google.com:19302")]))

    pcs.add(pc)
    # configure media
    media = {"audio": True, "video": True}
    if player and player.audio:
        pc.addTrack(player.audio)
        media["audio"] = True

    if player and player.video:
        pc.addTrack(player.video)
    else:
        pc.addTrack(VideoStreamTrack())

    channel = pc.createDataChannel("JanusDataChannel")
    print(channel, "-", "created by local party")

    async def send_data():
        sent_text = ""
        while True:
            # Check if there is new data to send
            if stdout and stdout[-1] != sent_text:
                # Update the sent_text to the last line of stdout
                sent_text = stdout[-1]
                # Send all lines in stdout at once
                channel_send(channel, "\n".join(stdout))
                # Clear the stdout after sending
                stdout.clear()
            # Wait for 3 seconds before the next check
            await asyncio.sleep(3)  # every 3s

    @channel.on("open")
    def on_open():
        asyncio.ensure_future(send_data())

    @channel.on("message")
    def on_message(message):
        print(channel, "<", message)

    # send offer
    await pc.setLocalDescription(await pc.createOffer())
    request = {"request": "publish", "display": "bot" ,"bitrate": 200000, "audiocodec": "pcmu"}
    request.update(media)
    response = await plugin.send(
        {
            "body": request,
            "jsep": {
                "sdp": pc.localDescription.sdp,
                "trickle": False,
                "type": pc.localDescription.type,
            },
        }
    )

    # apply answer
    await pc.setRemoteDescription(
        RTCSessionDescription(
            sdp=response["jsep"]["sdp"], type=response["jsep"]["type"]
        )
    )

async def subscribe(session, room, feed, recorder):
    pc = RTCPeerConnection()
    pcs.add(pc)
    audio_track_process = None
    audio_track = None

    @pc.on("track")
    async def on_track(track):
        nonlocal audio_track_process
        nonlocal audio_track
        print("Track %s received" % track.kind)
        if track.kind == "audio":
            audio_track = track
            recorder.addTrack(audio_track)
            await recorder.start()

    pc.createDataChannel("JanusDataChannel")
    @pc.on("datachannel")
    async def on_datachannel(channel):
        print(channel, "-", "created by remote party")

        @channel.on("message")
        async def on_message(message):
            nonlocal audio_track_process
            nonlocal audio_track
            print(channel, "<", message)
            if isinstance(message, str) and message.startswith("start"):
                # seeact_process.stop_process()
                await recorder.stop()
                await audio_track_process.start_speech_processing(audio_track)
            elif isinstance(message, str) and message.startswith("stop"):
                await audio_track_process.stop_speech_processing()
                await recorder.start()
                run_seeact_process()

    # subscribe
    plugin = await session.attach("janus.plugin.videoroom")
    response = await plugin.send(
        {"body": {"request": "join", "ptype": "subscriber", "room": room, "feed": feed}}
    )

    # apply offer
    await pc.setRemoteDescription(
        RTCSessionDescription(
            sdp=response["jsep"]["sdp"], type=response["jsep"]["type"]
        )
    )

    # send answer
    await pc.setLocalDescription(await pc.createAnswer())
    response = await plugin.send(
        {
            "body": {"request": "start"},
            "jsep": {
                "sdp": pc.localDescription.sdp,
                "trickle": False,
                "type": pc.localDescription.type,
            },
        }
    )
    audio_track_process = AudioTrackProcessor()

async def rtc_communication_process(player, recorder, room, session):
    await session.create()

    # join video room
    plugin = await session.attach("janus.plugin.videoroom")
    response = await plugin.send(
        {
            "body": {
                "display": "aiortc",
                "ptype": "publisher",
                "request": "join",
                "room": room,
            }
        }
    )
    publishers = response["plugindata"]["data"]["publishers"]
    for publisher in publishers:
        print("id: %(id)s, display: %(display)s" % publisher)

    # send video
    await publish(plugin=plugin, player=player)

    # receive video
    if recorder is not None and publishers:
        await subscribe(
            session=session, room=room, feed=publishers[0]["id"], recorder=recorder
        )

    # exchange media for 5 minutes
    print("Exchanging media")
    await asyncio.sleep(300)


def main(config, base_dir, player, recorder, room) -> None:
    # Get Janus config settings
    janus_config = config["janus"]
    janus_url = os.getenv("JANUS_URL")
    # Use Janus URL from environment variable if available, otherwise use config value
    if not janus_url:
        if janus_config["janus_url"] == "Your Janus URL Here":
            raise Exception(
                f"Please set your JANUS URL first. (in {os.path.join(base_dir, 'config', 'remote_mode.toml')} by default)"
            )
    else:
        janus_config["janus_url"] = janus_url

    # create signaling and peer connection
    session = JanusSession(janus_config["janus_url"])

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            rtc_communication_process(player=player, recorder=recorder, room=room, session=session)
        )
    except KeyboardInterrupt:
        pass
    finally:
        if recorder is not None:
            loop.run_until_complete(recorder.stop())
        loop.run_until_complete(session.destroy())

        # close peer connections
        coros = [pc.close() for pc in pcs]
        loop.run_until_complete(asyncio.gather(*coros))    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config_path", help="Path to the TOML configuration file.", type=str, metavar='config',
                        default=f"{os.path.join('config', 'remote_mode.toml')}")
    
    parser.add_argument(
        "--room",
        type=int,
        default=1234,
        help="The video room ID to join (default: 1234).",
    )
    parser.add_argument("--record-to", help="Write received media to a file.")
    parser.add_argument(
        "--play-without-decoding",
        help=(
            "Read the media without decoding it (experimental). "
            "For now it only works with an MPEGTS container with only H.264 video."
        ),
        action="store_true",
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Load configuration file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = None

     # create media source
    player = MediaPlayer('./images/LAM_bot_video.mp4', decode=not args.play_without_decoding)

    # create media sink
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    try:
        with open(os.path.join(base_dir, args.config_path) if not os.path.isabs(args.config_path) else args.config_path,
                  'r') as toml_config_file:
            config = toml.load(toml_config_file)
            print(f"Configuration File Loaded - {os.path.join(base_dir, args.config_path)}")
    except FileNotFoundError:
        print(f"Error: File '{args.config_path}' not found.")
    except toml.TomlDecodeError:
        print(f"Error: File '{args.config_path}' is not a valid TOML file.")

    asyncio.run(main(config, base_dir, player=player, recorder=recorder, room=args.room))