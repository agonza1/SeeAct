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

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceServer, RTCConfiguration
from aiortc.contrib.media import MediaPlayer, MediaRecorder, MediaBlackhole
import toml

global channel
channel = None
time_start = None
first_chunk = True
displayed_text = ""
audio_track_process = None
seeact_process = SeeactRunningProcess("python seeact.py -c config/remote_mode.toml")
pcs = set()

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
        global channel
        print("SeeAct LAM:", line)
        try:
            message_json = json.dumps(line)
            channel_send(channel, message_json)
        except:
            print("Error forwarding seeact logs via data channels")

    seeact_process.set_callback(line_received)
    seeact_process.set_error_callback(line_received)
    seeact_process.run()

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
    def __init__(self, track):
        self.start_requested = False
        self.track = track
        self.speech_recorder = None
        self.last_text_detected_time = None

    async def speech_to_text(self):
        def add_message_to_queue(type: str, content):
            global channel
            message = {
                "type": type,
                "content": content
            }
            print(message)
            message_json = json.dumps(message)
            channel_send(channel, message_json)

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
                add_message_to_queue("realtime", text)
                print(f"\r{text}", end='', flush=True)
                self.last_text_detected_time = datetime.now()

        def recording_started():
            add_message_to_queue("record_start", "") 

        # Initialize RealtimeSTT recorder
        recorder_config = {
            'spinner': False,
            'use_microphone': False,
            'model': 'base',
            'language': 'en',
            'silero_sensitivity': 0.4,
            'webrtc_sensitivity': 2,
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
            full_sentence = self.speech_recorder.text()
            add_message_to_queue('fullSentence', full_sentence)
            print(f"\rSentence: {full_sentence}")
            
        # Start the audio processing loop
        await get_audio_frames(self.track)

    async def start_speech_processing(self):
        self.start_requested = True
        await self.speech_to_text()

    async def stop_speech_processing(self):
        self.start_requested = False
        if self.speech_recorder:
            self.speech_recorder.stop()

def channel_send(channel, message):
    print(channel, ">", message)
    channel.send(message)

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

    global channel
    channel = pc.createDataChannel("JanusDataChannel")
    print(channel, "-", "created by local party")

    # async def send_data():
    #     while True:
    #         stdout_data = await process.stdout.readline()
    #         channel_send(channel, f"Output {i} %d",current_stamp())
    #         await asyncio.sleep(3) #every 3s

    # @channel.on("open")
    # def on_open():
    #     asyncio.ensure_future(send_data())

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

    @pc.on("track")
    def on_track(track):
        nonlocal audio_track_process
        print("Track %s received" % track.kind)
        if track.kind == "audio":
            audio_track_process = AudioTrackProcessor(track)

    pc.createDataChannel("JanusDataChannel")
    @pc.on("datachannel")
    async def on_datachannel(channel):
        print(channel, "-", "created by remote party")

        @channel.on("message")
        async def on_message(message):
            nonlocal audio_track_process
            print(channel, "<", message)
            if isinstance(message, str) and message.startswith("start"):
                run_seeact_process()
                await audio_track_process.start_speech_processing()
                recorder.addTrack(audio_track_process.track)
            elif isinstance(message, str) and message.startswith("stop"):
                seeact_process.stop_process()
                await audio_track_process.stop_speech_processing()
                await recorder.stop()

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
    await recorder.start()


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