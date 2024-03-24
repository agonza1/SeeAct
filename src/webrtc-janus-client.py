"""
TBC
"""

import argparse
import asyncio
import datetime
import json
import random
import string
import time
import logging
import os
import warnings
from dataclasses import dataclass
from av import VideoFrame

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceServer, RTCConfiguration
from aiortc.contrib.media import MediaPlayer, MediaRecorder
import cv2

import toml
# import torch
# from aioconsole import ainput, aprint
# from playwright.async_api import async_playwright

# Remove Huggingface internal warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

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

def transaction_id():
    return "".join(random.choice(string.ascii_letters) for x in range(12))

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

async def publish(plugin, player):
    """
    Send video to the room.
    """
    pc = RTCPeerConnection(configuration=RTCConfiguration(
            iceServers=[RTCIceServer("stun:stun.l.google.com:19302")]))
    pcs.add(pc)

    # configure media
    media = {"audio": False, "video": True}
    if player and player.audio:
        pc.addTrack(player.audio)
        media["audio"] = True

    if player and player.video:
        pc.addTrack(player.video)
    else:
        pc.addTrack(VideoStreamTrack())

    # send offer
    await pc.setLocalDescription(await pc.createOffer())
    request = {"request": "configure"}
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

    @pc.on("track")
    async def on_track(track):
        print("Track %s received" % track.kind)
        if track.kind == "audio":
            recorder.addTrack(track)

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

    # exchange media for 10 minutes
    print("Exchanging media")
    await asyncio.sleep(600)

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
    player = MediaPlayer('./images/LAM_bot_video.mp4', loop=True, decode=not args.play_without_decoding)

    # create media sink
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = None

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
