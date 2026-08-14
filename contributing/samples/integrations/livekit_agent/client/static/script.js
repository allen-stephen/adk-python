// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Browser client for the ADK × LiveKit dice agent. It uses LiveKit's own
// client SDK unmodified -- there is no ADK client code. The flow:
//   1. GET /token -> the backend mints a join token AND dispatches the worker.
//   2. Connect to the room and publish the mic (the inbound bridge's input).
//   3. Play the agent's audio track when it arrives (the outbound bridge).
//   4. Render transcripts and tool activity from the room data track, and send
//      typed messages back over the same track.

const { Room, RoomEvent, Track } = LivekitClient;

// Must match `DATA_TOPIC` in google.adk.integrations.livekit._livekit_runner.
const DATA_TOPIC = 'adk';

const talkBtn = document.getElementById('talk');
const stateEl = document.getElementById('state');
const dotEl = document.getElementById('dot');
const logEl = document.getElementById('log');
const audioContainer = document.getElementById('audio-container');
const transcriptEl = document.getElementById('transcript');
const sayForm = document.getElementById('say-form');
const sayInput = document.getElementById('say');
const sendBtn = document.getElementById('send');

const encoder = new TextEncoder();
const decoder = new TextDecoder();

let room = null;

function log(message) {
  const time = new Date().toLocaleTimeString();
  logEl.textContent += `[${time}] ${message}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

function setState(label, connected) {
  stateEl.textContent = label;
  dotEl.classList.toggle('connected', Boolean(connected));
}

function addTranscript(role, text) {
  const line = document.createElement('p');
  line.className = `line line-${role}`;
  line.textContent = text;
  transcriptEl.appendChild(line);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

// The connector publishes JSON on the `adk` topic: transcripts of both sides,
// plus function_call / function_response as the agent uses its tools.
function handleData(payload, _participant, _kind, topic) {
  if (topic !== DATA_TOPIC) {
    return;
  }
  let message;
  try {
    message = JSON.parse(decoder.decode(payload));
  } catch (err) {
    log(`Ignoring unparseable data message: ${err}`);
    return;
  }

  switch (message.type) {
    case 'transcript':
      addTranscript(message.role, message.text);
      break;
    case 'function_call':
      log(`Tool call: ${message.name}(${JSON.stringify(message.args)})`);
      break;
    case 'function_response':
      log(`Tool result: ${message.name} -> ${JSON.stringify(message.response)}`);
      break;
    default:
      log(`Unknown data message type: ${message.type}`);
  }
}

async function connect() {
  talkBtn.disabled = true;
  setState('Requesting token & dispatching agent…', false);
  log('Requesting token from backend (this also dispatches the worker)…');

  const resp = await fetch('/token');
  if (!resp.ok) {
    setState('Error', false);
    log(`Token request failed: ${resp.status} ${await resp.text()}`);
    talkBtn.disabled = false;
    return;
  }
  const { url, token, room: roomName } = await resp.json();
  log(`Dispatched "roll_dice" into room "${roomName}". Connecting…`);

  room = new Room();

  room.on(RoomEvent.TrackSubscribed, (track) => {
    if (track.kind === Track.Kind.Audio) {
      log('Agent audio track received — you should hear the dice agent now.');
      const el = track.attach();
      audioContainer.appendChild(el);
    }
  });

  room.on(RoomEvent.DataReceived, handleData);

  room.on(RoomEvent.ParticipantConnected, (participant) => {
    log(`Participant joined: ${participant.identity}`);
  });

  room.on(RoomEvent.Disconnected, () => {
    setState('Disconnected', false);
    log('Disconnected from room.');
    talkBtn.disabled = false;
    talkBtn.textContent = 'Start talking';
    sayInput.disabled = true;
    sendBtn.disabled = true;
  });

  await room.connect(url, token);
  setState('Connected — publishing mic', true);
  log('Connected. Enabling microphone…');

  await room.localParticipant.setMicrophoneEnabled(true);
  setState('Live — speak now', true);
  log('Microphone is live. Speak to the agent, or type below.');

  talkBtn.textContent = 'Hang up';
  talkBtn.disabled = false;
  sayInput.disabled = false;
  sendBtn.disabled = false;
}

async function disconnect() {
  if (room) {
    await room.disconnect();
    room = null;
  }
}

sayForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = sayInput.value.trim();
  if (!text || !room) {
    return;
  }
  sayInput.value = '';
  addTranscript('user', text);
  // Same data track the agent publishes on; the connector turns this into a
  // user turn via LiveRequestQueue.send_content.
  await room.localParticipant.publishData(
    encoder.encode(JSON.stringify({ type: 'text', text })),
    { reliable: true, topic: DATA_TOPIC },
  );
});

talkBtn.addEventListener('click', async () => {
  if (room) {
    await disconnect();
  } else {
    try {
      await connect();
    } catch (err) {
      setState('Error', false);
      log(`Error: ${err && err.message ? err.message : err}`);
      talkBtn.disabled = false;
    }
  }
});
