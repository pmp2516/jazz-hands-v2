$(document).ready(init);

async function init() {
  // --- 1) MIDI Setup ---
  let midiAccess, currentOutput = null;
  const midiSelect = $('#midiSelect')[0];
  let synth = new Tone.Synth().toDestination();

  // Populate outputs
  async function initMIDI() {
    try {
      midiAccess = await navigator.requestMIDIAccess();
      midiAccess.addEventListener('statechange', populateMIDIDevices);
      populateMIDIDevices();
    } catch (err) {
      console.warn('MIDI unavailable:', err);
    }
  }
  function populateMIDIDevices() {
    midiSelect.innerHTML = '';
    // Built-in option
    midiSelect.append(new Option('Built-in Synth', 'builtin'));
    if (!midiAccess) return;
    for (let output of midiAccess.outputs.values()) {
      midiSelect.append(new Option(output.name, output.id));
    }
  }
  $('#midiSelect').on('change', () => {
    const val = midiSelect.value;
    if (val === 'builtin') {
      currentOutput = 'builtin';
      synth = new Tone.Synth().toDestination();
    } else {
      currentOutput = midiAccess.outputs.get(val);
      synth = null;
    }
  });
  $('#saveSettings').on('click', () => {
    // TODO persist settings in indexedDB
    console.log('MIDI set to', currentOutput);
  });
  await initMIDI();

  // Stub for downstream MIDI handling
  function consumeMIDI(note, velocity) {
    // TODO Work with sequences.
    console.log(`→ MIDI note ${note}, vel ${velocity}`);
    if (currentOutput === 'builtin') {
      synth.triggerAttackRelease(Tone.Frequency(note, 'midi'), '8n', undefined, velocity / 127);
    } else if (currentOutput) {
      currentOutput.send([0x90, note, velocity]);
    }
  }

  // --- 2) Media Setup ---
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  const videoEl = $('#video')[0];
  videoEl.srcObject = stream;

  // --- 3) AudioWorklet + Meyda Chromagram ---
  const audioCtx = new AudioContext();
  await audioCtx.audioWorklet.addModule('recorder-processor.js');
  const micSource = audioCtx.createMediaStreamSource(stream);
  const recorderNode = new AudioWorkletNode(audioCtx, 'recorder-processor');
  micSource.connect(recorderNode);

  // Prepare OfflineAudioContext for chroma rendering
  const offlineCtx = new OfflineAudioContext(1, /*length set per‐buffer*/ 1024, audioCtx.sampleRate);

  let pendingPCM = null;
  recorderNode.port.onmessage = ev => {
    pendingPCM = ev.data;  // Float32Array of length ≈ bufferSize
  };

  // --- 4) Video Canvas Prep ---
  const vcanvas = $('#vcanvas')[0];
  const vctx = vcanvas.getContext('2d');

  // --- 5) Load TF.js Model ---
  const runner = await createGestureMidiRunner({
    audioChannels: 12,
    videoChannels: 5,
    nAudioHeads: 8,
    nVideoHeads: 8,
    latentDim: 64,
    notes: Array.from({length:12}, (_,i)=>60 + i),
    velos: [0,30,60,90,120]
  });

  // --- 6) Main Loop ---
  async function frameLoop() {
    if (pendingPCM) {
      // 1) Compute chroma via Meyda + OfflineAudioContext
      const buffer = offlineCtx.createBuffer(1, pendingPCM.length, audioCtx.sampleRate);
      buffer.copyToChannel(pendingPCM, 0);
      const src = offlineCtx.createBufferSource();
      src.buffer = buffer;
      src.connect(offlineCtx.destination);
      src.start();
      const rendered = await offlineCtx.startRendering();
      const chroma = Meyda.extract('chroma', rendered.getChannelData(0), {
        bufferSize: pendingPCM.length,
        hopSize: pendingPCM.length / 2,
        sampleRate: audioCtx.sampleRate
      });
      const audioTensor = tf.tensor(chroma, [chroma.length, 1]);

      // 2) Capture video frame
      vctx.drawImage(videoEl, 0, 0, vcanvas.width, vcanvas.height);
      const imgData = vctx.getImageData(0, 0, vcanvas.width, vcanvas.height);
      const videoTensor = tf.tidy(() =>
        tf.browser.fromPixels(imgData).toFloat().div(255).expandDims(0)
      );

      // 3) Model inference
      const { notes, velos } = await runner.step(audioTensor, videoTensor);
      consumeMIDI(notes, velos);

      // 4) Cleanup
      audioTensor.dispose();
      videoTensor.dispose();
      noteLogits.dispose();
      veloLogits.dispose();
      pendingPCM = null;
    }

    requestAnimationFrame(frameLoop);
  }

  frameLoop();
}

