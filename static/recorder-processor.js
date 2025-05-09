// Runs on the audio rendering thread, pushing raw Float32Array PCM to the main thread.
class RecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channelData = inputs[0][0];
    if (channelData) {
      // Copy to avoid holding onto the shared buffer
      this.port.postMessage(new Float32Array(channelData));
    }
    return true;
  }
}
registerProcessor('recorder-processor', RecorderProcessor);

