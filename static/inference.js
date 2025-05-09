// import * as tf from '@tensorflow/tfjs';

let saving = false;
const SAVE_INTERVAL_MS = 60 * 1000; // save at most once per minute
let lastSave = 0;

/**
 * Create (or restore) the model + runner.
 * Returns an object with .step(audioTensor, videoTensor) method.
 */
export async function createGestureMidiRunner(config) {
  let model;
  try {
    // 1) Try to load from IndexedDB
    model = await tf.loadLayersModel('indexeddb://gesture-midi');
    console.log('Loaded model from IndexedDB.');
  } catch (e) {
    console.warn('No model in IndexedDB, building fresh one.');
    // 2) Dynamically import modeling.js and build fresh
    const { WakeSleepModel } = await import('./modeling.js');
    model = WakeSleepModel(config);

    // Dummy forward to initialize weights
    const audioDummy = tf.zeros([1, 1, 1]);
    const videoDummy = tf.zeros([1, 1, 1, 1, 3]);
    const audioStateDummy = tf.zeros([1, config.nAudioHeads, config.latentDim]);
    const videoStateDummy = tf.zeros([1, config.nVideoHeads, config.latentDim]);
    model.predict([audioDummy, videoDummy, audioStateDummy, videoStateDummy]);

    // 3) Save fresh model to IndexedDB
    await model.save('indexeddb://gesture-midi');
    console.log('Saved fresh model to IndexedDB.');
  }

  // 4) Initialize recurrent states
  let audioState = tf.zeros([1, config.nAudioHeads, config.latentDim, config.latentDim]);
  let videoState = tf.zeros([1, config.nVideoHeads, config.latentDim, config.latentDim]);

  /** Performs one inference + plasticity step, returns logits */
  async function step(audioFrame, videoFrame) {
    return tf.tidy(() => {
      const aIn = audioFrame.expandDims(0);
      const vIn = videoFrame.expandDims(0);

      const [noteLogits, veloLogits, newAudioState, newVideoState,] =
        model.predict([aIn, vIn, audioState, videoState]);

      // Update states
      audioState.dispose();
      videoState.dispose();
      audioState = newAudioState;
      videoState = newVideoState;

      // Periodic save
      // FIXME Race condition on `saving`
      const now = Date.now();
      if (now - lastSave > SAVE_INTERVAL_MS && !saving) {
        saving = true;
        model.save('indexeddb://gesture-midi')
          .then(() => {
            console.log('Model auto-saved to IndexedDB.');
            lastSave = now;
          })
          .catch(err => console.error('Save failed:', err))
          .finally(() => { saving = false; });
      }

      return {
        noteLogits: noteLogits.squeeze(),
        veloLogits: veloLogits.squeeze()
      };
    });
  }

  return { step };
}
