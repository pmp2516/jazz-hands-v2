// import * as tf from '@tensorflow/tfjs';

let saving = false;
const SAVE_INTERVAL_MS = 60 * 1000; // save at most once per minute
let lastSave = 0;

function makePlastic(layer, { eta = 1e-3, alpha = 1e-4, gamma = 1e-2, eps = 1e-6 } = {}) {
  if (!layer.kernel) return layer;
  const origApply = layer.apply.bind(layer);
  layer.lastX = null;
  layer.lastY = null;
  layer.prevY = null;
  layer.eta = eta; layer.alpha = alpha; layer.gamma = gamma; layer.eps = eps;
  layer.apply = function(x) {
    const y = origApply(x);
    layer.lastX = x;
    layer.lastY = y;
    return y;
  };
  return layer;
}

function plasticUpdate(layer) {
  if (!layer.kernel || !layer.lastX || !layer.lastY) return;
  const yPrev = layer.prevY || layer.lastY.clone();
  const yCurr = layer.lastY;
  const e = yCurr.sub(yPrev);

  const xFlat = layer.lastX.flatten();
  const eFlat = e.flatten();
  const outer = tf.outerProduct(eFlat, xFlat).mul(layer.eta);

  const Y = yPrev.flatten().expandDims(1);
  const cov = Y.matMul(Y.transpose())
    .div(Y.shape[0])
    .add(tf.eye(Y.shape[0]).mul(layer.eps));
  const varGrad = tf.linalg.inv(cov).mul(layer.gamma);

  const W = layer.kernel.read();
  const dW = outer.add(varGrad).sub(W.mul(layer.alpha));
  layer.kernel.assign(W.add(dW));

  layer.prevY?.dispose();
  layer.prevY = yCurr.clone();
}

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
    const { buildGestureToMIDI } = await import('./modeling.js');
    model = buildGestureToMIDI(config);

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

  // 4) Wrap plastic layers
  function traverseLayers(layer, fn) {
    fn(layer);
    const children = layer.layers || layer._layers || [];
    children.forEach(child => traverseLayers(child, fn));
  }

  traverseLayers(model, makePlastic);

  // 5) Initialize recurrent states
  let audioState = tf.zeros([1, config.nAudioHeads, config.latentDim, config.latentDim]);
  let videoState = tf.zeros([1, config.nVideoHeads, config.latentDim, config.latentDim]);

  /** Performs one inference + plasticity step, returns logits */
  async function step(audioFrame, videoFrame) {
    return tf.tidy(() => {
      const aIn = audioFrame.expandDims(0);
      const vIn = videoFrame.expandDims(0);

      const [noteLogits, veloLogits, newAudioState, newVideoState,] =
        model.predict([aIn, vIn, audioState, videoState]);

      // Plasticity updates
      traverseLayers(model, plasticUpdate);

      // At time t, you have audioLatent_t  and  videoLatent_t
      // At t+1, you have audioLatent_{t+1} and videoLatent_{t+1}

      // Prediction error between modalities:
      const e_latent = videoLatent_t1.sub(audioLatent_t1);

      // Hebbian update on your cross-modal projection W_cp:
      // pre-synaptic = audioLatent_t1, post-synaptic = videoLatent_t1
      const W_cp = model.getLayer('cross_modal').kernel;
      const ΔW_cp = tf.outerProduct(e_latent.flatten(), audioLatent_t1.flatten()).mul(η)
        .sub(W_cp.mul(α));
      // (plus any variance regularizer if you keep it)
      W_cp.assign(W_cp.add(ΔW_cp));


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
