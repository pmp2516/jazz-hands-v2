import * as tf from '@tensorflow/tfjs';

/**
 * Mix-in to make a layer “plastic”: captures last inputs/outputs
 * and provides .kernel for weight updates.
 */
function makePlastic(layer, {eta=1e-3, alpha=1e-4, gamma=1e-2, eps=1e-6} = {}) {
  // Only apply to layers with a .kernel variable (Dense, Conv*)
  if (!layer.kernel) return layer;

  // Wrap apply() to capture x and y
  const origApply = layer.apply.bind(layer);
  layer.lastX = null;
  layer.lastY = null;
  layer.prevY = null;
  layer.eta = eta;
  layer.alpha = alpha;
  layer.gamma = gamma;
  layer.eps = eps;

  layer.apply = function(x) {
    const y = origApply(x);
    layer.lastX = x;
    layer.lastY = y;
    return y;
  };
  return layer;
}

/**
 * Creates and initializes the runner. Call .step() repeatedly.
 * @param {string} modelUrl – URL to your TF.js model.json
 */
export async function createGestureMidiRunner(modelUrl) {
  // 1) Load model
  const model = await tf.loadLayersModel(modelUrl);

  // 2) Make all layers plastic
  function traverseLayers(layer, fn) {
    fn(layer);
    // TF.js uses .layers for Sequential/Functional; custom Layers may expose _layers or subLayers
    const children = layer.layers || layer._layers || [];
    children.forEach(child => traverseLayers(child, fn));
  }

  // Then:
  traverseLayers(model, makePlastic);

  // 3) Initialize recurrent states to zeros (batch=1)
  const config = {
    nAudioHeads: model.inputs[2].shape[1],
    nVideoHeads: model.inputs[3].shape[1],
    latentDim: model.outputs[0].shape[1],
  };
  let audioState = tf.zeros([1, config.nAudioHeads, config.latentDim, config.latentDim]);
  let videoState = tf.zeros([1, config.nVideoHeads, config.latentDim, config.latentDim]);

  /**
   * Process one frame: audioFrame [T,1], videoFrame [D,H,W,3]
   * Returns { noteLogits: tf.Tensor1D, veloLogits: tf.Tensor1D }
   */
  async function step(audioFrame, videoFrame) {
    return tf.tidy(() => {
      // Ensure batch dim = 1
      const aIn = audioFrame.expandDims(0);
      const vIn = videoFrame.expandDims(0);

      // 1) Inference
      const [noteLogits, veloLogits, newAudioState, newVideoState] =
        model.predict([aIn, vIn, audioState, videoState]);

      // 2) Plasticity: update every plastic layer
      model.layers.forEach(layer => {
        if (!layer.kernel || !layer.lastX || !layer.lastY) return;

        // Two-step prediction error
        const yPrev = layer.prevY || layer.lastY.clone();
        const yCurr = layer.lastY;
        const e = yCurr.sub(yPrev);

        // Hebbian outer product: e ⊗ x
        const xFlat = layer.lastX.flatten();
        const eFlat = e.flatten();
        const outer = tf.outerProduct(eFlat, xFlat).mul(layer.eta);

        // Variance-maximization term: inv(cov + eps I)
        const Y = yPrev.flatten().expandDims(1);       // [D,1]
        const cov = Y.matMul(Y.transpose())
                     .div(Y.shape[0])
                     .add(tf.eye(Y.shape[0]).mul(layer.eps));
        const varGrad = tf.linalg.inv(cov).mul(layer.gamma);

        // Total weight update ΔW
        const W = layer.kernel.read();
        const dW = outer.add(varGrad).sub(W.mul(layer.alpha));

        // Apply update
        layer.kernel.assign(W.add(dW));

        // Save for next step
        layer.prevY?.dispose();
        layer.prevY = yCurr.clone();
      });

      // 3) Update recurrent state
      audioState.dispose();
      videoState.dispose();
      audioState = newAudioState;
      videoState = newVideoState;

      return {
        noteLogits: noteLogits.squeeze(),   // shape [notes]
        veloLogits: veloLogits.squeeze()    // shape [velocities]
      };
    });
  }

  return { step };
}
