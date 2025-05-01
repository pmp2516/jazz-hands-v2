import * as tf from '@tensorflow/tfjs';

class RNNBlock extends tf.layers.Layer {
  constructor(dModel, nHeads, kwargs) {
    super(kwargs);
    this.dModel = dModel;
    this.nHeads = nHeads;
    this.headDim = dModel / nHeads;

    // Projections: q, k, v
    this.qProj = this.addWeight('qProj', [dModel, dModel], 'float32', tf.initializers.glorotNormal());
    this.kProj = this.addWeight('kProj', [dModel, dModel], 'float32', tf.initializers.glorotNormal());
    this.vProj = this.addWeight('vProj', [dModel, dModel], 'float32', tf.initializers.glorotNormal());

    // Learnable decay γ per head
    this.gamma = this.addWeight('gamma', [nHeads, this.headDim], 'float32', tf.initializers.ones());

    // Gated MLP
    this.gateW = this.addWeight('gateW', [dModel, 2 * dModel], 'float32', tf.initializers.glorotNormal());
    this.downW = this.addWeight('downW', [dModel, dModel], 'float32', tf.initializers.glorotNormal());

    // Layer norms
    this.ln1 = tf.layers.layerNormalization({ axis: -1 });
    this.ln2 = tf.layers.layerNormalization({ axis: -1 });

    // Initialize recurrent state buffer
    this.state = tf.variable(tf.zeros([nHeads, this.headDim, this.headDim]));
  }

  build(inputShape) { super.build(inputShape); }

  call([x, prevState]) {
    // 1) Norm + linear projections
    const xNorm = this.ln1.apply(x);
    const q = xNorm.dot(this.qProj.read());
    const k = xNorm.dot(this.kProj.read());
    const v = xNorm.dot(this.vProj.read());

    // 2) Split into heads
    const batchSize = x.shape[0];
    const qh = tf.reshape(q, [batchSize, this.nHeads, this.headDim]);
    const kh = tf.reshape(k, [batchSize, this.nHeads, this.headDim]);
    const vh = tf.reshape(v, [batchSize, this.nHeads, this.headDim]);

    // 3) WKV‐style recurrent update
    // decay: state ← state * γ
    const γ = this.gamma.read().reshape([1, this.nHeads, this.headDim, 1]);
    let state = prevState.mul(γ);
    // outer(k, v): [batch, heads, headDim, 1] × [batch, heads, 1, headDim]
    const k_h = kh.reshape([batchSize, this.nHeads, this.headDim, 1]);
    const v_h = vh.reshape([batchSize, this.nHeads, 1, this.headDim]);
    state = state.add(tf.einsum('bhid,bhjd->bhij', k_h, v_h));

    // attention‐like output: o = (state ⋅ q) over last dim
    const q_h = qh.reshape([batchSize, this.nHeads, this.headDim, 1]);
    const o = state.matMul(q_h).reshape([batchSize, this.nHeads, this.headDim]);
    const attnOut = o.reshape([batchSize, this.dModel]);

    // 4) Gated MLP + residual
    const x2 = this.ln2.apply(x.add(attnOut));
    const gateMlpin = x2.dot(this.gateW.read());
    const [gate, mlpIn] = tf.split(gateMlpin, 2, -1);
    const mlpOut = tf.mul(tf.silu(mlpIn), tf.sigmoid(gate)).dot(this.downW.read());

    const out = tf.addN([x, attnOut, mlpOut]);
    return [out, state];
  }

  getConfig() {
    return { dModel: this.dModel, nHeads: this.nHeads };
  }
}
tf.serialization.registerClass(RNNBlock);

export function buildGestureToMIDI(config) {
  // 1) Inputs with dynamic spatial/temporal dims:
  //    - audio: [timeSteps, 1] → any length time-series, single channel
  //    - video: [depth, height, width, 3] → any depth/size RGB volume
  const audioIn = tf.input({ shape: [null, 1], name: 'audio_input' });
  const videoIn = tf.input({ shape: [null, null, null, 3], name: 'video_input' });

  // 2) Recurrent state inputs remain fixed by heads × latentDim²
  const audioStateIn = tf.input({
    shape: [config.nAudioHeads, config.latentDim, config.latentDim],
    name: 'audio_state'
  });
  const videoStateIn = tf.input({
    shape: [config.nVideoHeads, config.latentDim, config.latentDim],
    name: 'video_state'
  });

  // 3) Video encoder – conv3d with dynamic dims, then global mean‐pool
  let v = tf.layers.conv3d({
    filters: config.videoChannels,
    kernelSize: [3, 5, 5],
    strides: [1, 2, 2],
    padding: 'same'
  }).apply(videoIn);
  v = tf.relu(v);
  v = tf.layers.conv3d({
    filters: config.videoChannels * 2,
    kernelSize: 3,
    padding: 'same'
  }).apply(v);
  v = tf.relu(v);
  // mean over depth, height, width
  v = tf.mean(v, [1, 2, 3]);
  v = tf.layers.dense({ units: config.latentDim, name: 'cross_modal' }).apply(v);

  // 4) Video recurrent core
  const [videoLatent, videoState] = new RNNBlock(
    config.latentDim, config.nVideoHeads
  ).apply([v, videoStateIn]);

  // 5) Audio encoder – conv1d with dynamic time length
  let a = tf.layers.conv1d({
    filters: config.audioChannels,
    kernelSize: 5,
    padding: 'same'
  }).apply(audioIn);
  a = tf.relu(a);
  a = tf.layers.conv1d({
    filters: config.audioChannels * 2,
    kernelSize: 3,
    padding: 'same'
  }).apply(a);
  a = tf.relu(a);
  // mean over time
  a = tf.mean(a, [1]);
  a = tf.layers.dense({ units: config.latentDim }).apply(a);

  // 6) Audio recurrent core
  const [audioLatent, audioState] = new RNNBlock(
    config.latentDim, config.nAudioHeads
  ).apply([a, audioStateIn]);

  // 7) Cross‐modal fusion with Hopfield TODO
  // const hop = hopfield(videoLatent);

  const latent = videoLatent; // tf.add(videoLatent, hop);

  // 8) Output heads
  const noteLogits = tf.layers.dense({ units: config.notes.length })
    .apply(latent);
  const veloLogits = tf.layers.dense({ units: config.velocities.length })
    .apply(latent);

  return tf.model({
    inputs: [audioIn, videoIn, audioStateIn, videoStateIn],
    outputs: [noteLogits, veloLogits, audioState, videoState, audioLatent, videoLatent]
  });
}

