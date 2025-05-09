// import * as tf from '@tensorflow/tfjs';

class ModernHopfieldLayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.dim         = config.dim;
    this.maxMemory  = config.maxMemory || 1024;
    this.beta       = config.beta || 1.0;

    // 1) Preallocate fixed-shape buffers for keys & values:
    this.keysVar   = this.addWeight(
      'keys', [this.maxMemory, this.dim], 'float32',
      tf.initializers.zeros(), undefined, true
    );
    this.valuesVar = this.addWeight(
      'values', [this.maxMemory, this.dim], 'float32',
      tf.initializers.zeros(), undefined, true
    );

    // 2) Track how many patterns have been stored so far
    this.memSize   = 0;
    // 3) And an insertion index (for ring-buffer style)
    this.insertIdx = 0;
  }

  call([input]) {
    return tf.tidy(() => {
      const x = input;          // [B, D]

      // If empty memory, no recall—just write and return zeros
      if (this.memSize === 0) {
        this._writeMemory(x, x);
        return tf.zerosLike(x);
      }

      // 4) Compute attention-based recall:
      // [B, N] = x [B,D] · keys^T [D,N]
      const scores = tf.matMul(x, this.keysVar.read(), false, true)
                         .mul(this.beta);
      const p      = tf.softmax(scores, 1);      // [B, N]
      const recall = tf.matMul(p, this.valuesVar.read()); // [B, D]

      // 5) Append new patterns:
      this._writeMemory(x, x);

      return recall;
    });
  }

  _writeMemory(newKeys, newValues) {
    const B = newKeys.shape[0];
    // Compute the wrap‑around indices
    const endIdx = Math.min(this.insertIdx + B, this.maxMemory);
    const sliceSize = endIdx - this.insertIdx;

    // 1) Write first chunk
    this.keysVar.write(
      this.keysVar.read()
        .slice([0,0], [this.maxMemory, this.dim])
        .pad([[0,0],[0,0]]), // no-op, just to illustrate slicing
      // overwrite rows [insertIdx : endIdx]
      [[this.insertIdx, sliceSize, 0, this.dim]],
      newKeys.slice([0,0],[sliceSize, this.dim])
    );
    this.valuesVar.write(
      this.valuesVar.read(),
      [[this.insertIdx, sliceSize, 0, this.dim]],
      newValues.slice([0,0],[sliceSize, this.dim])
    );

    // 2) If B > space left, wrap and write remaining at front
    if (sliceSize < B) {
      const rem = B - sliceSize;
      this.keysVar.write(
        this.keysVar.read(),
        [[0, rem, 0, this.dim]],
        newKeys.slice([sliceSize,0],[rem, this.dim])
      );
      this.valuesVar.write(
        this.valuesVar.read(),
        [[0, rem, 0, this.dim]],
        newValues.slice([sliceSize,0],[rem, this.dim])
      );
      this.insertIdx = rem;
    } else {
      this.insertIdx = endIdx % this.maxMemory;
    }

    // Update memSize (capped to maxMemory)
    this.memSize = Math.min(this.memSize + B, this.maxMemory);
  }

  static get className() {
    return 'ModernHopfieldLayer';
  }
}
tf.serialization.registerClass(ModernHopfieldLayer);

class MeanLayer extends tf.layers.Layer {
  constructor(axes, config) {
    super(config);
    this.axes = axes;
  }
  // Adjust the output shape by setting reduced dims to 1
  computeOutputShape(inputShape) {
    const outShape = inputShape.slice();
    this.axes.forEach(ax => { outShape[ax] = 1; });
    return outShape;
  }
  call([input]) {
    return tf.mean(input, this.axes);
  }
  static get className() { return 'MeanLayer'; }
}
tf.serialization.registerClass(MeanLayer);

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
    this.state = tf.variable(tf.zeros([1, nHeads, this.headDim, this.headDim]));
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
    let state = prevState.reshape([1, this.nHeads, this.headDim, this.headDim]);
    state = state.mul(γ);
    // outer(k, v): [batch, heads, headDim, 1] × [batch, heads, 1, headDim]
    const k_h = kh.reshape([batchSize, this.nHeads, this.headDim, 1]);
    const v_h = vh.reshape([batchSize, this.nHeads, 1, this.headDim]);
    state = state.add(k_h.mul(v_h));

    // attention‐like output: o = (state ⋅ q) over last dim
    const q_h = qh.reshape([batchSize, this.nHeads, this.headDim, 1]);
    const o = state.matMul(q_h).reshape([batchSize, this.nHeads, this.headDim]);
    const attnOut = o.reshape([batchSize, this.dModel]);

    // 4) Gated MLP + residual
    const x2 = this.ln2.apply(x.add(attnOut));
    const gateMlpin = x2.dot(this.gateW.read());
    const [gate, mlpIn] = tf.split(gateMlpin, 2, -1);
    const mlpOut = tf.mul(mlpIn * tf.sigmoid(mlpIn), tf.sigmoid(gate)).dot(this.downW.read());

    const out = tf.addN([x, attnOut, mlpOut]);
    return [out, state];
  }

  getConfig() {
    return { dModel: this.dModel, nHeads: this.nHeads };
  }

  static get className() {
    return 'RNNBlock';
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
    shape: [config.nAudioHeads, config.latentDim],
    name: 'audio_state'
  });
  const videoStateIn = tf.input({
    shape: [config.nVideoHeads, config.latentDim],
    name: 'video_state'
  });

  // 3) Video encoder – conv3d with dynamic dims, then global mean‐pool
  let v = tf.layers.conv3d({
    filters: config.videoChannels,
    kernelSize: [3, 5, 5],
    strides: [1, 2, 2],
    padding: 'same'
  }).apply(videoIn);
  v = tf.layers.reLU().apply(v);
  v = tf.layers.conv3d({
    filters: config.videoChannels * 2,
    kernelSize: 3,
    padding: 'same'
  }).apply(v);
  v = tf.layers.reLU().apply(v);
  // mean over depth, height, width
  v = new MeanLayer([1,2,3]).apply(v);
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
  a = tf.layers.reLU().apply(a);
  a = tf.layers.conv1d({
    filters: config.audioChannels * 2,
    kernelSize: 3,
    padding: 'same'
  }).apply(a);
  a = tf.layers.reLU().apply(a);
  // mean over time
  v = new MeanLayer([1]).apply(v);
  a = tf.layers.dense({ units: config.latentDim }).apply(a);

  // 6) Audio recurrent core
  const [audioLatent, audioState] = new RNNBlock(
    config.latentDim, config.nAudioHeads
  ).apply([a, audioStateIn]);

  const recall = new ModernHopfieldLayer({ dim: config.latentDim }).apply(videoLatent);

  const latent = tf.layers.average().apply([videoLatent, recall]);

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

