// import * as tf from '@tensorflow/tfjs';
import { SpatialTransformer } from './stn';

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

/**
 * Helper delegation for Hebbian updates.
 * Determines update logic based on layer type.
 */
function hebbianUpdate(param, preAct, postAct, sleeping, type) {
  const lr = sleeping ? 1e-4 : 1e-3;
  let delta;
  switch (type) {
    case 'dense':
      delta = tf.matMul(preAct.transpose(), postAct).div(tf.scalar(preAct.shape[0]));
      break;
    case 'conv1d':
      const in1 = preAct.reshape([-1, preAct.shape[2]]);
      delta = tf.matMul(in1.transpose(), postAct).div(tf.scalar(in1.shape[0]));
      break;
    case 'conv2d':
      const in2 = preAct.reshape([preAct.shape[0], -1]);
      delta = tf.matMul(in2.transpose(), postAct).div(tf.scalar(preAct.shape[0]));
      break;
    case 'rnnBlock':
      const inR = preAct.reshape([-1, preAct.shape[2]]);
      const postR = postAct.reshape([-1, postAct.shape[1]]);
      delta = tf.matMul(inR.transpose(), postR).div(tf.scalar(inR.shape[0]));
      break;
    case 'stn':
      const inS = preAct.reshape([preAct.shape[0], -1]);
      delta = tf.matMul(inS.transpose(), postAct).div(tf.scalar(inS.shape[0]));
      break;
    default:
      throw new Error(`Unsupported Hebbian type: ${type}`);
  }
  const W = param.read();
  param.write(W.add(delta.mul(tf.scalar(lr))));
}

export class RNNBlock extends tf.layers.Layer {
  constructor(dModel, nHeads, kwargs) {
    super(kwargs);
    this.dModel = dModel;
    this.nHeads = nHeads;
    this.headDim = dModel / nHeads;
    this.qProj = this.addWeight('qProj', [dModel, dModel], 'float32', tf.initializers.glorotNormal());
    this.kProj = this.addWeight('kProj', [dModel, dModel], 'float32', tf.initializers.glorotNormal());
    this.vProj = this.addWeight('vProj', [dModel, dModel], 'float32', tf.initializers.glorotNormal());
    this.gamma = this.addWeight('gamma', [nHeads, this.headDim], 'float32', tf.initializers.ones());
    this.gateW = this.addWeight('gateW', [dModel, 2*dModel], 'float32', tf.initializers.glorotNormal());
    this.downW = this.addWeight('downW', [dModel, dModel], 'float32', tf.initializers.glorotNormal());
    this.ln1 = tf.layers.layerNormalization({axis:-1});
    this.ln2 = tf.layers.layerNormalization({axis:-1});
    this.initialState = tf.zeros([this.nHeads, this.headDim, this.headDim]);
  }

  step(x_t, prevState, sleeping=false) {
    // Hebbian update for input x_t and previous state before computing new
    hebbianUpdate(
      this.qProj,
      x_t,
      prevState.mean(-1),
      sleeping,
      'rnnBlock'
    );
    const xNorm = this.ln1.apply(x_t);
    const q = xNorm.dot(this.qProj.read());
    const k = xNorm.dot(this.kProj.read());
    const v = xNorm.dot(this.vProj.read());

    // Hebbian updates for kProj, vProj
    hebbianUpdate(
      this.kProj,
      x_t,
      prevState.mean(-1),
      sleeping,
      'rnnBlock'
    );
    hebbianUpdate(
      this.vProj,
      x_t,
      prevState.mean(-1),
      sleeping,
      'rnnBlock'
    );

    const [batch] = x_t.shape;
    const qh = q.reshape([batch, this.nHeads, this.headDim]);
    const kh = k.reshape([batch, this.nHeads, this.headDim]);
    const vh = v.reshape([batch, this.nHeads, this.headDim]);
    const γ = this.gamma.read().reshape([1, this.nHeads, this.headDim, 1]);
    let state = prevState.mul(γ);
    state = state.add(
      tf.einsum(
        'bhid,bhjd->bhij',
        kh.reshape([batch, this.nHeads, this.headDim, 1]),
        vh.reshape([batch, this.nHeads, 1, this.headDim])
      )
    );

    // Hebbian update for gamma
    hebbianUpdate(
      this.gamma,
      x_t,
      state.mean([1,3]),
      sleeping,
      'dense'
    );

    const o = state.matMul(
      qh.reshape([batch, this.nHeads, this.headDim, 1])
    ).reshape([batch, this.dModel]);
    const x2 = this.ln2.apply(x_t.add(o));
    const gateMlpin = x2.dot(this.gateW.read());
    const [gate, mlpIn] = tf.split(gateMlpin, 2, -1);

    // Hebbian update for gateW and downW
    hebbianUpdate(
      this.gateW,
      x_t,
      gate,
      sleeping,
      'rnnBlock'
    );
    hebbianUpdate(
      this.downW,
      mlpIn,
      mlpIn,
      sleeping,
      'dense'
    );

    const mlpOut = tf.mul(
      tf.silu(mlpIn),
      tf.sigmoid(gate)
    ).dot(this.downW.read());
    const out = tf.addN([x_t, o, mlpOut]);
    return [out, state];
  }

  applySequence(x_seq, initState=null, sleeping=false) {
    const [batch, time] = x_seq.shape;
    let state = initState ||
      tf.tile(this.initialState.expandDims(0), [batch,1,1,1]);
    const outputs = [];
    for (let t = 0; t < time; t++) {
      const x_t = x_seq
        .slice([0, t, 0], [batch, 1, this.dModel])
        .squeeze([1]);
      const [y, st] = this.step(x_t, state, sleeping);
      outputs.push(y.expandDims(1));
      state = st;
    }
    return [tf.concat(outputs, 1), state];
  }

  getConfig() {
    return { dModel: this.dModel, nHeads: this.nHeads };
  }
}

tf.serialization.registerClass(RNNBlock);

export class WakeSleepModel {
  constructor(config) {
    this.latentDim = config.latentDim;
    this.hopfield = new ModernHopfieldLayer({ dim: this.latentDim });
    this.convAudio = tf.layers.conv1d({
      filters: this.latentDim,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu'
    });
    this.poolAudio = tf.layers.maxPooling1d({ poolSize: 2 });
    this.stn = new SpatialTransformer({ inputChannels: config.videoChannels });
    this.convVideo2d = tf.layers.conv2d({
      filters: 16,
      kernelSize: [3, 3],
      padding: 'same',
      activation: 'relu'
    });
    this.poolVideo2d = tf.layers.maxPooling2d({ poolSize: [2, 2] });
    this.flattenVideo = tf.layers.flatten();
    this.projVideo = tf.layers.dense({ units: this.latentDim, activation: 'relu' });
    this.rnnA = new RNNBlock(this.latentDim, config.nAudioHeads);
    this.rnnV = new RNNBlock(this.latentDim, config.nVideoHeads);
    this.pitchDecoder = tf.layers.dense({
      units: config.notes.length,
      activation: 'softmax'
    });
    this.volumeDecoder = tf.layers.dense({
      units: config.velos.length,
      activation: 'softmax'
    });
  }

  step(aSeq, vSeq, prevStates=[null,null], sleeping=false) {
    return tf.tidy(() => {
      let za = this.convAudio.apply(aSeq);

      // Hebbian update immediately after convAudio
      hebbianUpdate(
        this.convAudio.trainableWeights.find(w => w.name.includes('kernel')),
        aSeq.reshape([-1, aSeq.shape[2]]),
        za.reshape([-1, za.shape[2]]),
        sleeping,
        'conv1d'
      );
      za = this.poolAudio.apply(za);

      const [b, t, h, w, c] = vSeq.shape;
      const fv = vSeq.reshape([b * t, h, w, c]);
      const stnOut = this.stn.apply(fv);

      // Hebbian update for STN
      this.stn.trainableWeights.forEach(wVar =>
        hebbianUpdate(
          wVar,
          fv.reshape([b * t, -1]),
          stnOut.reshape([b * t, -1]),
          sleeping,
          'stn'
        )
      );

      let zv = this.convVideo2d.apply(stnOut);

      // Hebbian update after convVideo2d
      hebbianUpdate(
        this.convVideo2d.trainableWeights.find(w => w.name.includes('kernel')),
        stnOut.reshape([b * t, -1]),
        zv.reshape([b * t, zv.shape[3]]),
        sleeping,
        'conv2d'
      );
      zv = this.poolVideo2d.apply(zv);
      zv = this.flattenVideo.apply(zv);
      zv = this.projVideo.apply(zv);

      // Hebbian update after projVideo
      hebbianUpdate(
        this.projVideo.trainableWeights.find(w => w.name.includes('kernel')),
        this.flattenVideo.apply(zv).reshape([b * t, this.latentDim]),
        zv.reshape([b * t, this.latentDim]),
        sleeping,
        'dense'
      );
      zv = zv.reshape([b, t, this.latentDim]);

      const [hA, sA] = this.rnnA.applySequence(za, prevStates[0], sleeping);
      const [hB, sB] = this.rnnV.applySequence(zv, prevStates[1], sleeping);
      const zA = hA.slice([0, hA.shape[1] - 1, 0], [b, 1, this.latentDim]).squeeze([1]);
      const zB = hB.slice([0, hB.shape[1] - 1, 0], [b, 1, this.latentDim]).squeeze([1]);
      const sim = tf.sum(zA.mul(zB), -1, true);
      const errA = zB.sub(sim.mul(zA));
      const errB = zA.sub(sim.mul(zB));
      const zF = zA.add(zB).div(2);
      const zR = this.hopfield.lookup(zF);
      const zFin = zF.add(zR).div(2);
      const seqLen = Math.max(hA.shape[1], hB.shape[1]);
      const zSeq = zFin.expandDims(1).tile([1, seqLen, 1]);

      // Hebbian coupling updates for RNNBlocks using predictive errors
      // Update RNN A using errA and audio latent zA
      ['qProj','kProj','vProj','gateW','downW'].forEach(name => {
        hebbianUpdate(
          this.rnnA[name].read ? this.rnnA[name] : this.rnnA[name],
          zA,
          errA,
          sleeping,
          'rnnBlock'
        );
      });
      // update gamma for RNN A
      hebbianUpdate(
        this.rnnA.gamma,
        zA,
        errA,
        sleeping,
        'dense'
      );
      // Update RNN B similarly
      ['qProj','kProj','vProj','gateW','downW'].forEach(name => {
        hebbianUpdate(
          this.rnnV[name].read ? this.rnnV[name] : this.rnnV[name],
          zB,
          errB,
          sleeping,
          'rnnBlock'
        );
      });
      hebbianUpdate(
        this.rnnV.gamma,
        zB,
        errB,
        sleeping,
        'dense'
      );
      const pLogits = this.pitchDecoder.apply(zSeq);
      const vLogits = this.volumeDecoder.apply(zSeq);

      // Hebbian for decoders
      hebbianUpdate(
        this.pitchDecoder.trainableWeights.find(w => w.name.includes('kernel')),
        zSeq.reshape([-1, this.latentDim]),
        pLogits.reshape([-1, pLogits.shape[2]]),
        sleeping,
        'dense'
      );
      hebbianUpdate(
        this.volumeDecoder.trainableWeights.find(w => w.name.includes('kernel')),
        zSeq.reshape([-1, this.latentDim]),
        vLogits.reshape([-1, vLogits.shape[2]]),
        sleeping,
        'dense'
      );

      return { pitchLogits: pLogits, volumeLogits: vLogits, states: [sA, sB] };
    });
  }
}

