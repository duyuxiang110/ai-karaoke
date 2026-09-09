/**
 * AudioWorklet Processor - 麦克风 PCM 采集
 *
 * 将浮点音频数据 [-1, 1] 转换为 16-bit PCM，
 * 按 2048 样本块通过 transferable ArrayBuffer 发送到主线程。
 *
 * 采样率由 AudioContext 决定（目标 44100Hz, mono）。
 * 渲染量子为 128 样本（AudioWorklet 标准）。
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this._chunks = []
    this._chunksTotalLength = 0
    this._targetSize = 2048
  }

  process(inputs) {
    const input = inputs[0]
    if (!input || !input[0]) {
      return true
    }

    const float32 = input[0]

    // Float32 [-1, 1] → Int16 [-32768, 32767]
    const int16 = new Int16Array(float32.length)
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]))
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff
    }

    this._chunks.push(int16)
    this._chunksTotalLength += int16.length

    if (this._chunksTotalLength >= this._targetSize) {
      const merged = new Int16Array(this._targetSize)
      let offset = 0
      const remainder = []

      for (const chunk of this._chunks) {
        const remaining = this._targetSize - offset
        if (chunk.length <= remaining) {
          merged.set(chunk, offset)
          offset += chunk.length
        } else {
          merged.set(chunk.subarray(0, remaining), offset)
          offset = this._targetSize
          if (remaining < chunk.length) {
            remainder.push(chunk.subarray(remaining))
          }
        }
        if (offset >= this._targetSize) {
          break
        }
      }

      this._chunks = remainder
      this._chunksTotalLength = remainder.reduce((sum, c) => sum + c.length, 0)

      const buffer = merged.buffer
      this.port.postMessage(buffer, [buffer])
    }

    return true
  }
}

registerProcessor('pcm-processor', PCMProcessor)
