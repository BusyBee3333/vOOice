import Foundation
import CoreML
import AVFoundation
import FluidAudio
import os.log

enum ParakeetTranscriptionError: LocalizedError {
    case modelValidationFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelValidationFailed(let message):
            return message
        }
    }
}

class ParakeetTranscriptionService: TranscriptionService {
    private var asrManager: AsrManager?
    private var vadManager: VadManager?
    private var activeVersion: AsrModelVersion?
    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink.parakeet", category: "ParakeetTranscriptionService")

    init() {
        logger.notice("🆕 ParakeetTranscriptionService initialized (v4 - raw audio, no preprocessing)")
    }

    // MARK: - Streaming Properties (using StreamingEouAsrManager for low-latency 160ms chunks)
    private var streamingEouManager: StreamingEouAsrManager?
    private var streamingTask: Task<Void, Never>?
    private var streamingContinuation: AsyncStream<String>.Continuation?
    private var streamAudioCallCount = 0
    private var lastPartialTranscript: String = ""

    private func version(for model: any TranscriptionModel) -> AsrModelVersion {
        model.name.lowercased().contains("v2") ? .v2 : .v3
    }

    private func ensureModelsLoaded(for version: AsrModelVersion) async throws {
        if let manager = asrManager, activeVersion == version {
            return
        }

        cleanup()

        // Validate models before loading
        let isValid = try await AsrModels.isModelValid(version: version)

        if !isValid {
            logger.error("Model validation failed for \(version == .v2 ? "v2" : "v3"). Models are corrupted.")
            throw ParakeetTranscriptionError.modelValidationFailed("Parakeet models are corrupted. Please delete and re-download the model.")
        }

        let manager = AsrManager(config: .default)
        let models = try await AsrModels.loadFromCache(
            configuration: nil,
            version: version
        )
        try await manager.initialize(models: models)
        self.asrManager = manager
        self.activeVersion = version
    }

    func loadModel(for model: ParakeetModel) async throws {
        try await ensureModelsLoaded(for: version(for: model))
    }

    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        let targetVersion = version(for: model)
        try await ensureModelsLoaded(for: targetVersion)

        guard let asrManager = asrManager else {
            throw ASRError.notInitialized
        }

        let audioSamples = try readAudioSamples(from: audioURL)

        let durationSeconds = Double(audioSamples.count) / 16000.0
        let isVADEnabled = UserDefaults.standard.object(forKey: "IsVADEnabled") as? Bool ?? true

        var speechAudio = audioSamples
        if durationSeconds >= 20.0, isVADEnabled {
            let vadConfig = VadConfig(defaultThreshold: 0.7)
            if vadManager == nil {
                do {
                    vadManager = try await VadManager(config: vadConfig)
                } catch {
                    logger.notice("VAD init failed; falling back to full audio: \(error.localizedDescription)")
                    vadManager = nil
                }
            }

            if let vadManager {
                do {
                    let segments = try await vadManager.segmentSpeechAudio(audioSamples)
                    speechAudio = segments.isEmpty ? audioSamples : segments.flatMap { $0 }
                } catch {
                    logger.notice("VAD segmentation failed; using full audio: \(error.localizedDescription)")
                    speechAudio = audioSamples
                }
            }
        }

        let result = try await asrManager.transcribe(speechAudio)

        return result.text
    }

    private func readAudioSamples(from url: URL) throws -> [Float] {
        do {
            let data = try Data(contentsOf: url)
            guard data.count > 44 else {
                throw ASRError.invalidAudioData
            }

            let floats = stride(from: 44, to: data.count, by: 2).map {
                return data[$0..<$0 + 2].withUnsafeBytes {
                    let short = Int16(littleEndian: $0.load(as: Int16.self))
                    return max(-1.0, min(Float(short) / 32767.0, 1.0))
                }
            }

            return floats
        } catch {
            throw ASRError.invalidAudioData
        }
    }

    func cleanup() {
        asrManager?.cleanup()
        asrManager = nil
        vadManager = nil
        activeVersion = nil
    }

    // MARK: - Streaming Transcription (Low-Latency EOU Mode)

    /// Gets the directory for EOU streaming models
    private func getEouModelsDirectory() -> URL {
        let applicationSupportURL = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let appDirectory = applicationSupportURL.appendingPathComponent("FluidAudio", isDirectory: true)
        return appDirectory.appendingPathComponent("Models/parakeet-eou-streaming/320ms", isDirectory: true)
    }

    /// Downloads EOU models if not already present
    private func ensureEouModelsDownloaded() async throws -> URL {
        let modelsDir = getEouModelsDirectory()
        let encoderPath = modelsDir.appendingPathComponent("streaming_encoder.mlmodelc")

        if !FileManager.default.fileExists(atPath: encoderPath.path) {
            logger.notice("🎙️ Downloading Parakeet EOU 320ms models for streaming preview...")
            let baseDir = modelsDir.deletingLastPathComponent().deletingLastPathComponent()
            try await DownloadUtils.downloadRepo(.parakeetEou320, to: baseDir)
            logger.notice("🎙️ EOU 320ms models downloaded successfully")
        }

        return modelsDir
    }

    /// Starts a streaming transcription session using StreamingEouAsrManager for near-instant results.
    /// Uses 160ms chunks for lowest latency (~160ms between updates).
    /// Returns an AsyncStream that emits transcription text updates as they arrive.
    func startStreaming(model: ParakeetModel) async throws -> AsyncStream<String> {
        logger.notice("🎙️ Starting low-latency EOU streaming transcription")

        // Reset state
        streamAudioCallCount = 0
        lastPartialTranscript = ""

        // Download EOU models if needed
        let modelsDir = try await ensureEouModelsDownloaded()

        // Create StreamingEouAsrManager with 320ms chunks for accuracy
        // In HYBRID mode: streaming provides visual feedback, batch provides final accuracy
        // EOU debounce of 1280ms means end-of-utterance detection after ~1.3s of silence
        let manager = StreamingEouAsrManager(chunkSize: .ms320, eouDebounceMs: 1280)
        streamingEouManager = manager

        // Load Parakeet EOU models
        try await manager.loadModels(modelDir: modelsDir)

        logger.notice("🎙️ EOU streaming preview started with 160ms chunks (batch will provide accuracy)")

        // Create stream using makeStream for proper continuation management
        let (stream, continuation) = AsyncStream<String>.makeStream()
        self.streamingContinuation = continuation

        // Set up partial callback BEFORE returning the stream (fixes race condition)
        await manager.setPartialCallback { [weak self] partialText in
            guard let self = self else { return }
            let trimmed = partialText.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty && trimmed != self.lastPartialTranscript {
                self.lastPartialTranscript = trimmed
                self.logger.notice("🎙️ Partial update: '\(trimmed.prefix(50))...'")
                continuation.yield(trimmed)
            }
        }

        // Note: Removed onTermination callback that called cancelStreaming()
        // This was causing a race condition where the manager was nullified
        // before finishStreaming() could call manager.finish()
        // Cleanup is handled by finishStreaming()'s defer block instead

        logger.notice("🎙️ Callback registered, streaming ready")
        return stream
    }

    /// Feeds raw audio samples to the streaming EOU transcription engine.
    /// Called from the audio thread - creates AVAudioPCMBuffer and forwards to manager.
    /// SDK handles resampling to 16kHz internally. No preprocessing applied (research shows it hurts accuracy).
    func streamAudio(samples: UnsafePointer<Float32>, frameCount: UInt32, sampleRate: Double, channels: UInt32) {
        streamAudioCallCount += 1

        // Create buffer at original sample rate
        // SDK's process() method handles resampling to 16kHz internally via AudioConverter
        guard let audioBuffer = createOriginalFormatBuffer(samples: samples, frameCount: frameCount, sampleRate: sampleRate, channels: channels) else {
            if streamAudioCallCount <= 5 {
                logger.warning("Failed to create audio buffer at chunk #\(self.streamAudioCallCount)")
            }
            return
        }

        guard streamingEouManager != nil else {
            return
        }

        // StreamingEouAsrManager.process is an actor method, dispatch to avoid blocking audio thread
        Task.detached { [weak self, audioBuffer] in
            do {
                _ = try await self?.streamingEouManager?.process(audioBuffer: audioBuffer)
            } catch {
                self?.logger.warning("EOU process error: \(error.localizedDescription)")
            }
        }
    }

    /// Creates a MONO AVAudioPCMBuffer from interleaved input samples.
    /// No preprocessing - research shows gain control and noise reduction HURT ASR accuracy.
    /// Just converts stereo to mono if needed, passes raw audio otherwise.
    private func createOriginalFormatBuffer(samples: UnsafePointer<Float32>, frameCount: UInt32, sampleRate: Double, channels: UInt32) -> AVAudioPCMBuffer? {
        // Create MONO non-interleaved format - simplest format for ASR
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,  // Output is MONO
            interleaved: false
        ) else {
            return nil
        }

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return nil
        }

        buffer.frameLength = frameCount

        guard let monoData = buffer.floatChannelData?[0] else {
            return nil
        }

        let channelCount = Int(channels)
        let frames = Int(frameCount)

        if channelCount == 1 {
            // Already mono - direct copy (no gain, no processing)
            for frame in 0..<frames {
                monoData[frame] = samples[frame]
            }
        } else {
            // Stereo or multi-channel - mix to mono (simple average, no gain)
            let channelWeight = 1.0 / Float(channelCount)
            for frame in 0..<frames {
                var sum: Float = 0
                for channel in 0..<channelCount {
                    // Input is interleaved: L0 R0 L1 R1 L2 R2 ...
                    sum += samples[frame * channelCount + channel]
                }
                monoData[frame] = sum * channelWeight
            }
        }

        return buffer
    }

    /// Finishes the streaming session and returns the final transcription.
    func finishStreaming() async throws -> String {
        defer {
            streamingTask?.cancel()
            streamingTask = nil
            streamingContinuation?.finish()
            streamingContinuation = nil
            streamingEouManager = nil
            lastPartialTranscript = ""
        }

        guard let manager = streamingEouManager else {
            return ""
        }
        let finalText = try await manager.finish()
        logger.notice("🎙️ EOU streaming finished with \(finalText.count) characters")
        return finalText
    }

    /// Cancels the streaming session without returning results.
    func cancelStreaming() async {
        streamingTask?.cancel()
        streamingTask = nil
        streamingContinuation?.finish()
        streamingContinuation = nil

        if let manager = streamingEouManager {
            await manager.reset()
            streamingEouManager = nil
            lastPartialTranscript = ""
            logger.notice("🎙️ Cancelled EOU streaming transcription")
        }
    }

}
