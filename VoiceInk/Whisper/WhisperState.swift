import Foundation
import SwiftUI
import AVFoundation
import SwiftData
import AppKit
import KeyboardShortcuts
import os

// MARK: - Recording State Machine
enum RecordingState: Equatable {
    case idle
    case recording
    case transcribing
    case enhancing
    case busy
}

@MainActor
class WhisperState: NSObject, ObservableObject {
    @Published var recordingState: RecordingState = .idle
    @Published var isModelLoaded = false
    @Published var loadedLocalModel: WhisperModel?
    @Published var currentTranscriptionModel: (any TranscriptionModel)?
    @Published var isModelLoading = false
    @Published var availableModels: [WhisperModel] = []
    @Published var allAvailableModels: [any TranscriptionModel] = PredefinedModels.models
    @Published var clipboardMessage = ""
    @Published var miniRecorderError: String?
    @Published var shouldCancelRecording = false

    // MARK: - Streaming Transcription Properties
    private var streamingUpdateTask: Task<Void, Never>?
    private var lastStreamedText: String = ""
    private var isStreamingActive: Bool = false


    @Published var recorderType: String = UserDefaults.standard.string(forKey: "RecorderType") ?? "mini" {
        didSet {
            if isMiniRecorderVisible {
                if oldValue == "notch" {
                    notchWindowManager?.hide()
                    notchWindowManager = nil
                } else {
                    miniWindowManager?.hide()
                    miniWindowManager = nil
                }
                Task { @MainActor in
                    try? await Task.sleep(nanoseconds: 50_000_000)
                    showRecorderPanel()
                }
            }
            UserDefaults.standard.set(recorderType, forKey: "RecorderType")
        }
    }
    
    @Published var isMiniRecorderVisible = false {
        didSet {
            if isMiniRecorderVisible {
                showRecorderPanel()
            } else {
                hideRecorderPanel()
            }
        }
    }
    
    var whisperContext: WhisperContext?
    let recorder = Recorder()
    var recordedFile: URL? = nil
    let whisperPrompt = WhisperPrompt()
    
    // Prompt detection service for trigger word handling
    private let promptDetectionService = PromptDetectionService()
    
    let modelContext: ModelContext
    
    internal var serviceRegistry: TranscriptionServiceRegistry!
    
    private var modelUrl: URL? {
        let possibleURLs = [
            Bundle.main.url(forResource: "ggml-base.en", withExtension: "bin", subdirectory: "Models"),
            Bundle.main.url(forResource: "ggml-base.en", withExtension: "bin"),
            Bundle.main.bundleURL.appendingPathComponent("Models/ggml-base.en.bin")
        ]
        
        for url in possibleURLs {
            if let url = url, FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }
        return nil
    }
    
    private enum LoadError: Error {
        case couldNotLocateModel
    }
    
    let modelsDirectory: URL
    let recordingsDirectory: URL
    let enhancementService: AIEnhancementService?
    var licenseViewModel: LicenseViewModel
    let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "WhisperState")
    var notchWindowManager: NotchWindowManager?
    var miniWindowManager: MiniWindowManager?
    
    // For model progress tracking
    @Published var downloadProgress: [String: Double] = [:]
    @Published var parakeetDownloadStates: [String: Bool] = [:]

    /// Returns true if the current transcription model supports streaming (Parakeet only)
    var isStreamingSupported: Bool {
        currentTranscriptionModel?.provider == .parakeet
    }

    init(modelContext: ModelContext, enhancementService: AIEnhancementService? = nil) {
        self.modelContext = modelContext
        let appSupportDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("com.prakashjoshipax.VoiceInk")
        
        self.modelsDirectory = appSupportDirectory.appendingPathComponent("WhisperModels")
        self.recordingsDirectory = appSupportDirectory.appendingPathComponent("Recordings")
        
        self.enhancementService = enhancementService
        self.licenseViewModel = LicenseViewModel()
        
        super.init()
        
        // Configure the session manager
        if let enhancementService = enhancementService {
            PowerModeSessionManager.shared.configure(whisperState: self, enhancementService: enhancementService)
        }

        // Initialize the transcription service registry
        self.serviceRegistry = TranscriptionServiceRegistry(whisperState: self, modelsDirectory: self.modelsDirectory)
        
        setupNotifications()
        createModelsDirectoryIfNeeded()
        createRecordingsDirectoryIfNeeded()
        loadAvailableModels()
        loadCurrentTranscriptionModel()
        refreshAllAvailableModels()
    }
    
    private func createRecordingsDirectoryIfNeeded() {
        do {
            try FileManager.default.createDirectory(at: recordingsDirectory, withIntermediateDirectories: true, attributes: nil)
        } catch {
            logger.error("Error creating recordings directory: \(error.localizedDescription)")
        }
    }
    
    func toggleRecord(powerModeId: UUID? = nil) async {
        if recordingState == .recording {
            await recorder.stopRecording()

            // Handle cancellation - clean up streaming if active
            if shouldCancelRecording {
                if isStreamingActive {
                    await cancelStreamingTranscription()
                }
                await MainActor.run {
                    recordingState = .idle
                }
                await cleanupModelResources()
                return
            }

            // Handle streaming transcription completion
            if isStreamingActive {
                await handleStreamingCompletion()
                return
            }

            // Non-streaming (batch) transcription
            if let recordedFile {
                let audioAsset = AVURLAsset(url: recordedFile)
                let duration = (try? CMTimeGetSeconds(await audioAsset.load(.duration))) ?? 0.0

                let transcription = Transcription(
                    text: "",
                    duration: duration,
                    audioFileURL: recordedFile.absoluteString,
                    transcriptionStatus: .pending
                )
                modelContext.insert(transcription)
                try? modelContext.save()
                NotificationCenter.default.post(name: .transcriptionCreated, object: transcription)

                await transcribeAudio(on: transcription)
            } else {
                logger.error("❌ No recorded file found after stopping recording")
                await MainActor.run {
                    recordingState = .idle
                }
            }
        } else {
            guard currentTranscriptionModel != nil else {
                await MainActor.run {
                    NotificationManager.shared.showNotification(
                        title: "No AI Model Selected",
                        type: .error
                    )
                }
                return
            }
            shouldCancelRecording = false
            requestRecordPermission { [self] granted in
                if granted {
                    Task {
                        do {
                            // --- Prepare permanent file URL ---
                            let fileName = "\(UUID().uuidString).wav"
                            let permanentURL = self.recordingsDirectory.appendingPathComponent(fileName)
                            self.recordedFile = permanentURL

                            // IMPORTANT: Set up streaming BEFORE starting recording to avoid losing early audio
                            // Check if we're using a Parakeet model and set up streaming first
                            let isParakeetModel = self.currentTranscriptionModel is ParakeetModel
                            if isParakeetModel {
                                self.logger.notice("🎙️ Detected Parakeet model, setting up streaming BEFORE recording...")
                                await self.startStreamingTranscription()
                            }

                            try await self.recorder.startRecording(toOutputFile: permanentURL)
                            self.logger.notice("🎙️ Recording started\(isParakeetModel ? " (streaming already active)" : "")")

                            await MainActor.run {
                                self.recordingState = .recording
                            }

                            // Detect and apply Power Mode for current app/website in background
                            Task {
                                await ActiveWindowService.shared.applyConfiguration(powerModeId: powerModeId)
                            }

                            // Load model and capture context in background without blocking
                            Task.detached { [weak self] in
                                guard let self = self else {
                                    print("⚠️ Self was deallocated in Task.detached!")
                                    return
                                }

                                // Debug: Check what model type we have
                                let modelType = await type(of: self.currentTranscriptionModel)
                                let modelName = await self.currentTranscriptionModel?.displayName ?? "nil"
                                print("🔍 DEBUG: Model type = \(modelType), name = \(modelName)")
                                print("🔍 DEBUG: Is ParakeetModel? \(await self.currentTranscriptionModel is ParakeetModel)")

                                // Only load model if it's a local model and not already loaded
                                // Note: Parakeet streaming is now set up BEFORE recording starts (above)
                                if let model = await self.currentTranscriptionModel, model.provider == .local {
                                    if let localWhisperModel = await self.availableModels.first(where: { $0.name == model.name }),
                                       await self.whisperContext == nil {
                                        do {
                                            try await self.loadModel(localWhisperModel)
                                        } catch {
                                            await self.logger.error("❌ Model loading failed: \(error.localizedDescription)")
                                        }
                                    }
                                } else if !(await self.currentTranscriptionModel is ParakeetModel) {
                                    // Non-Parakeet, non-local models - just log
                                    let modelDesc = await self.currentTranscriptionModel?.displayName ?? "nil"
                                    await self.logger.notice("🎙️ Model is not local or Parakeet: \(modelDesc)")
                                }

                                if let enhancementService = await self.enhancementService {
                                    await MainActor.run {
                                        enhancementService.captureClipboardContext()
                                    }
                                    await enhancementService.captureScreenContext()
                                }
                            }

                        } catch {
                            self.logger.error("❌ Failed to start recording: \(error.localizedDescription)")
                            await NotificationManager.shared.showNotification(title: "Recording failed to start", type: .error)
                            await self.dismissMiniRecorder()
                            // Do not remove the file on a failed start, to preserve all recordings.
                            self.recordedFile = nil
                        }
                    }
                } else {
                    logger.error("❌ Recording permission denied.")
                }
            }
        }
    }
    
    private func requestRecordPermission(response: @escaping (Bool) -> Void) {
        response(true)
    }

    // MARK: - Streaming Transcription Methods

    /// Starts streaming transcription for Parakeet models
    private func startStreamingTranscription() async {
        guard let parakeetModel = currentTranscriptionModel as? ParakeetModel else { return }

        // Capture direct reference to the service to avoid @MainActor isolation issues in audio callback
        let parakeetService = serviceRegistry.parakeetTranscriptionService

        // Set up audio callback BEFORE starting streaming to avoid losing early audio
        // Note: callback runs on audio thread, so we capture parakeetService directly
        // Audio will be silently dropped until manager is created (streamAudio has a guard)
        logger.notice("🎙️ Setting up streaming audio callback")
        recorder.setStreamingAudioCallback { samples, frameCount, sampleRate, channels in
            parakeetService.streamAudio(
                samples: samples,
                frameCount: frameCount,
                sampleRate: sampleRate,
                channels: channels
            )
        }

        do {
            let transcriptStream = try await parakeetService.startStreaming(model: parakeetModel)

            isStreamingActive = true
            lastStreamedText = ""

            // Enable streaming mode in CursorPaster to skip clipboard save/restore
            // This prevents race conditions during rapid paste operations
            CursorPaster.setStreamingMode(true)

            // Start task to handle streaming updates
            logger.notice("🎙️ Starting streaming update task...")
            streamingUpdateTask = Task {
                self.logger.notice("🎙️ Streaming update task running, waiting for transcripts...")
                for await text in transcriptStream {
                    self.logger.notice("🎙️ Got transcript from stream: '\(text.prefix(30))...'")
                    await self.handleStreamingUpdate(text)
                }
                self.logger.notice("🎙️ Streaming update task ended")
            }

            logger.notice("🎙️ Started streaming transcription - all setup complete")
        } catch {
            logger.error("❌ Failed to start streaming transcription: \(error.localizedDescription)")
            isStreamingActive = false
        }
    }

    /// Handles incoming streaming transcription updates by pasting text to active app
    /// Optimized to use differential updates when possible to reduce flicker
    private func handleStreamingUpdate(_ newText: String) async {
        guard isStreamingActive else { return }

        await MainActor.run {
            let oldText = self.lastStreamedText

            // Optimization: If new text starts with old text, just append the delta
            // This is the common case during continuous speech and avoids flicker
            if newText.hasPrefix(oldText) && !oldText.isEmpty {
                let deltaText = String(newText.dropFirst(oldText.count))
                if !deltaText.isEmpty {
                    self.lastStreamedText = newText
                    CursorPaster.pasteAtCursor(deltaText)
                    self.logger.notice("🎙️ Appended delta: '\(deltaText.prefix(30))...'")
                }
                return
            }

            // Full replacement needed (model corrected itself or first update)
            let charsToDelete = oldText.count

            // Step 1: Delete previously streamed text
            if charsToDelete > 0 {
                CursorPaster.deleteCharacters(count: charsToDelete)
            }

            // Step 2: Wait for deletions to complete before pasting
            let deleteWaitTime = max(0.02, Double(charsToDelete) * 0.002)  // ~2ms per char, min 20ms

            DispatchQueue.main.asyncAfter(deadline: .now() + deleteWaitTime) { [weak self] in
                guard let self = self, self.isStreamingActive else { return }

                self.lastStreamedText = newText
                CursorPaster.pasteAtCursor(newText)
                self.logger.notice("🎙️ Full replacement: '\(newText.prefix(30))...'")
            }
        }
    }

    /// Finishes streaming and returns the final transcription text
    private func finishStreamingTranscription() async -> String? {
        guard isStreamingActive else { return nil }

        // Stop receiving updates
        streamingUpdateTask?.cancel()
        streamingUpdateTask = nil

        // Clear the audio callback
        recorder.setStreamingAudioCallback(nil)

        // Get final text
        var finalText: String
        do {
            finalText = try await serviceRegistry.parakeetTranscriptionService.finishStreaming()
            // If EOU returns empty but we have streamed text, use that as fallback
            if finalText.isEmpty && !self.lastStreamedText.isEmpty {
                logger.warning("⚠️ EOU returned empty, using lastStreamedText fallback (\(self.lastStreamedText.count) chars)")
                finalText = self.lastStreamedText
            }
        } catch {
            logger.error("❌ Failed to finish streaming: \(error.localizedDescription)")
            finalText = self.lastStreamedText  // Fall back to last streamed text
        }

        // Delete the streamed preview text (will be replaced by batch transcription in hybrid mode)
        await MainActor.run {
            if !self.lastStreamedText.isEmpty {
                CursorPaster.deleteCharacters(count: self.lastStreamedText.count)
            }
        }

        self.isStreamingActive = false
        self.lastStreamedText = ""

        // Disable streaming mode - clipboard operations can resume normally
        CursorPaster.setStreamingMode(false)

        logger.notice("🎙️ Finished streaming transcription: \(finalText.count) characters")
        return finalText
    }

    /// Cancels streaming transcription
    private func cancelStreamingTranscription() async {
        guard isStreamingActive else { return }

        streamingUpdateTask?.cancel()
        streamingUpdateTask = nil
        recorder.setStreamingAudioCallback(nil)

        await serviceRegistry.parakeetTranscriptionService.cancelStreaming()

        // Delete any streamed text
        await MainActor.run {
            if !lastStreamedText.isEmpty {
                CursorPaster.deleteCharacters(count: lastStreamedText.count)
            }
        }

        isStreamingActive = false
        lastStreamedText = ""

        // Disable streaming mode - clipboard operations can resume normally
        CursorPaster.setStreamingMode(false)

        logger.notice("🎙️ Cancelled streaming transcription")
    }

    /// Handles completion of streaming transcription using HYBRID approach:
    /// 1. Streaming provided real-time preview (low accuracy, fast)
    /// 2. Now run BATCH transcription for accurate final result
    private func handleStreamingCompletion() async {
        guard let recordedFile = recordedFile else {
            await MainActor.run {
                recordingState = .idle
            }
            return
        }

        // Step 1: Clean up streaming and delete the preview text
        // We discard the streaming result and use batch transcription for accuracy
        _ = await finishStreamingTranscription()

        // If there was streamed text, it's already been deleted by finishStreamingTranscription()
        // Now we'll paste the accurate batch result

        // Play stop sound
        Task {
            let isSystemMuteEnabled = UserDefaults.standard.bool(forKey: "isSystemMuteEnabled")
            if isSystemMuteEnabled {
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            await MainActor.run {
                SoundManager.shared.playStopSound()
            }
        }

        // Step 2: Switch to transcribing state for batch processing
        await MainActor.run {
            recordingState = .transcribing
        }

        logger.notice("🎙️ HYBRID: Streaming preview done, now running accurate batch transcription...")

        // Get audio duration
        let audioAsset = AVURLAsset(url: recordedFile)
        let duration = (try? CMTimeGetSeconds(await audioAsset.load(.duration))) ?? 0.0

        // Create transcription record
        let transcription = Transcription(
            text: "",
            duration: duration,
            audioFileURL: recordedFile.absoluteString,
            transcriptionStatus: .pending
        )
        modelContext.insert(transcription)
        try? modelContext.save()
        NotificationCenter.default.post(name: .transcriptionCreated, object: transcription)

        // Step 3: Run BATCH transcription for accurate result
        // HYBRID MODE: Prefer Whisper for accuracy (2.7% WER) over Parakeet (6.05% WER)
        var text: String
        do {
            guard let model = currentTranscriptionModel else {
                throw WhisperStateError.transcriptionFailed
            }

            // Check if we should prefer Whisper for better accuracy
            var transcriptionModel: any TranscriptionModel = model
            var usedWhisper = false

            if model is ParakeetModel {
                // Parakeet was selected for streaming, but check if Whisper is available for better batch accuracy
                // Look for Whisper large-v3-turbo in available models (2.7% WER vs Parakeet's 6.05%)
                if let turboModel = allAvailableModels.first(where: {
                    $0.provider == .local && $0.name.contains("large-v3-turbo")
                }) {
                    // Check if this model is actually downloaded
                    let isDownloaded = availableModels.contains(where: { $0.name == turboModel.name })
                    if isDownloaded {
                        transcriptionModel = turboModel
                        usedWhisper = true
                        logger.notice("🎙️ HYBRID: Using Whisper turbo for accuracy: \(turboModel.name)")
                    }
                }
            }

            text = try await serviceRegistry.transcribe(audioURL: recordedFile, model: transcriptionModel)
            logger.notice("🎙️ HYBRID: Batch transcription complete\(usedWhisper ? " (Whisper)" : ""): \(text.prefix(50))...")
        } catch {
            logger.error("❌ Batch transcription failed: \(error.localizedDescription)")
            transcription.text = "Transcription Failed: \(error.localizedDescription)"
            transcription.transcriptionStatus = TranscriptionStatus.failed.rawValue
            try? modelContext.save()
            await MainActor.run {
                recordingState = .idle
            }
            await dismissMiniRecorder()
            return
        }

        // Step 4: Apply post-processing pipeline
        text = TranscriptionOutputFilter.filter(text)

        let shouldFormatText = UserDefaults.standard.object(forKey: "EnableTextFormatting") as? Bool ?? true
        if shouldFormatText {
            text = WhisperTextFormatter.format(text)
        }

        text = WordReplacementService.shared.applyReplacements(to: text, using: modelContext)

        // Update transcription record
        transcription.text = text
        transcription.transcriptionModelName = currentTranscriptionModel?.displayName

        // AI Enhancement (if enabled)
        var enhancedText: String?
        if let enhancementService = enhancementService,
           enhancementService.isEnhancementEnabled,
           enhancementService.isConfigured {
            await MainActor.run {
                recordingState = .enhancing
            }

            do {
                let (enhanced, enhancementDuration, promptName) = try await enhancementService.enhance(text)
                enhancedText = enhanced
                transcription.enhancedText = enhanced
                transcription.enhancementDuration = enhancementDuration
                transcription.promptName = promptName
            } catch {
                logger.error("❌ Enhancement failed: \(error.localizedDescription)")
            }
        }

        // Mark transcription as complete
        transcription.transcriptionStatus = TranscriptionStatus.completed.rawValue
        try? modelContext.save()

        NotificationCenter.default.post(name: .transcriptionCompleted, object: transcription)

        // Step 5: Paste the accurate final text
        let finalText = enhancedText ?? text
        await MainActor.run {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                CursorPaster.pasteAtCursor(finalText + " ")

                // Auto-send if Power Mode enabled
                let powerMode = PowerModeManager.shared
                if let activeConfig = powerMode.currentActiveConfiguration,
                   activeConfig.isAutoSendEnabled {
                    CursorPaster.pressEnter()
                }
            }
        }

        await MainActor.run {
            recordingState = .idle
        }
        await dismissMiniRecorder()
    }

    private func transcribeAudio(on transcription: Transcription) async {
        guard let urlString = transcription.audioFileURL, let url = URL(string: urlString) else {
            logger.error("❌ Invalid audio file URL in transcription object.")
            await MainActor.run {
                recordingState = .idle
            }
            transcription.text = "Transcription Failed: Invalid audio file URL"
            transcription.transcriptionStatus = TranscriptionStatus.failed.rawValue
            try? modelContext.save()
            return
        }

        if shouldCancelRecording {
            await MainActor.run {
                recordingState = .idle
            }
            await cleanupModelResources()
            return
        }

        await MainActor.run {
            recordingState = .transcribing
        }

        // Play stop sound when transcription starts with a small delay
        Task {
            let isSystemMuteEnabled = UserDefaults.standard.bool(forKey: "isSystemMuteEnabled")
            if isSystemMuteEnabled {
                try? await Task.sleep(nanoseconds: 200_000_000) // 200 milliseconds delay
            }
            await MainActor.run {
                SoundManager.shared.playStopSound()
            }
        }

        defer {
            if shouldCancelRecording {
                Task {
                    await cleanupModelResources()
                }
            }
        }

        logger.notice("🔄 Starting transcription...")
        
        var finalPastedText: String?
        var promptDetectionResult: PromptDetectionService.PromptDetectionResult?

        do {
            guard let model = currentTranscriptionModel else {
                throw WhisperStateError.transcriptionFailed
            }

            let transcriptionStart = Date()
            var text = try await serviceRegistry.transcribe(audioURL: url, model: model)
            logger.notice("📝 Raw transcript: \(text, privacy: .public)")
            text = TranscriptionOutputFilter.filter(text)
            logger.notice("📝 Output filter result: \(text, privacy: .public)")
            let transcriptionDuration = Date().timeIntervalSince(transcriptionStart)

            let powerModeManager = PowerModeManager.shared
            let activePowerModeConfig = powerModeManager.currentActiveConfiguration
            let powerModeName = (activePowerModeConfig?.isEnabled == true) ? activePowerModeConfig?.name : nil
            let powerModeEmoji = (activePowerModeConfig?.isEnabled == true) ? activePowerModeConfig?.emoji : nil

            if await checkCancellationAndCleanup() { return }

            text = text.trimmingCharacters(in: .whitespacesAndNewlines)

            if UserDefaults.standard.object(forKey: "IsTextFormattingEnabled") as? Bool ?? true {
                text = WhisperTextFormatter.format(text)
                logger.notice("📝 Formatted transcript: \(text, privacy: .public)")
            }

            text = WordReplacementService.shared.applyReplacements(to: text, using: modelContext)
            logger.notice("📝 WordReplacement: \(text, privacy: .public)")

            let audioAsset = AVURLAsset(url: url)
            let actualDuration = (try? CMTimeGetSeconds(await audioAsset.load(.duration))) ?? 0.0
            
            transcription.text = text
            transcription.duration = actualDuration
            transcription.transcriptionModelName = model.displayName
            transcription.transcriptionDuration = transcriptionDuration
            transcription.powerModeName = powerModeName
            transcription.powerModeEmoji = powerModeEmoji
            finalPastedText = text
            
            if let enhancementService = enhancementService, enhancementService.isConfigured {
                let detectionResult = await promptDetectionService.analyzeText(text, with: enhancementService)
                promptDetectionResult = detectionResult
                await promptDetectionService.applyDetectionResult(detectionResult, to: enhancementService)
            }

            if let enhancementService = enhancementService,
               enhancementService.isEnhancementEnabled,
               enhancementService.isConfigured {
                if await checkCancellationAndCleanup() { return }

                await MainActor.run { self.recordingState = .enhancing }
                let textForAI = promptDetectionResult?.processedText ?? text
                
                do {
                    let (enhancedText, enhancementDuration, promptName) = try await enhancementService.enhance(textForAI)
                    logger.notice("📝 AI enhancement: \(enhancedText, privacy: .public)")
                    transcription.enhancedText = enhancedText
                    transcription.aiEnhancementModelName = enhancementService.getAIService()?.currentModel
                    transcription.promptName = promptName
                    transcription.enhancementDuration = enhancementDuration
                    transcription.aiRequestSystemMessage = enhancementService.lastSystemMessageSent
                    transcription.aiRequestUserMessage = enhancementService.lastUserMessageSent
                    finalPastedText = enhancedText
                } catch {
                    transcription.enhancedText = "Enhancement failed: \(error)"
                  
                    if await checkCancellationAndCleanup() { return }
                }
            }

            transcription.transcriptionStatus = TranscriptionStatus.completed.rawValue

        } catch {
            let errorDescription = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            let recoverySuggestion = (error as? LocalizedError)?.recoverySuggestion ?? ""
            let fullErrorText = recoverySuggestion.isEmpty ? errorDescription : "\(errorDescription) \(recoverySuggestion)"

            transcription.text = "Transcription Failed: \(fullErrorText)"
            transcription.transcriptionStatus = TranscriptionStatus.failed.rawValue
        }

        // --- Finalize and save ---
        try? modelContext.save()
        
        if transcription.transcriptionStatus == TranscriptionStatus.completed.rawValue {
            NotificationCenter.default.post(name: .transcriptionCompleted, object: transcription)
        }

        if await checkCancellationAndCleanup() { return }

        if var textToPaste = finalPastedText, transcription.transcriptionStatus == TranscriptionStatus.completed.rawValue {
            if case .trialExpired = licenseViewModel.licenseState {
                textToPaste = """
                    Your trial has expired. Upgrade to VoiceInk Pro at tryvoiceink.com/buy
                    \n\(textToPaste)
                    """
            }

            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                CursorPaster.pasteAtCursor(textToPaste + " ")

                let powerMode = PowerModeManager.shared
                if let activeConfig = powerMode.currentActiveConfiguration, activeConfig.isAutoSendEnabled {
                    // Slight delay to ensure the paste operation completes
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                        CursorPaster.pressEnter()
                    }
                }
            }
        }

        if let result = promptDetectionResult,
           let enhancementService = enhancementService,
           result.shouldEnableAI {
            await promptDetectionService.restoreOriginalSettings(result, to: enhancementService)
        }

        await self.dismissMiniRecorder()

        shouldCancelRecording = false
    }

    func getEnhancementService() -> AIEnhancementService? {
        return enhancementService
    }
    
    private func checkCancellationAndCleanup() async -> Bool {
        if shouldCancelRecording {
            await cleanupModelResources()
            return true
        }
        return false
    }

    private func cleanupAndDismiss() async {
        await dismissMiniRecorder()
    }
}
