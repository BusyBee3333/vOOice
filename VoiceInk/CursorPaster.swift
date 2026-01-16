import Foundation
import AppKit
import os.log

class CursorPaster {
    private static let logger = Logger(subsystem: "com.jakeshore.VoiceInk", category: "CursorPaster")

    // MARK: - Streaming Mode
    // When streaming is active, we skip clipboard save/restore to avoid conflicts
    // with rapid consecutive paste operations
    private static var isStreamingMode: Bool = false

    /// Enable or disable streaming mode. When enabled, clipboard save/restore is skipped
    /// to prevent race conditions during rapid streaming text updates.
    static func setStreamingMode(_ enabled: Bool) {
        isStreamingMode = enabled
        logger.notice("📋 Streaming mode \(enabled ? "enabled" : "disabled")")
    }

    static func pasteAtCursor(_ text: String) {
        logger.notice("📋 pasteAtCursor called with \(text.count) chars: '\(text.prefix(50))...'")
        logger.notice("📋 AXIsProcessTrusted = \(AXIsProcessTrusted())")
        let pasteboard = NSPasteboard.general

        // During streaming mode, skip clipboard save/restore to avoid race conditions
        // with rapid consecutive paste operations
        let userWantsRestore = UserDefaults.standard.object(forKey: "restoreClipboardAfterPaste") as? Bool ?? true
        let shouldRestoreClipboard = userWantsRestore && !isStreamingMode

        var savedContents: [(NSPasteboard.PasteboardType, Data)] = []

        if shouldRestoreClipboard {
            let currentItems = pasteboard.pasteboardItems ?? []

            for item in currentItems {
                for type in item.types {
                    if let data = item.data(forType: type) {
                        savedContents.append((type, data))
                    }
                }
            }
        }

        ClipboardManager.setClipboard(text, transient: shouldRestoreClipboard)

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
            if UserDefaults.standard.bool(forKey: "UseAppleScriptPaste") {
                _ = pasteUsingAppleScript()
            } else {
                pasteUsingCommandV()
            }
        }

        if shouldRestoreClipboard {
            let restoreDelay = UserDefaults.standard.double(forKey: "clipboardRestoreDelay")
            let delay = restoreDelay > 0 ? restoreDelay : 1.0

            DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                if !savedContents.isEmpty {
                    pasteboard.clearContents()
                    for (type, data) in savedContents {
                        pasteboard.setData(data, forType: type)
                    }
                }
            }
        }
    }
    
    private static func pasteUsingAppleScript() -> Bool {
        guard AXIsProcessTrusted() else {
            return false
        }
        
        let script = """
        tell application "System Events"
            keystroke "v" using command down
        end tell
        """
        
        var error: NSDictionary?
        if let scriptObject = NSAppleScript(source: script) {
            _ = scriptObject.executeAndReturnError(&error)
            return error == nil
        }
        return false
    }
    
    private static func pasteUsingCommandV() {
        logger.notice("📋 pasteUsingCommandV called")
        guard AXIsProcessTrusted() else {
            logger.error("❌ pasteUsingCommandV: AXIsProcessTrusted() returned false!")
            return
        }

        let source = CGEventSource(stateID: .hidSystemState)

        let cmdDown = CGEvent(keyboardEventSource: source, virtualKey: 0x37, keyDown: true)
        let vDown = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: true)
        let vUp = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: false)
        let cmdUp = CGEvent(keyboardEventSource: source, virtualKey: 0x37, keyDown: false)

        cmdDown?.flags = .maskCommand
        vDown?.flags = .maskCommand
        vUp?.flags = .maskCommand
        cmdUp?.flags = .maskCommand  // Fix: cmdUp also needs .maskCommand flag

        cmdDown?.post(tap: .cghidEventTap)
        vDown?.post(tap: .cghidEventTap)
        vUp?.post(tap: .cghidEventTap)
        cmdUp?.post(tap: .cghidEventTap)
        logger.notice("📋 pasteUsingCommandV: Posted Cmd+V events")
    }

    // Simulate pressing the Return / Enter key
    static func pressEnter() {
        guard AXIsProcessTrusted() else { return }
        let source = CGEventSource(stateID: .hidSystemState)
        let enterDown = CGEvent(keyboardEventSource: source, virtualKey: 0x24, keyDown: true)
        let enterUp = CGEvent(keyboardEventSource: source, virtualKey: 0x24, keyDown: false)
        enterDown?.post(tap: .cghidEventTap)
        enterUp?.post(tap: .cghidEventTap)
    }

    /// Deletes the specified number of characters by simulating backspace key presses
    /// Includes inter-key delays to ensure reliable deletion across all applications
    static func deleteCharacters(count: Int) {
        logger.notice("📋 deleteCharacters called with count=\(count)")
        guard AXIsProcessTrusted() else {
            logger.error("❌ deleteCharacters: AXIsProcessTrusted() returned false!")
            return
        }
        guard count > 0 else { return }

        let source = CGEventSource(stateID: .hidSystemState)
        let backspaceKeyCode: CGKeyCode = 0x33  // Backspace key

        for i in 0..<count {
            let backspaceDown = CGEvent(keyboardEventSource: source, virtualKey: backspaceKeyCode, keyDown: true)
            let backspaceUp = CGEvent(keyboardEventSource: source, virtualKey: backspaceKeyCode, keyDown: false)
            backspaceDown?.post(tap: .cghidEventTap)
            backspaceUp?.post(tap: .cghidEventTap)

            // Add small delay every 5 keystrokes to let the system process them
            // This prevents keystroke loss in applications that can't handle rapid input
            if i % 5 == 4 && i < count - 1 {
                usleep(1500)  // 1.5ms pause every 5 keystrokes
            }
        }
        logger.notice("📋 deleteCharacters: Deleted \(count) characters")
    }
}
