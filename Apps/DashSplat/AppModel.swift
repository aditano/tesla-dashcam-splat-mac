import Combine
import Foundation

struct DashcamEventDTO: Identifiable, Codable, Hashable {
    var id: String
    var label: String
    var start: String
    var duration_sec: Double
    var cameras: [String]
    var location_label: String?
    var latitude: Double?
    var longitude: Double?
    var source_folder: String?
}

struct PipelineProgress: Equatable {
    var message: String
    var fraction: Double
}

@MainActor
final class AppModel: ObservableObject {
    @Published var events: [DashcamEventDTO] = []
    @Published var selectedEventIDs: Set<String> = []
    @Published var folderURL: URL?
    @Published var isBootstrapping = false
    @Published var isBuilding = false
    @Published var progress: PipelineProgress?
    @Published var statusMessage: String = "Drop a TeslaCam folder to begin"
    @Published var lastPLYURL: URL?
    @Published var bootstrapNeeded = true
    @Published var logLines: [String] = []

    private let bridge = PythonBridge()
    private var buildTask: Process?

    func appendLog(_ line: String) {
        logLines.append(line)
        if logLines.count > 400 {
            logLines.removeFirst(logLines.count - 400)
        }
    }

    func ensureBootstrap() async {
        guard bootstrapNeeded else { return }
        isBootstrapping = true
        statusMessage = "Installing Python environment & models…"
        progress = PipelineProgress(message: "Bootstrap", fraction: 0)
        defer {
            isBootstrapping = false
            progress = nil
        }
        do {
            let code = try await bridge.bootstrap { [weak self] msg, frac in
                Task { @MainActor in
                    self?.progress = PipelineProgress(message: msg, fraction: frac)
                    self?.appendLog(msg)
                }
            }
            bootstrapNeeded = code != 0
            statusMessage = code == 0 ? "Ready" : "Bootstrap failed (exit \(code))"
            appendLog("bootstrap exit \(code)")
        } catch {
            statusMessage = "Bootstrap error: \(error.localizedDescription)"
            appendLog(statusMessage)
        }
    }

    func ingest(folder: URL) async {
        folderURL = folder
        statusMessage = "Scanning \(folder.lastPathComponent)…"
        do {
            let listed = try await bridge.listEvents(folder: folder)
            events = listed
            selectedEventIDs = Set(listed.prefix(1).map(\.id))
            statusMessage = listed.isEmpty ? "No events found" : "\(listed.count) event(s)"
            appendLog(statusMessage)
        } catch {
            statusMessage = "Scan failed: \(error.localizedDescription)"
            appendLog(statusMessage)
        }
    }

    func toggleSelection(_ id: String) {
        if selectedEventIDs.contains(id) {
            selectedEventIDs.remove(id)
        } else {
            selectedEventIDs.insert(id)
        }
    }

    func cancelBuild() {
        bridge.requestCancel()
        buildTask?.terminate()
        isBuilding = false
        statusMessage = "Cancelled"
        progress = nil
    }

    func buildSelected() async {
        guard let folderURL else { return }
        let ids = Array(selectedEventIDs)
        guard !ids.isEmpty else {
            statusMessage = "Select at least one event"
            return
        }
        isBuilding = true
        progress = PipelineProgress(message: "Starting…", fraction: 0)
        defer {
            isBuilding = false
        }
        do {
            let result = try await bridge.build(
                folder: folderURL,
                eventIDs: ids,
                onProgress: { [weak self] msg, frac in
                    Task { @MainActor in
                        self?.progress = PipelineProgress(message: msg, fraction: frac)
                        self?.appendLog(msg)
                    }
                }
            )
            if let ply = result.plyURL {
                lastPLYURL = ply
            }
            statusMessage = result.ok
                ? "Built \(ids.count) event(s) → \(result.plyURL?.path ?? "")"
                : "Build failed (exit \(result.exitCode))"
            appendLog(statusMessage)
        } catch {
            statusMessage = "Build error: \(error.localizedDescription)"
            appendLog(statusMessage)
        }
        progress = nil
    }
}
