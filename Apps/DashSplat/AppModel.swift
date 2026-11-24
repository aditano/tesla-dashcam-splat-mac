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

struct LibraryItem: Identifiable, Codable, Hashable {
    var id: String { "\(event_id)-\(tier)" }
    var event_id: String
    var label: String?
    var tier: String
    var ply: String?
    var trainer: String?
    var elapsed_sec: Double?
    var num_gaussians: Int?
    var warning: String?
}

struct PipelineProgress: Equatable {
    var message: String
    var fraction: Double
    var stage: String?
    var etaSec: Double?
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
    @Published var library: [LibraryItem] = []
    @Published var healthWarning: String?
    @Published var canRefine = false
    @Published var showEgoCar = true

    private let bridge = PythonBridge()

    func appendLog(_ line: String) {
        logLines.append(line)
        if logLines.count > 400 {
            logLines.removeFirst(logLines.count - 400)
        }
    }

    func ensureBootstrap() async {
        if bridge.isBootstrapComplete() {
            bootstrapNeeded = false
            statusMessage = "Ready"
            return
        }
        guard bootstrapNeeded else { return }
        isBootstrapping = true
        statusMessage = "Installing Python environment, models & trainers…"
        progress = PipelineProgress(message: "Bootstrap", fraction: 0)
        defer {
            isBootstrapping = false
            progress = nil
        }
        do {
            let code = try await bridge.bootstrap { [weak self] msg, frac in
                Task { @MainActor in
                    self?.progress = PipelineProgress(message: msg, fraction: frac, stage: "bootstrap")
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
        healthWarning = nil
        do {
            let listed = try await bridge.listEvents(folder: folder)
            events = listed
            selectedEventIDs = Set(listed.prefix(1).map(\.id))
            statusMessage = listed.isEmpty ? "No events found" : "\(listed.count) event(s)"
            appendLog(statusMessage)
            await refreshLibrary()
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
        // Keep isBuilding true until the process actually exits (buttons stay disabled).
        statusMessage = "Cancelling…"
        appendLog("Cancel requested (SIGTERM→SIGKILL)")
        bridge.requestCancel()
    }

    func buildPreview() async {
        await runBuild(tier: PipelineContract.tierForPreview().rawValue)
    }

    func refineToFull() async {
        await runBuild(tier: PipelineContract.tierForRefine().rawValue)
    }

    private func runBuild(tier: String) async {
        guard let folderURL else { return }
        let ids = Array(selectedEventIDs)
        guard !ids.isEmpty else {
            statusMessage = "Select at least one event"
            return
        }
        isBuilding = true
        progress = PipelineProgress(message: tier == "full" ? "Refining to full…" : "Preview…", fraction: 0)
        healthWarning = nil
        defer {
            isBuilding = false
            progress = nil
        }
        do {
            let result = try await bridge.build(
                folder: folderURL,
                eventIDs: ids,
                tier: tier,
                onProgress: { [weak self] msg, frac, stage, eta in
                    Task { @MainActor in
                        if frac >= 0 {
                            self?.progress = PipelineProgress(
                                message: msg,
                                fraction: frac,
                                stage: stage,
                                etaSec: eta
                            )
                        }
                        self?.appendLog(msg)
                        if msg.lowercased().contains("encrypt") || msg.lowercased().contains("no sei") {
                            self?.healthWarning = msg
                        }
                    }
                },
                onLog: { [weak self] line, isStderr in
                    Task { @MainActor in
                        self?.appendLog(isStderr ? "[stderr] \(line)" : line)
                    }
                }
            )
            if let ply = result.plyURL {
                lastPLYURL = ply
            }
            canRefine = tier == "preview" && result.ok && !result.cpuFallback
            if let warn = result.warning {
                healthWarning = warn
            }
            if result.exitCode == 130 {
                statusMessage = "Cancelled"
            } else if result.ok {
                let trainerNote = result.cpuFallback
                    ? " (CPU seeded fallback — not Metal Full)"
                    : (result.trainer.map { " [\($0)]" } ?? "")
                statusMessage =
                    "\(tier.capitalized) done\(trainerNote) → \(result.plyURL?.lastPathComponent ?? "")"
            } else {
                statusMessage = "Build failed (exit \(result.exitCode))"
            }
            appendLog(statusMessage)
            await refreshLibrary()
        } catch {
            statusMessage = "Build error: \(error.localizedDescription)"
            appendLog(statusMessage)
        }
    }

    func refreshLibrary() async {
        guard let folderURL else { return }
        do {
            library = try await bridge.loadLibrary(folder: folderURL)
        } catch {
            appendLog("library refresh: \(error.localizedDescription)")
        }
    }

    func openLibraryItem(_ item: LibraryItem) {
        if let ply = item.ply {
            lastPLYURL = URL(fileURLWithPath: ply)
        }
    }
}
