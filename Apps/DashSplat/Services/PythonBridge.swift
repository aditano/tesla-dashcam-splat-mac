import Foundation

struct BuildResult {
    var ok: Bool
    var exitCode: Int
    var plyURL: URL?
    var warning: String?
}

enum PythonBridgeError: Error, LocalizedError {
    case pythonMissing
    case badJSON(String)
    case processFailed(Int, String)

    var errorDescription: String? {
        switch self {
        case .pythonMissing:
            return "Python environment not found. Run bootstrap."
        case .badJSON(let s):
            return "Bad JSON from pipeline: \(s)"
        case .processFailed(let code, let err):
            return "Process failed (\(code)): \(err)"
        }
    }
}

final class PythonBridge {
    private let supportDir: URL
    private let repoPython: URL
    private var cancelFlagURL: URL?

    init() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        supportDir = home
            .appendingPathComponent("Library/Application Support/DashSplat", isDirectory: true)
        if let bundled = Bundle.main.resourceURL?.appendingPathComponent("python") {
            repoPython = bundled
        } else {
            repoPython = URL(fileURLWithPath: #file)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("python")
        }
    }

    private var venvPython: URL {
        supportDir.appendingPathComponent("venv/bin/python")
    }

    private func resolvePython() -> URL? {
        if FileManager.default.isExecutableFile(atPath: venvPython.path) {
            return venvPython
        }
        let candidates = ["/usr/bin/python3", "/opt/homebrew/bin/python3"]
        for c in candidates where FileManager.default.isExecutableFile(atPath: c) {
            return URL(fileURLWithPath: c)
        }
        return nil
    }

    func requestCancel() {
        if let cancelFlagURL {
            try? "1".write(to: cancelFlagURL, atomically: true, encoding: .utf8)
        }
    }

    @discardableResult
    func bootstrap(onProgress: @escaping (String, Double) -> Void) async throws -> Int {
        guard let py = resolvePython() ?? URL(fileURLWithPath: "/usr/bin/python3") as URL? else {
            throw PythonBridgeError.pythonMissing
        }
        let args = [
            "-m", "dashsplat.cli", "bootstrap",
            "--support-dir", supportDir.path,
        ]
        return try await run(
            executable: py,
            arguments: args,
            cwd: repoPython,
            env: [
                "PYTHONPATH": repoPython.path,
                "DASHSPLAT_SUPPORT_DIR": supportDir.path,
            ],
            onProgress: { msg, frac, _, _ in onProgress(msg, frac) }
        ).exitCode
    }

    func listEvents(folder: URL) async throws -> [DashcamEventDTO] {
        let py = try requirePython()
        let result = try await run(
            executable: py,
            arguments: ["-m", "dashsplat.cli", "list", folder.path, "--json"],
            cwd: repoPython,
            env: defaultEnv(),
            onProgress: { _, _, _, _ in }
        )
        guard result.exitCode == 0 else {
            throw PythonBridgeError.processFailed(result.exitCode, result.stderr)
        }
        let data = Data(result.stdout.utf8)
        do {
            return try JSONDecoder().decode([DashcamEventDTO].self, from: data)
        } catch {
            if let start = result.stdout.firstIndex(of: "["),
               let end = result.stdout.lastIndex(of: "]") {
                let slice = String(result.stdout[start...end])
                if let d = slice.data(using: .utf8),
                   let decoded = try? JSONDecoder().decode([DashcamEventDTO].self, from: d) {
                    return decoded
                }
            }
            throw PythonBridgeError.badJSON(result.stdout)
        }
    }

    func loadLibrary(folder: URL) async throws -> [LibraryItem] {
        let out = supportDir.appendingPathComponent("jobs/\(folder.lastPathComponent)/library.json")
        guard FileManager.default.fileExists(atPath: out.path) else { return [] }
        let data = try Data(contentsOf: out)
        // Pipeline writes full result.json objects; map to LibraryItem
        struct Raw: Codable {
            var event_id: String?
            var label: String?
            var tier: String?
            var ply: String?
            var trainer: String?
            var elapsed_sec: Double?
            var num_gaussians: Int?
            var warning: String?
        }
        let raw = try JSONDecoder().decode([Raw].self, from: data)
        return raw.compactMap { r in
            guard let eid = r.event_id, let tier = r.tier else { return nil }
            return LibraryItem(
                event_id: eid,
                label: r.label,
                tier: tier,
                ply: r.ply,
                trainer: r.trainer,
                elapsed_sec: r.elapsed_sec,
                num_gaussians: r.num_gaussians,
                warning: r.warning
            )
        }
    }

    func build(
        folder: URL,
        eventIDs: [String],
        tier: String,
        onProgress: @escaping (String, Double, String?, Double?) -> Void
    ) async throws -> BuildResult {
        let py = try requirePython()
        let out = supportDir.appendingPathComponent("jobs/\(folder.lastPathComponent)", isDirectory: true)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        cancelFlagURL = out.appendingPathComponent("CANCEL")
        try? FileManager.default.removeItem(at: cancelFlagURL!)

        var args = [
            "-m", "dashsplat.cli", "build", folder.path,
            "--out", out.path,
            "--tier", tier,
        ]
        // Prefer Brush/msplat on device; seeded only when explicitly forced via env
        if ProcessInfo.processInfo.environment["DASHSPLAT_FORCE_SEEDED"] == "1" {
            args.append(contentsOf: ["--force-fallback-trainer", "--trainer", "seeded"])
        }
        if ProcessInfo.processInfo.environment["DASHSPLAT_NO_MLX"] == "1" {
            args.append("--no-mlx")
        }
        for id in eventIDs {
            args.append(contentsOf: ["--event", id])
        }

        let proc = try await run(
            executable: py,
            arguments: args,
            cwd: repoPython,
            env: defaultEnv(),
            onProgress: onProgress
        )
        var ply: URL?
        var warning: String?
        if let first = eventIDs.first {
            let candidate = out
                .appendingPathComponent("library/\(first.replacingOccurrences(of: "/", with: "_"))/\(tier)/splat.ply")
            if FileManager.default.fileExists(atPath: candidate.path) {
                ply = candidate
            }
            let resultJSON = candidate.deletingLastPathComponent().appendingPathComponent("result.json")
            if let data = try? Data(contentsOf: resultJSON),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                warning = obj["warning"] as? String
                if ply == nil, let p = obj["ply"] as? String {
                    ply = URL(fileURLWithPath: p)
                }
            }
        }
        // Scan stdout for encrypted / health messages
        for line in proc.stdout.split(whereSeparator: \.isNewline) {
            if let data = String(line).data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let msg = obj["message"] as? String,
               msg.lowercased().contains("encrypt") || msg.lowercased().contains("no sei") {
                warning = msg
            }
        }
        return BuildResult(ok: proc.exitCode == 0, exitCode: proc.exitCode, plyURL: ply, warning: warning)
    }

    private func requirePython() throws -> URL {
        guard let py = resolvePython() else { throw PythonBridgeError.pythonMissing }
        return py
    }

    private func defaultEnv() -> [String: String] {
        [
            "PYTHONPATH": repoPython.path,
            "DASHSPLAT_SUPPORT_DIR": supportDir.path,
            "PYTHONUNBUFFERED": "1",
        ]
    }

    private struct ProcResult {
        var exitCode: Int
        var stdout: String
        var stderr: String
    }

    private func run(
        executable: URL,
        arguments: [String],
        cwd: URL,
        env: [String: String],
        onProgress: @escaping (String, Double, String?, Double?) -> Void
    ) async throws -> ProcResult {
        try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            process.executableURL = executable
            process.arguments = arguments
            process.currentDirectoryURL = cwd
            var fullEnv = ProcessInfo.processInfo.environment
            for (k, v) in env { fullEnv[k] = v }
            process.environment = fullEnv

            let outPipe = Pipe()
            let errPipe = Pipe()
            process.standardOutput = outPipe
            process.standardError = errPipe

            var stdoutBox = ""
            var stderrBox = ""
            outPipe.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                guard !data.isEmpty, let chunk = String(data: data, encoding: .utf8) else { return }
                stdoutBox += chunk
                for line in chunk.split(whereSeparator: \.isNewline) {
                    if let parsed = Self.parseProgressLine(String(line)) {
                        onProgress(parsed.message, parsed.fraction, parsed.stage, parsed.etaSec)
                    }
                }
            }
            errPipe.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                if let chunk = String(data: data, encoding: .utf8) {
                    stderrBox += chunk
                }
            }

            process.terminationHandler = { proc in
                outPipe.fileHandleForReading.readabilityHandler = nil
                errPipe.fileHandleForReading.readabilityHandler = nil
                continuation.resume(
                    returning: ProcResult(
                        exitCode: Int(proc.terminationStatus),
                        stdout: stdoutBox,
                        stderr: stderrBox
                    )
                )
            }

            do {
                try process.run()
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }

    private struct ProgressParsed {
        var message: String
        var fraction: Double
        var stage: String?
        var etaSec: Double?
    }

    private static func parseProgressLine(_ line: String) -> ProgressParsed? {
        guard line.contains("\"type\""), let data = line.data(using: .utf8) else { return nil }
        guard
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let type = obj["type"] as? String
        else { return nil }
        if type == "progress" {
            let message = (obj["message"] as? String) ?? ""
            let frac = (obj["fraction"] as? Double) ?? 0
            let stage = obj["stage"] as? String
            let eta = obj["eta_sec"] as? Double
            return ProgressParsed(message: message, fraction: frac, stage: stage, etaSec: eta)
        }
        if type == "batch_progress" {
            let frac = (obj["fraction"] as? Double) ?? 0
            let index = (obj["index"] as? Int).map { $0 + 1 } ?? 0
            let total = obj["total"] as? Int ?? 0
            return ProgressParsed(
                message: "Batch \(index)/\(total)",
                fraction: frac,
                stage: "batch",
                etaSec: nil
            )
        }
        return nil
    }
}
