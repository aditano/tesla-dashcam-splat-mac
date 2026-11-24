import CryptoKit
import Foundation

struct BuildResult {
    var ok: Bool
    var exitCode: Int
    var plyURL: URL?
    var warning: String?
    var trainer: String?
    var cpuFallback: Bool
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

/// Stable, unique job directory name from a volume/folder URL (not just lastPathComponent).
enum JobPath {
    static func directoryName(for folder: URL) -> String {
        let path = folder.standardizedFileURL.path
        let digest = SHA256.hash(data: Data(path.utf8))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        let short = String(hex.prefix(16))
        let base = folder.lastPathComponent
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: ":", with: "_")
        return "\(base)-\(short)"
    }
}

final class PythonBridge {
    private let supportDir: URL
    private let repoPython: URL
    private var cancelFlagURL: URL?
    private let processLock = NSLock()
    private var activeProcess: Process?
    private var cancelRequested = false

    /// Bootstrap marker schema version — bump when deps/layout change.
    static let bootstrapMarkerVersion = "1.1.0"

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

    private var bootstrapMarkerURL: URL {
        supportDir.appendingPathComponent(".bootstrap_ok")
    }

    /// True when Application Support already has a completed bootstrap for this marker version.
    func isBootstrapComplete() -> Bool {
        let marker = bootstrapMarkerURL
        guard FileManager.default.fileExists(atPath: marker.path),
              FileManager.default.isExecutableFile(atPath: venvPython.path),
              let data = try? Data(contentsOf: marker),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let version = obj["version"] as? String
        else {
            return false
        }
        return version == Self.bootstrapMarkerVersion
    }

    func jobDirectory(for folder: URL) -> URL {
        supportDir.appendingPathComponent("jobs/\(JobPath.directoryName(for: folder))", isDirectory: true)
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
        cancelRequested = true
        if let cancelFlagURL {
            try? "1".write(to: cancelFlagURL, atomically: true, encoding: .utf8)
        }
        processLock.lock()
        let proc = activeProcess
        processLock.unlock()
        guard let proc, proc.isRunning else { return }
        let pid = proc.processIdentifier
        // Python is started as its own process group (os.setpgrp); kill the
        // whole group so brush/ffmpeg children die with it.
        kill(-pid, SIGTERM)
        proc.terminate() // also signal the leader
        DispatchQueue.global().async {
            let deadline = Date().addingTimeInterval(5.0)
            while proc.isRunning, Date() < deadline {
                Thread.sleep(forTimeInterval: 0.1)
            }
            if proc.isRunning {
                kill(-pid, SIGKILL)
                kill(pid, SIGKILL)
            }
        }
    }

    @discardableResult
    func bootstrap(onProgress: @escaping (String, Double) -> Void) async throws -> Int {
        if isBootstrapComplete() {
            onProgress("Bootstrap already complete", 1.0)
            return 0
        }
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
            onProgress: { msg, frac, _, _ in onProgress(msg, frac) },
            onLog: { _, _ in }
        ).exitCode
    }

    func listEvents(folder: URL) async throws -> [DashcamEventDTO] {
        let py = try requirePython()
        let result = try await run(
            executable: py,
            arguments: ["-m", "dashsplat.cli", "list", folder.path, "--json"],
            cwd: repoPython,
            env: defaultEnv(),
            onProgress: { _, _, _, _ in },
            onLog: { _, _ in }
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
        let out = jobDirectory(for: folder).appendingPathComponent("library.json")
        guard FileManager.default.fileExists(atPath: out.path) else { return [] }
        let data = try Data(contentsOf: out)
        struct Raw: Codable {
            var event_id: String?
            var label: String?
            var tier: String?
            var ply: String?
            var trainer: String?
            var elapsed_sec: Double?
            var num_gaussians: Int?
            var warning: String?
            var cpu_fallback: Bool?
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
        onProgress: @escaping (String, Double, String?, Double?) -> Void,
        onLog: @escaping (String, Bool) -> Void
    ) async throws -> BuildResult {
        let py = try requirePython()
        let out = jobDirectory(for: folder)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        cancelFlagURL = out.appendingPathComponent("CANCEL")
        cancelRequested = false
        try? FileManager.default.removeItem(at: cancelFlagURL!)

        var args = [
            "-m", "dashsplat.cli", "build", folder.path,
            "--out", out.path,
            "--tier", tier,
        ]
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
            onProgress: onProgress,
            onLog: onLog
        )
        var ply: URL?
        var warning: String?
        var trainer: String?
        var cpuFallback = false
        if let first = eventIDs.first {
            let safeID = first.replacingOccurrences(of: "/", with: "_")
            let candidate = out
                .appendingPathComponent("library/\(safeID)/\(tier)/splat.ply")
            if FileManager.default.fileExists(atPath: candidate.path) {
                ply = candidate
            }
            let resultJSON = candidate.deletingLastPathComponent().appendingPathComponent("result.json")
            if let data = try? Data(contentsOf: resultJSON),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                warning = obj["warning"] as? String
                trainer = obj["trainer"] as? String
                cpuFallback = (obj["cpu_fallback"] as? Bool) ?? (trainer == "seeded")
                if ply == nil, let p = obj["ply"] as? String {
                    ply = URL(fileURLWithPath: p)
                }
            }
        }
        for line in proc.stdout.split(whereSeparator: \.isNewline) {
            if let data = String(line).data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let msg = obj["message"] as? String,
               msg.lowercased().contains("encrypt") || msg.lowercased().contains("no sei") {
                warning = msg
            }
        }
        let exitCode = cancelRequested && proc.exitCode != 0 ? 130 : proc.exitCode
        return BuildResult(
            ok: exitCode == 0,
            exitCode: exitCode,
            plyURL: ply,
            warning: warning,
            trainer: trainer,
            cpuFallback: cpuFallback
        )
    }

    private func requirePython() throws -> URL {
        guard let py = resolvePython() else { throw PythonBridgeError.pythonMissing }
        return py
    }

    private func defaultEnv() -> [String: String] {
        let venvBin = supportDir.appendingPathComponent("venv/bin").path
        let pathExtras = ["/opt/homebrew/bin", "/usr/local/bin", venvBin]
        let existing = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin"
        var parts = pathExtras + existing.split(separator: ":").map(String.init)
        var seen = Set<String>()
        parts = parts.filter { seen.insert($0).inserted }
        return [
            "PYTHONPATH": repoPython.path,
            "DASHSPLAT_SUPPORT_DIR": supportDir.path,
            "PYTHONUNBUFFERED": "1",
            "DASHSPLAT_OWN_PGROUP": "1",
            "PATH": parts.joined(separator: ":"),
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
        onProgress: @escaping (String, Double, String?, Double?) -> Void,
        onLog: @escaping (_ line: String, _ isStderr: Bool) -> Void
    ) async throws -> ProcResult {
        try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            process.executableURL = executable
            process.arguments = arguments
            process.currentDirectoryURL = cwd
            var fullEnv = ProcessInfo.processInfo.environment
            for (k, v) in env { fullEnv[k] = v }
            // Force line-buffered Python stdio when possible
            fullEnv["PYTHONUNBUFFERED"] = "1"
            if fullEnv["DASHSPLAT_OWN_PGROUP"] == nil {
                fullEnv["DASHSPLAT_OWN_PGROUP"] = "1"
            }
            // Finder-launched apps often lack Homebrew on PATH — ensure ffmpeg/uv resolve.
            let venvBin = self.supportDir.appendingPathComponent("venv/bin").path
            let pathExtras = ["/opt/homebrew/bin", "/usr/local/bin", venvBin]
            let existingPath = fullEnv["PATH"] ?? "/usr/bin:/bin"
            var pathParts = pathExtras + existingPath.split(separator: ":").map(String.init)
            var seenPath = Set<String>()
            pathParts = pathParts.filter { seenPath.insert($0).inserted }
            fullEnv["PATH"] = pathParts.joined(separator: ":")
            process.environment = fullEnv

            let outPipe = Pipe()
            let errPipe = Pipe()
            process.standardOutput = outPipe
            process.standardError = errPipe

            let ioLock = NSLock()
            var stdoutBox = ""
            var stderrBox = ""
            // Buffer raw bytes and split on newline *before* UTF-8 decode so a
            // multibyte character split across read chunks is not dropped.
            var stdoutBytes = Data()
            var stderrBytes = Data()

            func drainByteBuffer(
                _ buffer: inout Data,
                accumulate: inout String,
                isStderr: Bool,
                flushAll: Bool
            ) {
                while true {
                    guard let nl = buffer.firstIndex(of: 0x0A) else { break }
                    let lineData = buffer.subdata(in: 0..<nl)
                    buffer.removeSubrange(0...nl)
                    let line = String(data: lineData, encoding: .utf8)
                        ?? String(decoding: lineData, as: UTF8.self)
                    let trimmed = line.trimmingCharacters(in: CharacterSet(charactersIn: "\r"))
                    accumulate += trimmed + "\n"
                    if !trimmed.isEmpty {
                        onLog(trimmed, isStderr)
                        if !isStderr, let parsed = Self.parseProgressLine(trimmed) {
                            onProgress(parsed.message, parsed.fraction, parsed.stage, parsed.etaSec)
                        }
                        if isStderr {
                            onProgress("[stderr] \(trimmed)", -1, "log", nil)
                        }
                    }
                }
                if flushAll, !buffer.isEmpty {
                    let line = String(data: buffer, encoding: .utf8)
                        ?? String(decoding: buffer, as: UTF8.self)
                    let trimmed = line.trimmingCharacters(in: CharacterSet(charactersIn: "\r"))
                    buffer.removeAll(keepingCapacity: false)
                    if !trimmed.isEmpty {
                        accumulate += trimmed + "\n"
                        onLog(trimmed, isStderr)
                    }
                }
            }

            outPipe.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                guard !data.isEmpty else { return }
                ioLock.lock()
                stdoutBytes.append(data)
                drainByteBuffer(&stdoutBytes, accumulate: &stdoutBox, isStderr: false, flushAll: false)
                ioLock.unlock()
            }
            errPipe.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                guard !data.isEmpty else { return }
                ioLock.lock()
                stderrBytes.append(data)
                drainByteBuffer(&stderrBytes, accumulate: &stderrBox, isStderr: true, flushAll: false)
                ioLock.unlock()
            }

            processLock.lock()
            activeProcess = process
            processLock.unlock()

            process.terminationHandler = { [weak self] proc in
                outPipe.fileHandleForReading.readabilityHandler = nil
                errPipe.fileHandleForReading.readabilityHandler = nil
                // Drain any remaining pipe bytes after the process exits
                let restOut = outPipe.fileHandleForReading.readDataToEndOfFile()
                let restErr = errPipe.fileHandleForReading.readDataToEndOfFile()
                ioLock.lock()
                if !restOut.isEmpty { stdoutBytes.append(restOut) }
                if !restErr.isEmpty { stderrBytes.append(restErr) }
                drainByteBuffer(&stdoutBytes, accumulate: &stdoutBox, isStderr: false, flushAll: true)
                drainByteBuffer(&stderrBytes, accumulate: &stderrBox, isStderr: true, flushAll: true)
                let out = stdoutBox
                let err = stderrBox
                ioLock.unlock()
                self?.processLock.lock()
                self?.activeProcess = nil
                self?.processLock.unlock()
                continuation.resume(
                    returning: ProcResult(
                        exitCode: Int(proc.terminationStatus),
                        stdout: out,
                        stderr: err
                    )
                )
            }

            do {
                try process.run()
            } catch {
                processLock.lock()
                activeProcess = nil
                processLock.unlock()
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
