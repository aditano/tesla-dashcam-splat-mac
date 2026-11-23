import XCTest

/// Smoke tests that shell out to the Python event scanner (source of truth).
final class EventScanTests: XCTestCase {
    /// Resolve the repo `python/` package without assuming `#file` depth.
    /// Prefers `DASHSPLAT_PYTHON_DIR` / `GITHUB_WORKSPACE`, then walks up from `#file`.
    private func resolvePythonDir() -> URL? {
        let env = ProcessInfo.processInfo.environment
        let candidates: [URL] = [
            env["DASHSPLAT_PYTHON_DIR"].map { URL(fileURLWithPath: $0) },
            env["GITHUB_WORKSPACE"].map { URL(fileURLWithPath: $0).appendingPathComponent("python") },
            env["SRCROOT"].map { URL(fileURLWithPath: $0).appendingPathComponent("python") },
        ].compactMap { $0 }

        for url in candidates where FileManager.default.fileExists(atPath: url.path) {
            return url
        }

        var dir = URL(fileURLWithPath: #file).deletingLastPathComponent()
        for _ in 0..<8 {
            let python = dir.appendingPathComponent("python")
            let marker = python.appendingPathComponent("dashsplat")
            if FileManager.default.fileExists(atPath: marker.path) {
                return python
            }
            let parent = dir.deletingLastPathComponent()
            if parent.path == dir.path { break }
            dir = parent
        }
        return nil
    }

    func testPythonListEventsExitCode() throws {
        guard let pythonDir = resolvePythonDir() else {
            throw XCTSkip("python/ not found (set DASHSPLAT_PYTHON_DIR or run from repo checkout)")
        }
        let fixtures = pythonDir
            .deletingLastPathComponent()
            .appendingPathComponent("fixtures/sample_teslacam")
        guard FileManager.default.fileExists(atPath: fixtures.path) else {
            throw XCTSkip("fixtures/sample_teslacam missing")
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = ["-m", "dashsplat.cli", "list", fixtures.path, "--json"]
        process.currentDirectoryURL = pythonDir
        var env = ProcessInfo.processInfo.environment
        env["PYTHONPATH"] = pythonDir.path
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0, "dashsplat list should exit 0")
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let text = String(data: data, encoding: .utf8) ?? ""
        XCTAssertTrue(text.contains("["), "expected JSON array, got: \(text)")
    }
}
