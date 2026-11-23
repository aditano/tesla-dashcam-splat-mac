import XCTest

/// Smoke tests that shell out to the Python event scanner (source of truth).
final class EventScanTests: XCTestCase {
    func testPythonListEventsExitCode() throws {
        let repo = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let pythonDir = repo.appendingPathComponent("python")
        let fixtures = repo.appendingPathComponent("fixtures/sample_teslacam")
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
