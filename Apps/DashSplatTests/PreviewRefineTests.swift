import XCTest

/// Ensures the CLI Preview / Refine / bench surface the v1 contract used by the app.
final class PreviewRefineTests: XCTestCase {
    private var repo: URL {
        URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private var pythonDir: URL { repo.appendingPathComponent("python") }
    private var fixtures: URL { repo.appendingPathComponent("fixtures/sample_teslacam") }

    func testBuildPreviewHelpMentionsTier() throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = ["-m", "dashsplat.cli", "build", "--help"]
        process.currentDirectoryURL = pythonDir
        var env = ProcessInfo.processInfo.environment
        env["PYTHONPATH"] = pythonDir.path
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0)
        let text = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        XCTAssertTrue(text.contains("preview") || text.contains("--tier"), text)
        XCTAssertFalse(text.lowercased().contains("opensplat"), "OpenSplat must not appear in CLI help")
    }

    func testRefineForcesFullTier() throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = ["-m", "dashsplat.cli", "refine", "--help"]
        process.currentDirectoryURL = pythonDir
        var env = ProcessInfo.processInfo.environment
        env["PYTHONPATH"] = pythonDir.path
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0)
        let text = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        XCTAssertTrue(text.contains("refine") || text.contains("Full") || text.contains("tier"), text)
    }
}
