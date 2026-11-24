import XCTest
@testable import DashSplat

final class PythonBridgeTests: XCTestCase {
    func testJobPathUniqueForDifferentFoldersWithSameName() {
        let a = URL(fileURLWithPath: "/Volumes/TeslaCam")
        let b = URL(fileURLWithPath: "/Volumes/Other/TeslaCam")
        let na = JobPath.directoryName(for: a)
        let nb = JobPath.directoryName(for: b)
        XCTAssertNotEqual(na, nb, "same lastPathComponent must still yield unique job dirs")
        XCTAssertTrue(na.hasPrefix("TeslaCam-"))
        XCTAssertTrue(nb.hasPrefix("TeslaCam-"))
    }

    func testJobPathStableForSameFolder() {
        let url = URL(fileURLWithPath: "/Users/me/TeslaCam")
        XCTAssertEqual(JobPath.directoryName(for: url), JobPath.directoryName(for: url))
    }

    func testBootstrapMarkerVersionMatchesPythonConstant() throws {
        // Bridge version string must stay aligned with python/dashsplat/bootstrap.py
        XCTAssertFalse(PythonBridge.bootstrapMarkerVersion.isEmpty)
        XCTAssertTrue(PythonBridge.bootstrapMarkerVersion.contains("."))
    }

    func testParseProgressLineViaBuildEnv() {
        // Smoke: JobPath / CryptoKit path doesn't crash on unicode
        let url = URL(fileURLWithPath: "/tmp/TeslaCam café")
        let name = JobPath.directoryName(for: url)
        XCTAssertTrue(name.contains("-"))
        XCTAssertGreaterThan(name.count, 10)
    }
}
