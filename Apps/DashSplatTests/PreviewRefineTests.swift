import XCTest
@testable import DashSplat

/// Preview / Refine contract — pure Swift, no repo-relative `python/` dependency.
final class PreviewRefineTests: XCTestCase {
    func testBuildPreviewHelpMentionsTier() {
        // Mirrors CLI `build --tier {preview,full}` and app Preview button.
        let tiers = PipelineContract.qualityTiers.map(\.rawValue)
        XCTAssertTrue(tiers.contains("preview"))
        XCTAssertTrue(tiers.contains("full"))
        XCTAssertEqual(PipelineContract.tierForPreview(), .preview)
        XCTAssertEqual(PipelineContract.trainerNames.sorted(), ["brush", "msplat", "seeded"])
        XCTAssertFalse(
            PipelineContract.trainerNames.contains(where: { $0.lowercased().contains("opensplat") }),
            "OpenSplat must not appear in trainer choices"
        )
        XCTAssertTrue(PipelineContract.rejectsOpenSplat("opensplat"))
        XCTAssertFalse(PipelineContract.isAllowedTrainer("opensplat"))
        XCTAssertTrue(PipelineContract.isAllowedTrainer("brush"))
    }

    func testRefineForcesFullTier() {
        // Mirrors CLI `refine` (forces --tier full) and app "Refine to full".
        XCTAssertEqual(PipelineContract.tierForRefine(), .full)
        XCTAssertNotEqual(PipelineContract.tierForRefine(), .preview)
        XCTAssertEqual(PipelineContract.etaPriorSeconds(for: .preview), 120)
        XCTAssertEqual(PipelineContract.etaPriorSeconds(for: .full), 720)
        XCTAssertTrue(PipelineContract.formatETA(48).contains("48s"))
        XCTAssertTrue(PipelineContract.formatETA(125).contains("2m"))
    }
}
