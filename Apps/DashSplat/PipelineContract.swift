import Foundation

/// App-side contract for Preview / Refine (mirrors python/dashsplat/tiers + trainers).
/// Kept in Swift so unit tests do not need a repo-relative `python/` directory.
enum QualityTier: String, CaseIterable {
    case preview
    case full
}

enum TrainerBackend: String, CaseIterable {
    case brush
    case msplat
    case seeded

    static let `default`: TrainerBackend = .brush
}

enum PipelineContract {
    /// Allowed CLI/app quality tiers. No modes picker beyond Preview → Refine.
    static let qualityTiers: [QualityTier] = QualityTier.allCases

    /// Permissive trainers only. OpenSplat (AGPL) is intentionally absent.
    static let trainers: [TrainerBackend] = TrainerBackend.allCases

    static var trainerNames: [String] { trainers.map(\.rawValue) }

    /// Refine always means Full quality.
    static func tierForRefine() -> QualityTier { .full }

    /// Default Build / Preview action.
    static func tierForPreview() -> QualityTier { .preview }

    static func isAllowedTrainer(_ name: String) -> Bool {
        TrainerBackend(rawValue: name.lowercased()) != nil
    }

    static func rejectsOpenSplat(_ name: String) -> Bool {
        name.lowercased().contains("opensplat")
    }

    /// ETA prior seconds (estimates on Max-class; UI labels them as estimates).
    static func etaPriorSeconds(for tier: QualityTier) -> Double {
        switch tier {
        case .preview: return 120
        case .full: return 720
        }
    }

    static func formatETA(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        if s < 60 { return "~\(s)s left" }
        return "~\(s / 60)m \(s % 60)s left"
    }
}
