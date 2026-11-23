import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var model: AppModel
    @State private var isDropTargeted = false

    var body: some View {
        HStack(spacing: 0) {
            sidebar
                .frame(width: 380)
            Divider()
            mainPane
        }
        .background(Color(red: 0.04, green: 0.04, blue: 0.04))
        .preferredColorScheme(.dark)
        .task {
            await model.ensureBootstrap()
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text("DashSplat")
                    .font(.system(size: 22, weight: .semibold, design: .default))
                    .tracking(-0.5)
                Text("TeslaCam → 3D Gaussian Splat")
                    .font(.system(size: 12))
                    .foregroundStyle(Color(white: 0.55))
            }
            .padding(.horizontal, 4)

            dropZone

            if let warn = model.healthWarning {
                Text(warn)
                    .font(.system(size: 11))
                    .foregroundStyle(Color(red: 0.95, green: 0.55, blue: 0.2))
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 10)
                            .fill(Color(red: 0.95, green: 0.55, blue: 0.2).opacity(0.12))
                    )
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("EVENTS")
                    .font(.system(size: 11, weight: .semibold))
                    .tracking(1.2)
                    .foregroundStyle(Color(white: 0.55))

                if model.events.isEmpty {
                    Text("No events yet")
                        .foregroundStyle(Color(white: 0.45))
                        .font(.system(size: 13))
                        .padding(.vertical, 12)
                } else {
                    ScrollView {
                        LazyVStack(spacing: 8) {
                            ForEach(model.events) { event in
                                EventRow(
                                    event: event,
                                    selected: model.selectedEventIDs.contains(event.id)
                                ) {
                                    model.toggleSelection(event.id)
                                }
                            }
                        }
                    }
                }
            }
            .frame(maxHeight: 220)

            if !model.library.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("RESULTS LIBRARY")
                        .font(.system(size: 11, weight: .semibold))
                        .tracking(1.2)
                        .foregroundStyle(Color(white: 0.55))
                    ScrollView {
                        LazyVStack(spacing: 6) {
                            ForEach(model.library) { item in
                                LibraryRow(item: item) {
                                    model.openLibraryItem(item)
                                }
                            }
                        }
                    }
                }
                .frame(maxHeight: 140)
            }

            Spacer(minLength: 0)

            VStack(spacing: 10) {
                if let progress = model.progress {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(progress.message)
                                .font(.system(size: 12))
                                .foregroundStyle(Color(white: 0.7))
                                .lineLimit(2)
                            Spacer()
                            if let eta = progress.etaSec, eta > 0 {
                                Text(formatETA(eta))
                                    .font(.system(size: 11, weight: .medium).monospacedDigit())
                                    .foregroundStyle(Color(white: 0.55))
                            }
                        }
                        if let stage = progress.stage {
                            Text(stage.uppercased())
                                .font(.system(size: 10, weight: .semibold))
                                .tracking(1.0)
                                .foregroundStyle(Color(white: 0.4))
                        }
                        ProgressView(value: progress.fraction)
                            .tint(Color(red: 0.24, green: 0.42, blue: 0.88))
                    }
                }

                HStack(spacing: 10) {
                    Button {
                        Task { await model.buildPreview() }
                    } label: {
                        Text(model.isBuilding ? "Working…" : "Preview")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(TeslaButtonStyle(filled: true))
                    .disabled(model.isBuilding || model.selectedEventIDs.isEmpty || model.isBootstrapping)

                    Button {
                        Task { await model.refineToFull() }
                    } label: {
                        Text("Refine to full")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(TeslaButtonStyle(filled: false))
                    .disabled(
                        model.isBuilding
                            || model.selectedEventIDs.isEmpty
                            || model.isBootstrapping
                            || (!model.canRefine && model.lastPLYURL == nil)
                    )
                }

                if model.isBuilding {
                    Button("Cancel") { model.cancelBuild() }
                        .buttonStyle(TeslaButtonStyle(filled: false))
                }

                Text(model.statusMessage)
                    .font(.system(size: 11))
                    .foregroundStyle(Color(white: 0.5))
                    .lineLimit(3)
            }
        }
        .padding(18)
        .background(Color(red: 0.086, green: 0.086, blue: 0.086).opacity(0.92))
    }

    private var dropZone: some View {
        VStack(spacing: 10) {
            Image(systemName: "folder.badge.plus")
                .font(.system(size: 28, weight: .light))
                .foregroundStyle(Color(red: 0.24, green: 0.42, blue: 0.88))
            Text("Drop TeslaCam folder")
                .font(.system(size: 14, weight: .medium))
            Text("SavedClips · SentryClips · RecentClips · MP4s")
                .font(.system(size: 11))
                .foregroundStyle(Color(white: 0.5))
            Button("Choose Folder…") {
                pickFolder()
            }
            .buttonStyle(TeslaButtonStyle(filled: false))
            .padding(.top, 4)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 22)
        .padding(.horizontal, 12)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .strokeBorder(
                    isDropTargeted
                        ? Color(red: 0.24, green: 0.42, blue: 0.88)
                        : Color.white.opacity(0.14),
                    style: StrokeStyle(lineWidth: 1.5, dash: [7, 5])
                )
                .background(
                    RoundedRectangle(cornerRadius: 16)
                        .fill(Color.white.opacity(isDropTargeted ? 0.06 : 0.02))
                )
        )
        .onDrop(of: [.fileURL], isTargeted: $isDropTargeted) { providers in
            handleDrop(providers)
        }
    }

    private var mainPane: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.06, green: 0.07, blue: 0.09),
                    Color(red: 0.03, green: 0.03, blue: 0.04),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            if let ply = model.lastPLYURL {
                SplatViewerRepresentable(plyURL: ply)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(16)
            } else {
                VStack(spacing: 12) {
                    Text("Viewer")
                        .font(.system(size: 13, weight: .semibold))
                        .tracking(1.0)
                        .foregroundStyle(Color(white: 0.45))
                    Text("Run Preview, then Refine to full")
                        .foregroundStyle(Color(white: 0.35))
                        .font(.system(size: 15))
                    Text("Estimate ~2 min Preview · ~10–15 min Full / 1-min event (Max-class)")
                        .font(.system(size: 11))
                        .foregroundStyle(Color(white: 0.3))
                }
            }
        }
    }

    private func formatETA(_ sec: Double) -> String {
        let s = Int(sec.rounded())
        if s < 60 { return "~\(s)s left" }
        return "~\(s / 60)m \(s % 60)s left"
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.folder, .mpeg4Movie]
        panel.message = "Select a TeslaCam folder or MP4"
        if panel.runModal() == .OK, let url = panel.url {
            Task { await model.ingest(folder: url) }
        }
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
            guard
                let data = item as? Data,
                let url = URL(dataRepresentation: data, relativeTo: nil)
            else { return }
            Task { @MainActor in
                await model.ingest(folder: url)
            }
        }
        return true
    }
}

struct LibraryRow: View {
    let item: LibraryItem
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(item.label ?? item.event_id)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(Color(white: 0.9))
                        .lineLimit(1)
                    Text("\(item.tier.capitalized) · \(item.trainer ?? "?") · \(item.num_gaussians.map { "\($0) gs" } ?? "")")
                        .font(.system(size: 10))
                        .foregroundStyle(Color(white: 0.45))
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.system(size: 10))
                    .foregroundStyle(Color(white: 0.35))
            }
            .padding(8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.white.opacity(0.04))
            )
        }
        .buttonStyle(.plain)
    }
}

struct EventRow: View {
    let event: DashcamEventDTO
    let selected: Bool
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(
                        selected
                            ? Color(red: 0.24, green: 0.42, blue: 0.88)
                            : Color.white.opacity(0.35)
                    )
                VStack(alignment: .leading, spacing: 4) {
                    Text(event.label)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(Color(white: 0.95))
                    Text(
                        "\(Int(event.duration_sec))s · \(event.cameras.joined(separator: ", "))"
                    )
                    .font(.system(size: 11))
                    .foregroundStyle(Color(white: 0.5))
                    if let loc = event.location_label {
                        Text(loc)
                            .font(.system(size: 11))
                            .foregroundStyle(Color(white: 0.45))
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.white.opacity(selected ? 0.08 : 0.03))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .strokeBorder(Color.white.opacity(0.08), lineWidth: 1)
                    )
            )
        }
        .buttonStyle(.plain)
    }
}

struct TeslaButtonStyle: ButtonStyle {
    var filled: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .semibold))
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(
                        filled
                            ? Color(red: 0.24, green: 0.42, blue: 0.88)
                                .opacity(configuration.isPressed ? 0.85 : 1)
                            : Color.white.opacity(configuration.isPressed ? 0.1 : 0.06)
                    )
            )
            .foregroundStyle(Color.white)
    }
}
