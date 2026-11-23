import SwiftUI
import WebKit

/// WKWebView host for the Spark / anti-aliased splat viewer (orbit + drag).
struct SplatViewerRepresentable: NSViewRepresentable {
    let plyURL: URL

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.preferences.setValue(true, forKey: "allowFileAccessFromFileURLs")
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.setValue(false, forKey: "drawsBackground")
        context.coordinator.webView = webView
        load(webView)
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        if context.coordinator.lastPLY != plyURL {
            context.coordinator.lastPLY = plyURL
            load(webView)
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    final class Coordinator {
        var webView: WKWebView?
        var lastPLY: URL?
    }

    private func load(_ webView: WKWebView) {
        guard let viewer = Bundle.main.url(forResource: "viewer", withExtension: "html", subdirectory: "Resources")
            ?? Bundle.main.url(forResource: "viewer", withExtension: "html")
            ?? repoViewerURL()
        else {
            webView.loadHTMLString("<html><body style='background:#0a0a0a;color:#888;font-family:sans-serif;padding:2rem'>Viewer missing</body></html>", baseURL: nil)
            return
        }
        var comps = URLComponents(url: viewer, resolvingAgainstBaseURL: false)
        comps?.queryItems = [URLQueryItem(name: "ply", value: plyURL.absoluteString)]
        if let url = comps?.url {
            webView.loadFileURL(url, allowingReadAccessTo: plyURL.deletingLastPathComponent())
        }
    }

    private func repoViewerURL() -> URL? {
        let url = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("viewer/viewer.html")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }
}
