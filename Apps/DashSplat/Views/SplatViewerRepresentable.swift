import SwiftUI
import WebKit

/// WKWebView host for the splat viewer (orbit + ego car on SEI path).
struct SplatViewerRepresentable: NSViewRepresentable {
    let plyURL: URL
    var showEgoCar: Bool = true

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
        let js = "window.__dashsplatSetEgoVisible && window.__dashsplatSetEgoVisible(\(showEgoCar ? "true" : "false"))"
        webView.evaluateJavaScript(js, completionHandler: nil)
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
        let jobDir = plyURL.deletingLastPathComponent()
        let egoPath = jobDir.appendingPathComponent("ego/ego_path.json")
        let egoGlb = jobDir.appendingPathComponent("ego/sedan_crossover.glb")
        var comps = URLComponents(url: viewer, resolvingAgainstBaseURL: false)
        var items = [URLQueryItem(name: "ply", value: plyURL.absoluteString)]
        if FileManager.default.fileExists(atPath: egoPath.path) {
            items.append(URLQueryItem(name: "egoPath", value: egoPath.absoluteString))
        }
        if FileManager.default.fileExists(atPath: egoGlb.path) {
            items.append(URLQueryItem(name: "egoGlb", value: egoGlb.absoluteString))
        }
        items.append(URLQueryItem(name: "showEgo", value: showEgoCar ? "1" : "0"))
        comps?.queryItems = items
        // Allow reading the whole job directory (PLY + ego/)
        if let url = comps?.url {
            webView.loadFileURL(url, allowingReadAccessTo: jobDir)
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
