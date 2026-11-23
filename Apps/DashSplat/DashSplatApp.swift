import SwiftUI

@main
struct DashSplatApp: App {
    @StateObject private var appModel = AppModel()

    var body: some Scene {
        WindowGroup("DashSplat") {
            ContentView()
                .environmentObject(appModel)
                .frame(minWidth: 1100, minHeight: 720)
        }
        .defaultSize(width: 1280, height: 800)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
}
