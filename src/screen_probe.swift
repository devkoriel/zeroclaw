#!/usr/bin/env swift
// ZeroClaw Hybrid Programmatic Grounding — Tier 1: Swift AXAPI Probe
// Compiled on first use by screen_state.rs, cached at /tmp/zeroclaw_screen_probe_*
//
// Reads the frontmost application's UI element tree via macOS Accessibility API
// and outputs a JSON ScreenState to stdout.

import Foundation
import ApplicationServices
import AppKit

// MARK: - Data types (match Rust ScreenState schema)

struct BBox: Codable {
    let x: Int
    let y: Int
    let w: Int
    let h: Int
}

struct UIElement: Codable {
    let id: String
    let role: String
    let name: String?
    let value: String?
    let bbox: BBox?
    let interactable: Bool
}

struct ScreenState: Codable {
    let status: String
    let source: String
    let app_name: String?
    let window_title: String?
    let elements: [UIElement]
}

// MARK: - Accessibility helpers

func isTrusted() -> Bool {
    let opts = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: false] as CFDictionary
    return AXIsProcessTrustedWithOptions(opts)
}

func getAttr(_ el: AXUIElement, _ attr: String) -> AnyObject? {
    var value: AnyObject?
    let err = AXUIElementCopyAttributeValue(el, attr as CFString, &value)
    return err == .success ? value : nil
}

func getStringAttr(_ el: AXUIElement, _ attr: String) -> String? {
    guard let val = getAttr(el, attr) else { return nil }
    if let s = val as? String, !s.isEmpty { return s }
    return nil
}

func getBoolAttr(_ el: AXUIElement, _ attr: String) -> Bool? {
    guard let val = getAttr(el, attr) else { return nil }
    if let n = val as? NSNumber { return n.boolValue }
    return nil
}

func getFrame(_ el: AXUIElement) -> BBox? {
    var posVal: AnyObject?
    var sizeVal: AnyObject?
    guard AXUIElementCopyAttributeValue(el, kAXPositionAttribute as CFString, &posVal) == .success,
          AXUIElementCopyAttributeValue(el, kAXSizeAttribute as CFString, &sizeVal) == .success
    else { return nil }

    var point = CGPoint.zero
    var size = CGSize.zero
    guard AXValueGetValue(posVal as! AXValue, .cgPoint, &point),
          AXValueGetValue(sizeVal as! AXValue, .cgSize, &size)
    else { return nil }

    let w = Int(size.width)
    let h = Int(size.height)
    // Skip invisible / zero-size elements
    guard w > 0 && h > 0 else { return nil }

    return BBox(x: Int(point.x), y: Int(point.y), w: w, h: h)
}

// MARK: - Role classification

let interactableRoles: Set<String> = [
    "AXButton", "AXTextField", "AXTextArea", "AXCheckBox",
    "AXRadioButton", "AXPopUpButton", "AXComboBox", "AXSlider",
    "AXLink", "AXMenuItem", "AXMenuBarItem", "AXTab", "AXTabGroup",
    "AXDisclosureTriangle", "AXIncrementor", "AXColorWell",
    "AXToolbar", "AXList", "AXOutline", "AXTable", "AXScrollBar",
    "AXCell", "AXRow", "AXSwitch", "AXToggle"
]

// Roles we skip entirely (container-only, no useful info for the agent)
let skipRoles: Set<String> = [
    "AXGroup", "AXSplitGroup", "AXSplitter", "AXScrollArea",
    "AXLayoutArea", "AXLayoutItem", "AXMatte", "AXRuler",
    "AXRulerMarker", "AXGrowArea", "AXUnknown"
]

// MARK: - Tree traversal

var elementCounter = 0
let maxDepth = 15
let maxElements = 500

func traverse(_ el: AXUIElement, depth: Int, elements: inout [UIElement]) {
    guard depth < maxDepth, elements.count < maxElements else { return }

    let role = getStringAttr(el, kAXRoleAttribute as String) ?? "AXUnknown"

    // Get name from title, description, or help text
    let name = getStringAttr(el, kAXTitleAttribute as String)
        ?? getStringAttr(el, kAXDescriptionAttribute as String)
        ?? getStringAttr(el, kAXHelpAttribute as String)

    // Get value (text content, checkbox state, etc.)
    var value: String? = nil
    if let raw = getAttr(el, kAXValueAttribute as String) {
        if let s = raw as? String, !s.isEmpty {
            // Truncate very long values (e.g. text editor content)
            value = s.count > 200 ? String(s.prefix(197)) + "..." : s
        } else if let n = raw as? NSNumber {
            value = n.stringValue
        }
    }

    let bbox = getFrame(el)
    let interactable = interactableRoles.contains(role)

    // Include element if it has useful info (skip noise)
    let hasName = name != nil
    let hasValue = value != nil
    let shouldSkip = skipRoles.contains(role) && !hasName && !hasValue

    if !shouldSkip && (hasName || hasValue || interactable) && bbox != nil {
        elementCounter += 1
        elements.append(UIElement(
            id: "ax_\(elementCounter)",
            role: role,
            name: name,
            value: value,
            bbox: bbox,
            interactable: interactable
        ))
    }

    // Traverse children
    guard let children = getAttr(el, kAXChildrenAttribute as String) as? [AXUIElement] else { return }
    for child in children {
        traverse(child, depth: depth + 1, elements: &elements)
    }
}

// MARK: - Main

func outputJSON(_ state: ScreenState) {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    guard let data = try? encoder.encode(state),
          let json = String(data: data, encoding: .utf8)
    else {
        print("{\"status\":\"error_encoding\",\"source\":\"swift_axapi\",\"elements\":[]}")
        return
    }
    print(json)
}

// 1. Check Accessibility permission
guard isTrusted() else {
    outputJSON(ScreenState(
        status: "error_no_accessibility",
        source: "swift_axapi",
        app_name: nil,
        window_title: nil,
        elements: []
    ))
    exit(1)
}

// 2. Get frontmost application
guard let frontApp = NSWorkspace.shared.frontmostApplication else {
    outputJSON(ScreenState(
        status: "error_no_frontmost_app",
        source: "swift_axapi",
        app_name: nil,
        window_title: nil,
        elements: []
    ))
    exit(1)
}

let pid = frontApp.processIdentifier
let appElement = AXUIElementCreateApplication(pid)
let appName = frontApp.localizedName

// 3. Get window title from the first window
var windowTitle: String? = nil
if let windows = getAttr(appElement, kAXWindowsAttribute as String) as? [AXUIElement],
   let firstWindow = windows.first {
    windowTitle = getStringAttr(firstWindow, kAXTitleAttribute as String)
}

// 4. Traverse UI element tree
var elements: [UIElement] = []
traverse(appElement, depth: 0, elements: &elements)

// 5. Output result
outputJSON(ScreenState(
    status: "ok",
    source: "swift_axapi",
    app_name: appName,
    window_title: windowTitle,
    elements: elements
))
