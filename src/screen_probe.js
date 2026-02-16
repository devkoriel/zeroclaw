// ZeroClaw Hybrid Programmatic Grounding — Tier 2: JXA System Events Probe
// Run via: osascript -l JavaScript screen_probe.js
//
// Uses System Events to traverse the frontmost application's UI element tree.
// Slower than Swift AXAPI (uses Apple Events IPC per property) but always available.

ObjC.import("Foundation");
ObjC.import("AppKit");

var elementId = 0;
var MAX_DEPTH = 10;
var MAX_ELEMENTS = 300;

var interactableRoles = {
  AXButton: true,
  AXTextField: true,
  AXTextArea: true,
  AXCheckBox: true,
  AXRadioButton: true,
  AXPopUpButton: true,
  AXComboBox: true,
  AXSlider: true,
  AXLink: true,
  AXMenuItem: true,
  AXMenuBarItem: true,
  AXTab: true,
  AXDisclosureTriangle: true,
  AXIncrementor: true,
  AXCell: true,
  AXRow: true,
  AXSwitch: true,
  AXToggle: true,
};

var skipRoles = {
  AXGroup: true,
  AXSplitGroup: true,
  AXSplitter: true,
  AXScrollArea: true,
  AXLayoutArea: true,
  AXLayoutItem: true,
  AXUnknown: true,
};

function safeGet(fn) {
  try {
    var v = fn();
    return v === null || v === undefined ? null : v;
  } catch (e) {
    return null;
  }
}

function traverseElement(el, depth, elements) {
  if (depth >= MAX_DEPTH || elements.length >= MAX_ELEMENTS) return;

  var role = safeGet(function () {
    return el.role();
  });
  if (!role) return;

  var name = safeGet(function () {
    return el.name();
  });
  if (!name)
    name = safeGet(function () {
      return el.description();
    });

  var value = safeGet(function () {
    var v = el.value();
    if (typeof v === "string") {
      return v.length > 200 ? v.substring(0, 197) + "..." : v;
    }
    if (typeof v === "number") return String(v);
    return null;
  });

  var bbox = null;
  try {
    var pos = el.position();
    var sz = el.size();
    if (pos && sz && sz[0] > 0 && sz[1] > 0) {
      bbox = { x: pos[0], y: pos[1], w: sz[0], h: sz[1] };
    }
  } catch (e) {}

  var interactable = !!interactableRoles[role];
  // JXA/System Events assigns generic names ("group", "scroll area") to containers.
  // These are noise — treat them as no name.
  var genericNames = {
    group: 1,
    "split group": 1,
    "scroll area": 1,
    "layout area": 1,
    "layout item": 1,
    splitter: 1,
    unknown: 1,
    matte: 1,
    "grow area": 1,
  };
  if (name && genericNames[name.toLowerCase()]) name = null;

  var hasName = name && name.length > 0;
  var hasValue = value && value.length > 0;
  var shouldSkip = !!skipRoles[role] && !hasName && !hasValue;

  if (!shouldSkip && (hasName || hasValue || interactable) && bbox) {
    elementId++;
    elements.push({
      id: "jxa_" + elementId,
      role: role,
      name: name || null,
      value: value || null,
      bbox: bbox,
      interactable: interactable,
    });
  }

  // Traverse children
  try {
    var children = el.uiElements();
    for (
      var i = 0;
      i < children.length && elements.length < MAX_ELEMENTS;
      i++
    ) {
      traverseElement(children[i], depth + 1, elements);
    }
  } catch (e) {}
}

function main() {
  var result = {
    status: "ok",
    source: "jxa_system_events",
    app_name: null,
    window_title: null,
    elements: [],
  };

  try {
    var se = Application("System Events");
    se.includeStandardAdditions = true;

    var procs = se.processes.whose({ frontmost: true });
    if (procs.length === 0) {
      result.status = "error_no_frontmost_process";
      writeStdout(JSON.stringify(result));
      return;
    }

    var frontProc = procs[0];
    result.app_name = safeGet(function () {
      return frontProc.name();
    });

    var windows = safeGet(function () {
      return frontProc.windows();
    });
    if (windows && windows.length > 0) {
      result.window_title = safeGet(function () {
        return windows[0].name();
      });
      traverseElement(windows[0], 0, result.elements);
    }
  } catch (e) {
    result.status = "error";
  }

  writeStdout(JSON.stringify(result));
}

function writeStdout(str) {
  var data = $.NSString.alloc
    .initWithString(str)
    .dataUsingEncoding($.NSUTF8StringEncoding);
  $.NSFileHandle.fileHandleWithStandardOutput.writeData(data);
}

main();
