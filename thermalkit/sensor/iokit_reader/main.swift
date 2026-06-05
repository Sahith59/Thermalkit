// Thermal sensor reader for Apple Silicon via IOKit HID event system.
// Requires IOHIDEventSystemClientCreate(type=0) — admin-level HID client.
// This works without sudo; the "admin" type here refers to HID access level,
// not OS privilege level.
//
// macOS 26.5 SDK: CopyServices, IOHIDServiceClientCopyProperty are public.
// Create, SetMatching, ServiceClientCopyEvent, EventGetFloatValue are private.
//
// Sensor layout on Mac16,8 (M4 Pro):
//   PMU tdie1..tdie14 — CPU/SoC die temperatures
//   PMU tdev1..tdev8  — device-area temps (GPU-adjacent; may be 0 at idle)
//
// Output: one JSON line per second on stdout.

import Foundation
import IOKit.hid

// MARK: - Private bindings

// IOHIDEventSystemClientCreate(type: Int32) — type 0 = admin (enables event reads)
@_silgen_name("IOHIDEventSystemClientCreate")
func _createEventSystemClient(_ type: Int32) -> IOHIDEventSystemClient

// IOHIDEventSystemClientSetMatching — filters which services are returned
@_silgen_name("IOHIDEventSystemClientSetMatching")
func _setMatching(_ client: IOHIDEventSystemClient, _ matching: CFDictionary)

@_silgen_name("IOHIDServiceClientCopyEvent")
func _copyEvent(
    _ service: IOHIDServiceClient,
    _ type: Int32,
    _ options: Int32,
    _ timeout: Int32
) -> AnyObject?

@_silgen_name("IOHIDEventGetFloatValue")
func _getFloatValue(_ event: AnyObject, _ field: Int32) -> Double

// kIOHIDEventTypeTemperature = 15; field = (type << 16) | 0
private let kTempType:  Int32 = 15
private let kTempField: Int32 = (15 << 16) | 0

// MARK: - Sensor classification (M4 Pro, Mac16,8)

private let kCPUNames: Set<String> = [
    "PMU tdie1",  "PMU tdie2",  "PMU tdie3",  "PMU tdie4",
    "PMU tdie5",  "PMU tdie6",  "PMU tdie7",  "PMU tdie8",
    "PMU tdie9",  "PMU tdie10", "PMU tdie11", "PMU tdie12",
    "PMU tdie13", "PMU tdie14"
]
private let kGPUNames: Set<String> = [
    "PMU tdev1", "PMU tdev2", "PMU tdev3", "PMU tdev4",
    "PMU tdev5", "PMU tdev6", "PMU tdev7", "PMU tdev8"
]

// MARK: - Snapshot

struct ThermalSnapshot {
    var cpuTempC: Double
    var gpuTempC: Double
    var timestamp: Double
}

func snapshot() -> ThermalSnapshot {
    let client = _createEventSystemClient(0)  // 0 = admin type, enables event reads

    // Filter to usage page 0xFF00 / usage 5 — thermal sensors only
    let matching: [String: Any] = ["PrimaryUsagePage": 0xFF00, "PrimaryUsage": 5]
    _setMatching(client, matching as CFDictionary)

    guard let services = IOHIDEventSystemClientCopyServices(client) as? [IOHIDServiceClient]
    else { return ThermalSnapshot(cpuTempC: 0, gpuTempC: 0, timestamp: Date().timeIntervalSince1970) }

    var cpuReadings: [Double] = []
    var gpuReadings: [Double] = []
    var seen = Set<String>()

    for service in services {
        guard
            let nameCF = IOHIDServiceClientCopyProperty(service, "Product" as CFString),
            let name   = nameCF as? String,
            !seen.contains(name)
        else { continue }
        seen.insert(name)

        guard let event = _copyEvent(service, kTempType, 0, 0) else { continue }
        let temp = _getFloatValue(event, kTempField)
        guard temp > 0.0 && temp < 150.0 else { continue }

        if kCPUNames.contains(name) { cpuReadings.append(temp) }
        if kGPUNames.contains(name) { gpuReadings.append(temp) }
    }

    return ThermalSnapshot(
        cpuTempC: cpuReadings.max() ?? 0.0,
        gpuTempC: gpuReadings.max() ?? 0.0,
        timestamp: Date().timeIntervalSince1970
    )
}

// MARK: - Main loop

signal(SIGTERM) { _ in exit(0) }
signal(SIGINT)  { _ in exit(0) }

while true {
    let start = Date()
    let s = snapshot()

    let output: [String: Any] = [
        "ts":         s.timestamp,
        "cpu_temp_c": s.cpuTempC,
        "gpu_temp_c": s.gpuTempC
    ]

    if let data = try? JSONSerialization.data(withJSONObject: output, options: .sortedKeys),
       let line = String(data: data, encoding: .utf8) {
        print(line)
        fflush(stdout)
    }

    let remaining = 1.0 - Date().timeIntervalSince(start)
    if remaining > 0 { Thread.sleep(forTimeInterval: remaining) }
}
