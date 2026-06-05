// Run this once to discover all available IOKit HID sensor keys on this machine.
// Usage: swiftc discover_sensors.swift -o discover_sensors && ./discover_sensors
import Foundation
import IOKit.hid

let client = IOHIDEventSystemClientCreateSimpleClient(kCFAllocatorDefault)

let services = IOHIDEventSystemClientCopyServices(client) as? [IOHIDServiceClient] ?? []

var found: [(String, Double)] = []

for service in services {
    guard let nameAny = IOHIDServiceClientCopyProperty(service, "Product" as CFString),
          let name = nameAny as? String else { continue }

    // Try to get a thermal/power event
    let event = IOHIDServiceClientCopySupportedEvents(service)
    let _ = event  // just checking existence

    // Try reading temperature value via IOHIDEventSystemClientCopyEvent
    let tempEvent = IOHIDServiceClientCopyEvent(service, Int32(kIOHIDEventTypeTemperature), 0, 0)
    if let tempEvent = tempEvent {
        let value = IOHIDEventGetFloatValue(tempEvent, IOHIDEventField(kIOHIDEventFieldTemperatureLevel))
        found.append((name, value))
    }
}

if found.isEmpty {
    print("No temperature sensors found via HID events. Trying SMC-style IOKit approach...")
} else {
    print("Found \(found.count) temperature sensors:")
    for (name, val) in found.sorted(by: { $0.0 < $1.0 }) {
        print(String(format: "  %-60s %.1f°C", name, val))
    }
}
